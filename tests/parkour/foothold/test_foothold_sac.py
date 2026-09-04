import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


def _load_sac_module():
    repo_root = Path(__file__).resolve().parents[3]
    learning_dir = repo_root / "source/instinctlab/instinctlab/learning"
    package = ModuleType("instinctlab.learning")
    package.__path__ = [str(learning_dir)]
    sys.modules["instinctlab.learning"] = package
    replay_path = learning_dir / "foothold_sac_replay.py"
    replay_spec = importlib.util.spec_from_file_location(
        "instinctlab.learning.foothold_sac_replay", replay_path
    )
    assert replay_spec is not None
    assert replay_spec.loader is not None
    replay_module = importlib.util.module_from_spec(replay_spec)
    sys.modules[replay_spec.name] = replay_module
    replay_spec.loader.exec_module(replay_module)
    module_path = learning_dir / "foothold_sac.py"
    spec = importlib.util.spec_from_file_location(
        "instinctlab.learning.foothold_sac", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sac_backup_uses_minimum_target_q_and_terminal_mask():
    module = _load_sac_module()
    target = module.sac_backup_target(
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([False, True]),
        target_q1=torch.tensor([3.0, 30.0]),
        target_q2=torch.tensor([4.0, 40.0]),
        next_log_prob=torch.tensor([0.5, 0.5]),
        alpha=torch.tensor(0.2),
        gamma=0.99,
    )

    torch.testing.assert_close(
        target,
        torch.tensor([1.0 + 0.99 * (3.0 - 0.2 * 0.5), 2.0]),
    )


def test_polyak_update_moves_target_toward_source():
    module = _load_sac_module()
    source = torch.nn.Linear(2, 1, bias=False)
    target = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(2.0)
        target.weight.zero_()

    module.polyak_update(target, source, tau=0.25)
    torch.testing.assert_close(target.weight, torch.full_like(target.weight, 0.5))


def test_sac_uses_event_scaled_update_credit_and_respects_cap():
    module = _load_sac_module()
    config = module.FootholdSACConfig(
        obs_dim=3,
        hidden_dims=(8, 8),
        batch_size=2,
        warmup_events=2,
        target_sample_ratio=1.0,
        max_updates_per_rollout=2,
    )
    sac = module.FootholdSAC(config, device="cpu")
    obs = torch.zeros(6, 3)
    actions = torch.zeros(6, 2)
    rewards = torch.ones(6)
    next_obs = torch.ones(6, 3)
    dones = torch.zeros(6, dtype=torch.bool)

    assert sac.update(new_event_count=0)["sac_update_count"] == 0.0
    sac.observe(obs, actions, rewards, next_obs, dones)
    # Three new events at batch size two produce 1.5 updates of credit; the
    # fractional remainder must be retained for the next rollout.
    diagnostics = sac.update(new_event_count=3)
    assert diagnostics["sac_update_count"] == 1.0
    assert diagnostics["sac_requested_update_count"] == 1.0
    assert diagnostics["replay_size"] == 6.0
    assert diagnostics["sac_replay_reward_mean"] == pytest.approx(1.0)
    assert diagnostics["sac_replay_reward_min"] == pytest.approx(1.0)
    assert diagnostics["sac_replay_reward_max"] == pytest.approx(1.0)
    assert "sac_target_q_mean" in diagnostics
    assert "sac_q_abs_max" in diagnostics
    torch.testing.assert_close(
        torch.tensor(diagnostics["sac_update_credit"]),
        torch.tensor(0.5),
    )

    # The retained half-credit combines with three new events to request two
    # updates, but the configured cap is still respected.
    diagnostics = sac.update(new_event_count=3)
    assert diagnostics["sac_update_count"] == 2.0
    assert diagnostics["sac_dropped_update_count"] == 0.0


def test_sac_drops_update_backlog_above_per_rollout_cap():
    module = _load_sac_module()
    config = module.FootholdSACConfig(
        obs_dim=2,
        hidden_dims=(8,),
        batch_size=2,
        warmup_events=0,
        target_sample_ratio=1.0,
        max_updates_per_rollout=2,
    )
    sac = module.FootholdSAC(config, device="cpu")
    sac.observe(
        torch.zeros(8, 2),
        torch.zeros(8, 2),
        torch.ones(8),
        torch.ones(8, 2),
        torch.zeros(8, dtype=torch.bool),
    )
    diagnostics = sac.update(new_event_count=8)
    assert diagnostics["sac_update_count"] == 2.0
    assert diagnostics["sac_requested_update_count"] == 4.0
    assert diagnostics["sac_dropped_update_count"] == 2.0
    assert diagnostics["sac_update_credit"] == 2.0


def test_sac_rejects_nonfinite_replay_and_keeps_parameters_finite():
    module = _load_sac_module()
    config = module.FootholdSACConfig(
        obs_dim=2,
        hidden_dims=(8,),
        batch_size=2,
        warmup_events=0,
        target_sample_ratio=1.0,
        max_updates_per_rollout=1,
    )
    sac = module.FootholdSAC(config, device="cpu")
    sac.observe(
        torch.tensor([[0.0, float("nan")], [0.0, float("nan")]]),
        torch.zeros(2, 2),
        torch.zeros(2),
        torch.zeros(2, 2),
        torch.zeros(2, dtype=torch.bool),
    )
    diagnostics = sac.update(new_event_count=2)
    assert diagnostics["sac_update_count"] == 0.0
    assert diagnostics["sac_skipped_update_count"] == 1.0


def test_sac_can_encode_raw_replay_into_compact_planner_features():
    module = _load_sac_module()
    planner = torch.nn.Linear(2, 2)
    log_std = torch.nn.Parameter(torch.zeros(2))

    def distribution(features):
        return module.Normal(planner(features), log_std.exp().expand(features.shape[0], -1))

    config = module.FootholdSACConfig(
        obs_dim=2,
        replay_obs_dim=4,
        hidden_dims=(8,),
        batch_size=2,
        warmup_events=0,
        target_sample_ratio=1.0,
        max_updates_per_rollout=1,
    )
    sac = module.FootholdSAC(
        config,
        device="cpu",
        actor_distribution_fn=distribution,
        actor_parameters=tuple(planner.parameters()) + (log_std,),
        feature_fn=lambda obs: obs[..., :2],
    )
    before = [parameter.detach().clone() for parameter in planner.parameters()]
    sac.observe(
        torch.randn(2, 4),
        torch.zeros(2, 2),
        torch.ones(2),
        torch.randn(2, 4),
        torch.zeros(2, dtype=torch.bool),
    )
    assert sac.update(new_event_count=2)["sac_update_count"] == 1.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, planner.parameters())
    )


def test_event_accumulator_closes_only_at_next_planner_event():
    module = _load_sac_module()
    accumulator = module.PlannerEventAccumulator(
        num_envs=1,
        obs_dim=2,
        action_dim=2,
        device="cpu",
    )
    recorded = []

    def record(obs, actions, rewards, next_obs, dones):
        recorded.append((obs.clone(), actions.clone(), rewards.clone(), next_obs.clone(), dones.clone()))

    accumulator.process_step(
        observations=torch.tensor([[0.0, 0.0]]),
        actions=torch.tensor([[0.1, 0.2]]),
        rewards=torch.tensor([1.0]),
        next_observations=torch.tensor([[1.0, 1.0]]),
        dones=torch.tensor([False]),
        event_mask=torch.tensor([True]),
        record=record,
    )
    accumulator.process_step(
        observations=torch.tensor([[1.0, 1.0]]),
        actions=torch.tensor([[0.0, 0.0]]),
        rewards=torch.tensor([0.5]),
        next_observations=torch.tensor([[2.0, 2.0]]),
        dones=torch.tensor([False]),
        event_mask=torch.tensor([False]),
        record=record,
    )
    assert recorded == []
    accumulator.process_step(
        observations=torch.tensor([[2.0, 2.0]]),
        actions=torch.tensor([[0.3, 0.4]]),
        rewards=torch.tensor([2.0]),
        next_observations=torch.tensor([[3.0, 3.0]]),
        dones=torch.tensor([False]),
        event_mask=torch.tensor([True]),
        record=record,
    )

    assert len(recorded) == 1
    obs, actions, rewards, next_obs, dones = recorded[0]
    torch.testing.assert_close(obs, torch.tensor([[0.0, 0.0]]))
    torch.testing.assert_close(actions, torch.tensor([[0.1, 0.2]]))
    torch.testing.assert_close(rewards, torch.tensor([1.5]))
    torch.testing.assert_close(next_obs, torch.tensor([[2.0, 2.0]]))
    torch.testing.assert_close(dones, torch.tensor([False]))


def test_event_accumulator_closes_pending_event_as_terminal_on_done():
    module = _load_sac_module()
    accumulator = module.PlannerEventAccumulator(
        num_envs=1,
        obs_dim=2,
        action_dim=2,
        device="cpu",
    )
    recorded = []
    accumulator.process_step(
        observations=torch.zeros(1, 2),
        actions=torch.ones(1, 2),
        rewards=torch.tensor([1.0]),
        next_observations=torch.ones(1, 2),
        dones=torch.tensor([True]),
        event_mask=torch.tensor([True]),
        record=lambda *values: recorded.append(values),
    )

    assert len(recorded) == 1
    _, _, rewards, next_obs, dones = recorded[0]
    torch.testing.assert_close(rewards, torch.tensor([1.0]))
    torch.testing.assert_close(next_obs, torch.ones(1, 2))
    torch.testing.assert_close(dones, torch.tensor([True]))


def test_event_accumulator_marks_previous_event_terminal_at_reset_boundary():
    module = _load_sac_module()
    accumulator = module.PlannerEventAccumulator(
        num_envs=1,
        obs_dim=2,
        action_dim=2,
        device="cpu",
    )
    recorded = []
    accumulator.process_step(
        observations=torch.zeros(1, 2),
        actions=torch.ones(1, 2),
        rewards=torch.tensor([1.0]),
        next_observations=torch.ones(1, 2),
        dones=torch.tensor([False]),
        event_mask=torch.tensor([True]),
        record=lambda *values: recorded.append(values),
    )
    accumulator.process_step(
        observations=torch.full((1, 2), 2.0),
        actions=torch.full((1, 2), 3.0),
        rewards=torch.tensor([2.0]),
        next_observations=torch.full((1, 2), 4.0),
        dones=torch.tensor([True]),
        event_mask=torch.tensor([True]),
        record=lambda *values: recorded.append(values),
    )

    assert len(recorded) == 2
    _, _, reward, next_obs, dones = recorded[0]
    torch.testing.assert_close(reward, torch.tensor([1.0]))
    torch.testing.assert_close(next_obs, torch.full((1, 2), 4.0))
    torch.testing.assert_close(dones, torch.tensor([True]))


def test_radial_action_transform_stays_inside_unit_disk_with_correct_jacobian():
    module = _load_sac_module()
    raw = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    action = module.radial_squash(raw)
    assert torch.all(torch.linalg.vector_norm(action, dim=-1) < 1.0)
    torch.testing.assert_close(
        module.radial_squash_log_abs_det_jacobian(raw),
        -2.0 * torch.log1p(raw.square().sum(dim=-1)),
    )


def test_sac_raw_rollout_action_matches_its_bounded_sac_action():
    module = _load_sac_module()
    config = module.FootholdSACConfig(obs_dim=3, hidden_dims=(8,))
    sac = module.FootholdSAC(config, device="cpu")
    features = torch.zeros(2, 3)
    torch.manual_seed(4)
    raw_action, raw_log_prob = sac.act_raw_with_log_prob(features)
    torch.manual_seed(4)
    bounded_action, bounded_log_prob = sac.act_with_log_prob(features)
    torch.testing.assert_close(
        module.radial_squash(raw_action),
        bounded_action,
    )
    torch.testing.assert_close(raw_log_prob, bounded_log_prob)
