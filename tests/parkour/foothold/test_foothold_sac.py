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


def test_sac_skips_until_warmup_then_reports_updates():
    module = _load_sac_module()
    config = module.FootholdSACConfig(
        obs_dim=3,
        hidden_dims=(8, 8),
        batch_size=4,
        warmup_events=4,
        updates_per_rollout=1,
    )
    sac = module.FootholdSAC(config, device="cpu")
    obs = torch.zeros(4, 3)
    actions = torch.zeros(4, 2)
    rewards = torch.ones(4)
    next_obs = torch.ones(4, 3)
    dones = torch.zeros(4, dtype=torch.bool)

    assert sac.update()["sac_update_count"] == 0.0
    sac.observe(obs, actions, rewards, next_obs, dones)
    diagnostics = sac.update()
    assert diagnostics["sac_update_count"] == 1.0
    assert diagnostics["replay_size"] == 4.0


def test_sac_rejects_nonfinite_replay_and_keeps_parameters_finite():
    module = _load_sac_module()
    config = module.FootholdSACConfig(
        obs_dim=2,
        hidden_dims=(8,),
        batch_size=2,
        warmup_events=0,
        updates_per_rollout=1,
    )
    sac = module.FootholdSAC(config, device="cpu")
    sac.observe(
        torch.tensor([[0.0, float("nan")], [0.0, float("nan")]]),
        torch.zeros(2, 2),
        torch.zeros(2),
        torch.zeros(2, 2),
        torch.zeros(2, dtype=torch.bool),
    )
    diagnostics = sac.update()
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
        updates_per_rollout=1,
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
    assert sac.update()["sac_update_count"] == 1.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, planner.parameters())
    )
