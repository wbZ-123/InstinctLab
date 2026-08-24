import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_ppo_module():
    repo_root = Path(__file__).resolve().parents[3]
    learning_dir = (
        repo_root
        / "source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py"
    ).parent
    package = ModuleType("instinctlab.learning")
    package.__path__ = [str(learning_dir)]
    sys.modules["instinctlab.learning"] = package

    policy_path = learning_dir / "independent_foothold_actor_critic.py"
    policy_spec = importlib.util.spec_from_file_location(
        "instinctlab.learning.independent_foothold_actor_critic",
        policy_path,
    )
    assert policy_spec is not None
    assert policy_spec.loader is not None
    policy_module = importlib.util.module_from_spec(policy_spec)
    sys.modules[policy_spec.name] = policy_module
    policy_spec.loader.exec_module(policy_module)

    storage_path = learning_dir / "foothold_rollout_storage.py"
    storage_spec = importlib.util.spec_from_file_location(
        "instinctlab.learning.foothold_rollout_storage",
        storage_path,
    )
    assert storage_spec is not None
    assert storage_spec.loader is not None
    storage_module = importlib.util.module_from_spec(storage_spec)
    sys.modules[storage_spec.name] = storage_module
    storage_spec.loader.exec_module(storage_module)

    module_path = learning_dir / "event_gated_foothold_ppo.py"
    spec = importlib.util.spec_from_file_location(
        "instinctlab.learning.event_gated_foothold_ppo",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.IndependentFootholdMoEActorCritic = (
        policy_module.IndependentFootholdMoEActorCritic
    )
    return module


def _make_algorithm(module):
    obs_format = {
        "policy": {"obs": (4,)},
        "critic": {"obs": (4,)},
        "amp_policy": {"state": (3,)},
        "amp_reference": {"state": (3,)},
    }
    actor_critic = module.IndependentFootholdMoEActorCritic(
        obs_format=obs_format,
        num_actions=31,
        motor_action_dim=29,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        foothold_hidden_dims=[8, 4],
        num_moe_experts=2,
        init_noise_std=0.9,
        num_rewards=2,
    )
    algorithm = module.EventGatedWasabiPPO(
        actor_critic=actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        schedule="adaptive",
        device="cpu",
        discriminator_kwargs={
            "hidden_sizes": [8],
            "nonlinearity": "ReLU",
        },
        motor_action_dim=29,
        execution_reward_index=0,
        foothold_reward_index=1,
        foothold_initial_std_m=(0.05, 0.05),
        foothold_min_std_m=(0.02, 0.02),
        foothold_max_std_m=(0.05, 0.05),
        foothold_reachability_radii_m=(0.42, 0.25),
        foothold_learning_rate=5.0e-4,
        foothold_desired_kl=0.01,
        foothold_kl_stop_multiplier=2.0,
    )
    return algorithm, obs_format


def _make_minibatch(module, actor_critic, *, event_mask=None):
    batch_size = 4
    if event_mask is None:
        event_mask = torch.tensor([True, False, True, False])
    old_sigma = torch.cat(
        (actor_critic.motor_std, actor_critic.foothold_std),
    ).detach().expand(batch_size, -1).clone()
    return module.FootholdMiniBatch(
        obs=torch.zeros(batch_size, 4),
        critic_obs=torch.zeros(batch_size, 4),
        actions=torch.zeros(batch_size, 31),
        values=torch.zeros(batch_size, 2),
        advantages=torch.tensor(
            [[1.0, 2.0], [1.0, 50.0], [-1.0, -2.0], [-1.0, -50.0]]
        ),
        returns=torch.zeros(batch_size, 2),
        old_actions_log_prob=torch.zeros(batch_size, 1),
        old_mu=torch.zeros(batch_size, 31),
        old_sigma=old_sigma,
        hidden_states=None,
        masks=None,
        foothold_action_event=event_mask,
    )


def test_physical_std_uses_reachability_source():
    module = _load_ppo_module()

    result = module.normalized_foothold_std(
        std_m=(0.05, 0.05),
        radii_m=(0.42, 0.25),
    )

    torch.testing.assert_close(
        result,
        torch.tensor([0.05 / 0.42, 0.05 / 0.25]),
    )


def test_full_finite_scan_cadence_is_periodic_and_keeps_first_iteration():
    module = _load_ppo_module()

    assert module.should_run_full_finite_check(0, 100)
    assert not module.should_run_full_finite_check(1, 100)
    assert not module.should_run_full_finite_check(99, 100)
    assert module.should_run_full_finite_check(100, 100)

    with pytest.raises(ValueError, match="positive"):
        module.should_run_full_finite_check(0, 0)


def test_non_event_foothold_changes_do_not_change_motor_log_probability():
    module = _load_ppo_module()
    distribution = torch.distributions.Normal(
        torch.zeros(2, 31),
        torch.ones(2, 31),
    )
    actions = torch.zeros(2, 31)
    actions[1, 29:] = torch.tensor([3.0, -4.0])

    motor_log_prob, foothold_log_prob = module.grouped_log_prob(
        distribution,
        actions,
        motor_action_dim=29,
    )

    torch.testing.assert_close(motor_log_prob[0], motor_log_prob[1])
    assert foothold_log_prob[0] != foothold_log_prob[1]


def test_planner_loss_uses_only_event_rows_and_planner_advantage():
    module = _load_ppo_module()
    common = dict(
        new_motor_log_prob=torch.zeros(2),
        old_motor_log_prob=torch.zeros(2),
        old_foothold_log_prob=torch.zeros(2),
        execution_advantage=torch.tensor([100.0, -100.0]),
        foothold_advantage=torch.tensor([2.0, 50.0]),
        event_mask=torch.tensor([True, False]),
        clip_param=0.2,
    )

    _, planner_loss_a = module.grouped_clipped_surrogates(
        new_foothold_log_prob=torch.tensor([0.1, -8.0]),
        **common,
    )
    _, planner_loss_b = module.grouped_clipped_surrogates(
        new_foothold_log_prob=torch.tensor([0.1, 8.0]),
        **common,
    )

    torch.testing.assert_close(planner_loss_a, planner_loss_b)


def test_no_event_minibatch_has_zero_planner_loss_with_gradient_path():
    module = _load_ppo_module()
    values = torch.tensor([3.0, 7.0], requires_grad=True)

    result = module.event_masked_mean(
        values,
        torch.tensor([False, False]),
    )

    assert result.item() == 0.0
    result.backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))


def test_algorithm_uses_foothold_storage_and_physical_initial_std():
    module = _load_ppo_module()
    algorithm, obs_format = _make_algorithm(module)

    algorithm.init_storage(
        num_envs=2,
        num_transitions_per_env=1,
        obs_format=obs_format,
        num_actions=31,
        num_rewards=2,
    )

    assert isinstance(algorithm.storage, module.FootholdRolloutStorage)
    assert isinstance(algorithm.transition, module.FootholdTransition)
    torch.testing.assert_close(
        algorithm.actor_critic.foothold_std.detach(),
        torch.tensor([0.05 / 0.42, 0.05 / 0.25]),
    )


def test_motor_and_foothold_optimizers_are_disjoint_and_exhaustive():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)

    motor_ids = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    foothold_ids = {
        id(parameter)
        for group in algorithm.foothold_optimizer.param_groups
        for parameter in group["params"]
    }
    all_ids = {id(parameter) for parameter in algorithm.actor_critic.parameters()}

    assert motor_ids.isdisjoint(foothold_ids)
    assert motor_ids | foothold_ids == all_ids


def test_physical_foothold_std_bounds_do_not_modify_motor_std():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)
    motor_before = algorithm.actor_critic.motor_std.detach().clone()

    with torch.no_grad():
        algorithm.actor_critic.foothold_std.copy_(
            torch.tensor([0.001, 1.0])
        )
    algorithm._clip_foothold_std_to_physical_bounds()

    torch.testing.assert_close(
        algorithm.actor_critic.foothold_std.detach(),
        torch.tensor([0.02 / 0.42, 0.05 / 0.25]),
    )
    torch.testing.assert_close(
        algorithm.actor_critic.motor_std.detach(),
        motor_before,
    )


def test_policy_gradient_step_can_update_each_group_in_isolation():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)
    policy = algorithm.actor_critic

    motor_before = [parameter.detach().clone() for parameter in policy.motor_parameters()]
    foothold_before = [
        parameter.detach().clone() for parameter in policy.foothold_parameters()
    ]
    observations = torch.randn(4, 4)
    means = policy.act_inference(observations)
    algorithm._policy_gradient_step(
        motor_loss=means[..., :29].square().mean(),
        foothold_loss=means[..., 29:].square().mean(),
        run_foothold=False,
        average_stats={"motor_grad_norm": 0.0, "foothold_grad_norm": 0.0},
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(motor_before, policy.motor_parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(foothold_before, policy.foothold_parameters())
    )

    motor_before = [parameter.detach().clone() for parameter in policy.motor_parameters()]
    foothold_before = [
        parameter.detach().clone() for parameter in policy.foothold_parameters()
    ]
    means = policy.act_inference(observations)
    algorithm._policy_gradient_step(
        motor_loss=means[..., :29].sum() * 0.0,
        foothold_loss=means[..., 29:].square().mean(),
        run_foothold=True,
        run_motor=False,
        average_stats={"motor_grad_norm": 0.0, "foothold_grad_norm": 0.0},
    )

    assert all(
        torch.equal(before, after)
        for before, after in zip(motor_before, policy.motor_parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(foothold_before, policy.foothold_parameters())
    )


def test_policy_gradient_step_steps_each_enabled_optimizer_once():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)
    observations = torch.randn(4, 4)
    means = algorithm.actor_critic.act_inference(observations)
    motor_step_count = 0
    foothold_step_count = 0
    original_motor_step = algorithm.optimizer.step
    original_foothold_step = algorithm.foothold_optimizer.step

    def count_motor_step(*args, **kwargs):
        nonlocal motor_step_count
        motor_step_count += 1
        return original_motor_step(*args, **kwargs)

    def count_foothold_step(*args, **kwargs):
        nonlocal foothold_step_count
        foothold_step_count += 1
        return original_foothold_step(*args, **kwargs)

    algorithm.optimizer.step = count_motor_step
    algorithm.foothold_optimizer.step = count_foothold_step

    algorithm._policy_gradient_step(
        motor_loss=means[..., :29].square().mean(),
        foothold_loss=means[..., 29:].square().mean(),
        run_foothold=True,
        average_stats={},
    )

    assert motor_step_count == 1
    assert foothold_step_count == 1


def test_excess_foothold_kl_blocks_only_foothold_update_and_lr():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)
    motor_lr_before = algorithm.learning_rate
    foothold_lr_before = algorithm.foothold_learning_rate

    assert not algorithm._foothold_update_allowed(
        torch.tensor(0.021),
        event_count=torch.tensor(1.0),
    )
    algorithm._adjust_foothold_learning_rate_once(torch.tensor(0.021))

    assert algorithm.learning_rate == motor_lr_before
    assert algorithm.foothold_learning_rate < foothold_lr_before


def test_independent_policy_checkpoint_round_trip_restores_both_optimizers():
    module = _load_ppo_module()
    algorithm, obs_format = _make_algorithm(module)
    algorithm.init_storage(2, 1, obs_format, 31, num_rewards=2)
    observations = torch.randn(4, 4)
    means = algorithm.actor_critic.act_inference(observations)
    algorithm._policy_gradient_step(
        motor_loss=means[..., :29].square().mean(),
        foothold_loss=means[..., 29:].square().mean(),
        run_foothold=True,
        average_stats={},
    )
    algorithm.learning_rate = 7.0e-4
    algorithm.foothold_learning_rate = 3.0e-4
    state = algorithm.state_dict()

    restored, restored_obs_format = _make_algorithm(module)
    restored.init_storage(2, 1, restored_obs_format, 31, num_rewards=2)
    restored.load_state_dict(state)

    assert restored.learning_rate == pytest.approx(7.0e-4)
    assert restored.foothold_learning_rate == pytest.approx(3.0e-4)
    assert restored.optimizer.state_dict()["state"]
    assert restored.foothold_optimizer.state_dict()["state"]
    for key, value in algorithm.actor_critic.state_dict().items():
        torch.testing.assert_close(
            restored.actor_critic.state_dict()[key],
            value,
        )


def test_process_env_step_rejects_missing_causal_event():
    module = _load_ppo_module()
    algorithm, obs_format = _make_algorithm(module)
    algorithm.init_storage(2, 1, obs_format, 31, num_rewards=2)

    with pytest.raises(KeyError, match="learned_foothold_action_event"):
        algorithm.process_env_step(
            rewards=torch.zeros(2, 2),
            dones=torch.zeros(2, dtype=torch.long),
            infos={"observations": {}, "step": {}},
            next_obs=torch.zeros(2, 4),
            next_critic_obs=torch.zeros(2, 4),
        )


def test_compute_losses_reports_separate_policy_groups():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)
    minibatch = _make_minibatch(module, algorithm.actor_critic)

    losses, _, stats = algorithm.compute_losses(minibatch)

    assert {
        "motor_surrogate_loss",
        "foothold_surrogate_loss",
    }.issubset(losses)
    assert {
        "motor_kl",
        "foothold_kl",
        "foothold_event_count",
    }.issubset(stats)
    assert stats["foothold_event_count"].item() == 2


def test_compute_losses_reports_event_only_foothold_action_diagnostics():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)
    minibatch = _make_minibatch(module, algorithm.actor_critic)
    # Event rows are 0 and 2.  Row 0 is raw-out-of-range and requires
    # radial projection after square clipping; row 2 is already valid.
    minibatch.actions[0, -2:] = torch.tensor([1.2, 0.8])
    minibatch.actions[1, -2:] = torch.tensor([9.0, 9.0])
    minibatch.actions[2, -2:] = torch.tensor([0.3, 0.4])

    _, _, stats = algorithm.compute_losses(minibatch)

    assert stats["foothold_raw_out_of_range_fraction"].item() == pytest.approx(
        0.5
    )
    assert stats["foothold_ellipse_projection_fraction"].item() == pytest.approx(
        0.5
    )


def test_compute_losses_rejects_nonfinite_minibatch():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)
    minibatch = _make_minibatch(module, algorithm.actor_critic)
    minibatch.actions[0, 0] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match=r"PPO minibatch\.actions.*nonfinite_count=1.*first_index=",
    ):
        algorithm.compute_losses(minibatch)


def test_finite_check_names_the_failing_tensor():
    module = _load_ppo_module()

    with pytest.raises(
        FloatingPointError,
        match=r"diagnostic\.bad.*nonfinite_count=1.*first_index=\(1,\)",
    ):
        module._require_finite(
            "diagnostic",
            {
                "good": torch.tensor([1.0, 2.0]),
                "bad": torch.tensor([3.0, float("inf")]),
            },
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Mixed optimizer-state devices require CUDA.",
)
def test_finite_check_accepts_mixed_cpu_and_cuda_optimizer_state():
    module = _load_ppo_module()

    module._require_finite(
        "mixed optimizer state",
        (
            torch.tensor(1.0, device="cpu"),
            torch.tensor([2.0, 3.0], device="cuda:0"),
        ),
    )


class _FiniteForwardNaNBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter):
        ctx.shape = parameter.shape
        return parameter.sum() * 0.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(
            ctx.shape,
            float("nan"),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )


def test_nonfinite_gradient_prevents_optimizer_step():
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)
    step_calls = 0

    def count_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1

    algorithm.optimizer.step = count_step
    loss = _FiniteForwardNaNBackward.apply(
        algorithm.actor_critic.motor_std
    )

    with pytest.raises(FloatingPointError, match="gradient"):
        algorithm.gradient_step(loss, {"grad_norm": 0.0})

    assert step_calls == 0


def test_gradient_step_does_not_scan_optimizer_state_per_minibatch(monkeypatch):
    module = _load_ppo_module()
    algorithm, _ = _make_algorithm(module)

    def fail_if_scanned(*args, **kwargs):
        raise AssertionError("optimizer state must be checked once per update")

    monkeypatch.setattr(module, "_optimizer_state_tensors", fail_if_scanned)
    loss = algorithm.actor_critic.motor_std.square().mean()

    algorithm.gradient_step(loss, {"grad_norm": 0.0})


def test_nonfinite_discriminator_gradient_prevents_optimizer_step():
    module = _load_ppo_module()
    algorithm, obs_format = _make_algorithm(module)
    algorithm.init_storage(2, 1, obs_format, 31, num_rewards=2)
    step_calls = 0

    def count_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1

    algorithm.discriminator_optimizer.step = count_step
    parameter = next(algorithm.discriminator.parameters())
    loss = _FiniteForwardNaNBackward.apply(parameter.flatten()[:31])

    with pytest.raises(FloatingPointError, match="gradient"):
        algorithm.wasabi_gradient_step(loss, {})

    assert step_calls == 0


def test_motor_lr_adapts_per_minibatch_while_foothold_lr_adapts_once(monkeypatch):
    module = _load_ppo_module()
    algorithm, obs_format = _make_algorithm(module)
    algorithm.init_storage(2, 1, obs_format, 31, num_rewards=2)
    minibatches = (object(), object(), object())
    events = []

    algorithm.storage.mini_batch_generator = lambda *args: iter(minibatches)

    def fake_compute_losses(_minibatch):
        events.append("compute")
        return (
            {
                "motor_surrogate_loss": torch.tensor(0.0),
                "foothold_surrogate_loss": torch.tensor(0.0),
                "motor_value_loss": torch.tensor(0.0),
                "foothold_value_loss": torch.tensor(0.0),
                "motor_entropy": torch.tensor(0.0),
                "foothold_entropy": torch.tensor(0.0),
            },
            {},
            {
                "motor_kl": torch.tensor(0.001),
                "foothold_kl": torch.tensor(0.01),
                "foothold_event_count": torch.tensor(0.0),
            },
        )

    monkeypatch.setattr(algorithm, "compute_losses", fake_compute_losses)
    monkeypatch.setattr(
        algorithm,
        "_policy_gradient_step",
        lambda **kwargs: events.append("step"),
    )
    monkeypatch.setattr(
        algorithm,
        "_adjust_learning_rate_once",
        lambda _kl: events.append("motor_lr"),
    )
    monkeypatch.setattr(
        algorithm,
        "_adjust_foothold_learning_rate_once",
        lambda _kl: events.append("foothold_lr"),
    )

    algorithm._update_policy(7)

    assert events == [
        "compute",
        "motor_lr",
        "step",
        "compute",
        "motor_lr",
        "step",
        "compute",
        "motor_lr",
        "step",
        "foothold_lr",
    ]


def test_registration_is_explicit():
    module = _load_ppo_module()
    import instinct_rl.algorithms as algorithms

    if hasattr(algorithms, "EventGatedWasabiPPO"):
        delattr(algorithms, "EventGatedWasabiPPO")
    assert not hasattr(algorithms, "EventGatedWasabiPPO")

    module.register_event_gated_foothold_algorithm()

    assert algorithms.EventGatedWasabiPPO is module.EventGatedWasabiPPO
