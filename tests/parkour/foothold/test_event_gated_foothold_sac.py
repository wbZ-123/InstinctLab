"""Focused tests for the hybrid motor-PPO / planner-SAC boundary."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


def _load_modules():
    repo_root = Path(__file__).resolve().parents[3]
    learning_dir = repo_root / "source/instinctlab/instinctlab/learning"
    package = ModuleType("instinctlab.learning")
    package.__path__ = [str(learning_dir)]
    sys.modules["instinctlab.learning"] = package

    policy_path = learning_dir / "independent_foothold_actor_critic.py"
    policy_spec = importlib.util.spec_from_file_location(
        "instinctlab.learning.independent_foothold_actor_critic", policy_path
    )
    assert policy_spec is not None and policy_spec.loader is not None
    policy_module = importlib.util.module_from_spec(policy_spec)
    sys.modules[policy_spec.name] = policy_module
    policy_spec.loader.exec_module(policy_module)

    for name in ("foothold_rollout_storage", "foothold_sac_replay", "foothold_sac"):
        path = learning_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(
            f"instinctlab.learning.{name}", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    algo_path = learning_dir / "event_gated_foothold_ppo.py"
    algo_spec = importlib.util.spec_from_file_location(
        "instinctlab.learning.event_gated_foothold_ppo", algo_path
    )
    assert algo_spec is not None and algo_spec.loader is not None
    algo_module = importlib.util.module_from_spec(algo_spec)
    sys.modules[algo_spec.name] = algo_module
    algo_spec.loader.exec_module(algo_module)
    return algo_module, policy_module


def _make_sac_algorithm(module, policy_module):
    policy = policy_module.IndependentFootholdMoEActorCritic(
        obs_format={"policy": {"obs": (4,)}, "critic": {"obs": (4,)}},
        num_actions=31,
        motor_action_dim=29,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        foothold_hidden_dims=[8, 4],
        num_moe_experts=2,
        init_noise_std=0.9,
        num_rewards=2,
    )
    return module.EventGatedWasabiSAC(
        actor_critic=policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        schedule="fixed",
        device="cpu",
        discriminator_kwargs={"hidden_sizes": [8], "nonlinearity": "ReLU"},
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
        sac_batch_size=2,
        sac_warmup_events=0,
        sac_target_sample_ratio=1.0,
        sac_max_updates_per_rollout=1,
    )


def test_hybrid_algorithm_is_registered_and_has_separate_planner_learner():
    module, policy_module = _load_modules()
    algorithm = _make_sac_algorithm(module, policy_module)
    algorithm.init_storage(
        2,
        2,
        {
            "policy": {"obs": (4,)},
            "critic": {"obs": (4,)},
            "amp_policy": {"state": (3,)},
            "amp_reference": {"state": (3,)},
        },
        31,
        num_rewards=2,
    )
    assert isinstance(algorithm.sac, module.FootholdSAC)
    assert algorithm.motor_action_dim == 29
    assert algorithm.sac.config.action_dim == 2
    assert algorithm.use_foothold_ppo is False


def test_non_event_replay_is_a_noop_and_event_is_recorded():
    module, policy_module = _load_modules()
    algorithm = _make_sac_algorithm(module, policy_module)
    algorithm.init_storage(
        2,
        2,
        {
            "policy": {"obs": (4,)},
            "critic": {"obs": (4,)},
            "amp_policy": {"state": (3,)},
            "amp_reference": {"state": (3,)},
        },
        31,
        num_rewards=2,
    )

    obs = torch.zeros(2, 4)
    algorithm.transition.observations = obs
    algorithm.transition.critic_observations = obs
    algorithm.transition.actions = torch.zeros(2, 31)
    algorithm.transition.actions[1, 29:] = torch.tensor([3.0, 4.0])
    algorithm.transition.values = torch.zeros(2, 2)
    algorithm.transition.actions_log_prob = torch.zeros(2, 1)
    algorithm.transition.action_mean = torch.zeros(2, 31)
    algorithm.transition.action_sigma = torch.ones(2, 31)
    algorithm.transition.foothold_action_event = torch.tensor([False, True])
    algorithm.transition.foothold_nominal_safe_event = torch.tensor([False, True])
    algorithm.transition.foothold_nominal_unsafe_event = torch.tensor([False, False])

    algorithm.process_env_step(
        torch.zeros(2, 2),
        torch.zeros(2, dtype=torch.bool),
        {
            "observations": {
                "policy": obs,
                "amp_policy": torch.zeros(2, 3),
                "amp_reference": torch.zeros(2, 3),
            },
            "time_outs": torch.zeros(2),
            "learned_foothold_action_event": torch.tensor([False, True]),
            "learned_foothold_nominal_safe_event": torch.tensor([False, True]),
            "learned_foothold_nominal_unsafe_event": torch.tensor([False, False]),
            "learned_foothold_event_reward": torch.tensor([0.0, 0.73]),
            "step": {},
        },
        obs + 1.0,
        obs + 1.0,
    )
    # The first event is pending; a post-step control frame is not a valid
    # planner next state and must not enter replay yet.
    assert len(algorithm.sac.replay) == 0

    algorithm.transition.observations = obs + 1.0
    algorithm.transition.critic_observations = obs + 1.0
    algorithm.transition.actions = torch.zeros(2, 31)
    algorithm.transition.values = torch.zeros(2, 2)
    algorithm.transition.actions_log_prob = torch.zeros(2, 1)
    algorithm.transition.action_mean = torch.zeros(2, 31)
    algorithm.transition.action_sigma = torch.ones(2, 31)
    algorithm.process_env_step(
        torch.zeros(2, 2),
        torch.zeros(2, dtype=torch.bool),
        {
            "observations": {
                "policy": obs + 1.0,
                "amp_policy": torch.zeros(2, 3),
                "amp_reference": torch.zeros(2, 3),
            },
            "time_outs": torch.zeros(2),
            "learned_foothold_action_event": torch.tensor([False, True]),
            "learned_foothold_nominal_safe_event": torch.tensor([False, True]),
            "learned_foothold_nominal_unsafe_event": torch.tensor([False, False]),
            "learned_foothold_event_reward": torch.tensor([0.0, 0.0]),
            "step": {},
        },
        obs + 2.0,
        obs + 2.0,
    )
    assert len(algorithm.sac.replay) == 1
    torch.testing.assert_close(
        algorithm.sac.replay.rewards[0],
        torch.tensor(0.73),
    )
    torch.testing.assert_close(
        algorithm.sac.replay.actions[0],
        torch.tensor([3.0, 4.0]) / 26.0**0.5,
    )


def test_hybrid_sac_updates_from_event_replay_without_ppo_planner_step():
    module, policy_module = _load_modules()
    algorithm = _make_sac_algorithm(module, policy_module)
    algorithm.init_storage(
        2,
        1,
        {
            "policy": {"obs": (4,)},
            "critic": {"obs": (4,)},
            "amp_policy": {"state": (3,)},
            "amp_reference": {"state": (3,)},
        },
        31,
        num_rewards=2,
    )
    sac = algorithm.sac
    assert sac is not None
    before = [parameter.detach().clone() for parameter in algorithm.actor_critic.planner_policy_parameters()]
    sac.observe(
        torch.randn(2, 4),
        torch.zeros(2, 2),
        torch.ones(2),
        torch.randn(2, 4),
        torch.zeros(2, dtype=torch.bool),
    )
    stats = sac.update(new_event_count=2)
    assert stats["sac_update_count"] == 1.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, algorithm.actor_critic.planner_policy_parameters())
    )


def test_planner_encoder_is_owned_by_critic_not_actor():
    module, policy_module = _load_modules()
    policy = policy_module.IndependentFootholdEncoderMoEActorCritic(
        obs_format={
            "policy": {"obs": (4,), "depth_image": (1, 8, 8)},
            "critic": {"obs": (4,)},
        },
        num_actions=31,
        motor_action_dim=29,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        foothold_hidden_dims=[8, 4],
        foothold_depth_output_size=4,
        num_moe_experts=2,
        init_noise_std=0.9,
        num_rewards=2,
        encoder_configs={
            "depth": {
                "class_name": "Conv2dHeadModel",
                "component_names": ["depth_image"],
                "output_size": 4,
                "channels": [2],
                "kernel_sizes": [3],
                "strides": [1],
                "paddings": [1],
                "hidden_sizes": [4],
                "nonlinearity": "ReLU",
                "use_maxpool": False,
                "takeout_input_components": True,
            }
        },
    )
    actor_params = policy.planner_actor_parameters()
    encoder_params = policy.planner_encoder_parameters()
    assert encoder_params
    assert actor_params
    assert not ({id(p) for p in actor_params} & {id(p) for p in encoder_params})


def test_hybrid_sac_requires_unscaled_event_reward_extra():
    module, policy_module = _load_modules()
    algorithm = _make_sac_algorithm(module, policy_module)
    algorithm.init_storage(
        2,
        1,
        {
            "policy": {"obs": (4,)},
            "critic": {"obs": (4,)},
            "amp_policy": {"state": (3,)},
            "amp_reference": {"state": (3,)},
        },
        31,
        num_rewards=2,
    )
    algorithm.transition.observations = torch.zeros(2, 4)
    algorithm.transition.critic_observations = torch.zeros(2, 4)
    algorithm.transition.actions = torch.zeros(2, 31)
    algorithm.transition.values = torch.zeros(2, 2)
    algorithm.transition.actions_log_prob = torch.zeros(2, 1)
    algorithm.transition.action_mean = torch.zeros(2, 31)
    algorithm.transition.action_sigma = torch.ones(2, 31)

    with pytest.raises(KeyError, match="learned_foothold_event_reward"):
        algorithm.process_env_step(
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.bool),
            {
                "observations": {
                    "policy": torch.zeros(2, 4),
                    "amp_policy": torch.zeros(2, 3),
                    "amp_reference": torch.zeros(2, 3),
                },
                "time_outs": torch.zeros(2),
                "learned_foothold_action_event": torch.tensor([True, False]),
                "learned_foothold_nominal_safe_event": torch.tensor([True, False]),
                "learned_foothold_nominal_unsafe_event": torch.tensor([False, False]),
                "step": {},
            },
            torch.zeros(2, 4),
            torch.zeros(2, 4),
        )


def test_hybrid_act_keeps_motor_actions_and_replaces_only_planner_slice():
    module, policy_module = _load_modules()
    algorithm = _make_sac_algorithm(module, policy_module)
    algorithm.init_storage(
        2,
        1,
        {
            "policy": {"obs": (4,)},
            "critic": {"obs": (4,)},
            "amp_policy": {"state": (3,)},
            "amp_reference": {"state": (3,)},
        },
        31,
        num_rewards=2,
    )
    actions = algorithm.act(torch.zeros(2, 4), torch.zeros(2, 4))
    assert actions.shape == (2, 31)
    torch.testing.assert_close(
        actions[..., :29],
        algorithm.transition.actions[..., :29],
    )
    torch.testing.assert_close(
        actions[..., 29:],
        algorithm.transition.actions[..., 29:],
    )


def test_hybrid_sac_checkpoint_round_trip_restores_replay_and_temperature():
    module, policy_module = _load_modules()
    algorithm = _make_sac_algorithm(module, policy_module)
    storage_kwargs = {
        "num_envs": 2,
        "num_transitions_per_env": 1,
        "obs_format": {
            "policy": {"obs": (4,)},
            "critic": {"obs": (4,)},
            "amp_policy": {"state": (3,)},
            "amp_reference": {"state": (3,)},
        },
        "num_actions": 31,
        "num_rewards": 2,
    }
    algorithm.init_storage(**storage_kwargs)
    assert algorithm.sac is not None
    algorithm.sac.observe(
        torch.randn(2, 4),
        torch.zeros(2, 2),
        torch.ones(2),
        torch.randn(2, 4),
        torch.zeros(2, dtype=torch.bool),
    )
    algorithm.sac.log_alpha.data.fill_(-0.7)
    algorithm.sac.update_credit = 0.375
    state = algorithm.state_dict()

    restored = _make_sac_algorithm(module, policy_module)
    restored.init_storage(**storage_kwargs)
    restored.load_state_dict(state)
    assert restored.sac is not None
    assert len(restored.sac.replay) == 2
    assert restored.sac.log_alpha.item() == pytest.approx(-0.7)
    assert restored.sac.update_credit == pytest.approx(0.375)


def test_legacy_event_sac_checkpoint_discards_incompatible_replay(capsys):
    module, policy_module = _load_modules()
    algorithm = _make_sac_algorithm(module, policy_module)
    storage_kwargs = {
        "num_envs": 2,
        "num_transitions_per_env": 1,
        "obs_format": {
            "policy": {"obs": (4,)},
            "critic": {"obs": (4,)},
            "amp_policy": {"state": (3,)},
            "amp_reference": {"state": (3,)},
        },
        "num_actions": 31,
        "num_rewards": 2,
    }
    algorithm.init_storage(**storage_kwargs)
    assert algorithm.sac is not None
    algorithm.sac.observe(
        torch.randn(2, 4),
        torch.zeros(2, 2),
        torch.ones(2),
        torch.randn(2, 4),
        torch.zeros(2, dtype=torch.bool),
    )
    state = algorithm.state_dict()
    state["foothold_sac_version"] = 1

    restored = _make_sac_algorithm(module, policy_module)
    restored.init_storage(**storage_kwargs)
    restored.load_state_dict(state)
    assert restored.sac is not None
    assert len(restored.sac.replay) == 0
    assert "Legacy foothold SAC state" in capsys.readouterr().out
