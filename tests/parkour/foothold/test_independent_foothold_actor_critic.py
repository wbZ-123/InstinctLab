import importlib.util
from pathlib import Path

import torch


def _load_policy_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/learning/independent_foothold_actor_critic.py"
    )
    spec = importlib.util.spec_from_file_location(
        "independent_foothold_actor_critic",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_policy_class():
    return _load_policy_module().IndependentFootholdMoEActorCritic


def _make_policy():
    policy_class = _load_policy_class()
    return policy_class(
        obs_format={
            "policy": {"obs": (4,)},
            "critic": {"obs": (4,)},
        },
        num_actions=31,
        motor_action_dim=29,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        foothold_hidden_dims=[8, 4],
        num_moe_experts=2,
        init_noise_std=0.9,
        num_rewards=2,
    )


def _make_encoded_policy(*, foothold_depth_output_size=6):
    module = _load_policy_module()
    encoder_cfg = {
        "depth_encoder": {
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
    }
    return module.IndependentFootholdEncoderMoEActorCritic(
        obs_format={
            "policy": {
                "state": (4,),
                "depth_image": (4, 8, 8),
            },
            "critic": {
                "state": (4,),
                "depth_image": (4, 8, 8),
            },
        },
        num_actions=31,
        motor_action_dim=29,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        foothold_hidden_dims=[8, 4],
        foothold_depth_output_size=foothold_depth_output_size,
        encoder_configs=encoder_cfg,
        critic_encoder_configs=encoder_cfg.copy(),
        num_moe_experts=2,
        init_noise_std=0.9,
        num_rewards=2,
    )


def _parameter_ids(parameters):
    return {id(parameter) for parameter in parameters}


def _assert_no_gradient(parameters):
    for parameter in parameters:
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_independent_policy_preserves_environment_action_shape():
    policy = _make_policy()
    observations = torch.zeros(3, 4)

    actions = policy.act_inference(observations)
    forward_actions = policy(observations)
    values = policy.evaluate(observations)

    assert actions.shape == (3, 31)
    torch.testing.assert_close(forward_actions, actions)
    assert policy.actor(observations).shape == (3, 29)
    assert policy.foothold_actor(observations.detach()).shape == (3, 2)
    assert values.shape == (3, 2)
    assert policy.motor_std.shape == (29,)
    assert policy.foothold_std.shape == (2,)


def test_motor_and_foothold_parameter_groups_are_disjoint_and_exhaustive():
    policy = _make_policy()
    motor_ids = _parameter_ids(policy.motor_parameters())
    foothold_ids = _parameter_ids(policy.foothold_parameters())
    all_ids = _parameter_ids(policy.parameters())

    assert motor_ids.isdisjoint(foothold_ids)
    assert motor_ids | foothold_ids == all_ids


def test_foothold_actor_backward_does_not_reach_motor_parameters():
    policy = _make_policy()
    observations = torch.randn(3, 4)

    policy.act_inference(observations)[..., 29:].sum().backward()

    _assert_no_gradient(policy.motor_parameters())
    assert any(parameter.grad is not None for parameter in policy.foothold_parameters())


def test_motor_actor_backward_does_not_reach_foothold_parameters():
    policy = _make_policy()
    observations = torch.randn(3, 4)

    policy.act_inference(observations)[..., :29].sum().backward()

    _assert_no_gradient(policy.foothold_parameters())
    assert any(parameter.grad is not None for parameter in policy.motor_parameters())


def test_critics_have_independent_gradient_paths():
    policy = _make_policy()
    observations = torch.randn(3, 4)

    values = policy.evaluate(observations)
    values[..., 0].sum().backward()
    _assert_no_gradient(policy.foothold_parameters())

    policy.zero_grad(set_to_none=True)
    values = policy.evaluate(observations)
    values[..., 1].sum().backward()
    _assert_no_gradient(policy.motor_parameters())


def test_encoded_policy_keeps_motor_path_separate_from_planner_depth_path():
    policy = _make_encoded_policy()
    observations = torch.randn(3, 4 + 4 * 8 * 8)

    actions_before = policy.act_inference(observations)
    with torch.no_grad():
        for parameter in policy.foothold_depth_encoder.parameters():
            parameter.add_(0.25)
    actions_after = policy.act_inference(observations)

    torch.testing.assert_close(
        actions_before[..., :29],
        actions_after[..., :29],
    )
    assert not torch.allclose(
        actions_before[..., 29:],
        actions_after[..., 29:],
    )


def test_saved_legacy_encoded_config_keeps_original_policy_shape():
    policy = _make_encoded_policy(foothold_depth_output_size=0)
    observations = torch.randn(3, 4 + 4 * 8 * 8)

    actions = policy.act_inference(observations)

    assert actions.shape == (3, 31)
    assert not hasattr(policy, "foothold_depth_encoder")
    assert policy.foothold_actor[0].in_features == policy.mlp_input_dim_a


def test_planner_depth_loss_updates_only_planner_depth_parameters():
    policy = _make_encoded_policy()
    observations = torch.randn(3, 4 + 4 * 8 * 8)
    motor_before = [
        parameter.detach().clone() for parameter in policy.motor_parameters()
    ]
    depth_before = [
        parameter.detach().clone()
        for parameter in policy.foothold_depth_encoder.parameters()
    ]
    optimizer = torch.optim.Adam(policy.foothold_parameters(), lr=1.0e-3)

    policy.act_inference(observations)[..., 29:].sum().backward()

    _assert_no_gradient(policy.motor_parameters())
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.foothold_depth_encoder.parameters()
    )
    optimizer.step()
    for before, after in zip(motor_before, policy.motor_parameters()):
        torch.testing.assert_close(before, after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            depth_before,
            policy.foothold_depth_encoder.parameters(),
        )
    )
