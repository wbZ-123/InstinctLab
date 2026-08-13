import torch
import instinctlab_foothold

from instinctlab_foothold.flat_provider import (
    FlatProviderConfig,
    FlatTargetBatch,
    TerrainCorridor,
    sample_flat_targets,
)


def test_default_velocity_lookahead_is_short_enough_for_training_curriculum():
    assert FlatProviderConfig().velocity_lookahead_s == 0.10


def test_targets_stay_reachable_and_do_not_cross_legs():
    num_envs = 4096
    cfg = FlatProviderConfig()

    swing_side = torch.arange(num_envs).remainder(2)

    result = sample_flat_targets(
        stance_xy=torch.zeros((num_envs, 2)),
        swing_side=swing_side,
        desired_velocity=torch.zeros((num_envs, 3)),
        level=torch.zeros(num_envs, dtype=torch.long),
        generator=torch.Generator().manual_seed(7),
        cfg=cfg,
    )

    normalized_radius = (
        (result.position_f[:, 0] / cfg.outer_radius_x).square()
        + (result.position_f[:, 1] / cfg.outer_radius_y).square()
    )

    assert torch.all(normalized_radius <= 1.0 + 1.0e-6)

    left_targets = result.position_f[swing_side == 0]
    right_targets = result.position_f[swing_side == 1]

    assert torch.all(left_targets[:, 1] >= cfg.min_lateral_separation)
    assert torch.all(right_targets[:, 1] <= -cfg.min_lateral_separation)


def test_sampling_is_repeatable_for_same_seed():
    cfg = FlatProviderConfig()
    kwargs = {
        "stance_xy": torch.zeros((32, 2)),
        "swing_side": torch.arange(32).remainder(2),
        "desired_velocity": torch.zeros((32, 3)),
        "level": torch.zeros(32, dtype=torch.long),
        "cfg": cfg,
    }

    first = sample_flat_targets(
        generator=torch.Generator().manual_seed(3),
        **kwargs,
    )
    repeated = sample_flat_targets(
        generator=torch.Generator().manual_seed(3),
        **kwargs,
    )
    different = sample_flat_targets(
        generator=torch.Generator().manual_seed(4),
        **kwargs,
    )

    torch.testing.assert_close(first.position_f, repeated.position_f)
    assert not torch.equal(first.position_f, different.position_f)


def test_curriculum_limits_residual_around_nominal_foothold():
    num_envs = 4096
    swing_side = torch.arange(num_envs).remainder(2)
    side_sign = torch.where(swing_side == 0, 1.0, -1.0)

    cfg = FlatProviderConfig(
        outer_radius_x=0.50,
        outer_radius_y=0.40,
        min_lateral_separation=0.06,
        nominal_step_width=0.18,
        curriculum_radius_x=(0.04, 0.08, 0.12),
        curriculum_radius_y=(0.02, 0.04, 0.08),
    )

    for level_value in range(3):
        result = sample_flat_targets(
            stance_xy=torch.zeros((num_envs, 2)),
            swing_side=swing_side,
            desired_velocity=torch.zeros((num_envs, 3)),
            level=torch.full(
                (num_envs,),
                level_value,
                dtype=torch.long,
            ),
            generator=torch.Generator().manual_seed(7),
            cfg=cfg,
        )

        residual_x = result.position_f[:, 0]
        residual_y = (
            result.position_f[:, 1]
            - side_sign * cfg.nominal_step_width
        )

        normalized_residual = (
            residual_x / cfg.curriculum_radius_x[level_value]
        ).square() + (
            residual_y / cfg.curriculum_radius_y[level_value]
        ).square()

        assert torch.all(normalized_residual <= 1.0 + 1.0e-6)


def test_targets_are_positioned_relative_to_stance_foot():
    cfg = FlatProviderConfig(
        outer_radius_x=0.50,
        outer_radius_y=0.40,
        min_lateral_separation=0.06,
        nominal_step_width=0.18,
        curriculum_radius_x=(0.0, 0.0, 0.0),
        curriculum_radius_y=(0.0, 0.0, 0.0),
    )

    stance_xy = torch.tensor(
        [
            [0.30, -0.10],
            [-0.20, 0.20],
        ]
    )

    result = sample_flat_targets(
        stance_xy=stance_xy,
        swing_side=torch.tensor([0, 1]),
        desired_velocity=torch.zeros((2, 3)),
        level=torch.zeros(2, dtype=torch.long),
        generator=torch.Generator().manual_seed(7),
        cfg=cfg,
    )

    expected = torch.tensor(
        [
            [0.30, 0.08, 0.0],
            [-0.20, 0.02, 0.0],
        ]
    )

    torch.testing.assert_close(result.position_f, expected)


def test_desired_velocity_shifts_nominal_foothold():
    cfg = FlatProviderConfig(
        outer_radius_x=0.50,
        outer_radius_y=0.40,
        min_lateral_separation=0.06,
        nominal_step_width=0.18,
        curriculum_radius_x=(0.0, 0.0, 0.0),
        curriculum_radius_y=(0.0, 0.0, 0.0),
        velocity_lookahead_s=0.20,
    )

    result = sample_flat_targets(
        stance_xy=torch.zeros((2, 2)),
        swing_side=torch.tensor([0, 1]),
        desired_velocity=torch.tensor(
            [
                [0.50, 0.10, 0.0],
                [0.50, 0.10, 0.0],
            ]
        ),
        level=torch.zeros(2, dtype=torch.long),
        generator=torch.Generator().manual_seed(7),
        cfg=cfg,
    )

    expected = torch.tensor(
        [
            [0.10, 0.20, 0.0],
            [0.10, -0.16, 0.0],
        ]
    )

    torch.testing.assert_close(result.position_f, expected)


def test_flat_provider_returns_complete_flat_contract():
    num_envs = 4

    result = sample_flat_targets(
        stance_xy=torch.zeros((num_envs, 2)),
        swing_side=torch.arange(num_envs).remainder(2),
        desired_velocity=torch.zeros((num_envs, 3)),
        level=torch.zeros(num_envs, dtype=torch.long),
        generator=torch.Generator().manual_seed(7),
        cfg=FlatProviderConfig(),
    )

    assert result.position_f.shape == (num_envs, 3)
    assert result.yaw_f.shape == (num_envs,)
    assert result.normal_f.shape == (num_envs, 3)
    assert result.feasible_velocity_f.shape == (num_envs, 3)
    assert result.curriculum_residual_f.shape == (num_envs, 2)
    assert result.curriculum_radius_f.shape == (num_envs, 2)
    assert result.curriculum_usage.shape == (num_envs,)
    assert result.valid.shape == (num_envs,)
    assert result.valid.dtype == torch.bool

    torch.testing.assert_close(
        result.position_f[:, 2],
        torch.zeros(num_envs),
    )
    torch.testing.assert_close(
        result.yaw_f,
        torch.zeros(num_envs),
    )
    torch.testing.assert_close(
        result.normal_f,
        torch.tensor([0.0, 0.0, 1.0]).expand(num_envs, 3),
    )
    torch.testing.assert_close(
        result.feasible_velocity_f,
        torch.zeros((num_envs, 3)),
    )
    torch.testing.assert_close(
        result.curriculum_radius_f,
        torch.tensor([0.04, 0.02]).expand(num_envs, 2),
    )
    assert torch.all(result.curriculum_usage <= 1.0 + 1.0e-6)

    assert result.terrain.heights.shape == (num_envs, 8)
    assert result.terrain.confidences.shape == (num_envs, 8)

    torch.testing.assert_close(
        result.terrain.heights,
        torch.zeros((num_envs, 8)),
    )
    torch.testing.assert_close(
        result.terrain.confidences,
        torch.ones((num_envs, 8)),
    )
    torch.testing.assert_close(
        result.terrain.support_margin,
        torch.ones(num_envs),
    )
    torch.testing.assert_close(
        result.terrain.edge_risk,
        torch.zeros(num_envs),
    )
    torch.testing.assert_close(
        result.terrain.unknown_fraction,
        torch.zeros(num_envs),
    )


def test_yaw_intent_is_limited_by_curriculum_level():
    cfg = FlatProviderConfig(
        curriculum_yaw_limit_rad=(0.0, 0.10, 0.20),
        velocity_lookahead_s=0.20,
    )

    result = sample_flat_targets(
        stance_xy=torch.zeros((3, 2)),
        swing_side=torch.tensor([0, 1, 0]),
        desired_velocity=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
            ]
        ),
        level=torch.tensor([0, 1, 2]),
        generator=torch.Generator().manual_seed(7),
        cfg=cfg,
    )

    torch.testing.assert_close(
        result.yaw_f,
        torch.tensor([0.0, 0.10, 0.20]),
    )


def test_learned_mode_can_disable_nominal_curriculum_residual_without_changing_command_intent():
    """The learned policy needs one deterministic analytic prior per HOLD."""
    cfg = FlatProviderConfig(
        outer_radius_x=0.50,
        outer_radius_y=0.40,
        nominal_step_width=0.18,
        velocity_lookahead_s=0.20,
        curriculum_radius_x=(0.04, 0.08, 0.12),
        curriculum_radius_y=(0.02, 0.04, 0.06),
    )
    kwargs = {
        "stance_xy": torch.zeros((2, 2)),
        "swing_side": torch.tensor([0, 1]),
        "desired_velocity": torch.tensor([[0.30, 0.10, 0.0]]).expand(2, -1),
        "level": torch.tensor([2, 2]),
        "cfg": cfg,
        "enable_curriculum_residual": False,
    }

    first = sample_flat_targets(
        generator=torch.Generator().manual_seed(3),
        **kwargs,
    )
    second = sample_flat_targets(
        generator=torch.Generator().manual_seed(4),
        **kwargs,
    )

    torch.testing.assert_close(first.curriculum_residual_f, torch.zeros((2, 2)))
    torch.testing.assert_close(first.curriculum_usage, torch.zeros(2))
    torch.testing.assert_close(first.position_f, second.position_f)
    torch.testing.assert_close(
        first.position_f,
        torch.tensor([[0.06, 0.20, 0.0], [0.06, -0.16, 0.0]]),
    )


def test_flat_provider_exports_curriculum_residual_radius_and_usage():
    cfg = FlatProviderConfig(
        curriculum_radius_x=(0.04, 0.08, 0.12),
        curriculum_radius_y=(0.02, 0.04, 0.06),
    )
    level = torch.tensor([0, 1, 2])

    result = sample_flat_targets(
        stance_xy=torch.zeros((3, 2)),
        swing_side=torch.tensor([0, 1, 0]),
        desired_velocity=torch.zeros((3, 3)),
        level=level,
        generator=torch.Generator().manual_seed(7),
        cfg=cfg,
    )

    torch.testing.assert_close(
        result.curriculum_radius_f,
        torch.tensor(
            [
                [0.04, 0.02],
                [0.08, 0.04],
                [0.12, 0.06],
            ]
        ),
    )
    normalized = torch.where(
        result.curriculum_radius_f > 1.0e-6,
        torch.abs(result.curriculum_residual_f)
        / result.curriculum_radius_f,
        torch.zeros_like(result.curriculum_residual_f),
    )
    torch.testing.assert_close(
        result.curriculum_usage,
        torch.linalg.norm(normalized, dim=-1),
    )
    assert torch.all(result.curriculum_usage <= 1.0 + 1.0e-6)


def test_feasible_velocity_matches_the_clipped_plan():
    cfg = FlatProviderConfig(
        outer_radius_x=0.20,
        outer_radius_y=0.25,
        min_lateral_separation=0.06,
        nominal_step_width=0.18,
        velocity_lookahead_s=0.20,
        curriculum_radius_x=(0.0, 0.0, 0.0),
        curriculum_radius_y=(0.0, 0.0, 0.0),
        curriculum_yaw_limit_rad=(0.0, 0.10, 0.20),
    )

    requested_velocity = torch.tensor(
        [[10.0, 0.0, 10.0]]
    )

    result = sample_flat_targets(
        stance_xy=torch.zeros((1, 2)),
        swing_side=torch.tensor([0]),
        desired_velocity=requested_velocity,
        level=torch.tensor([2]),
        generator=torch.Generator().manual_seed(7),
        cfg=cfg,
    )

    realized_vx = (
        result.position_f[0, 0] / cfg.velocity_lookahead_s
    )
    realized_vy = (
        (
            result.position_f[0, 1]
            - cfg.nominal_step_width
        )
        / cfg.velocity_lookahead_s
    )
    realized_wz = (
        result.yaw_f[0] / cfg.velocity_lookahead_s
    )

    expected_velocity = torch.stack(
        (realized_vx, realized_vy, realized_wz)
    ).unsqueeze(0)

    torch.testing.assert_close(
        result.feasible_velocity_f,
        expected_velocity,
    )

    assert result.feasible_velocity_f[0, 0] < requested_velocity[0, 0]
    assert result.feasible_velocity_f[0, 2] < requested_velocity[0, 2]


def test_flat_provider_types_are_public_package_api():
    assert (
        instinctlab_foothold.FlatProviderConfig
        is FlatProviderConfig
    )
    assert (
        instinctlab_foothold.FlatTargetBatch
        is FlatTargetBatch
    )
    assert (
        instinctlab_foothold.TerrainCorridor
        is TerrainCorridor
    )
    assert (
        instinctlab_foothold.sample_flat_targets
        is sample_flat_targets
    )
