from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlatProviderConfig:
    # Fixed reachability ellipse in the frozen support-foot frame.
    outer_radius_x: float = 1.00
    # Experimental lateral proposal bound.  This is an action/reach envelope,
    # not a calibrated physical limit of the G1.
    outer_radius_y: float = 0.50
    min_lateral_separation: float = 0.06

    nominal_step_width: float = 0.26
    velocity_lookahead_s: float = 0.10
    # Kept as zero-valued compatibility fields for existing telemetry and
    # checkpoints. The nominal XY foothold no longer receives curriculum
    # random residuals at any level.
    curriculum_radius_x: tuple[float, ...] = (0.0, 0.0, 0.0)
    curriculum_radius_y: tuple[float, ...] = (0.0, 0.0, 0.0)

    curriculum_yaw_limit_rad: tuple[float, ...] = (
        0.0,
        0.10,
        0.20,
    )


@dataclass(frozen=True)
class TerrainCorridor:
    heights: torch.Tensor
    confidences: torch.Tensor
    support_margin: torch.Tensor
    edge_risk: torch.Tensor
    unknown_fraction: torch.Tensor


@dataclass(frozen=True)
class FlatTargetBatch:
    position_f: torch.Tensor
    yaw_f: torch.Tensor
    normal_f: torch.Tensor
    feasible_velocity_f: torch.Tensor
    curriculum_residual_f: torch.Tensor
    curriculum_radius_f: torch.Tensor
    curriculum_usage: torch.Tensor
    valid: torch.Tensor
    terrain: TerrainCorridor


def sample_flat_targets(
    stance_xy: torch.Tensor,
    swing_side: torch.Tensor,
    desired_velocity: torch.Tensor,
    level: torch.Tensor,
    generator: torch.Generator,
    cfg: FlatProviderConfig,
    *,
    enable_curriculum_residual: bool = True,
) -> FlatTargetBatch:
    num_envs = stance_xy.shape[0]

    radius_x_by_level = torch.as_tensor(
        cfg.curriculum_radius_x,
        device=stance_xy.device,
        dtype=stance_xy.dtype,
    )
    radius_y_by_level = torch.as_tensor(
        cfg.curriculum_radius_y,
        device=stance_xy.device,
        dtype=stance_xy.dtype,
    )

    level_index = level.clamp(
        min=0,
        max=len(cfg.curriculum_radius_x) - 1,
    )

    curriculum_radius_f = torch.stack(
        (
            radius_x_by_level[level_index],
            radius_y_by_level[level_index],
        ),
        dim=-1,
    )
    if enable_curriculum_residual:
        random_values = torch.rand(
            (num_envs, 2),
            generator=generator,
            device=stance_xy.device,
            dtype=stance_xy.dtype,
        )
        disk_radius = torch.sqrt(random_values[:, 0])
        disk_angle = 2.0 * torch.pi * random_values[:, 1]
        residual_x = (
            disk_radius * torch.cos(disk_angle) * curriculum_radius_f[:, 0]
        )
        residual_y = (
            disk_radius * torch.sin(disk_angle) * curriculum_radius_f[:, 1]
        )
    else:
        residual_x = torch.zeros(
            num_envs,
            device=stance_xy.device,
            dtype=stance_xy.dtype,
        )
        residual_y = torch.zeros_like(residual_x)
    curriculum_residual_f = torch.stack(
        (residual_x, residual_y),
        dim=-1,
    )
    normalized_curriculum_residual = torch.where(
        curriculum_radius_f > 1.0e-6,
        torch.abs(curriculum_residual_f) / curriculum_radius_f,
        torch.zeros_like(curriculum_residual_f),
    )
    curriculum_usage = torch.linalg.norm(
        normalized_curriculum_residual,
        dim=-1,
    )

    side_sign = torch.where(
        swing_side == 0,
        torch.ones_like(swing_side),
        -torch.ones_like(swing_side),
    ).to(dtype=stance_xy.dtype)

    velocity_offset = (
        desired_velocity[:, :2] * cfg.velocity_lookahead_s
    )

    delta_x = residual_x + velocity_offset[:, 0]
    delta_y = (
        side_sign * cfg.nominal_step_width
        + residual_y
        + velocity_offset[:, 1]
    )

    # 防交叉约束针对“目标相对支撑脚的横向距离”。
    lateral_magnitude = torch.clamp(
        side_sign * delta_y,
        min=cfg.min_lateral_separation,
        max=cfg.outer_radius_y,
    )
    delta_y = side_sign * lateral_magnitude

    # 外椭圆同样以支撑脚为中心。
    max_abs_delta_x = cfg.outer_radius_x * torch.sqrt(
        torch.clamp(
            1.0
            - (delta_y / cfg.outer_radius_y).square(),
            min=0.0,
        )
    )
    delta_x = torch.clamp(
        delta_x,
        min=-max_abs_delta_x,
        max=max_abs_delta_x,
    )

    target_delta = torch.stack(
        (delta_x, delta_y),
        dim=-1,
    )
    position_xy_f = stance_xy + target_delta

    height_f = torch.zeros(
        (num_envs, 1),
        device=stance_xy.device,
        dtype=stance_xy.dtype,
    )
    position_f = torch.cat((position_xy_f, height_f), dim=-1)

    yaw_limit_by_level = torch.as_tensor(
        cfg.curriculum_yaw_limit_rad,
        device=stance_xy.device,
        dtype=stance_xy.dtype,
    )
    yaw_level_index = level.clamp(
        min=0,
        max=len(cfg.curriculum_yaw_limit_rad) - 1,
    )
    yaw_limit = yaw_limit_by_level[yaw_level_index]

    requested_yaw = (
        desired_velocity[:, 2] * cfg.velocity_lookahead_s
    )
    yaw_f = torch.clamp(
        requested_yaw,
        min=-yaw_limit,
        max=yaw_limit,
    )

    normal_f = torch.zeros(
        (num_envs, 3),
        device=stance_xy.device,
        dtype=stance_xy.dtype,
    )
    normal_f[:, 2] = 1.0

    realized_velocity_x = (
        delta_x - residual_x
    ) / cfg.velocity_lookahead_s

    realized_velocity_y = (
        delta_y
        - side_sign * cfg.nominal_step_width
        - residual_y
    ) / cfg.velocity_lookahead_s

    realized_yaw_velocity = (
        yaw_f / cfg.velocity_lookahead_s
    )

    feasible_velocity_f = torch.stack(
        (
            realized_velocity_x,
            realized_velocity_y,
            realized_yaw_velocity,
        ),
        dim=-1,
    )
    valid = torch.ones(
        num_envs,
        device=stance_xy.device,
        dtype=torch.bool,
    )

    terrain = TerrainCorridor(
        heights=torch.zeros(
            (num_envs, 8),
            device=stance_xy.device,
            dtype=stance_xy.dtype,
        ),
        confidences=torch.ones(
            (num_envs, 8),
            device=stance_xy.device,
            dtype=stance_xy.dtype,
        ),
        support_margin=torch.ones(
            num_envs,
            device=stance_xy.device,
            dtype=stance_xy.dtype,
        ),
        edge_risk=torch.zeros(
            num_envs,
            device=stance_xy.device,
            dtype=stance_xy.dtype,
        ),
        unknown_fraction=torch.zeros(
            num_envs,
            device=stance_xy.device,
            dtype=stance_xy.dtype,
        ),
    )

    return FlatTargetBatch(
        position_f=position_f,
        yaw_f=yaw_f,
        normal_f=normal_f,
        feasible_velocity_f=feasible_velocity_f,
        curriculum_residual_f=curriculum_residual_f,
        curriculum_radius_f=curriculum_radius_f,
        curriculum_usage=curriculum_usage,
        valid=valid,
        terrain=terrain,
    )
