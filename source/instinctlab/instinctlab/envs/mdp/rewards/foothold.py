from __future__ import annotations

import torch


def _foothold_planner_data(env, sensor_name: str):
    return env.scene.sensors[sensor_name].data


def _sync_desired_velocity_command(
    env,
    sensor_name: str,
    command_name: str | None,
) -> None:
    if command_name is None or not hasattr(env, "command_manager"):
        return

    planner = env.scene.sensors[sensor_name]
    planner.set_desired_velocity(
        env.command_manager.get_command(command_name),
    )


def _is_swing_mode(gait_mode: torch.Tensor) -> torch.Tensor:
    return (gait_mode == 1) | (gait_mode == 2)


def _is_touchdown_confirm_mode(gait_mode: torch.Tensor) -> torch.Tensor:
    return gait_mode == 3


def _is_late_touchdown_tracking_mode(
    gait_mode: torch.Tensor,
    phase: torch.Tensor,
    min_phase: float,
) -> torch.Tensor:
    late_swing = _is_swing_mode(gait_mode) & (phase >= min_phase)
    overdue = gait_mode == 5
    return late_swing | overdue


def _swing_foot_contact(data) -> torch.Tensor:
    env_ids = torch.arange(
        data.foot_contact.shape[0],
        device=data.foot_contact.device,
    )
    swing_side = data.swing_side.long().clamp(0, 1)
    return data.foot_contact[env_ids, swing_side].bool()


def _is_mode(
    gait_mode: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    return (gait_mode == mode).float()


def foothold_swing_tracking_exp(
    env,
    sensor_name: str = "foothold_planner",
    command_name: str | None = "base_velocity",
    std: float = 0.15,
):
    """Reward swing foot center tracking of the planner reference path."""
    _sync_desired_velocity_command(env, sensor_name, command_name)
    data = _foothold_planner_data(env, sensor_name)

    position_error = data.actual_swing_foot_pos_w - data.swing_reference_pos_w
    squared_error = torch.sum(torch.square(position_error), dim=-1)
    reward = torch.exp(-squared_error / (std * std))

    return reward * _is_swing_mode(data.gait_mode).float()


def foothold_touchdown_tracking_exp(
    env,
    sensor_name: str = "foothold_planner",
    command_name: str | None = "base_velocity",
    std: float = 0.10,
):
    """Reward touchdown-confirm foot placement near the planner target."""
    _sync_desired_velocity_command(env, sensor_name, command_name)
    data = _foothold_planner_data(env, sensor_name)

    position_error = data.actual_swing_foot_pos_w - data.target_foothold_w
    squared_error = torch.sum(torch.square(position_error), dim=-1)
    reward = torch.exp(-squared_error / (std * std))

    return (
        reward
        * _is_touchdown_confirm_mode(data.gait_mode).float()
    )


def foothold_swing_contact_indicator(
    env,
    sensor_name: str = "foothold_planner",
    min_phase: float = 0.20,
):
    """Return 1 when the planned swing foot is still in contact too late.

    The small phase grace avoids penalizing the first few frames of a newly
    planned swing, where the foot may not have lifted yet.
    """
    data = _foothold_planner_data(env, sensor_name)
    swing_contact = _swing_foot_contact(data)
    after_liftoff_grace = data.phase >= min_phase
    return (
        _is_swing_mode(data.gait_mode)
        & after_liftoff_grace
        & swing_contact
    ).float()


def foothold_no_liftoff_indicator(
    env,
    sensor_name: str = "foothold_planner",
    min_phase: float = 0.35,
):
    """Return 1 when swing has progressed but the foot never lifted.

    This is different from swing contact: it uses the planner's confirmed
    liftoff latch, so brief contact noise after a real liftoff is not counted
    as "never lifted".
    """
    data = _foothold_planner_data(env, sensor_name)
    after_liftoff_deadline = data.phase >= min_phase
    return (
        _is_swing_mode(data.gait_mode)
        & after_liftoff_deadline
        & ~data.swing_has_lifted.bool()
    ).float()


def foothold_swing_height_under_error_l1(
    env,
    sensor_name: str = "foothold_planner",
    max_error_m: float = 0.25,
):
    """Return positive height deficit below the swing reference.

    Only under-shooting the reference height is penalized.  Overshooting is
    left to the base tracking/regularization terms, because the current failure
    mode is dragging the swing foot too low.
    """
    data = _foothold_planner_data(env, sensor_name)
    height_deficit = (
        data.swing_reference_pos_w[:, 2]
        - data.actual_swing_foot_pos_w[:, 2]
    ).clamp(min=0.0, max=max_error_m)
    return height_deficit * _is_swing_mode(data.gait_mode).float()


def foothold_swing_xy_error_l2(
    env,
    sensor_name: str = "foothold_planner",
    max_error_m: float = 0.30,
):
    """Return planar distance from swing foot to the reference trajectory."""
    data = _foothold_planner_data(env, sensor_name)
    xy_error = torch.linalg.norm(
        (
            data.actual_swing_foot_pos_w[:, :2]
            - data.swing_reference_pos_w[:, :2]
        ),
        dim=-1,
    ).clamp_max(max_error_m)
    return xy_error * _is_swing_mode(data.gait_mode).float()


def foothold_touchdown_xy_error_l2(
    env,
    sensor_name: str = "foothold_planner",
    min_phase: float = 0.65,
    max_error_m: float = 0.30,
):
    """Return touchdown XY error during the landing part of swing."""
    data = _foothold_planner_data(env, sensor_name)
    active = _is_late_touchdown_tracking_mode(
        data.gait_mode,
        data.phase,
        min_phase,
    )
    return data.touchdown_xy_error.clamp_max(max_error_m) * active.float()


def foothold_touchdown_z_error_l1(
    env,
    sensor_name: str = "foothold_planner",
    min_phase: float = 0.65,
    max_error_m: float = 0.20,
):
    """Return touchdown height error during the landing part of swing."""
    data = _foothold_planner_data(env, sensor_name)
    active = _is_late_touchdown_tracking_mode(
        data.gait_mode,
        data.phase,
        min_phase,
    )
    return data.touchdown_z_error.clamp_max(max_error_m) * active.float()


def foothold_swing_mode_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner state machine is in left/right swing."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_swing_mode(data.gait_mode).float()


def foothold_reset_mode_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner state machine is holding after reset."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 0)


def foothold_left_swing_mode_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner state machine is in left swing."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 1)


def foothold_right_swing_mode_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner state machine is in right swing."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 2)


def foothold_touchdown_confirm_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner accepted touchdown and swapped support."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_touchdown_confirm_mode(data.gait_mode).float()


def foothold_early_contact_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner reports early swing-foot contact."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 4)


def foothold_overdue_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the swing phase is overdue."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 5)


def foothold_stance_lost_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the support foot loses confirmed contact."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 6)


def foothold_gait_anomaly_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 for any planner gait anomaly, max-pooled across reasons."""
    data = _foothold_planner_data(env, sensor_name)
    failure_mode = (
        (data.gait_mode == 4)
        | (data.gait_mode == 5)
        | (data.gait_mode == 6)
        | (data.gait_mode == 7)
        | (data.gait_mode == 8)
    )
    planner_invalid = ~data.planner_valid.bool()
    return (failure_mode | planner_invalid).float()


def foothold_recovery_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner is in recovery after a gait failure."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 8)


def foothold_clearance_safe_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the adjusted swing trajectory is clearance-safe."""
    data = _foothold_planner_data(env, sensor_name)
    return data.swing_clearance_safe.float()


def foothold_clearance_penetration_l1(
    env,
    sensor_name: str = "foothold_planner",
    max_penetration_m: float = 0.15,
):
    """Return swing trajectory penetration depth into edge obstacles."""
    data = _foothold_planner_data(env, sensor_name)
    return data.swing_clearance_penetration.clamp_max(max_penetration_m)


def foothold_touchdown_accepted_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the current swing has physically touched down."""
    data = _foothold_planner_data(env, sensor_name)
    return data.touchdown_accepted.float()


def foothold_plan_invalid_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner reports no valid foothold plan."""
    data = _foothold_planner_data(env, sensor_name)
    return (~data.planner_valid).float()
