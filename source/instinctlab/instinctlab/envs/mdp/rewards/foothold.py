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
    """Reward accepted touchdown close to the planner target foothold."""
    _sync_desired_velocity_command(env, sensor_name, command_name)
    data = _foothold_planner_data(env, sensor_name)

    position_error = data.actual_swing_foot_pos_w - data.target_foothold_w
    squared_error = torch.sum(torch.square(position_error), dim=-1)
    reward = torch.exp(-squared_error / (std * std))

    return (
        reward
        * data.touchdown_accepted.float()
        * _is_touchdown_confirm_mode(data.gait_mode).float()
    )


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
):
    """Return swing trajectory penetration depth into edge obstacles."""
    data = _foothold_planner_data(env, sensor_name)
    return data.swing_clearance_penetration

def foothold_touchdown_accepted_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the current swing touchdown is within planner tolerance."""
    data = _foothold_planner_data(env, sensor_name)
    return data.touchdown_accepted.float()


def foothold_plan_invalid_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner reports no valid foothold plan."""
    data = _foothold_planner_data(env, sensor_name)
    return (~data.planner_valid).float()
