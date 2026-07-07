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


def foothold_touchdown_confirm_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner accepted touchdown and swapped support."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_touchdown_confirm_mode(data.gait_mode).float()


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
