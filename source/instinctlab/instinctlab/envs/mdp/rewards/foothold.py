from __future__ import annotations

import torch


def _foothold_planner_data(env, sensor_name: str):
    return env.scene.sensors[sensor_name].data


def _is_swing_mode(gait_mode: torch.Tensor) -> torch.Tensor:
    return (gait_mode == 1) | (gait_mode == 2)


def _is_touchdown_confirm_mode(gait_mode: torch.Tensor) -> torch.Tensor:
    return gait_mode == 3


def foothold_swing_tracking_exp(
    env,
    sensor_name: str = "foothold_planner",
    std: float = 0.15,
):
    """Reward swing foot center tracking of the planner reference path."""
    data = _foothold_planner_data(env, sensor_name)

    position_error = data.actual_swing_foot_pos_w - data.swing_reference_pos_w
    squared_error = torch.sum(torch.square(position_error), dim=-1)
    reward = torch.exp(-squared_error / (std * std))

    return reward * _is_swing_mode(data.gait_mode).float()


def foothold_touchdown_tracking_exp(
    env,
    sensor_name: str = "foothold_planner",
    std: float = 0.10,
):
    """Reward accepted touchdown close to the planner target foothold."""
    data = _foothold_planner_data(env, sensor_name)

    position_error = data.actual_swing_foot_pos_w - data.target_foothold_w
    squared_error = torch.sum(torch.square(position_error), dim=-1)
    reward = torch.exp(-squared_error / (std * std))

    return (
        reward
        * data.touchdown_accepted.float()
        * _is_touchdown_confirm_mode(data.gait_mode).float()
    )
