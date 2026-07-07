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


def foothold_planner_observation(
    env,
    sensor_name: str = "foothold_planner",
    command_name: str | None = "base_velocity",
) -> torch.Tensor:
    """Expose the compact foothold planner state to the policy."""
    _sync_desired_velocity_command(env, sensor_name, command_name)
    data = _foothold_planner_data(env, sensor_name)

    phase = data.phase.unsqueeze(-1)
    swing_side_sign = (
        data.swing_side.to(dtype=data.phase.dtype)
        * 2.0
        - 1.0
    ).unsqueeze(-1)

    return torch.cat(
        (
            data.target_foothold_f,
            data.feasible_velocity_f,
            phase,
            swing_side_sign,
        ),
        dim=-1,
    )
