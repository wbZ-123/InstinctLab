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

    stabilization_active = getattr(data, "stabilization_active", None)
    if stabilization_active is None:
        stabilization_active = data.recovery_step_active
    stabilization_active = stabilization_active.bool()

    phase = data.phase.unsqueeze(-1)
    swing_side_sign = (
        data.swing_side.to(dtype=data.phase.dtype)
        * 2.0
        - 1.0
    ).unsqueeze(-1)

    swing_apex_height = data.swing_apex_height.unsqueeze(-1)
    swing_apex_delta = (
        data.swing_apex_height - data.default_swing_apex_height
    ).unsqueeze(-1)
    swing_clearance_safe = data.swing_clearance_safe.to(
        dtype=data.phase.dtype
    ).unsqueeze(-1)
    swing_clearance_penetration = data.swing_clearance_penetration.unsqueeze(-1)
    reference_error_w = data.swing_reference_pos_w - data.actual_swing_foot_pos_w
    target_error_w = data.target_foothold_w - data.actual_swing_foot_pos_w
    stabilization_column = stabilization_active.to(dtype=data.phase.dtype).unsqueeze(-1)
    target_foothold_f = torch.where(
        stabilization_column.bool(),
        torch.zeros_like(data.target_foothold_f),
        data.target_foothold_f,
    )
    feasible_velocity_f = torch.where(
        stabilization_column.bool(),
        torch.zeros_like(data.feasible_velocity_f),
        data.feasible_velocity_f,
    )
    reference_error_w = torch.where(
        stabilization_column.bool(),
        torch.zeros_like(reference_error_w),
        reference_error_w,
    )
    target_error_w = torch.where(
        stabilization_column.bool(),
        torch.zeros_like(target_error_w),
        target_error_w,
    )

    env_ids = torch.arange(
        data.foot_contact.shape[0],
        device=data.foot_contact.device,
    )
    swing_side = data.swing_side.long().clamp(0, 1)
    swing_foot_contact = data.foot_contact[env_ids, swing_side].to(
        dtype=data.phase.dtype,
    ).unsqueeze(-1)
    swing_has_lifted = data.swing_has_lifted.to(
        dtype=data.phase.dtype,
    ).unsqueeze(-1)
    obs = torch.cat(
        (
            target_foothold_f,
            feasible_velocity_f,
            phase,
            swing_side_sign,
            swing_apex_height,
            swing_apex_delta,
            swing_clearance_safe,
            swing_clearance_penetration,
            reference_error_w,
            target_error_w,
            swing_foot_contact,
            swing_has_lifted,
            stabilization_column,
        ),
        dim=-1,
    )
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def nominal_foothold_observation(
    env,
    sensor_name: str = "foothold_planner",
    command_name: str | None = "base_velocity",
) -> torch.Tensor:
    """Return the current nominal foothold in the support-foot frame."""

    _sync_desired_velocity_command(env, sensor_name, command_name)
    data = _foothold_planner_data(env, sensor_name)
    nominal_foothold_f = data.raw_unclipped_foothold_f
    if nominal_foothold_f is None:
        raise RuntimeError(
            "Learned foothold observation requires "
            "raw_unclipped_foothold_f."
        )
    return torch.nan_to_num(
        nominal_foothold_f,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
