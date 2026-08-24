from __future__ import annotations

import torch


def _foothold_planner_data(env, sensor_name: str):
    return env.scene.sensors[sensor_name].data


def _world_vector_to_planner_frame(
    vector_w: torch.Tensor,
    frame_yaw_w: torch.Tensor,
) -> torch.Tensor:
    """Rotate a world vector into the frozen planner/support frame.

    The planner publishes ``*_f`` quantities in the support-foot frame whose
    yaw is captured when the HOLD transaction is prepared.  Position errors
    are translations, so only the yaw rotation is applied; the vertical
    component is intentionally preserved instead of being mixed by roll or
    pitch.
    """
    cos_yaw = torch.cos(frame_yaw_w)
    sin_yaw = torch.sin(frame_yaw_w)
    vector_f = vector_w.clone()
    vector_f[..., 0] = (
        cos_yaw * vector_w[..., 0] + sin_yaw * vector_w[..., 1]
    )
    vector_f[..., 1] = (
        -sin_yaw * vector_w[..., 0] + cos_yaw * vector_w[..., 1]
    )
    return vector_f


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
    planner_frame_yaw_w = getattr(data, "nominal_frame_yaw_w", None)
    if planner_frame_yaw_w is None:
        raise RuntimeError(
            "Foothold planner observations require nominal_frame_yaw_w "
            "to express world position errors in the frozen planner frame."
        )
    reference_error_f = _world_vector_to_planner_frame(
        reference_error_w,
        planner_frame_yaw_w,
    )
    target_error_f = _world_vector_to_planner_frame(
        target_error_w,
        planner_frame_yaw_w,
    )
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
    reference_error_f = torch.where(
        stabilization_column.bool(),
        torch.zeros_like(reference_error_f),
        reference_error_f,
    )
    target_error_f = torch.where(
        stabilization_column.bool(),
        torch.zeros_like(target_error_f),
        target_error_f,
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
            reference_error_f,
            target_error_f,
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
