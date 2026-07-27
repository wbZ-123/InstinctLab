from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import math


GAIT_MODE_NAMES = {
    0: "HOLD",
    1: "LEFT_SWING",
    2: "RIGHT_SWING",
    3: "TOUCHDOWN_CONFIRM",
    4: "EARLY_CONTACT",
    5: "OVERDUE",
    6: "STANCE_LOST",
    7: "PLAN_INVALID",
    8: "RECOVERY",
    9: "HOLD_CONTACT_LOST",
}

STARTUP_DIAGNOSTIC_JOINT_NAMES = [
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

STARTUP_DIAGNOSTIC_FOOT_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


def _safe_getattr(obj: Any, name: str) -> Any | None:
    return getattr(obj, name, None) if obj is not None else None


def _select_env_value(value: Any, env_id: int = 0) -> Any | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        tensor = value.detach()
        if tensor.ndim > 0:
            tensor = tensor[env_id]
        tensor = tensor.cpu()
        if tensor.ndim == 0:
            item = tensor.item()
            if isinstance(item, bool):
                return bool(item)
            if isinstance(item, int):
                return int(item)
            return round(float(item), 5)
        return [
            bool(item) if isinstance(item, bool) else round(float(item), 5)
            for item in tensor.reshape(-1).tolist()
        ]
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return list(value[env_id])
        return list(value)
    return value


def _select_env_indices(value: Any, env_id: int, indices: list[int] | None) -> Any | None:
    selected = _select_env_value(value, env_id)
    if selected is None or not indices:
        return selected
    if not isinstance(selected, list):
        return selected
    if len(selected) == len(indices):
        return selected
    if max(indices) >= len(selected) or min(indices) < 0:
        return selected
    return [selected[index] for index in indices]


def _get_planner_contact_body_ids(sensor: Any | None) -> list[int] | None:
    left_id = _safe_getattr(sensor, "_left_contact_body_id")
    right_id = _safe_getattr(sensor, "_right_contact_body_id")
    if isinstance(left_id, int) and isinstance(right_id, int):
        return [left_id, right_id]
    return None


def _get_planner_contact_body_names(
    sensor: Any | None,
    contact_body_ids: list[int] | None,
) -> list[str] | None:
    contact_body_names = _safe_getattr(sensor, "_contact_body_names")
    if not isinstance(contact_body_names, (list, tuple)) or not contact_body_ids:
        return None
    if min(contact_body_ids) < 0 or max(contact_body_ids) >= len(contact_body_names):
        return None
    return [str(contact_body_names[index]) for index in contact_body_ids]


def _get_sensor(base_env: Any, sensor_name: str) -> Any | None:
    scene = _safe_getattr(base_env, "scene")
    sensors = _safe_getattr(scene, "sensors")
    if isinstance(sensors, Mapping):
        return sensors.get(sensor_name)
    return None


def _get_sensor_data(base_env: Any, sensor_name: str) -> Any | None:
    return _safe_getattr(_get_sensor(base_env, sensor_name), "data")


def _get_command(base_env: Any, command_name: str, env_id: int) -> Any | None:
    command_manager = _safe_getattr(base_env, "command_manager")
    if command_manager is None:
        return None
    try:
        command = command_manager.get_command(command_name)
    except (AttributeError, KeyError, RuntimeError):
        return None
    return _select_env_value(command, env_id)


def _get_scene_entity(scene: Any, name: str) -> Any | None:
    if scene is None:
        return None
    try:
        return scene[name]
    except Exception:
        pass
    articulations = _safe_getattr(scene, "articulations")
    if isinstance(articulations, Mapping):
        return articulations.get(name)
    return _safe_getattr(scene, name)


def _resolve_entity_indices(
    entity: Any | None,
    finder_name: str,
    names: list[str],
) -> tuple[list[int] | None, list[str] | None]:
    finder = _safe_getattr(entity, finder_name)
    if finder is None:
        return None, None
    try:
        indices, resolved_names = finder(names, preserve_order=True)
    except Exception:
        return None, None
    if indices is None:
        return None, None
    return list(indices), [str(name) for name in resolved_names]


def _select_named_tensor_values(
    value: Any,
    *,
    env_id: int,
    indices: list[int] | None,
    names: list[str] | None,
) -> dict[str, float] | None:
    if value is None or indices is None or names is None:
        return None
    selected = _select_env_indices(value, env_id, indices)
    if not isinstance(selected, list):
        return None
    return {
        name: selected[index]
        for index, name in enumerate(names)
        if index < len(selected)
    }


def _select_named_vector_values(
    value: Any,
    *,
    env_id: int,
    indices: list[int] | None,
    names: list[str] | None,
) -> dict[str, list[float]] | None:
    if value is None or indices is None or names is None:
        return None
    if not hasattr(value, "detach"):
        return None
    tensor = value.detach()
    if tensor.ndim < 3:
        return None
    if max(indices) >= tensor.shape[1] or min(indices) < 0:
        return None
    selected = tensor[env_id, indices].cpu()
    return {
        name: [round(float(item), 5) for item in selected[row_index].reshape(-1).tolist()]
        for row_index, name in enumerate(names)
    }


def _quat_wxyz_to_rpy(quat: list[float]) -> list[float] | None:
    if len(quat) < 4:
        return None
    w, x, y, z = [float(item) for item in quat[:4]]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [round(roll, 5), round(pitch, 5), round(yaw, 5)]


def _select_named_body_rpy_w(
    body_quat_w: Any,
    *,
    env_id: int,
    indices: list[int] | None,
    names: list[str] | None,
) -> dict[str, list[float]] | None:
    quat_by_body = _select_named_vector_values(
        body_quat_w,
        env_id=env_id,
        indices=indices,
        names=names,
    )
    if quat_by_body is None:
        return None
    return {
        name: rpy
        for name, quat in quat_by_body.items()
        if (rpy := _quat_wxyz_to_rpy(quat)) is not None
    }


def _build_startup_pose_diagnostics(
    base_env: Any,
    *,
    env_id: int,
    actions: Any | None = None,
) -> dict[str, Any]:
    scene = _safe_getattr(base_env, "scene")
    robot = _get_scene_entity(scene, "robot")
    robot_data = _safe_getattr(robot, "data")

    joint_ids, joint_names = _resolve_entity_indices(
        robot,
        "find_joints",
        STARTUP_DIAGNOSTIC_JOINT_NAMES,
    )
    body_ids, body_names = _resolve_entity_indices(
        robot,
        "find_bodies",
        STARTUP_DIAGNOSTIC_FOOT_BODY_NAMES,
    )

    return {
        "ankle_joint_ids": joint_ids,
        "ankle_joint_names": joint_names,
        "ankle_joint_pos": _select_named_tensor_values(
            _safe_getattr(robot_data, "joint_pos"),
            env_id=env_id,
            indices=joint_ids,
            names=joint_names,
        ),
        "ankle_joint_vel": _select_named_tensor_values(
            _safe_getattr(robot_data, "joint_vel"),
            env_id=env_id,
            indices=joint_ids,
            names=joint_names,
        ),
        "ankle_action": _select_named_tensor_values(
            actions,
            env_id=env_id,
            indices=joint_ids,
            names=joint_names,
        ),
        "foot_body_ids": body_ids,
        "foot_body_names": body_names,
        "foot_pos_w": _select_named_vector_values(
            _safe_getattr(robot_data, "body_pos_w"),
            env_id=env_id,
            indices=body_ids,
            names=body_names,
        ),
        "foot_rpy_w": _select_named_body_rpy_w(
            _safe_getattr(robot_data, "body_quat_w"),
            env_id=env_id,
            indices=body_ids,
            names=body_names,
        ),
    }


def capture_reset_debug_snapshot(base_env: Any, *, env_id: int = 0) -> dict[str, Any]:
    """Capture reset-sensitive state before env.step can auto-reset buffers."""

    termination_manager = _safe_getattr(base_env, "termination_manager")
    snapshot: dict[str, Any] = {
        "episode_length": _select_env_value(
            _safe_getattr(base_env, "episode_length_buf"),
            env_id,
        ),
        "terminated": _select_env_value(
            _safe_getattr(termination_manager, "terminated"),
            env_id,
        ),
        "time_out": _select_env_value(
            _safe_getattr(termination_manager, "time_outs"),
            env_id,
        ),
    }

    scene = _safe_getattr(base_env, "scene")
    robot_data = _safe_getattr(_get_scene_entity(scene, "robot"), "data")
    snapshot["root_pos_w"] = _select_env_value(
        _safe_getattr(robot_data, "root_pos_w"),
        env_id,
    )
    snapshot["root_quat_w"] = _select_env_value(
        _safe_getattr(robot_data, "root_quat_w"),
        env_id,
    )
    snapshot["root_lin_vel_w"] = _select_env_value(
        _safe_getattr(robot_data, "root_lin_vel_w"),
        env_id,
    )
    snapshot["root_ang_vel_w"] = _select_env_value(
        _safe_getattr(robot_data, "root_ang_vel_w"),
        env_id,
    )

    contact_data = _get_sensor_data(base_env, "contact_forces")
    snapshot["contact_time_s"] = _select_env_value(
        _safe_getattr(contact_data, "current_contact_time"),
        env_id,
    )
    snapshot["air_time_s"] = _select_env_value(
        _safe_getattr(contact_data, "current_air_time"),
        env_id,
    )
    snapshot.update(_build_startup_pose_diagnostics(base_env, env_id=env_id))
    return snapshot


def _touchdown_errors(
    actual_swing_w: Any | None,
    target_w: Any | None,
) -> tuple[float | None, float | None]:
    if actual_swing_w is None or target_w is None:
        return None, None
    if len(actual_swing_w) < 3 or len(target_w) < 3:
        return None, None

    dx = actual_swing_w[0] - target_w[0]
    dy = actual_swing_w[1] - target_w[1]
    dz = abs(actual_swing_w[2] - target_w[2])
    return round(math.sqrt(dx * dx + dy * dy), 5), round(dz, 5)


def _point_errors(
    actual_w: Any | None,
    reference_w: Any | None,
) -> tuple[float | None, float | None]:
    if actual_w is None or reference_w is None:
        return None, None
    if len(actual_w) < 3 or len(reference_w) < 3:
        return None, None

    dx = actual_w[0] - reference_w[0]
    dy = actual_w[1] - reference_w[1]
    dz = abs(actual_w[2] - reference_w[2])
    return round(math.sqrt(dx * dx + dy * dy), 5), round(dz, 5)


def _foot_widths(
    left_sole_w: Any | None,
    right_sole_w: Any | None,
) -> tuple[float | None, float | None]:
    if left_sole_w is None or right_sole_w is None:
        return None, None
    if len(left_sole_w) < 3 or len(right_sole_w) < 3:
        return None, None

    dx = left_sole_w[0] - right_sole_w[0]
    dy = left_sole_w[1] - right_sole_w[1]
    return round(abs(dy), 5), round(math.sqrt(dx * dx + dy * dy), 5)


def _left_right_sole_positions(
    swing_side: Any | None,
    actual_swing_w: Any | None,
    actual_stance_w: Any | None,
) -> tuple[Any | None, Any | None]:
    if swing_side == 0:
        return actual_swing_w, actual_stance_w
    if swing_side == 1:
        return actual_stance_w, actual_swing_w
    return None, None


def _planned_width(target_delta_f: Any | None) -> float | None:
    if target_delta_f is None or len(target_delta_f) < 2:
        return None
    return round(abs(float(target_delta_f[1])), 5)


def _actual_delta_in_target_frame(
    actual_swing_w: Any | None,
    actual_stance_w: Any | None,
    target_f: Any | None,
    target_w: Any | None,
) -> list[float] | None:
    """Project actual swing-minus-stance foot delta into the planner target frame.

    The planner stores the target foot point in a support-foot local frame and
    composes it to world as:
        target_w_xy = stance_w_xy + R(yaw) @ target_f_xy
    Reconstruct the yaw rotation from that already-logged target pair, then
    rotate the actual swing delta back into the same local frame.  This avoids
    using world-y foot width for calibration, which is yaw-dependent.
    """
    if (
        actual_swing_w is None
        or actual_stance_w is None
        or target_f is None
        or target_w is None
    ):
        return None
    if (
        len(actual_swing_w) < 3
        or len(actual_stance_w) < 3
        or len(target_f) < 2
        or len(target_w) < 2
    ):
        return None

    fx = float(target_f[0])
    fy = float(target_f[1])
    target_norm_sq = fx * fx + fy * fy
    if target_norm_sq < 1.0e-12:
        return None

    twx = float(target_w[0]) - float(actual_stance_w[0])
    twy = float(target_w[1]) - float(actual_stance_w[1])
    cos_yaw = (twx * fx + twy * fy) / target_norm_sq
    sin_yaw = (twy * fx - twx * fy) / target_norm_sq
    yaw_norm = math.sqrt(cos_yaw * cos_yaw + sin_yaw * sin_yaw)
    if yaw_norm < 1.0e-12:
        return None
    cos_yaw /= yaw_norm
    sin_yaw /= yaw_norm

    dx_w = float(actual_swing_w[0]) - float(actual_stance_w[0])
    dy_w = float(actual_swing_w[1]) - float(actual_stance_w[1])
    dz_w = float(actual_swing_w[2]) - float(actual_stance_w[2])
    dx_f = cos_yaw * dx_w + sin_yaw * dy_w
    dy_f = -sin_yaw * dx_w + cos_yaw * dy_w
    return [round(dx_f, 5), round(dy_f, 5), round(dz_w, 5)]


def _actual_width(actual_delta_f: Any | None) -> float | None:
    if actual_delta_f is None or len(actual_delta_f) < 2:
        return None
    return round(abs(float(actual_delta_f[1])), 5)


def _actual_minus_planned_width(
    actual_width_f: float | None,
    planned_width_f: float | None,
) -> float | None:
    if actual_width_f is None or planned_width_f is None:
        return None
    return round(actual_width_f - planned_width_f, 5)


def _has_active_foothold_plan(target_delta_f: Any | None) -> bool:
    """Return whether the planner currently has a non-zero local foothold target."""

    if target_delta_f is None or len(target_delta_f) < 2:
        return False
    return abs(float(target_delta_f[0])) > 1.0e-6 or abs(float(target_delta_f[1])) > 1.0e-6


def _is_touchdown_error_meaningful(gait_mode_int: int | None) -> bool:
    return gait_mode_int in {
        1,  # LEFT_SWING
        2,  # RIGHT_SWING
        3,  # TOUCHDOWN_CONFIRM
        5,  # OVERDUE
        8,  # RECOVERY
    }


def _is_reference_error_meaningful(gait_mode_int: int | None) -> bool:
    return gait_mode_int in {
        1,  # LEFT_SWING
        2,  # RIGHT_SWING
        5,  # OVERDUE
        8,  # RECOVERY
    }


def build_foothold_debug_payload(
    base_env: Any,
    *,
    env_id: int = 0,
    command_name: str = "base_velocity",
    sensor_name: str = "foothold_planner",
    actions: Any | None = None,
) -> dict[str, Any]:
    sensor = _get_sensor(base_env, sensor_name)
    data = _get_sensor_data(base_env, sensor_name)
    gait_state = _safe_getattr(sensor, "_gait_state")
    gait_mode = _select_env_value(_safe_getattr(data, "gait_mode"), env_id)
    gait_mode_int = int(gait_mode) if isinstance(gait_mode, int) else None
    actual_swing_w = _select_env_value(
        _safe_getattr(data, "actual_swing_foot_pos_w"),
        env_id,
    )
    actual_stance_w = _select_env_value(
        _safe_getattr(data, "actual_stance_foot_pos_w"),
        env_id,
    )
    target_w = _select_env_value(
        _safe_getattr(data, "target_foothold_w"),
        env_id,
    )
    target_f = _select_env_value(_safe_getattr(data, "target_foothold_f"), env_id)
    swing_reference_w = _select_env_value(
        _safe_getattr(data, "swing_reference_pos_w"),
        env_id,
    )
    swing_start_w = _select_env_value(
        _safe_getattr(data, "swing_start_pos_w"),
        env_id,
    )
    touchdown_xy_error, touchdown_z_error = _touchdown_errors(
        actual_swing_w,
        target_w,
    )
    reference_xy_error, reference_z_error = _point_errors(
        actual_swing_w,
        swing_reference_w,
    )
    swing_side = _select_env_value(_safe_getattr(data, "swing_side"), env_id)
    left_sole_w, right_sole_w = _left_right_sole_positions(
        swing_side,
        actual_swing_w,
        actual_stance_w,
    )
    sole_width_y_w, sole_width_xy_w = _foot_widths(left_sole_w, right_sole_w)
    target_delta_f = _select_env_value(
        _safe_getattr(data, "target_delta_f"),
        env_id,
    )
    planned_width_f = _planned_width(target_delta_f)
    actual_delta_f = _actual_delta_in_target_frame(
        actual_swing_w,
        actual_stance_w,
        target_f,
        target_w,
    )
    actual_width_f = _actual_width(actual_delta_f)
    contact_body_ids = _get_planner_contact_body_ids(sensor)
    contact_body_names = _get_planner_contact_body_names(sensor, contact_body_ids)
    contact_data = _get_sensor_data(base_env, "contact_forces")
    air_time_s = _select_env_indices(
        _safe_getattr(contact_data, "current_air_time"),
        env_id,
        contact_body_ids,
    )
    last_air_time_s = _select_env_indices(
        _safe_getattr(contact_data, "last_air_time"),
        env_id,
        contact_body_ids,
    )
    contact_time_s = _select_env_indices(
        _safe_getattr(contact_data, "current_contact_time"),
        env_id,
        contact_body_ids,
    )
    swing_air_time_s = None
    if isinstance(air_time_s, list) and isinstance(swing_side, int):
        if 0 <= swing_side < len(air_time_s):
            swing_air_time_s = air_time_s[swing_side]
    has_active_plan = _has_active_foothold_plan(target_delta_f)
    if not has_active_plan or not _is_touchdown_error_meaningful(gait_mode_int):
        touchdown_xy_error = None
        touchdown_z_error = None
    if not has_active_plan or not _is_reference_error_meaningful(gait_mode_int):
        reference_xy_error = None
        reference_z_error = None

    payload = {
        "command": _get_command(base_env, command_name, env_id),
        "gait_mode": GAIT_MODE_NAMES.get(gait_mode_int, gait_mode),
        "swing_side": swing_side,
        "phase": _select_env_value(_safe_getattr(data, "phase"), env_id),
        "hold_elapsed_s": _select_env_value(
            _safe_getattr(gait_state, "hold_elapsed_s"), env_id
        ),
        "hold_required_s": _select_env_value(
            _safe_getattr(gait_state, "hold_required_s"), env_id
        ),
        "contact_elapsed_s": _select_env_value(
            _safe_getattr(gait_state, "contact_elapsed_s"), env_id
        ),
        "no_contact_elapsed_s": _select_env_value(
            _safe_getattr(gait_state, "no_contact_elapsed_s"), env_id
        ),
        "air_time_s": air_time_s,
        "last_air_time_s": last_air_time_s,
        "contact_time_s": contact_time_s,
        "contact_body_ids": contact_body_ids,
        "contact_body_names": contact_body_names,
        "swing_air_time_s": swing_air_time_s,
        "foot_contact": _select_env_value(_safe_getattr(data, "foot_contact"), env_id),
        "planner_valid": _select_env_value(_safe_getattr(data, "planner_valid"), env_id),
        "touchdown_accepted": _select_env_value(
            _safe_getattr(data, "touchdown_accepted"), env_id
        ),
        "touchdown_swing_contact": _select_env_value(
            _safe_getattr(data, "touchdown_swing_contact"), env_id
        ),
        "touchdown_xy_ok": _select_env_value(
            _safe_getattr(data, "touchdown_xy_ok"), env_id
        ),
        "touchdown_z_ok": _select_env_value(
            _safe_getattr(data, "touchdown_z_ok"), env_id
        ),
        "touchdown_within_tolerance": _select_env_value(
            _safe_getattr(data, "touchdown_within_tolerance"), env_id
        ),
        "swing_has_lifted": _select_env_value(
            _safe_getattr(data, "swing_has_lifted"), env_id
        ),
        "recovery_step_active": _select_env_value(
            _safe_getattr(data, "recovery_step_active"), env_id
        ),
        "safe_target_search": _select_env_value(
            _safe_getattr(data, "safe_target_search_performed"), env_id
        ),
        "safe_target_valid": _select_env_value(
            _safe_getattr(data, "safe_target_final_valid"), env_id
        ),
        "safe_target_fallback": _select_env_value(
            _safe_getattr(data, "safe_target_used_fallback"), env_id
        ),
        "safe_target_score": _select_env_value(
            _safe_getattr(data, "safe_target_score"), env_id
        ),
        "target_f": target_f,
        "target_w": target_w,
        "swing_reference_w": swing_reference_w,
        "swing_start_w": swing_start_w,
        "actual_swing_w": actual_swing_w,
        "actual_stance_w": actual_stance_w,
        "left_sole_w": left_sole_w,
        "right_sole_w": right_sole_w,
        "sole_width_y_w": sole_width_y_w,
        "sole_width_xy_w": sole_width_xy_w,
        "planned_width_f": planned_width_f,
        "actual_delta_f": actual_delta_f,
        "actual_width_f": actual_width_f,
        "actual_minus_planned_width_f": _actual_minus_planned_width(
            actual_width_f,
            planned_width_f,
        ),
        "reference_xy_error": reference_xy_error,
        "reference_z_error": reference_z_error,
        "touchdown_xy_error": touchdown_xy_error,
        "touchdown_z_error": touchdown_z_error,
        "swing_apex_height": _select_env_value(
            _safe_getattr(data, "swing_apex_height"), env_id
        ),
        "default_swing_apex_height": _select_env_value(
            _safe_getattr(data, "default_swing_apex_height"), env_id
        ),
        "flat_target_level": _select_env_value(
            _safe_getattr(data, "flat_target_level"), env_id
        ),
        "velocity_lookahead_s": _select_env_value(
            _safe_getattr(data, "velocity_lookahead_s"), env_id
        ),
        "target_delta_f": target_delta_f,
        "curriculum_residual_f": _select_env_value(
            _safe_getattr(data, "curriculum_residual_f"), env_id
        ),
        "curriculum_radius_f": _select_env_value(
            _safe_getattr(data, "curriculum_radius_f"), env_id
        ),
        "curriculum_usage": _select_env_value(
            _safe_getattr(data, "curriculum_usage"), env_id
        ),
        "target_ellipse_max_x": _select_env_value(
            _safe_getattr(data, "target_ellipse_max_x"), env_id
        ),
        "target_ellipse_usage": _select_env_value(
            _safe_getattr(data, "target_ellipse_usage"), env_id
        ),
        "feasible_velocity_f": _select_env_value(
            _safe_getattr(data, "feasible_velocity_f"), env_id
        ),
    }
    payload.update(
        _build_startup_pose_diagnostics(
            base_env,
            env_id=env_id,
            actions=actions,
        )
    )
    return payload


def build_reset_debug_payload(
    base_env: Any,
    *,
    env_id: int = 0,
    done: bool | None = None,
    pre_step_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    termination_manager = _safe_getattr(base_env, "termination_manager")
    term_names = _safe_getattr(termination_manager, "_term_names") or []
    last_episode_dones = _safe_getattr(
        termination_manager,
        "_last_episode_dones",
    )
    active_terms: list[str] = []
    if last_episode_dones is not None and term_names:
        done_row = _select_env_value(last_episode_dones, env_id)
        if isinstance(done_row, list):
            active_terms = [
                name
                for name, active in zip(term_names, done_row, strict=False)
                if bool(active)
            ]

    pre_step_snapshot = pre_step_snapshot or {}
    return {
        "env_id": env_id,
        "done": bool(done) if done is not None else None,
        "terminated": _select_env_value(
            _safe_getattr(termination_manager, "terminated"), env_id
        ),
        "time_out": _select_env_value(
            _safe_getattr(termination_manager, "time_outs"), env_id
        ),
        "terminated_pre_step": pre_step_snapshot.get("terminated"),
        "time_out_pre_step": pre_step_snapshot.get("time_out"),
        "episode_length": pre_step_snapshot.get(
            "episode_length",
            _select_env_value(_safe_getattr(base_env, "episode_length_buf"), env_id),
        ),
        "root_pos_w_pre_step": pre_step_snapshot.get("root_pos_w"),
        "root_quat_w_pre_step": pre_step_snapshot.get("root_quat_w"),
        "root_lin_vel_w_pre_step": pre_step_snapshot.get("root_lin_vel_w"),
        "root_ang_vel_w_pre_step": pre_step_snapshot.get("root_ang_vel_w"),
        "contact_time_s_pre_step": pre_step_snapshot.get("contact_time_s"),
        "air_time_s_pre_step": pre_step_snapshot.get("air_time_s"),
        "ankle_joint_pos_pre_step": pre_step_snapshot.get("ankle_joint_pos"),
        "ankle_joint_vel_pre_step": pre_step_snapshot.get("ankle_joint_vel"),
        "foot_pos_w_pre_step": pre_step_snapshot.get("foot_pos_w"),
        "foot_rpy_w_pre_step": pre_step_snapshot.get("foot_rpy_w"),
        "active_terms": active_terms,
    }


def format_reset_debug_line(timestep: int, payload: dict[str, Any]) -> str:
    return (
        f"[RESET_DEBUG] step={timestep} "
        f"env_id={payload['env_id']} "
        f"done={payload['done']} "
        f"terminated={payload['terminated']} "
        f"time_out={payload['time_out']} "
        f"terminated_pre_step={payload['terminated_pre_step']} "
        f"time_out_pre_step={payload['time_out_pre_step']} "
        f"episode_length={payload['episode_length']} "
        f"root_pos_w_pre_step={payload['root_pos_w_pre_step']} "
        f"root_quat_w_pre_step={payload['root_quat_w_pre_step']} "
        f"root_lin_vel_w_pre_step={payload['root_lin_vel_w_pre_step']} "
        f"root_ang_vel_w_pre_step={payload['root_ang_vel_w_pre_step']} "
        f"contact_time_s_pre_step={payload['contact_time_s_pre_step']} "
        f"air_time_s_pre_step={payload['air_time_s_pre_step']} "
        f"ankle_pos_pre_step={payload['ankle_joint_pos_pre_step']} "
        f"ankle_vel_pre_step={payload['ankle_joint_vel_pre_step']} "
        f"foot_pos_w_pre_step={payload['foot_pos_w_pre_step']} "
        f"foot_rpy_w_pre_step={payload['foot_rpy_w_pre_step']} "
        f"active_terms={payload['active_terms']}"
    )


def format_foothold_debug_line(
    timestep: int,
    payload: dict[str, Any],
    *,
    zero_act_active: bool | None = None,
) -> str:
    return (
        f"[PLAY_DEBUG] step={timestep} "
        f"zero_act_active={zero_act_active} "
        f"command={payload['command']} "
        f"mode={payload['gait_mode']} "
        f"swing_side={payload['swing_side']} "
        f"phase={payload['phase']} "
        f"hold_elapsed_s={payload['hold_elapsed_s']} "
        f"hold_required_s={payload['hold_required_s']} "
        f"contact_elapsed_s={payload['contact_elapsed_s']} "
        f"no_contact_elapsed_s={payload['no_contact_elapsed_s']} "
        f"air_time_s={payload['air_time_s']} "
        f"last_air_time_s={payload['last_air_time_s']} "
        f"contact_time_s={payload['contact_time_s']} "
        f"contact_body_ids={payload['contact_body_ids']} "
        f"swing_air_time_s={payload['swing_air_time_s']} "
        f"contact={payload['foot_contact']} "
        f"planner_valid={payload['planner_valid']} "
        f"touchdown={payload['touchdown_accepted']} "
        f"td_contact={payload['touchdown_swing_contact']} "
        f"td_xy_ok={payload['touchdown_xy_ok']} "
        f"td_z_ok={payload['touchdown_z_ok']} "
        f"td_within_tol={payload['touchdown_within_tolerance']} "
        f"lifted={payload['swing_has_lifted']} "
        f"recovery_step={payload['recovery_step_active']} "
        f"safe_search={payload['safe_target_search']} "
        f"safe_valid={payload['safe_target_valid']} "
        f"fallback={payload['safe_target_fallback']} "
        f"score={payload['safe_target_score']} "
        f"target_f={payload['target_f']} "
        f"target_w={payload['target_w']} "
        f"swing_ref_w={payload['swing_reference_w']} "
        f"swing_start_w={payload['swing_start_w']} "
        f"actual_swing_w={payload['actual_swing_w']} "
        f"actual_stance_w={payload['actual_stance_w']} "
        f"left_sole_w={payload['left_sole_w']} "
        f"right_sole_w={payload['right_sole_w']} "
        f"sole_width_y_w={payload['sole_width_y_w']} "
        f"sole_width_xy_w={payload['sole_width_xy_w']} "
        f"planned_width_f={payload['planned_width_f']} "
        f"actual_delta_f={payload['actual_delta_f']} "
        f"actual_width_f={payload['actual_width_f']} "
        f"actual_minus_planned_width_f={payload['actual_minus_planned_width_f']} "
        f"ref_xy_err={payload['reference_xy_error']} "
        f"ref_z_err={payload['reference_z_error']} "
        f"td_xy_err={payload['touchdown_xy_error']} "
        f"td_z_err={payload['touchdown_z_error']} "
        f"apex={payload['swing_apex_height']} "
        f"default_apex={payload['default_swing_apex_height']} "
        f"flat_level={payload['flat_target_level']} "
        f"lookahead_s={payload['velocity_lookahead_s']} "
        f"target_delta_f={payload['target_delta_f']} "
        f"curriculum_residual_f={payload['curriculum_residual_f']} "
        f"curriculum_radius_f={payload['curriculum_radius_f']} "
        f"curriculum_usage={payload['curriculum_usage']} "
        f"ellipse_max_x={payload['target_ellipse_max_x']} "
        f"ellipse_usage={payload['target_ellipse_usage']} "
        f"feasible_velocity_f={payload['feasible_velocity_f']}"
        f" ankle_pos={payload['ankle_joint_pos']}"
        f" ankle_vel={payload['ankle_joint_vel']}"
        f" ankle_action={payload['ankle_action']}"
        f" foot_pos_w={payload['foot_pos_w']}"
        f" foot_rpy_w={payload['foot_rpy_w']}"
    )


def is_foothold_debug_anomaly(
    payload: dict[str, Any],
    *,
    reference_xy_error_threshold: float = 0.10,
    touchdown_xy_error_threshold: float = 0.10,
    min_phase_for_liftoff_check: float = 0.20,
) -> bool:
    if payload.get("gait_mode") in {
        "EARLY_CONTACT",
        "OVERDUE",
        "STANCE_LOST",
        "PLAN_INVALID",
        "RECOVERY",
        "HOLD_CONTACT_LOST",
    }:
        return True

    reference_xy_error = payload.get("reference_xy_error")
    if (
        isinstance(reference_xy_error, (int, float))
        and reference_xy_error > reference_xy_error_threshold
    ):
        return True

    touchdown_xy_error = payload.get("touchdown_xy_error")
    if (
        isinstance(touchdown_xy_error, (int, float))
        and touchdown_xy_error > touchdown_xy_error_threshold
    ):
        return True

    phase = payload.get("phase")
    if (
        isinstance(phase, (int, float))
        and phase >= min_phase_for_liftoff_check
        and payload.get("swing_has_lifted") is False
        and payload.get("gait_mode") in {"LEFT_SWING", "RIGHT_SWING"}
    ):
        return True

    return False
