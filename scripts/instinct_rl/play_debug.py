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
}


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


def _get_sensor_data(base_env: Any, sensor_name: str) -> Any | None:
    scene = _safe_getattr(base_env, "scene")
    sensors = _safe_getattr(scene, "sensors")
    if isinstance(sensors, Mapping):
        sensor = sensors.get(sensor_name)
    else:
        sensor = None
    return _safe_getattr(sensor, "data")


def _get_command(base_env: Any, command_name: str, env_id: int) -> Any | None:
    command_manager = _safe_getattr(base_env, "command_manager")
    if command_manager is None:
        return None
    try:
        command = command_manager.get_command(command_name)
    except (AttributeError, KeyError, RuntimeError):
        return None
    return _select_env_value(command, env_id)


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


def _is_touchdown_error_meaningful(gait_mode_int: int | None) -> bool:
    return gait_mode_int in {
        1,  # LEFT_SWING
        2,  # RIGHT_SWING
        3,  # TOUCHDOWN_CONFIRM
        5,  # OVERDUE
        8,  # RECOVERY
    }


def build_foothold_debug_payload(
    base_env: Any,
    *,
    env_id: int = 0,
    command_name: str = "base_velocity",
    sensor_name: str = "foothold_planner",
) -> dict[str, Any]:
    data = _get_sensor_data(base_env, sensor_name)
    gait_mode = _select_env_value(_safe_getattr(data, "gait_mode"), env_id)
    gait_mode_int = int(gait_mode) if isinstance(gait_mode, int) else None
    actual_swing_w = _select_env_value(
        _safe_getattr(data, "actual_swing_foot_pos_w"),
        env_id,
    )
    target_w = _select_env_value(
        _safe_getattr(data, "target_foothold_w"),
        env_id,
    )
    touchdown_xy_error, touchdown_z_error = _touchdown_errors(
        actual_swing_w,
        target_w,
    )
    if not _is_touchdown_error_meaningful(gait_mode_int):
        touchdown_xy_error = None
        touchdown_z_error = None

    return {
        "command": _get_command(base_env, command_name, env_id),
        "gait_mode": GAIT_MODE_NAMES.get(gait_mode_int, gait_mode),
        "swing_side": _select_env_value(_safe_getattr(data, "swing_side"), env_id),
        "phase": _select_env_value(_safe_getattr(data, "phase"), env_id),
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
        "target_f": _select_env_value(_safe_getattr(data, "target_foothold_f"), env_id),
        "target_w": target_w,
        "actual_swing_w": actual_swing_w,
        "touchdown_xy_error": touchdown_xy_error,
        "touchdown_z_error": touchdown_z_error,
        "feasible_velocity_f": _select_env_value(
            _safe_getattr(data, "feasible_velocity_f"), env_id
        ),
    }


def format_foothold_debug_line(timestep: int, payload: dict[str, Any]) -> str:
    return (
        f"[PLAY_DEBUG] step={timestep} "
        f"command={payload['command']} "
        f"mode={payload['gait_mode']} "
        f"swing_side={payload['swing_side']} "
        f"phase={payload['phase']} "
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
        f"actual_swing_w={payload['actual_swing_w']} "
        f"td_xy_err={payload['touchdown_xy_error']} "
        f"td_z_err={payload['touchdown_z_error']} "
        f"feasible_velocity_f={payload['feasible_velocity_f']}"
    )
