#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


PLAY_DEBUG_PREFIX = "[PLAY_DEBUG]"
FIELD_PATTERN = re.compile(r"(\w+)=([^=]*?)(?= \w+=|$)")
NOMINAL_STEP_WIDTH_M = 0.18
CURRICULUM_RADIUS_X_BY_LEVEL = (0.04, 0.08, 0.12)
CURRICULUM_RADIUS_Y_BY_LEVEL = (0.02, 0.04, 0.06)
REPORT_SERIES_KEYS = (
    "swing_air_time_s",
    "completed_air_time_s",
    "last_air_time_left_s",
    "last_air_time_right_s",
    "target_delta_x_f",
    "target_delta_y_f",
    "curriculum_residual_x_f",
    "curriculum_residual_y_f",
    "curriculum_usage_x",
    "curriculum_usage_y",
    "curriculum_usage_norm",
    "ellipse_usage",
    "sole_width_y_w",
    "sole_width_xy_w",
    "planned_width_f",
    "actual_minus_planned_width_y_w",
    "actual_width_f",
    "actual_minus_planned_width_f",
    "ref_xy_err",
    "td_xy_err",
)


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw == "None":
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def parse_play_debug_line(line: str) -> dict[str, Any]:
    if PLAY_DEBUG_PREFIX not in line:
        raise ValueError("Line does not contain a PLAY_DEBUG payload.")

    payload_text = line.split(PLAY_DEBUG_PREFIX, maxsplit=1)[1].strip()
    payload: dict[str, Any] = {}
    for match in FIELD_PATTERN.finditer(payload_text):
        key = match.group(1)
        value = _parse_value(match.group(2))
        payload[key] = value
    return payload


def _append_number(series: list[float], value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        series.append(float(value))


def _append_positive_number(series: list[float], value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)) and value > 0.0:
        series.append(float(value))


def _percentile_nearest(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile for an empty series.")
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(mean(values), 5),
        "min": round(min(values), 5),
        "max": round(max(values), 5),
        "p50": round(_percentile_nearest(values, 0.50), 5),
        "p75": round(_percentile_nearest(values, 0.75), 5),
        "p90": round(_percentile_nearest(values, 0.90), 5),
    }


def _empty_side_stats() -> dict[str, list[float]]:
    return {
        "swing_air_time_s": [],
        "target_delta_x_f": [],
        "curriculum_residual_x_f": [],
        "curriculum_residual_y_f": [],
        "curriculum_usage_x": [],
        "curriculum_usage_y": [],
        "curriculum_usage_norm": [],
        "sole_width_y_w": [],
        "sole_width_xy_w": [],
        "planned_width_f": [],
        "actual_minus_planned_width_y_w": [],
        "actual_width_f": [],
        "actual_minus_planned_width_f": [],
        "ref_xy_err": [],
        "td_xy_err": [],
    }


def _side_name(side: Any) -> str | None:
    if side == 0:
        return "left"
    if side == 1:
        return "right"
    return None


def _is_nonzero_vector(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    numeric_values = [
        float(item)
        for item in value
        if not isinstance(item, bool) and isinstance(item, (int, float))
    ]
    return bool(numeric_values) and any(abs(item) > 1e-6 for item in numeric_values)


def _is_calibration_payload(payload: dict[str, Any]) -> bool:
    """Return whether a PLAY_DEBUG row is suitable for planner parameter calibration."""

    if payload.get("zero_act_active") is True:
        return False
    if payload.get("recovery_step") is True:
        return False
    if payload.get("mode") not in {"LEFT_SWING", "RIGHT_SWING"}:
        return False
    if payload.get("planner_valid") is False:
        return False
    if not _is_nonzero_vector(payload.get("target_w")):
        return False

    planned_width = payload.get("planned_width_f")
    if not isinstance(planned_width, (int, float)) or planned_width <= 0.05:
        return False

    actual_width = payload.get("actual_width_f")
    if not isinstance(actual_width, (int, float)) or actual_width <= 0.0:
        return False

    td_xy_err = payload.get("td_xy_err")
    if isinstance(td_xy_err, (int, float)) and td_xy_err > 0.5:
        return False

    return True


def _level_radius(level: Any, radii: tuple[float, ...]) -> float | None:
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        return None
    index = int(level)
    if index < 0:
        index = 0
    if index >= len(radii):
        index = len(radii) - 1
    return radii[index]


def _curriculum_residual_metrics(payload: dict[str, Any]) -> dict[str, float] | None:
    direct_residual = payload.get("curriculum_residual_f")
    direct_radius = payload.get("curriculum_radius_f")
    direct_usage = payload.get("curriculum_usage")
    if (
        isinstance(direct_residual, list)
        and len(direct_residual) >= 2
        and isinstance(direct_radius, list)
        and len(direct_radius) >= 2
    ):
        residual_x = float(direct_residual[0])
        residual_y = float(direct_residual[1])
        radius_x = float(direct_radius[0])
        radius_y = float(direct_radius[1])
        usage_x = abs(residual_x) / radius_x if radius_x > 0.0 else 0.0
        usage_y = abs(residual_y) / radius_y if radius_y > 0.0 else 0.0
        if isinstance(direct_usage, bool) or not isinstance(
            direct_usage,
            (int, float),
        ):
            usage_norm = (usage_x * usage_x + usage_y * usage_y) ** 0.5
        else:
            usage_norm = float(direct_usage)
        return {
            "curriculum_residual_x_f": round(residual_x, 5),
            "curriculum_residual_y_f": round(residual_y, 5),
            "curriculum_usage_x": round(usage_x, 5),
            "curriculum_usage_y": round(usage_y, 5),
            "curriculum_usage_norm": round(usage_norm, 5),
        }

    target_delta = payload.get("target_delta_f")
    command = payload.get("command")
    lookahead = payload.get("lookahead_s")
    side = payload.get("swing_side")
    if (
        not isinstance(target_delta, list)
        or len(target_delta) < 2
        or not isinstance(command, list)
        or len(command) < 2
        or isinstance(lookahead, bool)
        or not isinstance(lookahead, (int, float))
    ):
        return None

    radius_x = _level_radius(
        payload.get("flat_level"),
        CURRICULUM_RADIUS_X_BY_LEVEL,
    )
    radius_y = _level_radius(
        payload.get("flat_level"),
        CURRICULUM_RADIUS_Y_BY_LEVEL,
    )
    if radius_x is None or radius_y is None:
        return None

    side_sign = _side_name(side)
    if side_sign == "left":
        nominal_y = NOMINAL_STEP_WIDTH_M
    elif side_sign == "right":
        nominal_y = -NOMINAL_STEP_WIDTH_M
    else:
        return None

    residual_x = float(target_delta[0]) - float(command[0]) * float(lookahead)
    residual_y = (
        float(target_delta[1])
        - nominal_y
        - float(command[1]) * float(lookahead)
    )
    usage_x = abs(residual_x) / radius_x if radius_x > 0.0 else 0.0
    usage_y = abs(residual_y) / radius_y if radius_y > 0.0 else 0.0
    usage_norm = (usage_x * usage_x + usage_y * usage_y) ** 0.5
    return {
        "curriculum_residual_x_f": round(residual_x, 5),
        "curriculum_residual_y_f": round(residual_y, 5),
        "curriculum_usage_x": round(usage_x, 5),
        "curriculum_usage_y": round(usage_y, 5),
        "curriculum_usage_norm": round(usage_norm, 5),
    }


def analyze_play_debug_lines(
    lines: list[str],
    *,
    skip_until_step: int = 0,
    exclude_zero_act: bool = False,
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for line in lines:
        if PLAY_DEBUG_PREFIX not in line:
            continue
        payload = parse_play_debug_line(line)
        step = payload.get("step")
        if isinstance(step, int) and step < skip_until_step:
            continue
        if exclude_zero_act and payload.get("zero_act_active") is True:
            continue
        payloads.append(payload)

    report = _summarize_play_debug_payloads(payloads)
    calibration_payloads = [
        payload for payload in payloads if _is_calibration_payload(payload)
    ]
    report["calibration_subset"] = _summarize_play_debug_payloads(
        calibration_payloads
    )
    return report


def _summarize_play_debug_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    swing_air_time_s: list[float] = []
    completed_air_time_s: list[float] = []
    last_air_time_left_s: list[float] = []
    last_air_time_right_s: list[float] = []
    target_delta_x_f: list[float] = []
    target_delta_y_f: list[float] = []
    curriculum_residual_x_f: list[float] = []
    curriculum_residual_y_f: list[float] = []
    curriculum_usage_x: list[float] = []
    curriculum_usage_y: list[float] = []
    curriculum_usage_norm: list[float] = []
    ellipse_usage: list[float] = []
    ref_xy_err: list[float] = []
    td_xy_err: list[float] = []
    lookahead_s: list[float] = []
    sole_width_y_w: list[float] = []
    sole_width_xy_w: list[float] = []
    planned_width_f: list[float] = []
    actual_minus_planned_width_y_w: list[float] = []
    actual_width_f: list[float] = []
    actual_minus_planned_width_f: list[float] = []
    by_side = {
        "left": _empty_side_stats(),
        "right": _empty_side_stats(),
    }

    for payload in payloads:
        _append_number(swing_air_time_s, payload.get("swing_air_time_s"))
        _append_number(ellipse_usage, payload.get("ellipse_usage"))
        _append_number(ref_xy_err, payload.get("ref_xy_err"))
        _append_number(td_xy_err, payload.get("td_xy_err"))
        _append_number(lookahead_s, payload.get("lookahead_s"))
        _append_number(sole_width_y_w, payload.get("sole_width_y_w"))
        _append_number(sole_width_xy_w, payload.get("sole_width_xy_w"))
        _append_number(planned_width_f, payload.get("planned_width_f"))
        _append_number(actual_width_f, payload.get("actual_width_f"))
        _append_number(
            actual_minus_planned_width_f,
            payload.get("actual_minus_planned_width_f"),
        )
        if isinstance(payload.get("sole_width_y_w"), (int, float)) and isinstance(
            payload.get("planned_width_f"), (int, float)
        ):
            _append_number(
                actual_minus_planned_width_y_w,
                payload["sole_width_y_w"] - payload["planned_width_f"],
            )

        last_air_time = payload.get("last_air_time_s")
        if isinstance(last_air_time, list) and len(last_air_time) >= 2:
            _append_number(last_air_time_left_s, last_air_time[0])
            _append_number(last_air_time_right_s, last_air_time[1])
            _append_positive_number(completed_air_time_s, last_air_time[0])
            _append_positive_number(completed_air_time_s, last_air_time[1])

        target_delta = payload.get("target_delta_f")
        if isinstance(target_delta, list) and len(target_delta) >= 2:
            _append_number(target_delta_x_f, target_delta[0])
            _append_number(target_delta_y_f, target_delta[1])

        curriculum_metrics = _curriculum_residual_metrics(payload)
        if curriculum_metrics is not None:
            _append_number(
                curriculum_residual_x_f,
                curriculum_metrics["curriculum_residual_x_f"],
            )
            _append_number(
                curriculum_residual_y_f,
                curriculum_metrics["curriculum_residual_y_f"],
            )
            _append_number(
                curriculum_usage_x,
                curriculum_metrics["curriculum_usage_x"],
            )
            _append_number(
                curriculum_usage_y,
                curriculum_metrics["curriculum_usage_y"],
            )
            _append_number(
                curriculum_usage_norm,
                curriculum_metrics["curriculum_usage_norm"],
            )

        side_name = _side_name(payload.get("swing_side"))
        if side_name is None:
            continue
        side_stats = by_side[side_name]
        _append_number(
            side_stats["swing_air_time_s"],
            payload.get("swing_air_time_s"),
        )
        if isinstance(target_delta, list) and len(target_delta) >= 2:
            _append_number(side_stats["target_delta_x_f"], target_delta[0])
        if curriculum_metrics is not None:
            for key in (
                "curriculum_residual_x_f",
                "curriculum_residual_y_f",
                "curriculum_usage_x",
                "curriculum_usage_y",
                "curriculum_usage_norm",
            ):
                _append_number(side_stats[key], curriculum_metrics[key])
        _append_number(side_stats["sole_width_y_w"], payload.get("sole_width_y_w"))
        _append_number(side_stats["sole_width_xy_w"], payload.get("sole_width_xy_w"))
        _append_number(side_stats["planned_width_f"], payload.get("planned_width_f"))
        _append_number(side_stats["actual_width_f"], payload.get("actual_width_f"))
        _append_number(
            side_stats["actual_minus_planned_width_f"],
            payload.get("actual_minus_planned_width_f"),
        )
        if isinstance(payload.get("sole_width_y_w"), (int, float)) and isinstance(
            payload.get("planned_width_f"), (int, float)
        ):
            _append_number(
                side_stats["actual_minus_planned_width_y_w"],
                payload["sole_width_y_w"] - payload["planned_width_f"],
            )
        _append_number(side_stats["ref_xy_err"], payload.get("ref_xy_err"))
        _append_number(side_stats["td_xy_err"], payload.get("td_xy_err"))

    side_report: dict[str, Any] = {}
    for name, stats in by_side.items():
        side_report[name] = {
            "count": len(stats["swing_air_time_s"]),
            "swing_air_time_s": summarize(stats["swing_air_time_s"]),
            "target_delta_x_f": summarize(stats["target_delta_x_f"]),
            "curriculum_residual_x_f": summarize(
                stats["curriculum_residual_x_f"]
            ),
            "curriculum_residual_y_f": summarize(
                stats["curriculum_residual_y_f"]
            ),
            "curriculum_usage_x": summarize(stats["curriculum_usage_x"]),
            "curriculum_usage_y": summarize(stats["curriculum_usage_y"]),
            "curriculum_usage_norm": summarize(stats["curriculum_usage_norm"]),
            "sole_width_y_w": summarize(stats["sole_width_y_w"]),
            "sole_width_xy_w": summarize(stats["sole_width_xy_w"]),
            "planned_width_f": summarize(stats["planned_width_f"]),
            "actual_minus_planned_width_y_w": summarize(
                stats["actual_minus_planned_width_y_w"]
            ),
            "actual_width_f": summarize(stats["actual_width_f"]),
            "actual_minus_planned_width_f": summarize(
                stats["actual_minus_planned_width_f"]
            ),
            "ref_xy_err": summarize(stats["ref_xy_err"]),
            "td_xy_err": summarize(stats["td_xy_err"]),
        }

    suggested_lookahead = None
    if completed_air_time_s:
        suggested_lookahead = summarize(completed_air_time_s)["p50"]

    current_lookahead = lookahead_s[-1] if lookahead_s else None
    if current_lookahead is not None:
        current_lookahead = round(float(current_lookahead), 5)

    return {
        "play_debug_count": len(payloads),
        "current_lookahead_s": current_lookahead,
        "suggested_velocity_lookahead_s": suggested_lookahead,
        "swing_air_time_s": summarize(swing_air_time_s),
        "completed_air_time_s": summarize(completed_air_time_s),
        "last_air_time_left_s": summarize(last_air_time_left_s),
        "last_air_time_right_s": summarize(last_air_time_right_s),
        "target_delta_x_f": summarize(target_delta_x_f),
        "target_delta_y_f": summarize(target_delta_y_f),
        "curriculum_residual_x_f": summarize(curriculum_residual_x_f),
        "curriculum_residual_y_f": summarize(curriculum_residual_y_f),
        "curriculum_usage_x": summarize(curriculum_usage_x),
        "curriculum_usage_y": summarize(curriculum_usage_y),
        "curriculum_usage_norm": summarize(curriculum_usage_norm),
        "ellipse_usage": summarize(ellipse_usage),
        "sole_width_y_w": summarize(sole_width_y_w),
        "sole_width_xy_w": summarize(sole_width_xy_w),
        "planned_width_f": summarize(planned_width_f),
        "actual_minus_planned_width_y_w": summarize(
            actual_minus_planned_width_y_w
        ),
        "actual_width_f": summarize(actual_width_f),
        "actual_minus_planned_width_f": summarize(actual_minus_planned_width_f),
        "ref_xy_err": summarize(ref_xy_err),
        "td_xy_err": summarize(td_xy_err),
        "by_swing_side": side_report,
    }


def analyze_play_debug_file(
    path: Path,
    *,
    skip_until_step: int = 0,
    exclude_zero_act: bool = False,
) -> dict[str, Any]:
    return analyze_play_debug_lines(
        path.read_text().splitlines(),
        skip_until_step=skip_until_step,
        exclude_zero_act=exclude_zero_act,
    )


def _print_text_report(report: dict[str, Any]) -> None:
    print(f"play_debug_count: {report['play_debug_count']}")
    print(f"current_lookahead_s: {report['current_lookahead_s']}")
    print(
        "suggested_velocity_lookahead_s_from_p50_completed_air_time: "
        f"{report['suggested_velocity_lookahead_s']}"
    )
    for key in REPORT_SERIES_KEYS:
        print(f"{key}: {report[key]}")
    print("by_swing_side:")
    print(json.dumps(report["by_swing_side"], indent=2, ensure_ascii=False))
    calibration_subset = report.get("calibration_subset")
    if isinstance(calibration_subset, dict):
        print("calibration_subset:")
        print(f"  play_debug_count: {calibration_subset['play_debug_count']}")
        print(f"  current_lookahead_s: {calibration_subset['current_lookahead_s']}")
        print(
            "  suggested_velocity_lookahead_s_from_p50_completed_air_time: "
            f"{calibration_subset['suggested_velocity_lookahead_s']}"
        )
        for key in REPORT_SERIES_KEYS:
            print(f"  {key}: {calibration_subset[key]}")
        print("  by_swing_side:")
        print(
            json.dumps(
                calibration_subset["by_swing_side"],
                indent=2,
                ensure_ascii=False,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze PLAY_DEBUG foothold logs for parameter calibration.",
    )
    parser.add_argument("logfile", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )
    parser.add_argument(
        "--skip-until-step",
        type=int,
        default=0,
        help="Ignore PLAY_DEBUG rows before this printed step.",
    )
    parser.add_argument(
        "--exclude-zero-act",
        action="store_true",
        help="Ignore rows printed while --zero_act_until was active.",
    )
    args = parser.parse_args()

    report = analyze_play_debug_file(
        args.logfile,
        skip_until_step=args.skip_until_step,
        exclude_zero_act=args.exclude_zero_act,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_text_report(report)


if __name__ == "__main__":
    main()
