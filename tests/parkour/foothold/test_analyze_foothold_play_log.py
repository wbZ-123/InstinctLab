from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_analyzer_module():
    path = (
        Path(__file__).resolve().parent
        / "analyze_foothold_play_log.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_foothold_play_log_under_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_play_debug_line_extracts_lists_scalars_and_modes():
    module = _load_analyzer_module()

    payload = module.parse_play_debug_line(
        "[PLAY_DEBUG] step=20 command=[0.6, 0.0, 0.1] "
        "mode=LEFT_SWING swing_side=0 phase=0.5 "
        "air_time_s=[0.12, 0.03] last_air_time_s=[0.31, 0.28] "
        "contact_time_s=[0.0, 0.40] swing_air_time_s=0.12 "
        "target_delta_f=[0.18, 0.17] ellipse_usage=0.64 "
        "lookahead_s=0.10 ref_xy_err=0.04 td_xy_err=None"
    )

    assert payload["step"] == 20
    assert payload["mode"] == "LEFT_SWING"
    assert payload["swing_side"] == 0
    assert payload["command"] == [0.6, 0.0, 0.1]
    assert payload["air_time_s"] == [0.12, 0.03]
    assert payload["swing_air_time_s"] == 0.12
    assert payload["target_delta_f"] == [0.18, 0.17]
    assert payload["ellipse_usage"] == 0.64
    assert payload["td_xy_err"] is None


def test_analyze_play_debug_lines_reports_percentiles_and_side_balance():
    module = _load_analyzer_module()
    lines = [
        "[PLAY_DEBUG] step=10 mode=LEFT_SWING swing_side=0 "
        "swing_air_time_s=0.10 last_air_time_s=[0.20, 0.15] "
        "target_delta_f=[0.10, 0.18] ellipse_usage=0.30 "
        "sole_width_y_w=0.22 sole_width_xy_w=0.23 planned_width_f=0.18 "
        "actual_width_f=0.20 actual_minus_planned_width_f=0.02 "
        "lookahead_s=0.10 ref_xy_err=0.06 td_xy_err=0.04",
        "[PLAY_DEBUG] step=20 mode=RIGHT_SWING swing_side=1 "
        "swing_air_time_s=0.20 last_air_time_s=[0.22, 0.30] "
        "target_delta_f=[0.20, -0.18] ellipse_usage=0.60 "
        "sole_width_y_w=0.24 sole_width_xy_w=0.25 planned_width_f=0.18 "
        "actual_width_f=0.16 actual_minus_planned_width_f=-0.02 "
        "lookahead_s=0.10 ref_xy_err=0.08 td_xy_err=None",
        "[PLAY_DEBUG] step=30 mode=LEFT_SWING swing_side=0 "
        "swing_air_time_s=0.30 last_air_time_s=[0.40, 0.35] "
        "target_delta_f=[0.30, 0.18] ellipse_usage=0.90 "
        "sole_width_y_w=0.26 sole_width_xy_w=0.28 planned_width_f=0.18 "
        "actual_width_f=0.28 actual_minus_planned_width_f=0.10 "
        "lookahead_s=0.10 ref_xy_err=None td_xy_err=0.12",
    ]

    report = module.analyze_play_debug_lines(lines)

    assert report["play_debug_count"] == 3
    assert report["current_lookahead_s"] == 0.10
    assert report["suggested_velocity_lookahead_s"] == 0.22
    assert report["completed_air_time_s"]["p50"] == 0.22
    assert report["swing_air_time_s"]["p50"] == 0.20
    assert report["swing_air_time_s"]["p90"] == 0.30
    assert report["target_delta_x_f"]["max"] == 0.30
    assert report["ellipse_usage"]["p75"] == 0.90
    assert report["sole_width_y_w"]["p50"] == 0.24
    assert report["sole_width_xy_w"]["p50"] == 0.25
    assert report["planned_width_f"]["p50"] == 0.18
    assert report["actual_minus_planned_width_y_w"]["p50"] == 0.06
    assert report["actual_width_f"]["p50"] == 0.20
    assert report["actual_minus_planned_width_f"]["p50"] == 0.02
    assert report["by_swing_side"]["left"]["count"] == 2
    assert report["by_swing_side"]["right"]["count"] == 1
    assert report["by_swing_side"]["left"]["sole_width_y_w"]["p50"] == 0.22
    assert report["by_swing_side"]["left"]["actual_width_f"]["p50"] == 0.20
    assert report["by_swing_side"]["right"]["actual_minus_planned_width_f"]["p50"] == -0.02


def test_analyze_play_debug_lines_can_filter_warmup_and_zero_action_rows():
    module = _load_analyzer_module()
    lines = [
        "[PLAY_DEBUG] step=10 zero_act_active=True mode=LEFT_SWING swing_side=0 "
        "swing_air_time_s=0.10 last_air_time_s=[0.10, 0.10] "
        "target_delta_f=[0.10, 0.18] ellipse_usage=0.30 lookahead_s=0.10 "
        "actual_width_f=0.30 actual_minus_planned_width_f=0.12",
        "[PLAY_DEBUG] step=60 zero_act_active=False mode=LEFT_SWING swing_side=0 "
        "swing_air_time_s=0.20 last_air_time_s=[0.20, 0.20] "
        "target_delta_f=[0.20, 0.18] ellipse_usage=0.60 lookahead_s=0.10 "
        "actual_width_f=0.22 actual_minus_planned_width_f=0.04",
        "[PLAY_DEBUG] step=70 mode=RIGHT_SWING swing_side=1 "
        "swing_air_time_s=0.30 last_air_time_s=[0.30, 0.30] "
        "target_delta_f=[0.30, -0.18] ellipse_usage=0.90 lookahead_s=0.10 "
        "actual_width_f=0.18 actual_minus_planned_width_f=0.00",
    ]

    report = module.analyze_play_debug_lines(
        lines,
        skip_until_step=50,
        exclude_zero_act=True,
    )

    assert report["play_debug_count"] == 2
    assert report["actual_width_f"]["max"] == 0.22
    assert report["actual_width_f"]["min"] == 0.18
    assert report["swing_air_time_s"]["p50"] == 0.20


def test_analyze_play_debug_lines_reports_calibration_subset_without_reset_pollution():
    module = _load_analyzer_module()
    lines = [
        "[PLAY_DEBUG] step=10 zero_act_active=False mode=HOLD swing_side=0 "
        "planner_valid=True target_w=[0.0, 0.0, 0.0] planned_width_f=0.0 "
        "actual_width_f=0.0 actual_minus_planned_width_f=0.0 "
        "target_delta_f=[0.0, 0.0] ellipse_usage=0.0 lookahead_s=0.256 "
        "last_air_time_s=[0.01, 0.01] ref_xy_err=None td_xy_err=None",
        "[PLAY_DEBUG] step=20 zero_act_active=True mode=LEFT_SWING swing_side=0 "
        "planner_valid=True target_w=[1.0, 2.0, 0.0] planned_width_f=0.18 "
        "actual_width_f=0.30 actual_minus_planned_width_f=0.12 "
        "target_delta_f=[0.10, 0.18] ellipse_usage=0.30 lookahead_s=0.256 "
        "last_air_time_s=[0.25, 0.25] ref_xy_err=0.03 td_xy_err=0.04",
        "[PLAY_DEBUG] step=30 zero_act_active=False mode=LEFT_SWING swing_side=0 "
        "swing_air_time_s=0.12 "
        "planner_valid=True target_w=[1.0, 2.0, 0.0] planned_width_f=0.18 "
        "actual_width_f=0.20 actual_minus_planned_width_f=0.02 "
        "target_delta_f=[0.10, 0.18] ellipse_usage=0.30 lookahead_s=0.256 "
        "last_air_time_s=[0.25, 0.26] ref_xy_err=0.03 td_xy_err=0.04",
        "[PLAY_DEBUG] step=40 zero_act_active=False mode=RIGHT_SWING swing_side=1 "
        "swing_air_time_s=0.14 "
        "planner_valid=True target_w=[1.1, 2.0, 0.0] planned_width_f=0.19 "
        "actual_width_f=0.17 actual_minus_planned_width_f=-0.02 "
        "target_delta_f=[0.12, -0.19] ellipse_usage=0.40 lookahead_s=0.256 "
        "last_air_time_s=[0.24, 0.27] ref_xy_err=0.02 td_xy_err=0.05",
        "[PLAY_DEBUG] step=50 zero_act_active=False mode=LEFT_SWING swing_side=0 "
        "planner_valid=True target_w=[1.0, 2.0, 0.0] planned_width_f=0.18 "
        "actual_width_f=0.22 actual_minus_planned_width_f=0.04 "
        "target_delta_f=[0.10, 0.18] ellipse_usage=0.30 lookahead_s=0.256 "
        "last_air_time_s=[0.25, 0.26] ref_xy_err=0.03 td_xy_err=1.20",
        "[PLAY_DEBUG] step=60 zero_act_active=False mode=RIGHT_SWING swing_side=1 "
        "planner_valid=False target_w=[1.1, 2.0, 0.0] planned_width_f=0.19 "
        "actual_width_f=0.19 actual_minus_planned_width_f=0.00 "
        "target_delta_f=[0.12, -0.19] ellipse_usage=0.40 lookahead_s=0.256 "
        "last_air_time_s=[0.24, 0.27] ref_xy_err=0.02 td_xy_err=0.05",
    ]

    report = module.analyze_play_debug_lines(lines)
    subset = report["calibration_subset"]

    assert report["play_debug_count"] == 6
    assert subset["play_debug_count"] == 2
    assert subset["planned_width_f"]["p50"] == 0.18
    assert subset["actual_width_f"]["p50"] == 0.17
    assert subset["actual_minus_planned_width_f"]["p50"] == -0.02
    assert subset["td_xy_err"]["max"] == 0.05
    assert subset["by_swing_side"]["left"]["count"] == 1
    assert subset["by_swing_side"]["right"]["count"] == 1


def test_analyze_play_debug_lines_reports_curriculum_residual_usage():
    module = _load_analyzer_module()
    lines = [
        "[PLAY_DEBUG] step=10 mode=LEFT_SWING swing_side=0 "
        "command=[0.4, 0.0, 0.0] "
        "flat_level=2 lookahead_s=0.25 target_delta_f=[0.15, 0.20] "
        "feasible_velocity_f=[0.4, 0.0, 0.0] planned_width_f=0.20 "
        "actual_width_f=0.19 actual_minus_planned_width_f=-0.01 "
        "ellipse_usage=0.20 last_air_time_s=[0.25, 0.25] "
        "ref_xy_err=0.01 td_xy_err=0.02",
        "[PLAY_DEBUG] step=20 mode=RIGHT_SWING swing_side=1 "
        "command=[0.0, 0.04, 0.0] "
        "flat_level=1 lookahead_s=0.25 target_delta_f=[0.08, -0.19] "
        "feasible_velocity_f=[0.0, 0.04, 0.0] planned_width_f=0.19 "
        "actual_width_f=0.18 actual_minus_planned_width_f=-0.01 "
        "ellipse_usage=0.10 last_air_time_s=[0.25, 0.25] "
        "ref_xy_err=0.01 td_xy_err=0.02",
    ]

    report = module.analyze_play_debug_lines(lines)

    # level 2 x residual = 0.15 - 0.4 * 0.25 = 0.05; radius_x = 0.12.
    assert report["curriculum_residual_x_f"]["p50"] == 0.05
    assert report["curriculum_usage_x"]["p50"] == 0.41667

    # right foot uses side_sign=-1:
    # y residual = -0.19 - (-0.18) - 0.04 * 0.25 = -0.02; radius_y level 1 = 0.04.
    assert report["curriculum_residual_y_f"]["min"] == -0.02
    assert report["curriculum_usage_y"]["p50"] == 0.33333


def test_analyze_play_debug_lines_prefers_direct_curriculum_fields():
    module = _load_analyzer_module()
    lines = [
        "[PLAY_DEBUG] step=10 mode=LEFT_SWING swing_side=0 "
        "command=[9.0, 9.0, 0.0] "
        "flat_level=2 lookahead_s=0.25 target_delta_f=[0.15, 0.20] "
        "curriculum_residual_f=[0.03, -0.015] "
        "curriculum_radius_f=[0.12, 0.06] "
        "curriculum_usage=0.35355 "
        "feasible_velocity_f=[0.4, 0.0, 0.0] planned_width_f=0.20 "
        "actual_width_f=0.19 actual_minus_planned_width_f=-0.01 "
        "ellipse_usage=0.20 last_air_time_s=[0.25, 0.25] "
        "ref_xy_err=0.01 td_xy_err=0.02",
    ]

    report = module.analyze_play_debug_lines(lines)

    assert report["curriculum_residual_x_f"]["p50"] == 0.03
    assert report["curriculum_residual_y_f"]["p50"] == -0.015
    assert report["curriculum_usage_x"]["p50"] == 0.25
    assert report["curriculum_usage_y"]["p50"] == 0.25
    assert report["curriculum_usage_norm"]["p50"] == 0.35355
