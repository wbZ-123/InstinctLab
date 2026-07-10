from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_inspector_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "parkour"
        / "foothold"
        / "inspect_foothold_tensorboard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "inspect_foothold_tensorboard_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_series_reports_basic_statistics_and_trend():
    module = _load_inspector_module()

    summary = module.summarize_series([1.0, 2.0, 4.0, 8.0])

    assert summary.last == 8.0
    assert summary.mean == 3.75
    assert summary.min == 1.0
    assert summary.max == 8.0
    assert summary.trend == "up"


def test_build_report_marks_core_foothold_metrics():
    module = _load_inspector_module()
    scalars = {
        "Step_Monitor/foothold_planner_nonfinite_fraction": [
            0.0,
            0.0,
        ],
        "Step_Monitor/foothold_planner_plan_invalid_fraction": [
            0.0,
            0.0,
        ],
        "Step_Monitor/foothold_planner_safe_target_valid_fraction": [
            0.9,
            0.85,
        ],
        "Train/time/mean_reward_0": [0.1, 0.3, 0.6],
    }

    rows = module.build_report_rows(scalars)
    by_name = {row.name: row for row in rows}

    assert by_name["foothold_planner_nonfinite_fraction"].status == "OK"
    assert by_name["foothold_planner_plan_invalid_fraction"].status == "OK"
    assert by_name["foothold_planner_safe_target_valid_fraction"].status == "OK"
    assert by_name["mean_reward_0"].summary.trend == "up"
