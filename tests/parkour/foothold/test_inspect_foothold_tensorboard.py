from __future__ import annotations

import importlib.util
import json
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
    assert summary.first_window_mean == 1.0
    assert summary.last_window_mean == 8.0
    assert summary.delta == 7.0
    assert summary.delta_percent == 700.0
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
        "Step_Monitor/foothold_planner_safe_target_final_valid_fraction": [
            0.9,
            0.85,
        ],
        "Step_Monitor/foothold_planner_safe_target_nominal_valid_fraction": [
            0.4,
            0.3,
        ],
        "Step_Monitor/foothold_planner_safe_target_candidate_valid_count_mean": [
            1.0,
            3.0,
        ],
        "Train/time/mean_reward_0": [0.1, 0.3, 0.6],
    }

    rows = module.build_report_rows(scalars)
    by_name = {row.name: row for row in rows}

    assert by_name["foothold_planner_nonfinite_fraction"].status == "OK"
    assert by_name["foothold_planner_plan_invalid_fraction"].status == "OK"
    assert by_name["foothold_planner_safe_target_final_valid_fraction"].status == "OK"
    assert by_name["foothold_planner_safe_target_nominal_valid_fraction"].status == "BAD"
    assert by_name["foothold_planner_safe_target_candidate_valid_count_mean"].status == "WATCH"
    assert by_name["mean_reward_0"].summary.trend == "up"


def test_candidate_valid_count_zero_is_not_bad_when_no_candidate_search_is_needed():
    module = _load_inspector_module()
    scalars = {
        "Step_Monitor/foothold_planner_safe_target_final_valid_fraction": [1.0, 1.0],
        "Step_Monitor/foothold_planner_safe_target_nominal_valid_fraction": [1.0, 1.0],
        "Step_Monitor/foothold_planner_safe_target_fallback_fraction": [0.0, 0.0],
        "Step_Monitor/foothold_planner_safe_target_candidate_count_mean": [0.0, 0.0],
        "Step_Monitor/foothold_planner_safe_target_candidate_valid_count_mean": [0.0, 0.0],
    }

    rows = module.build_report_rows(scalars)
    by_name = {row.name: row for row in rows}

    assert by_name["foothold_planner_safe_target_final_valid_fraction"].status == "OK"
    assert by_name["foothold_planner_safe_target_candidate_valid_count_mean"].status == "INFO"


def test_format_rows_includes_quantified_trend_columns():
    module = _load_inspector_module()
    rows = module.build_report_rows(
        {
            "Train/time/mean_reward_0": [1.0, 2.0, 4.0, 8.0],
        }
    )

    table = module.format_rows(rows)

    assert "first_q" in table
    assert "last_q" in table
    assert "delta" in table
    assert "delta_%" in table
    assert "700" in table


def test_save_report_writes_machine_readable_json(tmp_path):
    module = _load_inspector_module()
    run_dir = Path("logs/instinct_rl/g1_parkour/example_run")
    rows = module.build_report_rows(
        {
            "Train/time/mean_reward_0": [1.0, 2.0, 4.0, 8.0],
        }
    )

    report_path = module.save_report(
        run_dir=run_dir,
        rows=rows,
        save_dir=tmp_path,
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["run"] == "logs/instinct_rl/g1_parkour/example_run"
    assert payload["rows"][0]["metric"] == "mean_reward_0"
    assert payload["rows"][0]["tag"] == "Train/time/mean_reward_0"
    assert payload["rows"][0]["summary"]["delta_percent"] == 700.0
