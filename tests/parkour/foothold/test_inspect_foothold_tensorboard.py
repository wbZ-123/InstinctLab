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
        "Step_Monitor/foothold_planner_left_swing_fraction": [0.05, 0.08],
        "Step_Monitor/foothold_planner_right_swing_fraction": [0.04, 0.07],
        "Step_Monitor/foothold_planner_swing_entry_step_rate": [0.02, 0.03],
        "Step_Monitor/foothold_planner_mean_swing_duration_steps": [3.0, 2.5],
        "Step_Monitor/foothold_planner_left_touchdown_accepted_step_rate": [0.01, 0.02],
        "Step_Monitor/foothold_planner_right_touchdown_accepted_step_rate": [0.01, 0.015],
        "Step_Monitor/foothold_planner_left_touchdown_confirm_step_rate": [0.005, 0.01],
        "Step_Monitor/foothold_planner_right_touchdown_confirm_step_rate": [0.005, 0.008],
        "Step_Monitor/foothold_planner_recovery_entry_step_rate": [0.001, 0.002],
        "Step_Monitor/foothold_planner_stance_lost_per_swing_entry": [0.05, 0.12],
        "Step_Monitor/foothold_planner_recovery_per_swing_entry": [0.02, 0.08],
        "Step_Monitor/foothold_planner_left_swing_stance_lost_per_swing_entry": [0.01, 0.04],
        "Step_Monitor/foothold_planner_right_swing_stance_lost_per_swing_entry": [0.2, 0.35],
        "Step_Monitor/foothold_planner_plan_invalid_mode_fraction": [0.0, 0.0],
        "Step_Monitor/foothold_planner_hold_contact_lost_fraction": [0.0, 0.01],
        "Step_Monitor/foothold_planner_hold_contact_lost_entry_step_rate": [0.0, 0.005],
        "Step_Monitor/foothold_planner_hold_contact_lost_per_swing_entry": [0.05, 0.35],
        "Step_Monitor/foothold_planner_recovery_fraction": [0.01, 0.02],
        "Step_Monitor/foothold_planner_recovery_step_fraction": [0.0, 0.01],
        "Step_Monitor/foothold_planner_recovery_step_entry_step_rate": [0.0, 0.005],
        "Train/time/mean_reward_0": [0.1, 0.3, 0.6],
        "Train/motor_kl": [0.01, 0.02],
        "Train/foothold_kl": [0.001, 0.003],
        "Train/foothold_event_count": [10.0, 12.0],
        "Train/foothold_raw_out_of_range_fraction": [0.3, 0.2],
        "Train/foothold_ellipse_projection_fraction": [0.2, 0.1],
        "Train/grad_norm": [0.4, 0.5],
        "Train/motor_grad_norm": [0.4, 0.5],
        "Train/foothold_grad_norm": [0.2, 0.3],
        "Train/motor_learning_rate": [1.0e-3, 7.5e-4],
        "Train/foothold_learning_rate": [5.0e-4, 3.3e-4],
        "Train/foothold_kl_skip_count": [0.0, 2.0],
        "Train/foothold_std_normalized_x": [0.12, 0.08],
        "Train/foothold_std_normalized_y": [0.20, 0.10],
        "Train/foothold_std_m_x": [0.05, 0.034],
        "Train/foothold_std_m_y": [0.05, 0.025],
        "Loss/learning_rate": [1.0e-3, 7.5e-4],
        "Train/mean_reward_0": [80.0, 90.0],
        "Train/mean_reward_1": [0.2, 0.4],
    }

    rows = module.build_report_rows(scalars)
    by_name = {row.name: row for row in rows}

    assert by_name["foothold_planner_nonfinite_fraction"].status == "OK"
    assert by_name["foothold_planner_plan_invalid_fraction"].status == "OK"
    assert by_name["foothold_planner_safe_target_final_valid_fraction"].status == "OK"
    assert by_name["foothold_planner_safe_target_nominal_valid_fraction"].status == "BAD"
    assert by_name["foothold_planner_safe_target_candidate_valid_count_mean"].status == "WATCH"
    assert by_name["foothold_planner_left_swing_fraction"].status == "INFO"
    assert by_name["foothold_planner_right_swing_fraction"].status == "INFO"
    assert by_name["foothold_planner_swing_entry_step_rate"].status == "INFO"
    assert by_name["foothold_planner_mean_swing_duration_steps"].status == "INFO"
    assert by_name["foothold_planner_left_touchdown_accepted_step_rate"].status == "INFO"
    assert by_name["foothold_planner_right_touchdown_accepted_step_rate"].status == "INFO"
    assert by_name["foothold_planner_left_touchdown_confirm_step_rate"].status == "INFO"
    assert by_name["foothold_planner_right_touchdown_confirm_step_rate"].status == "INFO"
    assert by_name["foothold_planner_recovery_entry_step_rate"].status == "INFO"
    assert by_name["foothold_planner_stance_lost_per_swing_entry"].status == "WATCH"
    assert by_name["foothold_planner_recovery_per_swing_entry"].status == "OK"
    assert by_name["foothold_planner_left_swing_stance_lost_per_swing_entry"].status == "OK"
    assert by_name["foothold_planner_right_swing_stance_lost_per_swing_entry"].status == "BAD"
    assert by_name["foothold_planner_plan_invalid_mode_fraction"].status == "OK"
    assert by_name["foothold_planner_hold_contact_lost_fraction"].status == "BAD"
    assert (
        by_name["foothold_planner_hold_contact_lost_entry_step_rate"].status
        == "INFO"
    )
    assert by_name["foothold_planner_hold_contact_lost_per_swing_entry"].status == "BAD"
    assert by_name["foothold_planner_recovery_fraction"].status == "OK"
    assert by_name["foothold_planner_recovery_step_fraction"].status == "INFO"
    assert (
        by_name["foothold_planner_recovery_step_entry_step_rate"].status
        == "INFO"
    )
    assert by_name["mean_reward_0"].summary.trend == "up"
    assert by_name["motor_kl"].status == "INFO"
    assert by_name["foothold_kl"].status == "INFO"
    assert by_name["foothold_event_count"].status == "OK"
    assert by_name["foothold_raw_out_of_range_fraction"].status == "INFO"
    assert by_name["foothold_ellipse_projection_fraction"].status == "INFO"
    assert by_name["grad_norm"].status == "INFO"
    assert by_name["motor_grad_norm"].status == "INFO"
    assert by_name["foothold_grad_norm"].status == "INFO"
    assert by_name["motor_learning_rate"].status == "INFO"
    assert by_name["foothold_learning_rate"].status == "INFO"
    assert by_name["foothold_kl_skip_count"].status == "INFO"
    assert by_name["foothold_std_m_x"].summary.last == 0.034
    assert by_name["learning_rate"].status == "INFO"
    assert by_name["mean_reward_1"].summary.trend == "up"


def test_event_gated_training_tag_is_bad_when_no_events_are_observed():
    module = _load_inspector_module()

    rows = module.build_report_rows(
        {"Train/foothold_event_count": [0.0, 0.0]}
    )

    assert rows[0].status == "BAD"


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
