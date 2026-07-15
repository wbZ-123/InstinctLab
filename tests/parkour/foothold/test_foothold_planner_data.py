from dataclasses import fields
from collections.abc import Sequence
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_foothold_planner_data_class():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "foothold_planner_data_for_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FootholdPlannerData


def _load_foothold_planner_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    )
    source = module_path.read_text()
    start = source.index("def _yaw_from_quat_wxyz")
    end = source.index("class FootholdPlanner")
    namespace = {
        "FootholdPlannerData": object,
        "Sequence": Sequence,
        "torch": torch,
    }
    exec(source[start:end], namespace)
    return SimpleNamespace(
        clear_safe_target_event_buffers=namespace[
            "_clear_safe_target_event_buffers"
        ],
        yaw_from_quat_wxyz=namespace["_yaw_from_quat_wxyz"],
        compose_world_from_frame=namespace["_compose_world_from_frame"],
        adaptive_step_hold_s=namespace["_adaptive_step_hold_s"],
    )


def test_foothold_planner_data_exposes_clearance_debug_fields():
    FootholdPlannerData = _load_foothold_planner_data_class()
    field_names = {field.name for field in fields(FootholdPlannerData)}

    assert {
        "default_swing_reference_pos_w",
        "default_swing_apex_height",
        "swing_apex_height",
        "swing_clearance_safe",
        "swing_clearance_penetration",
    }.issubset(field_names)


def test_foothold_planner_data_exposes_touchdown_debug_fields():
    FootholdPlannerData = _load_foothold_planner_data_class()
    field_names = {field.name for field in fields(FootholdPlannerData)}

    assert {
        "touchdown_xy_error",
        "touchdown_z_error",
        "touchdown_xy_ok",
        "touchdown_z_ok",
        "touchdown_swing_contact",
        "touchdown_within_tolerance",
        "swing_has_lifted",
        "recovery_step_active",
    }.issubset(field_names)


def test_clear_safe_target_event_buffers_resets_event_fields_only_for_selected_envs():
    module = _load_foothold_planner_module()
    data = SimpleNamespace(
        safe_target_search_performed=torch.ones(3, dtype=torch.bool),
        safe_target_final_valid=torch.zeros(3, dtype=torch.bool),
        safe_target_used_fallback=torch.ones(3, dtype=torch.bool),
        safe_target_score=torch.ones(3),
        safe_target_nominal_inside_ellipse=torch.zeros(3, dtype=torch.bool),
        safe_target_nominal_obstacle_safe=torch.zeros(3, dtype=torch.bool),
        safe_target_nominal_valid=torch.zeros(3, dtype=torch.bool),
        safe_target_candidate_count=torch.ones(3) * 32.0,
        safe_target_candidate_inside_ellipse_count=torch.ones(3) * 24.0,
        safe_target_candidate_obstacle_safe_count=torch.ones(3) * 8.0,
        safe_target_candidate_valid_count=torch.ones(3) * 4.0,
    )

    module.clear_safe_target_event_buffers(data, torch.tensor([0, 2]))

    torch.testing.assert_close(
        data.safe_target_search_performed,
        torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(
        data.safe_target_final_valid,
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        data.safe_target_used_fallback,
        torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(data.safe_target_score, torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(
        data.safe_target_nominal_inside_ellipse,
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        data.safe_target_nominal_obstacle_safe,
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        data.safe_target_nominal_valid,
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        data.safe_target_candidate_count, torch.tensor([0.0, 32.0, 0.0])
    )
    torch.testing.assert_close(
        data.safe_target_candidate_inside_ellipse_count,
        torch.tensor([0.0, 24.0, 0.0]),
    )
    torch.testing.assert_close(
        data.safe_target_candidate_obstacle_safe_count,
        torch.tensor([0.0, 8.0, 0.0]),
    )
    torch.testing.assert_close(
        data.safe_target_candidate_valid_count,
        torch.tensor([0.0, 4.0, 0.0]),
    )


def test_compose_world_from_frame_rotates_body_target_by_yaw():
    module = _load_foothold_planner_module()
    origin_w = torch.tensor([[10.0, 20.0, 0.5]])
    target_f = torch.tensor([[0.10, 0.00, 0.0]])
    yaw_w = torch.tensor([torch.pi / 2.0])

    target_w = module.compose_world_from_frame(origin_w, target_f, yaw_w)

    torch.testing.assert_close(
        target_w,
        torch.tensor([[10.0, 20.10, 0.5]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_yaw_from_quat_wxyz_reads_heading_about_world_z():
    module = _load_foothold_planner_module()
    half_yaw = math.pi / 4.0
    quat_wxyz = torch.tensor(
        [[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]]
    )

    yaw = module.yaw_from_quat_wxyz(quat_wxyz)

    torch.testing.assert_close(
        yaw,
        torch.tensor([torch.pi / 2.0]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_adaptive_step_hold_s_shortens_hold_as_command_speed_increases():
    module = _load_foothold_planner_module()

    desired_velocity_f = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )

    hold_s = module.adaptive_step_hold_s(
        desired_velocity_f,
        base_hold_s=0.04,
        min_hold_s=0.0,
        velocity_scale_s_per_mps=0.02,
    )

    torch.testing.assert_close(
        hold_s,
        torch.tensor([0.04, 0.02, 0.0]),
    )


def test_foothold_planner_cfg_exposes_all_state_machine_timing_fields():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py"
    ).read_text()
    planner_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    for field in (
        "contact_confirm_s",
        "early_contact_phase",
        "overdue_s",
        "recovery_hold_s",
        "step_hold_s",
        "step_hold_min_s",
        "step_hold_velocity_scale_s_per_mps",
        "recovery_step_length_m",
        "recovery_step_velocity_lookahead_s",
        "recovery_step_max_length_m",
        "recovery_step_width_m",
    ):
        assert f"{field}:" in cfg_text
    for field in (
        "contact_confirm_s",
        "early_contact_phase",
        "overdue_s",
        "recovery_hold_s",
        "step_hold_s",
    ):
        assert f"{field}=cfg.{field}" in planner_text


def test_touchdown_error_uses_fresh_sole_positions_before_acceptance():
    repo_root = Path(__file__).resolve().parents[3]
    planner_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    left_sole_index = planner_text.index("left_sole_pos_w =")
    actual_swing_index = planner_text.index(
        "self._data.actual_swing_foot_pos_w[env_ids] = torch.where"
    )
    foot_error_index = planner_text.index("foot_target_error =")
    touchdown_index = planner_text.index("self._data.touchdown_accepted[env_ids] =")

    assert left_sole_index < actual_swing_index < foot_error_index < touchdown_index


def test_touchdown_acceptance_requires_confirmed_liftoff():
    repo_root = Path(__file__).resolve().parents[3]
    planner_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    assignment = planner_text[
        planner_text.index("self._data.touchdown_accepted[env_ids] =") :
        planner_text.index("gait_state = advance_gait(")
    ]

    assert "current_swing_has_lifted" in assignment
    assert "& current_swing_has_lifted" in assignment


def test_new_swing_target_uses_state_machine_swing_side_after_transition():
    repo_root = Path(__file__).resolve().parents[3]
    planner_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    assert "new_swing_side = gait_state.swing_side[new_swing]" in planner_text
    assert "new_swing_side = swing_side[new_swing]" not in planner_text
