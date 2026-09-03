from dataclasses import fields
from collections.abc import Sequence
import importlib.util
import math
from pathlib import Path
import re
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
        "FlatProviderConfig": __import__(
            "instinctlab_foothold",
            fromlist=["FlatProviderConfig"],
        ).FlatProviderConfig,
        "GaitMachineState": __import__(
            "instinctlab_foothold",
            fromlist=["GaitMachineState"],
        ).GaitMachineState,
        "GaitState": __import__(
            "instinctlab_foothold",
            fromlist=["GaitState"],
        ).GaitState,
        "clear_learned_foothold_buffers": __import__(
            "instinctlab_foothold",
            fromlist=["clear_learned_foothold_buffers"],
        ).clear_learned_foothold_buffers,
        "FootholdPlannerData": object,
        "Sequence": Sequence,
        "replace": __import__("dataclasses", fromlist=["replace"]).replace,
        "re": re,
        "torch": torch,
    }
    exec(source[start:end], namespace)
    return SimpleNamespace(
        clear_safe_target_event_buffers=namespace[
            "_clear_safe_target_event_buffers"
        ],
        clear_foothold_plan_buffers=namespace[
            "_clear_foothold_plan_buffers"
        ],
        apply_startup_hold_gate=namespace["_apply_startup_hold_gate"],
        yaw_from_quat_wxyz=namespace["_yaw_from_quat_wxyz"],
        select_sole_roles=namespace["_select_sole_roles"],
        compose_world_from_frame=namespace["_compose_world_from_frame"],
        make_required_body_paths_glob=namespace[
            "_make_required_body_paths_glob"
        ],
        adaptive_step_hold_s=namespace["_adaptive_step_hold_s"],
        apply_terrain_height_to_target=namespace[
            "_apply_terrain_height_to_target"
        ],
        derive_flat_provider_config=namespace["_derive_flat_provider_config"],
        flat_target_level_from_curriculum_scale=namespace[
            "flat_target_level_from_curriculum_scale"
        ],
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


def test_foothold_planner_data_exposes_monotonic_learned_action_event_generation():
    FootholdPlannerData = _load_foothold_planner_data_class()

    assert (
        "learned_foothold_event_generation"
        in FootholdPlannerData.__annotations__
    )


def test_foothold_planner_data_exposes_authoritative_hold_frame():
    FootholdPlannerData = _load_foothold_planner_data_class()
    field_names = {field.name for field in fields(FootholdPlannerData)}

    assert {
        "nominal_frame_origin_w",
        "nominal_frame_yaw_w",
    }.issubset(field_names)


def test_foothold_planner_data_exposes_safe_target_penetration_debug_field():
    FootholdPlannerData = _load_foothold_planner_data_class()
    field_names = {field.name for field in fields(FootholdPlannerData)}

    assert "safe_target_final_max_penetration_depth" in field_names


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
        safe_target_final_max_penetration_depth=torch.ones(3) * 0.03,
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
    torch.testing.assert_close(
        data.safe_target_final_max_penetration_depth,
        torch.tensor([0.0, 0.03, 0.0]),
    )


def test_clear_foothold_plan_buffers_resets_stale_targets_only_for_selected_envs():
    module = _load_foothold_planner_module()
    data = SimpleNamespace(
        target_foothold_w=torch.ones(3, 3),
        target_foothold_f=torch.ones(3, 3) * 2.0,
        swing_start_pos_w=torch.ones(3, 3) * 3.0,
        raw_unclipped_foothold_f=torch.ones(3, 3) * 4.0,
        feasible_velocity_f=torch.ones(3, 3) * 5.0,
        target_delta_f=torch.ones(3, 2) * 6.0,
        curriculum_residual_f=torch.ones(3, 2) * 6.5,
        curriculum_radius_f=torch.ones(3, 2) * 6.75,
        curriculum_usage=torch.ones(3) * 6.875,
        target_ellipse_max_x=torch.ones(3) * 7.0,
        target_ellipse_usage=torch.ones(3) * 8.0,
        default_swing_apex_height=torch.ones(3) * 0.08,
        swing_apex_height=torch.ones(3) * 0.16,
        swing_clearance_safe=torch.tensor([False, False, False]),
        swing_clearance_penetration=torch.ones(3) * 0.03,
    )

    module.clear_foothold_plan_buffers(data, torch.tensor([0, 2]))

    torch.testing.assert_close(data.target_foothold_w[0], torch.zeros(3))
    torch.testing.assert_close(data.target_foothold_w[1], torch.ones(3))
    torch.testing.assert_close(data.target_foothold_w[2], torch.zeros(3))
    torch.testing.assert_close(data.target_foothold_f[0], torch.zeros(3))
    torch.testing.assert_close(data.target_foothold_f[1], torch.ones(3) * 2.0)
    torch.testing.assert_close(data.swing_start_pos_w[0], torch.zeros(3))
    torch.testing.assert_close(data.raw_unclipped_foothold_f[2], torch.zeros(3))
    torch.testing.assert_close(data.feasible_velocity_f[0], torch.zeros(3))
    torch.testing.assert_close(data.target_delta_f[2], torch.zeros(2))
    torch.testing.assert_close(data.curriculum_residual_f[2], torch.zeros(2))
    torch.testing.assert_close(data.curriculum_radius_f[2], torch.zeros(2))
    torch.testing.assert_close(data.curriculum_usage, torch.tensor([0.0, 6.875, 0.0]))
    torch.testing.assert_close(data.target_ellipse_max_x, torch.tensor([0.0, 7.0, 0.0]))
    torch.testing.assert_close(data.target_ellipse_usage, torch.tensor([0.0, 8.0, 0.0]))
    torch.testing.assert_close(data.default_swing_apex_height, torch.tensor([0.0, 0.08, 0.0]))
    torch.testing.assert_close(data.swing_apex_height, torch.tensor([0.0, 0.16, 0.0]))
    torch.testing.assert_close(data.swing_clearance_safe, torch.tensor([True, False, True]))
    torch.testing.assert_close(data.swing_clearance_penetration, torch.tensor([0.0, 0.03, 0.0]))


def test_startup_hold_gate_freezes_selected_envs_and_clears_active_plans():
    module = _load_foothold_planner_module()
    gait_state = SimpleNamespace(
        mode=torch.tensor([1, 2, 8], dtype=torch.long),
        elapsed_s=torch.tensor([0.12, 0.20, 0.04]),
        hold_elapsed_s=torch.tensor([0.12, 0.20, 0.04]),
        hold_required_s=torch.tensor([0.04, 0.04, 0.04]),
        swing_has_lifted=torch.tensor([True, True, False]),
        recovery_step_pending=torch.tensor([False, True, True]),
        recovery_step_active=torch.tensor([False, True, True]),
    )
    data = SimpleNamespace(
        gait_mode=torch.tensor([1, 2, 8], dtype=torch.long),
        phase=torch.tensor([0.4, 0.6, 0.0]),
        target_foothold_w=torch.ones(3, 3),
        target_foothold_f=torch.ones(3, 3) * 2.0,
        swing_start_pos_w=torch.ones(3, 3) * 3.0,
        raw_unclipped_foothold_f=torch.ones(3, 3) * 4.0,
        feasible_velocity_f=torch.ones(3, 3) * 5.0,
        target_delta_f=torch.ones(3, 2) * 6.0,
        curriculum_residual_f=torch.ones(3, 2) * 6.5,
        curriculum_radius_f=torch.ones(3, 2) * 6.75,
        curriculum_usage=torch.ones(3) * 6.875,
        target_ellipse_max_x=torch.ones(3) * 7.0,
        target_ellipse_usage=torch.ones(3) * 8.0,
        touchdown_accepted=torch.tensor([True, True, True]),
        swing_has_lifted=torch.tensor([True, True, False]),
        recovery_step_active=torch.tensor([False, True, True]),
    )

    module.apply_startup_hold_gate(
        data=data,
        gait_state=gait_state,
        selected_env_ids=torch.tensor([0, 1, 2]),
        startup_hold_mask=torch.tensor([True, False, True]),
        reset_hold_s=0.4,
    )

    torch.testing.assert_close(gait_state.mode, torch.tensor([0, 2, 0]))
    torch.testing.assert_close(gait_state.elapsed_s, torch.tensor([0.0, 0.20, 0.0]))
    torch.testing.assert_close(
        gait_state.hold_elapsed_s,
        torch.tensor([0.12, 0.20, 0.04]),
    )
    torch.testing.assert_close(gait_state.hold_required_s, torch.tensor([0.4, 0.04, 0.4]))
    torch.testing.assert_close(gait_state.swing_has_lifted, torch.tensor([False, True, False]))
    torch.testing.assert_close(gait_state.recovery_step_pending, torch.tensor([False, True, False]))
    torch.testing.assert_close(gait_state.recovery_step_active, torch.tensor([False, True, False]))

    torch.testing.assert_close(data.gait_mode, torch.tensor([0, 2, 0]))
    torch.testing.assert_close(data.phase, torch.tensor([0.0, 0.6, 0.0]))
    torch.testing.assert_close(data.target_foothold_w[0], torch.zeros(3))
    torch.testing.assert_close(data.target_foothold_w[1], torch.ones(3))
    torch.testing.assert_close(data.target_foothold_w[2], torch.zeros(3))
    torch.testing.assert_close(data.target_delta_f[0], torch.zeros(2))
    torch.testing.assert_close(data.target_delta_f[1], torch.ones(2) * 6.0)
    torch.testing.assert_close(data.target_delta_f[2], torch.zeros(2))
    torch.testing.assert_close(data.touchdown_accepted, torch.tensor([False, True, False]))
    torch.testing.assert_close(data.swing_has_lifted, torch.tensor([False, True, False]))
    torch.testing.assert_close(data.recovery_step_active, torch.tensor([False, True, False]))


def test_startup_hold_does_not_prepare_or_score_discarded_footholds():
    planner_text = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    assert "startup_hold_mask = self._startup_hold_mask(selected_env_ids)" in planner_text
    prepare_nominal_block = planner_text[
        planner_text.index("prepare_nominal = nominal_foothold_prepare_mask(") :
        planner_text.index("prepare_learned, _ = learned_foothold_event_masks(")
    ]
    assert "hold_contact_ready=hold_contact_ready" in prepare_nominal_block
    assert "startup_hold=startup_hold_mask" in prepare_nominal_block
    assert "prepare_learned &= ~startup_hold_mask" in planner_text
    assert "entered_hold &= ~startup_hold_mask" in planner_text


def test_planner_uses_event_gated_hold_contact_readiness():
    planner_text = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    assert "confirmed_contact = (" in planner_text
    assert "stance_side = 1 - previous_gait_state.swing_side" in planner_text
    assert "new_support_confirmed = confirmed_contact[" in planner_text
    assert "confirmed_contact_lost = (" in planner_text
    assert "new_support_lost = confirmed_contact_lost[" in planner_text
    assert "strict_double_support = (" in planner_text
    assert "startup_hold_mask" in planner_text
    assert "initial_stabilization_hold = (" in planner_text
    assert "previous_gait_state.hold_required_s" in planner_text
    assert "previous_gait_state.recovery_step_pending" in planner_text
    recovery_gate_block = planner_text[
        planner_text.index("recovery_contact_stable = torch.all(") :
        planner_text.index("event = torch.full_like(", planner_text.index("recovery_contact_stable = torch.all("))
    ]
    assert "stabilization_ready = support_available" in recovery_gate_block
    assert "stability_ready(" not in recovery_gate_block
    assert planner_text.count("hold_contact_ready=hold_contact_ready") >= 2
    assert "hold_contact_lost=hold_contact_lost" in planner_text
    assert "step_hold_s = torch.zeros_like(" in planner_text


def test_command_updates_preserve_the_current_hold_plan_transaction():
    planner_text = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()
    setter_start = planner_text.index("    def set_desired_velocity(")
    setter_end = planner_text.index(
        "    def set_flat_target_curriculum_scale(",
        setter_start,
    )
    setter_text = planner_text[setter_start:setter_end]

    assert "self._data.desired_velocity_f[resolved_env_ids] = desired_velocity_f" in setter_text
    assert "_invalidate_nominal_on_command_change" not in setter_text


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


def test_sole_roles_follow_the_post_transition_swing_side():
    """A toggled gait side must immediately toggle stance/swing geometry."""
    module = _load_foothold_planner_module()
    left_sole_w = torch.tensor(
        [[1.00, 0.18, 0.0], [2.00, 0.18, 0.0]]
    )
    right_sole_w = torch.tensor(
        [[1.00, 0.00, 0.0], [2.00, 0.00, 0.0]]
    )

    stance_w, swing_w = module.select_sole_roles(
        left_sole_w=left_sole_w,
        right_sole_w=right_sole_w,
        # env 0: next swing is right, so the new support is left.
        # env 1: next swing is left, so the new support is right.
        swing_side=torch.tensor([1, 0]),
    )

    torch.testing.assert_close(stance_w[0], left_sole_w[0])
    torch.testing.assert_close(swing_w[0], right_sole_w[0])
    torch.testing.assert_close(stance_w[1], right_sole_w[1])
    torch.testing.assert_close(swing_w[1], left_sole_w[1])

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


def test_apply_terrain_height_to_target_updates_world_and_local_z_only_when_valid():
    module = _load_foothold_planner_module()
    target_w = torch.tensor(
        [
            [1.0, 0.2, -0.50],
            [2.0, -0.1, -0.30],
        ]
    )
    target_f = torch.tensor(
        [
            [0.3, 0.18, 0.0],
            [0.4, -0.18, 0.0],
        ]
    )
    stance_w = torch.tensor(
        [
            [0.7, 0.02, -0.50],
            [1.6, 0.08, -0.30],
        ]
    )
    terrain_height = torch.tensor([0.20, float("inf")])
    terrain_valid = torch.tensor([True, False])

    corrected_w, corrected_f, valid = module.apply_terrain_height_to_target(
        target_foothold_w=target_w,
        target_foothold_f=target_f,
        stance_pos_w=stance_w,
        terrain_height_w=terrain_height,
        terrain_valid=terrain_valid,
    )

    torch.testing.assert_close(
        corrected_w,
        torch.tensor(
            [
                [1.0, 0.2, 0.20],
                [2.0, -0.1, -0.30],
            ]
        ),
    )
    torch.testing.assert_close(
        corrected_f,
        torch.tensor(
            [
                [0.3, 0.18, 0.70],
                [0.4, -0.18, 0.0],
            ]
        ),
    )
    torch.testing.assert_close(valid, torch.tensor([True, False]))


def test_flat_target_level_follows_reward_curriculum_scale():
    module = _load_foothold_planner_module()

    scale = torch.tensor([0.0, 0.32, 0.34, 0.66, 0.67, 1.0])

    level = module.flat_target_level_from_curriculum_scale(
        scale,
        num_levels=3,
    )

    torch.testing.assert_close(
        level,
        torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
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
        "hold_contact_lost_confirm_s",
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
        "hold_contact_lost_confirm_s",
        "overdue_s",
        "recovery_hold_s",
        "step_hold_s",
    ):
        assert f"{field}=cfg.{field}" in planner_text
    assert "early_contact_phase: float = 0.65" in cfg_text
    assert "recovery_step_width_m: float = 0.30" in cfg_text


def test_parkour_cfg_updates_foothold_planner_at_control_period_not_physics_period():
    repo_root = Path(__file__).resolve().parents[3]
    env_cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_control_dt = self.sim.dt * self.decimation" in env_cfg_text
    assert (
        "self.scene.foothold_planner.update_period = foothold_control_dt"
        in env_cfg_text
    )
    assert "self.scene.foothold_planner.control_dt_s = foothold_control_dt" in env_cfg_text
    assert "self.scene.foothold_planner.update_period = self.sim.dt" not in env_cfg_text


def test_parkour_cfg_uses_short_startup_hold_to_reduce_command_planner_conflict():
    repo_root = Path(__file__).resolve().parents[3]
    env_cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "startup_hold_s=0.15" in env_cfg_text
    assert "startup_hold_s=0.40" not in env_cfg_text


def test_parkour_cfg_uses_short_reset_hold_after_recovery():
    repo_root = Path(__file__).resolve().parents[3]
    env_cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "reset_hold_s=0.15" in env_cfg_text


def test_flat_provider_lookahead_is_derived_from_planner_swing_duration():
    module = _load_foothold_planner_module()
    cfg = SimpleNamespace(
        swing_duration_s=0.32,
        flat_target_lookahead_phase=0.8,
    )

    flat_cfg = module.derive_flat_provider_config(cfg)

    assert flat_cfg.velocity_lookahead_s == 0.256


def test_foothold_planner_cfg_marks_flat_target_lookahead_phase_as_temporary_calibration_parameter():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py"
    ).read_text()

    assert "flat_target_lookahead_phase: float = 0.8" in cfg_text


def test_foothold_planner_sole_geometry_matches_active_g1_shoe_urdf_envelope():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py"
    ).read_text()

    assert "sole_center_offset_b: tuple[float, float, float] = (0.039, 0.0, -0.058)" in cfg_text
    assert "sole_half_length: float = 0.093" in cfg_text
    assert "sole_half_width: float = 0.036" in cfg_text
    assert "safe_target_foot_length_m: float = 0.186" in cfg_text
    assert "safe_target_foot_width_m: float = 0.072" in cfg_text


def test_foothold_planner_rigid_body_view_only_matches_required_bodies():
    module = _load_foothold_planner_module()

    glob = module.make_required_body_paths_glob(
        "/World/envs/env_.*/Robot",
        (
            "pelvis",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ),
    )

    assert glob == (
        "/World/envs/env_*/Robot/"
        "(left_ankle_roll_link|pelvis|right_ankle_roll_link)"
    )
    assert "torso_link" not in glob

    planner_text = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()
    assert "required_body_names = (" in planner_text
    assert (
        "body_paths_glob = _make_required_body_paths_glob(\n"
        "            robot_prim_path,\n"
        "            required_body_names,\n"
        "        )"
    ) in planner_text


def test_foothold_planner_contact_view_only_matches_foot_contact_bodies():
    planner_text = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    assert "required_contact_body_names = (" in planner_text
    assert (
        "contact_body_paths_glob = _make_required_body_paths_glob(\n"
        "            robot_prim_path,\n"
        "            required_contact_body_names,\n"
        "        )"
    ) in planner_text
    assert (
        "max_contact_data_count=len(required_contact_body_names)\n"
        "                * self._num_envs,"
    ) in planner_text
    assert "self._contact_body_names = contact_body_names" not in planner_text
    assert "self._contact_physx_view.prim_paths" not in planner_text


def test_touchdown_error_uses_fresh_sole_positions_before_acceptance():
    repo_root = Path(__file__).resolve().parents[3]
    planner_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    left_sole_index = planner_text.index("left_sole_pos_w =")
    sole_role_index = planner_text.index(
        "stance_sole_pos_w, swing_sole_pos_w = _select_sole_roles"
    )
    actual_swing_index = planner_text.index(
        "self._data.actual_swing_foot_pos_w[env_ids] = swing_sole_pos_w"
    )
    foot_error_index = planner_text.index("foot_target_error =")
    touchdown_index = planner_text.index("self._data.touchdown_accepted[env_ids] =")

    assert (
        left_sole_index
        < sole_role_index
        < actual_swing_index
        < foot_error_index
        < touchdown_index
    )


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
    assert "& touchdown_xy_ok" in assignment
    assert "& touchdown_z_ok" in assignment


def test_new_swing_target_uses_state_machine_swing_side_after_transition():
    repo_root = Path(__file__).resolve().parents[3]
    planner_text = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    assert "new_swing_side = gait_state.swing_side[new_swing]" in planner_text
    assert "new_swing_side = swing_side[new_swing]" not in planner_text


def test_contact_adaptive_config_gate_does_not_invert_python_bool_mask():
    planner_text = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    assert "& ~self.cfg.enable_contact_adaptive_recovery" not in planner_text


def test_post_transition_sole_roles_are_refreshed_before_hold_plan_is_cached():
    planner_text = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    ).read_text()

    advance_index = planner_text.index("gait_state = advance_gait(")
    refresh_index = planner_text.index(
        "stance_sole_pos_w, swing_sole_pos_w = _select_sole_roles(",
        advance_index,
    )
    entered_hold_index = planner_text.index("entered_hold = (", refresh_index)
    prepare_index = planner_text.index(
        "self._prepare_nominal_footholds(",
        planner_text.index("prepare_nominal = nominal_foothold_prepare_mask(") ,
    )

    assert prepare_index < advance_index < refresh_index
    entered_hold_end = planner_text.index("_apply_startup_hold_gate(", entered_hold_index)
    entered_hold_block = planner_text[entered_hold_index:entered_hold_end]
    assert "clear_learned_foothold_buffers(" in entered_hold_block
    assert "self._data.nominal_foothold_prepared" in entered_hold_block
    assert "self._prepare_nominal_footholds(" not in entered_hold_block
