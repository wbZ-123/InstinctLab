from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_play_debug_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "instinct_rl"
        / "play_debug.py"
    )
    spec = importlib.util.spec_from_file_location("play_debug_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCommandManager:
    def __init__(self, command: torch.Tensor, term=None):
        self._command = command
        self._terms = {"base_velocity": term} if term is not None else {}

    def get_command(self, name: str) -> torch.Tensor:
        assert name == "base_velocity"
        return self._command


class FakeTerminationManager:
    def __init__(self):
        self._term_names = ["base_contact", "time_out", "illegal_reset_contact"]
        self._last_episode_dones = torch.tensor(
            [[False, True, False], [True, False, True]]
        )
        self.terminated = torch.tensor([False, True])
        self.time_outs = torch.tensor([True, False])


class FakeRobot:
    def __init__(self):
        self.data = SimpleNamespace(
            joint_pos=torch.tensor(
                [
                    [
                        -0.31,
                        0.66,
                        -0.42,
                        0.02,
                        -0.30,
                        0.64,
                        -0.35,
                        -0.01,
                    ]
                ]
            ),
            joint_vel=torch.tensor(
                [
                    [
                        0.1,
                        0.2,
                        -0.3,
                        0.4,
                        0.5,
                        0.6,
                        -0.7,
                        0.8,
                    ]
                ]
            ),
            body_pos_w=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 2.0, 0.05],
                        [1.2, 1.8, 0.06],
                    ]
                ]
            ),
            body_quat_w=torch.tensor(
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.99875, 0.04998, 0.0, 0.0],
                        [0.995, 0.0, 0.09983, 0.0],
                    ]
                ]
            ),
        )

    def find_joints(self, names, preserve_order=False):
        assert preserve_order is True
        joint_order = [
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]
        return [joint_order.index(name) for name in names], names

    def find_bodies(self, names, preserve_order=False):
        assert preserve_order is True
        body_order = ["torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
        return [body_order.index(name) for name in names], names


def test_build_foothold_debug_payload_reads_command_and_planner_data():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([1]),
        swing_side=torch.tensor([1]),
        phase=torch.tensor([0.25]),
        foot_contact=torch.tensor([[True, False]]),
        planner_valid=torch.tensor([True]),
        learned_foothold_prepared_valid=torch.tensor([True]),
        learned_foothold_geometric_valid=torch.tensor([True]),
        learned_foothold_height_valid=torch.tensor([True]),
        learned_foothold_safety_valid=torch.tensor([True]),
        learned_foothold_evaluated=torch.tensor([True]),
        learned_foothold_route_event=torch.tensor([True]),
        learned_foothold_route_use_nominal=torch.tensor([True]),
        learned_foothold_route_use_learned=torch.tensor([False]),
        learned_foothold_route_initial_executable=torch.tensor([True]),
        learned_foothold_lock_geometric_valid=torch.tensor([True]),
        target_terrain_valid=torch.tensor([True]),
        nominal_geometric_valid=torch.tensor([True]),
        nominal_safety_valid=torch.tensor([True]),
        swing_clearance_safe=torch.tensor([True]),
        touchdown_accepted=torch.tensor([False]),
        touchdown_swing_contact=torch.tensor([False]),
        touchdown_xy_ok=torch.tensor([True]),
        touchdown_z_ok=torch.tensor([True]),
        touchdown_within_tolerance=torch.tensor([True]),
        swing_has_lifted=torch.tensor([True]),
        recovery_step_active=torch.tensor([True]),
        safe_target_search_performed=torch.tensor([True]),
        safe_target_final_valid=torch.tensor([True]),
        safe_target_used_fallback=torch.tensor([False]),
        safe_target_score=torch.tensor([0.0]),
        safe_target_final_max_penetration_depth=torch.tensor([0.012]),
        safe_target_candidate_count=torch.tensor([32.0]),
        safe_target_candidate_obstacle_safe_count=torch.tensor([24.0]),
        safe_target_candidate_valid_count=torch.tensor([8.0]),
        target_foothold_f=torch.tensor([[0.2, -0.1, 0.0]]),
        target_foothold_w=torch.tensor([[1.0, 2.0, 0.3]]),
        actual_swing_foot_pos_w=torch.tensor([[1.03, 1.96, 0.35]]),
        actual_stance_foot_pos_w=torch.tensor([[0.95, 2.20, 0.31]]),
        swing_reference_pos_w=torch.tensor([[1.01, 1.99, 0.32]]),
        swing_start_pos_w=torch.tensor([[0.8, 1.7, 0.25]]),
        swing_apex_height=torch.tensor([0.18]),
        default_swing_apex_height=torch.tensor([0.12]),
        feasible_velocity_f=torch.tensor([[0.5, 0.0, 0.0]]),
        flat_target_level=torch.tensor([2]),
        velocity_lookahead_s=torch.tensor([0.16]),
        target_delta_f=torch.tensor([[0.2, -0.1]]),
        curriculum_residual_f=torch.tensor([[0.03, -0.02]]),
        curriculum_radius_f=torch.tensor([[0.12, 0.06]]),
        curriculum_usage=torch.tensor([0.41667]),
        target_ellipse_max_x=torch.tensor([0.3]),
        target_ellipse_usage=torch.tensor([0.75]),
    )
    contact_data = SimpleNamespace(
        current_air_time=torch.tensor([[0.12, 0.03]]),
        last_air_time=torch.tensor([[0.30, 0.28]]),
        current_contact_time=torch.tensor([[0.0, 0.40]]),
    )
    gait_state = SimpleNamespace(
        hold_elapsed_s=torch.tensor([0.12]),
        hold_required_s=torch.tensor([0.04]),
        contact_elapsed_s=torch.tensor([[0.02, 0.04]]),
        no_contact_elapsed_s=torch.tensor([[0.10, 0.0]]),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(
            torch.tensor([[0.5, 0.0, -0.2]]),
            term=SimpleNamespace(
                pos_command_w=torch.tensor([[1.3, 2.4, 0.3]]),
                pos_command_b=torch.tensor([[0.3, 0.4, 0.0]]),
                max_command_b=torch.tensor([[0.8, 0.0, 1.0]]),
                is_standing_env=torch.tensor([False]),
                cfg=SimpleNamespace(target_dis_threshold=0.4),
            ),
        ),
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(
                    data=data,
                    _gait_state=gait_state,
                ),
                "contact_forces": SimpleNamespace(data=contact_data),
            }
        ),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_foothold_debug_line(12, payload)

    assert payload["command"] == [0.5, 0.0, -0.2]
    assert payload["env_id"] == 0
    assert payload["command_target_w"] == [1.3, 2.4, 0.3]
    assert payload["command_target_b"] == [0.3, 0.4, 0.0]
    assert payload["command_target_dist_xy"] == 0.5
    assert payload["command_target_threshold"] == 0.4
    assert payload["command_max_b"] == [0.8, 0.0, 1.0]
    assert payload["command_is_standing_env"] is False
    assert payload["gait_mode"] == "LEFT_SWING"
    assert payload["foot_contact"] == [True, False]
    assert payload["touchdown_swing_contact"] is False
    assert payload["touchdown_xy_ok"] is True
    assert payload["touchdown_z_ok"] is True
    assert payload["touchdown_within_tolerance"] is True
    assert payload["swing_has_lifted"] is True
    assert payload["recovery_step_active"] is True
    assert payload["learned_prepared_valid"] is True
    assert payload["learned_geometric_valid"] is True
    assert payload["learned_height_valid"] is True
    assert payload["learned_safety_valid"] is True
    assert payload["learned_evaluated"] is True
    assert payload["route_event"] is True
    assert payload["route_use_nominal"] is True
    assert payload["route_use_learned"] is False
    assert payload["route_executable"] is True
    assert payload["lock_geometric_valid"] is True
    assert payload["target_terrain_valid"] is True
    assert payload["nominal_geometric_valid"] is True
    assert payload["nominal_safety_valid"] is True
    assert payload["swing_clearance_safe"] is True
    assert payload["actual_swing_w"] == [1.03, 1.96, 0.35]
    assert payload["actual_stance_w"] == [0.95, 2.2, 0.31]
    assert payload["left_sole_w"] == [0.95, 2.2, 0.31]
    assert payload["right_sole_w"] == [1.03, 1.96, 0.35]
    assert payload["sole_width_y_w"] == 0.24
    assert payload["sole_width_xy_w"] == 0.25298
    assert payload["planned_width_f"] == 0.1
    assert payload["target_w"] == [1.0, 2.0, 0.3]
    assert payload["touchdown_xy_error"] == 0.05
    assert payload["touchdown_z_error"] == 0.05
    assert payload["swing_reference_w"] == [1.01, 1.99, 0.32]
    assert payload["swing_start_w"] == [0.8, 1.7, 0.25]
    assert payload["reference_xy_error"] == 0.03606
    assert payload["reference_z_error"] == 0.03
    assert payload["swing_apex_height"] == 0.18
    assert payload["default_swing_apex_height"] == 0.12
    assert payload["hold_elapsed_s"] == 0.12
    assert payload["hold_required_s"] == 0.04
    assert payload["contact_elapsed_s"] == [0.02, 0.04]
    assert payload["no_contact_elapsed_s"] == [0.1, 0.0]
    assert payload["air_time_s"] == [0.12, 0.03]
    assert payload["last_air_time_s"] == [0.3, 0.28]
    assert payload["contact_time_s"] == [0.0, 0.4]
    assert payload["swing_air_time_s"] == 0.03
    assert payload["flat_target_level"] == 2
    assert payload["velocity_lookahead_s"] == 0.16
    assert payload["target_delta_f"] == [0.2, -0.1]
    assert payload["curriculum_residual_f"] == [0.03, -0.02]
    assert payload["curriculum_radius_f"] == [0.12, 0.06]
    assert payload["curriculum_usage"] == 0.41667
    assert payload["target_ellipse_max_x"] == 0.3
    assert payload["target_ellipse_usage"] == 0.75
    assert payload["safe_target_final_max_penetration_depth"] == 0.012
    assert payload["safe_target_candidate_count"] == 32.0
    assert payload["safe_target_candidate_obstacle_safe_count"] == 24.0
    assert payload["safe_target_candidate_valid_count"] == 8.0
    assert "step=12" in line
    assert "env_id=0" in line
    assert "mode=LEFT_SWING" in line
    assert "command=[0.5, 0.0, -0.2]" in line
    assert "command_target_dist_xy=0.5" in line
    assert "command_target_threshold=0.4" in line
    assert "command_max_b=[0.8, 0.0, 1.0]" in line
    assert "command_is_standing=False" in line
    assert "td_contact=False" in line
    assert "td_xy_ok=True" in line
    assert "td_z_ok=True" in line
    assert "td_within_tol=True" in line
    assert "lifted=True" in line
    assert "recovery_step=True" in line
    assert "actual_swing_w=[1.03, 1.96, 0.35]" in line
    assert "actual_stance_w=[0.95, 2.2, 0.31]" in line
    assert "left_sole_w=[0.95, 2.2, 0.31]" in line
    assert "right_sole_w=[1.03, 1.96, 0.35]" in line
    assert "sole_width_y_w=0.24" in line
    assert "sole_width_xy_w=0.25298" in line
    assert "planned_width_f=0.1" in line
    assert "target_w=[1.0, 2.0, 0.3]" in line
    assert "final_penetration=0.012" in line
    assert "candidate_valid=8.0/32.0" in line
    assert "swing_ref_w=[1.01, 1.99, 0.32]" in line
    assert "swing_start_w=[0.8, 1.7, 0.25]" in line
    assert "ref_xy_err=0.03606" in line
    assert "ref_z_err=0.03" in line
    assert "apex=0.18" in line
    assert "default_apex=0.12" in line
    assert "air_time_s=[0.12, 0.03]" in line
    assert "swing_air_time_s=0.03" in line
    assert "flat_level=2" in line
    assert "lookahead_s=0.16" in line
    assert "curriculum_residual_f=[0.03, -0.02]" in line
    assert "curriculum_radius_f=[0.12, 0.06]" in line
    assert "curriculum_usage=0.41667" in line
    assert "ellipse_usage=0.75" in line
    assert "hold_elapsed_s=0.12" in line
    assert "hold_required_s=0.04" in line
    assert "contact_elapsed_s=[0.02, 0.04]" in line
    assert "no_contact_elapsed_s=[0.1, 0.0]" in line
    assert "td_xy_err=0.05" in line
    assert "td_z_err=0.05" in line


def test_build_foothold_debug_payload_reports_recovery_stability_diagnostics():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([8]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.0]),
        foot_contact=torch.tensor([[True, True]]),
        confirmed_foot_contact=torch.tensor([[True, True]]),
        body_tilt_rad=torch.tensor([0.40]),
        body_angular_speed_rad_s=torch.tensor([0.8]),
        body_horizontal_speed_m_s=torch.tensor([0.10]),
        support_slip_m_s=torch.tensor([0.01]),
        stabilization_active=torch.tensor([True]),
        stabilization_ready=torch.tensor([True]),
        event_response=torch.tensor([5]),
    )
    gait_state = SimpleNamespace(
        hold_elapsed_s=torch.tensor([0.0]),
        hold_required_s=torch.tensor([0.2]),
        contact_elapsed_s=torch.tensor([[0.2, 0.2]]),
        no_contact_elapsed_s=torch.tensor([[0.0, 0.0]]),
        stabilization_elapsed_s=torch.tensor([0.06]),
    )
    sensor = SimpleNamespace(
        data=data,
        _gait_state=gait_state,
        _stability_bounds=SimpleNamespace(
            max_tilt_rad=0.35,
            max_angular_speed_rad_s=1.5,
            max_horizontal_speed_m_s=0.35,
            max_support_slip_m_s=0.05,
            dwell_s=0.1,
        ),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.0, 0.0, 0.0]])),
        scene=SimpleNamespace(sensors={"foothold_planner": sensor}),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_foothold_debug_line(12, payload)

    assert payload["confirmed_foot_contact"] == [True, True]
    assert payload["body_tilt_rad"] == 0.4
    assert payload["body_angular_speed_rad_s"] == 0.8
    assert payload["body_horizontal_speed_m_s"] == 0.1
    assert payload["support_slip_m_s"] == 0.01
    assert payload["stabilization_active"] is True
    assert payload["stabilization_ready"] is True
    assert payload["stabilization_elapsed_s"] == 0.06
    assert payload["event_response"] == "STABILIZE"
    assert payload["stability_current"] is True
    assert payload["stability_gate"] is True
    assert payload["stability_fail_reasons"] == []
    assert "stability_gate=True" in line
    assert "event_response=STABILIZE" in line


def test_build_foothold_debug_payload_reports_missing_second_contact_as_gate():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([8]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.0]),
        foot_contact=torch.tensor([[True, False]]),
        confirmed_foot_contact=torch.tensor([[True, False]]),
        body_tilt_rad=torch.tensor([0.10]),
        body_angular_speed_rad_s=torch.tensor([0.2]),
        body_horizontal_speed_m_s=torch.tensor([0.10]),
        support_slip_m_s=torch.tensor([0.01]),
        stabilization_active=torch.tensor([True]),
        stabilization_ready=torch.tensor([False]),
        event_response=torch.tensor([5]),
    )
    gait_state = SimpleNamespace(
        hold_elapsed_s=torch.tensor([0.0]),
        hold_required_s=torch.tensor([0.2]),
        contact_elapsed_s=torch.tensor([[0.2, 0.0]]),
        no_contact_elapsed_s=torch.tensor([[0.0, 0.2]]),
        stabilization_elapsed_s=torch.tensor([0.0]),
    )
    sensor = SimpleNamespace(
        data=data,
        _gait_state=gait_state,
        _stability_bounds=SimpleNamespace(
            max_tilt_rad=0.35,
            max_angular_speed_rad_s=1.5,
            max_horizontal_speed_m_s=0.35,
            max_support_slip_m_s=0.05,
            dwell_s=0.1,
        ),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.0, 0.0, 0.0]])),
        scene=SimpleNamespace(sensors={"foothold_planner": sensor}),
    )

    payload = module.build_foothold_debug_payload(env)

    assert payload["stability_current"] is False
    assert payload["stability_gate"] is False
    assert payload["stability_fail_reasons"] == ["contact"]


def test_is_foothold_debug_plan_event_detects_safe_search_frame():
    module = _load_play_debug_module()

    assert module.is_foothold_debug_plan_event({"safe_target_search": True}) is True
    assert module.is_foothold_debug_plan_event({"safe_target_search": False}) is False
    assert module.is_foothold_debug_plan_event({"safe_target_search": None}) is False


def test_build_foothold_debug_payload_reports_startup_pose_and_action_diagnostics():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([0]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.0]),
        foot_contact=torch.tensor([[True, True]]),
    )
    robot = FakeRobot()
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.0, 0.0, 0.0]])),
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=data)},
            articulations={"robot": robot},
        ),
    )
    actions = torch.tensor([[0.0, 0.0, 0.25, -0.1, 0.0, 0.0, -0.2, 0.15]])

    payload = module.build_foothold_debug_payload(env, actions=actions)
    line = module.format_foothold_debug_line(12, payload)

    assert payload["ankle_joint_pos"] == {
        "left_ankle_pitch_joint": -0.42,
        "left_ankle_roll_joint": 0.02,
        "right_ankle_pitch_joint": -0.35,
        "right_ankle_roll_joint": -0.01,
    }
    assert payload["ankle_joint_vel"] == {
        "left_ankle_pitch_joint": -0.3,
        "left_ankle_roll_joint": 0.4,
        "right_ankle_pitch_joint": -0.7,
        "right_ankle_roll_joint": 0.8,
    }
    assert payload["ankle_action"] == {
        "left_ankle_pitch_joint": 0.25,
        "left_ankle_roll_joint": -0.1,
        "right_ankle_pitch_joint": -0.2,
        "right_ankle_roll_joint": 0.15,
    }
    assert payload["foot_pos_w"] == {
        "left_ankle_roll_link": [1.0, 2.0, 0.05],
        "right_ankle_roll_link": [1.2, 1.8, 0.06],
    }
    assert payload["foot_rpy_w"] == {
        "left_ankle_roll_link": [0.1, 0.0, 0.0],
        "right_ankle_roll_link": [0.0, 0.19999, 0.0],
    }
    assert "ankle_pos={" in line
    assert "'left_ankle_pitch_joint': -0.42" in line
    assert "ankle_action={" in line
    assert "'right_ankle_roll_joint': 0.15" in line
    assert "foot_rpy_w={" in line
    assert "'right_ankle_roll_link': [0.0, 0.19999, 0.0]" in line


def test_build_foothold_debug_payload_selects_foot_contact_times_by_planner_body_ids():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([2]),
        swing_side=torch.tensor([1]),
        phase=torch.tensor([0.5]),
        foot_contact=torch.tensor([[True, False]]),
        planner_valid=torch.tensor([True]),
        touchdown_accepted=torch.tensor([False]),
        touchdown_swing_contact=torch.tensor([False]),
        touchdown_xy_ok=torch.tensor([True]),
        touchdown_z_ok=torch.tensor([True]),
        touchdown_within_tolerance=torch.tensor([True]),
        swing_has_lifted=torch.tensor([True]),
        recovery_step_active=torch.tensor([False]),
        safe_target_search_performed=torch.tensor([False]),
        safe_target_final_valid=torch.tensor([True]),
        safe_target_used_fallback=torch.tensor([False]),
        safe_target_score=torch.tensor([0.0]),
        target_foothold_f=torch.tensor([[0.2, -0.1, 0.0]]),
        target_foothold_w=torch.tensor([[1.0, 2.0, 0.3]]),
        actual_swing_foot_pos_w=torch.tensor([[1.03, 1.96, 0.35]]),
        swing_reference_pos_w=torch.tensor([[1.01, 1.99, 0.32]]),
        swing_start_pos_w=torch.tensor([[0.8, 1.7, 0.25]]),
        swing_apex_height=torch.tensor([0.18]),
        default_swing_apex_height=torch.tensor([0.12]),
        feasible_velocity_f=torch.tensor([[0.5, 0.0, 0.0]]),
        flat_target_level=torch.tensor([1]),
        velocity_lookahead_s=torch.tensor([0.10]),
        target_delta_f=torch.tensor([[0.2, -0.1]]),
        target_ellipse_max_x=torch.tensor([0.3]),
        target_ellipse_usage=torch.tensor([0.75]),
    )
    contact_data = SimpleNamespace(
        current_air_time=torch.tensor([[9.0, 9.1, 0.02, 0.18]]),
        last_air_time=torch.tensor([[0.0, 0.0, 0.27, 0.31]]),
        current_contact_time=torch.tensor([[0.0, 0.0, 0.30, 0.0]]),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.5, 0.0, -0.2]])),
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(
                    data=data,
                    _left_contact_body_id=0,
                    _right_contact_body_id=1,
                    _contact_body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                ),
                "contact_forces": SimpleNamespace(
                    body_names=[
                        "pelvis",
                        "torso",
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                    data=contact_data,
                ),
            }
        ),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_foothold_debug_line(12, payload)

    assert payload["contact_body_ids"] == [2, 3]
    assert payload["contact_body_names"] == [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    assert payload["air_time_s"] == [0.02, 0.18]
    assert payload["last_air_time_s"] == [0.27, 0.31]
    assert payload["contact_time_s"] == [0.3, 0.0]
    assert payload["swing_air_time_s"] == 0.18
    assert "contact_body_ids=[2, 3]" in line
    assert "air_time_s=[0.02, 0.18]" in line
    assert "swing_air_time_s=0.18" in line


def test_build_foothold_debug_payload_reports_actual_width_in_planner_frame():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([1]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.5]),
        foot_contact=torch.tensor([[False, True]]),
        target_foothold_f=torch.tensor([[0.20, 0.18, 0.0]]),
        target_foothold_w=torch.tensor([[1.20, 2.18, 0.30]]),
        target_delta_f=torch.tensor([[0.20, 0.18]]),
        actual_stance_foot_pos_w=torch.tensor([[1.00, 2.00, 0.30]]),
        actual_swing_foot_pos_w=torch.tensor([[1.25, 2.22, 0.32]]),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.5, 0.0, 0.0]])),
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=data)}
        ),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_foothold_debug_line(12, payload)

    assert payload["actual_delta_f"] == [0.25, 0.22, 0.02]
    assert payload["actual_width_f"] == 0.22
    assert payload["actual_minus_planned_width_f"] == 0.04
    assert "actual_delta_f=[0.25, 0.22, 0.02]" in line
    assert "actual_width_f=0.22" in line
    assert "actual_minus_planned_width_f=0.04" in line


def test_build_reset_debug_payload_reports_last_episode_termination_terms():
    module = _load_play_debug_module()
    env = SimpleNamespace(
        termination_manager=FakeTerminationManager(),
        episode_length_buf=torch.tensor([0, 3]),
    )

    payload = module.build_reset_debug_payload(
        env,
        env_id=1,
        done=True,
    )
    line = module.format_reset_debug_line(17, payload)

    assert payload["done"] is True
    assert payload["terminated"] is True
    assert payload["time_out"] is False
    assert payload["episode_length"] == 3
    assert payload["active_terms"] == [
        "base_contact",
        "illegal_reset_contact",
    ]
    assert "step=17" in line
    assert "env_id=1" in line
    assert "terminated=True" in line
    assert "active_terms=['base_contact', 'illegal_reset_contact']" in line


def test_build_reset_debug_payload_prefers_pre_step_snapshot_for_reset_sensitive_state():
    module = _load_play_debug_module()
    env = SimpleNamespace(
        termination_manager=FakeTerminationManager(),
        episode_length_buf=torch.tensor([0, 0]),
    )
    pre_step_snapshot = {
        "episode_length": 42,
        "terminated": False,
        "time_out": False,
    }

    payload = module.build_reset_debug_payload(
        env,
        env_id=0,
        done=True,
        pre_step_snapshot=pre_step_snapshot,
    )
    line = module.format_reset_debug_line(43, payload)

    assert payload["episode_length"] == 42
    assert payload["terminated_pre_step"] is False
    assert payload["time_out_pre_step"] is False
    assert "episode_length=42" in line
    assert "terminated_pre_step=False" in line
    assert "time_out_pre_step=False" in line


def test_capture_reset_debug_snapshot_includes_startup_pose_diagnostics():
    module = _load_play_debug_module()
    robot = FakeRobot()
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([7]),
        termination_manager=FakeTerminationManager(),
        scene=SimpleNamespace(articulations={"robot": robot}, sensors={}),
    )

    snapshot = module.capture_reset_debug_snapshot(env, env_id=0)

    assert snapshot["ankle_joint_pos"] == {
        "left_ankle_pitch_joint": -0.42,
        "left_ankle_roll_joint": 0.02,
        "right_ankle_pitch_joint": -0.35,
        "right_ankle_roll_joint": -0.01,
    }
    assert snapshot["foot_rpy_w"] == {
        "left_ankle_roll_link": [0.1, 0.0, 0.0],
        "right_ankle_roll_link": [0.0, 0.19999, 0.0],
    }


def test_format_foothold_debug_line_can_mark_zero_action_warmup():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([0]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.0]),
        foot_contact=torch.tensor([[True, True]]),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.0, 0.0, 0.0]])),
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=data)}
        ),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_foothold_debug_line(
        12,
        payload,
        zero_act_active=True,
    )

    assert "zero_act_active=True" in line


def test_is_learned_foothold_debug_event_selects_only_evaluation_or_route():
    module = _load_play_debug_module()

    assert (
        module.is_learned_foothold_debug_event(
            {"learned_evaluated": True, "route_event": False}
        )
        is True
    )
    assert (
        module.is_learned_foothold_debug_event(
            {"learned_evaluated": False, "route_event": True}
        )
        is True
    )
    assert (
        module.is_learned_foothold_debug_event(
            {"learned_evaluated": False, "route_event": False}
        )
        is False
    )


def test_learned_foothold_debug_reports_nominal_deviation_without_side_gate():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([0]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.0]),
        foot_contact=torch.tensor([[True, True]]),
        learned_foothold_action_normalized=torch.tensor([[0.4, -0.2]]),
        learned_foothold_decoded_f=torch.tensor([[0.168, -0.05, 0.0]]),
        raw_unclipped_foothold_f=torch.tensor([[0.168, 0.05, 0.0]]),
        learned_foothold_prepared_f=torch.tensor([[0.168, -0.05, 0.10]]),
        learned_foothold_prepared_w=torch.tensor([[1.168, 1.95, 0.40]]),
        learned_foothold_height_valid=torch.tensor([True]),
        learned_foothold_geometric_valid=torch.tensor([False]),
        learned_foothold_prepared_valid=torch.tensor([False]),
        learned_foothold_safety_valid=torch.tensor([True]),
        learned_foothold_evaluated=torch.tensor([True]),
        learned_foothold_event_generation=torch.tensor([7]),
        learned_foothold_penetrating_point_count=torch.tensor([2.0]),
        learned_foothold_penetrating_point_ratio=torch.tensor([0.07692]),
        learned_foothold_total_penetration_depth=torch.tensor([0.012]),
        learned_foothold_safety_score=torch.tensor([-0.2]),
        learned_foothold_route_event=torch.tensor([False]),
        learned_foothold_route_use_nominal=torch.tensor([False]),
        learned_foothold_route_use_learned=torch.tensor([False]),
        learned_foothold_route_initial_executable=torch.tensor([False]),
        learned_foothold_route_outcome=torch.tensor([3]),
    )
    sensor = SimpleNamespace(
        data=data,
        cfg=SimpleNamespace(max_foothold_step_height_m=0.25),
        _flat_provider_cfg=SimpleNamespace(
            outer_radius_x=0.42,
            outer_radius_y=0.25,
        ),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.5, 0.0, 0.0]])),
        scene=SimpleNamespace(sensors={"foothold_planner": sensor}),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_learned_foothold_debug_line(12, payload)

    assert payload["learned_action_normalized"] == [0.4, -0.2]
    assert payload["learned_decoded_f"] == [0.168, -0.05, 0.0]
    assert payload["learned_prepared_f"] == [0.168, -0.05, 0.1]
    assert payload["learned_prepared_w"] == [1.168, 1.95, 0.4]
    assert payload["learned_relative_height_m"] == 0.1
    assert payload["learned_step_height_valid"] is True
    assert payload["learned_max_step_height_m"] == 0.25
    assert payload["nominal_delta_f"] == [0.0, -0.1]
    assert payload["nominal_deviation_cost"] == 0.2
    assert payload["reward_branch"] == "geometry_invalid"
    assert payload["learned_event_generation"] == 7
    assert "[LEARNED_FOOTHOLD_DEBUG]" in line
    assert "step_height_valid=True" in line
    assert "nominal_deviation_cost=0.2" in line
    assert "reward_branch=geometry_invalid" in line
    assert "route_learned=False" in line
    assert payload["route_outcome"] == 3
    assert payload["route_outcome_name"] == "geometric_invalid"
    assert "route_outcome=geometric_invalid" in line


def test_build_foothold_debug_payload_hides_reference_and_touchdown_error_in_hold():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([0]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.0]),
        foot_contact=torch.tensor([[True, True]]),
        planner_valid=torch.tensor([True]),
        touchdown_accepted=torch.tensor([False]),
        target_foothold_w=torch.tensor([[1.0, 2.0, 0.3]]),
        swing_reference_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
        actual_swing_foot_pos_w=torch.tensor([[10.0, 20.0, 0.35]]),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.5, 0.0, -0.2]])),
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=data)}
        ),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_foothold_debug_line(12, payload)

    assert payload["gait_mode"] == "HOLD"
    assert payload["reference_xy_error"] is None
    assert payload["reference_z_error"] is None
    assert payload["touchdown_xy_error"] is None
    assert payload["touchdown_z_error"] is None
    assert "ref_xy_err=None" in line
    assert "ref_z_err=None" in line
    assert "td_xy_err=None" in line
    assert "td_z_err=None" in line


def test_build_foothold_debug_payload_hides_errors_without_active_plan_in_recovery():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([8]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.0]),
        foot_contact=torch.tensor([[False, True]]),
        planner_valid=torch.tensor([True]),
        touchdown_accepted=torch.tensor([False]),
        target_foothold_f=torch.tensor([[0.0, 0.0, 0.0]]),
        target_foothold_w=torch.tensor([[0.0, 0.0, 0.0]]),
        swing_reference_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        actual_swing_foot_pos_w=torch.tensor([[-11.85, -35.96, 0.08]]),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.0, 0.0, -1.0]])),
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=data)}
        ),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_foothold_debug_line(250, payload)

    assert payload["gait_mode"] == "RECOVERY"
    assert payload["reference_xy_error"] is None
    assert payload["reference_z_error"] is None
    assert payload["touchdown_xy_error"] is None
    assert payload["touchdown_z_error"] is None
    assert "ref_xy_err=None" in line
    assert "td_xy_err=None" in line


def test_is_foothold_debug_anomaly_detects_modes_errors_and_no_liftoff():
    module = _load_play_debug_module()

    assert not module.is_foothold_debug_anomaly(
        {
            "gait_mode": "LEFT_SWING",
            "phase": 0.1,
            "swing_has_lifted": True,
            "reference_xy_error": 0.02,
            "touchdown_xy_error": 0.03,
        }
    )
    assert module.is_foothold_debug_anomaly(
        {
            "gait_mode": "RECOVERY",
            "phase": 0.0,
            "swing_has_lifted": False,
            "reference_xy_error": 0.0,
            "touchdown_xy_error": None,
        }
    )
    assert module.is_foothold_debug_anomaly(
        {
            "gait_mode": "LEFT_SWING",
            "phase": 0.3,
            "swing_has_lifted": True,
            "reference_xy_error": 0.2,
            "touchdown_xy_error": 0.0,
        }
    )
    assert module.is_foothold_debug_anomaly(
        {
            "gait_mode": "RIGHT_SWING",
            "phase": 0.3,
            "swing_has_lifted": False,
            "reference_xy_error": 0.0,
            "touchdown_xy_error": 0.0,
        }
    )
