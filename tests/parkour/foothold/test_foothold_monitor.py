from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import pytest


def _load_monitor_module():
    source_root = (
        Path(__file__).resolve().parents[3] / "source" / "instinctlab"
    )
    instinctlab_package = ModuleType("instinctlab")
    instinctlab_package.__path__ = [str(source_root / "instinctlab")]
    monitors_package = ModuleType("instinctlab.monitors")
    monitors_package.__path__ = [
        str(source_root / "instinctlab" / "monitors")
    ]
    manager_module = ModuleType("instinctlab.monitors.monitor_manager")

    class MonitorTerm:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.device = env.device

    path = source_root / "instinctlab" / "monitors" / "foothold.py"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "instinctlab", instinctlab_package)
        monkeypatch.setitem(
            sys.modules, "instinctlab.monitors", monitors_package
        )
        monkeypatch.setitem(
            sys.modules,
            "instinctlab.monitors.monitor_manager",
            manager_module,
        )
        setattr(manager_module, "MonitorTerm", MonitorTerm)
        spec = importlib.util.spec_from_file_location(
            "instinctlab.monitors.foothold_under_test", path
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command

    def get_command(self, command_name: str) -> torch.Tensor:
        assert command_name == "base_velocity"
        return self.command


class _FakeScene:
    def __init__(self, sensors=None, robot=None):
        self.sensors = sensors or {}
        self._items = {}
        if robot is not None:
            self._items["robot"] = robot

    def __getitem__(self, name: str):
        return self._items[name]


def _make_env(
    data,
    *,
    common_step_counter: int = 0,
    episode_length_buf: torch.Tensor | None = None,
    command: torch.Tensor | None = None,
    root_lin_vel_b: torch.Tensor | None = None,
    curriculum_scale: torch.Tensor | None = None,
):
    num_envs = data.gait_mode.shape[0]
    if curriculum_scale is None:
        curriculum_scale = torch.zeros(num_envs)
    planner = SimpleNamespace(
        data=data,
        flat_target_curriculum_scale=curriculum_scale,
    )
    if episode_length_buf is None:
        episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    if command is None:
        command = torch.zeros(num_envs, 3)
    if root_lin_vel_b is None:
        root_lin_vel_b = torch.zeros(num_envs, 3)
    robot = SimpleNamespace(
        data=SimpleNamespace(root_lin_vel_b=root_lin_vel_b)
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        common_step_counter=common_step_counter,
        episode_length_buf=episode_length_buf,
        command_manager=_FakeCommandManager(command),
        scene=_FakeScene(
            sensors={"foothold_planner": planner},
            robot=robot,
        ),
    )


def _make_cfg(**params):
    return SimpleNamespace(
        params={"sensor_name": "foothold_planner", **params}
    )


def _make_data(num_envs=2):
    return SimpleNamespace(
        gait_mode=torch.zeros(num_envs, dtype=torch.long),
        swing_side=torch.zeros(num_envs, dtype=torch.long),
        touchdown_accepted=torch.zeros(num_envs, dtype=torch.bool),
        swing_clearance_safe=torch.ones(num_envs, dtype=torch.bool),
        swing_clearance_penetration=torch.zeros(num_envs),
        default_swing_apex_height=torch.full((num_envs,), 0.08),
        swing_apex_height=torch.full((num_envs,), 0.08),
        planner_valid=torch.ones(num_envs, dtype=torch.bool),
        safe_target_search_performed=torch.zeros(num_envs, dtype=torch.bool),
        safe_target_final_valid=torch.ones(num_envs, dtype=torch.bool),
        safe_target_used_fallback=torch.zeros(num_envs, dtype=torch.bool),
        safe_target_score=torch.zeros(num_envs),
        safe_target_nominal_inside_ellipse=torch.ones(
            num_envs, dtype=torch.bool
        ),
        safe_target_nominal_obstacle_safe=torch.ones(
            num_envs, dtype=torch.bool
        ),
        safe_target_nominal_valid=torch.ones(num_envs, dtype=torch.bool),
        safe_target_candidate_count=torch.zeros(num_envs),
        safe_target_candidate_inside_ellipse_count=torch.zeros(num_envs),
        safe_target_candidate_obstacle_safe_count=torch.zeros(num_envs),
        safe_target_candidate_valid_count=torch.zeros(num_envs),
        learned_foothold_evaluated=torch.zeros(
            num_envs, dtype=torch.bool
        ),
        learned_foothold_geometric_valid=torch.zeros(
            num_envs, dtype=torch.bool
        ),
        learned_foothold_safety_valid=torch.zeros(
            num_envs, dtype=torch.bool
        ),
        learned_foothold_safety_score=torch.zeros(num_envs),
        learned_foothold_penetrating_point_ratio=torch.zeros(num_envs),
        learned_foothold_total_penetration_depth=torch.zeros(num_envs),
        learned_foothold_route_event=torch.zeros(
            num_envs, dtype=torch.bool
        ),
        learned_foothold_route_use_nominal=torch.zeros(
            num_envs, dtype=torch.bool
        ),
        learned_foothold_route_use_learned=torch.zeros(
            num_envs, dtype=torch.bool
        ),
        learned_foothold_route_initial_executable=torch.zeros(
            num_envs, dtype=torch.bool
        ),
        learned_foothold_used=torch.zeros(num_envs, dtype=torch.bool),
        recovery_step_active=torch.zeros(num_envs, dtype=torch.bool),
    )


def test_monitor_constructs_compact_per_environment_buffers():
    module = _load_monitor_module()
    data = _make_data(num_envs=3)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    assert monitor._step_count.shape == (3,)
    assert monitor._step_count.device.type == "cpu"
    assert monitor._step_count.dtype == torch.float32


def test_update_counts_states_events_and_clearance_without_double_counting():
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    env = _make_env(data)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), env)

    data.gait_mode[:] = torch.tensor([1, 4])
    data.touchdown_accepted[:] = torch.tensor([True, False])
    data.swing_clearance_safe[:] = torch.tensor([False, True])
    data.swing_clearance_penetration[:] = torch.tensor([0.03, 0.50])
    data.swing_apex_height[:] = torch.tensor([0.14, 0.20])
    data.planner_valid[:] = torch.tensor([True, False])
    monitor.update(dt=0.02)
    monitor.update(dt=0.02)

    torch.testing.assert_close(monitor._step_count, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(
        monitor._swing_step_count, torch.tensor([2.0, 0.0])
    )
    torch.testing.assert_close(
        monitor._touchdown_accepted_count, torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(
        monitor._early_contact_count, torch.tensor([0.0, 2.0])
    )
    torch.testing.assert_close(
        monitor._clearance_sample_count, torch.tensor([2.0, 0.0])
    )
    torch.testing.assert_close(
        monitor._penetration_sum, torch.tensor([0.06, 0.0])
    )
    torch.testing.assert_close(
        monitor._penetration_max, torch.tensor([0.03, 0.0])
    )
    torch.testing.assert_close(
        monitor._apex_delta_sum, torch.tensor([0.12, 0.0])
    )
    torch.testing.assert_close(
        monitor._invalid_plan_count, torch.tensor([0.0, 2.0])
    )


def test_update_counts_safe_target_search_diagnostics():
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.safe_target_search_performed[:] = torch.tensor([True, False])
    data.safe_target_final_valid[:] = torch.tensor([True, False])
    data.safe_target_used_fallback[:] = torch.tensor([True, False])
    data.safe_target_score[:] = torch.tensor([0.04, 0.20])
    data.safe_target_nominal_inside_ellipse[:] = torch.tensor([True, False])
    data.safe_target_nominal_obstacle_safe[:] = torch.tensor([False, True])
    data.safe_target_nominal_valid[:] = torch.tensor([False, False])
    data.safe_target_candidate_count[:] = torch.tensor([32.0, 99.0])
    data.safe_target_candidate_inside_ellipse_count[:] = torch.tensor([28.0, 99.0])
    data.safe_target_candidate_obstacle_safe_count[:] = torch.tensor([12.0, 99.0])
    data.safe_target_candidate_valid_count[:] = torch.tensor([4.0, 99.0])
    monitor.update(dt=0.02)

    data.safe_target_search_performed[:] = torch.tensor([False, True])
    data.safe_target_final_valid[:] = torch.tensor([True, True])
    data.safe_target_used_fallback[:] = torch.tensor([False, True])
    data.safe_target_score[:] = torch.tensor([0.00, 0.10])
    data.safe_target_nominal_inside_ellipse[:] = torch.tensor([False, True])
    data.safe_target_nominal_obstacle_safe[:] = torch.tensor([False, False])
    data.safe_target_nominal_valid[:] = torch.tensor([False, False])
    data.safe_target_candidate_count[:] = torch.tensor([99.0, 32.0])
    data.safe_target_candidate_inside_ellipse_count[:] = torch.tensor([99.0, 20.0])
    data.safe_target_candidate_obstacle_safe_count[:] = torch.tensor([99.0, 8.0])
    data.safe_target_candidate_valid_count[:] = torch.tensor([99.0, 2.0])
    monitor.update(dt=0.02)

    torch.testing.assert_close(
        monitor._safe_target_search_count, torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(
        monitor._safe_target_final_valid_count, torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(
        monitor._safe_target_fallback_count, torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(
        monitor._safe_target_score_sum, torch.tensor([0.04, 0.10])
    )
    torch.testing.assert_close(
        monitor._safe_target_score_max, torch.tensor([0.04, 0.10])
    )
    torch.testing.assert_close(
        monitor._safe_target_nominal_inside_ellipse_count,
        torch.tensor([1.0, 1.0]),
    )
    torch.testing.assert_close(
        monitor._safe_target_nominal_obstacle_safe_count,
        torch.tensor([0.0, 0.0]),
    )
    torch.testing.assert_close(
        monitor._safe_target_nominal_valid_count, torch.tensor([0.0, 0.0])
    )
    torch.testing.assert_close(
        monitor._safe_target_candidate_count_sum,
        torch.tensor([32.0, 32.0]),
    )
    torch.testing.assert_close(
        monitor._safe_target_candidate_inside_ellipse_count_sum,
        torch.tensor([28.0, 20.0]),
    )
    torch.testing.assert_close(
        monitor._safe_target_candidate_obstacle_safe_count_sum,
        torch.tensor([12.0, 8.0]),
    )
    torch.testing.assert_close(
        monitor._safe_target_candidate_valid_count_sum,
        torch.tensor([4.0, 2.0]),
    )

    log = monitor.get_log()
    torch.testing.assert_close(
        log["safe_target_final_valid_fraction"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["safe_target_fallback_fraction"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["safe_target_score_mean"], torch.tensor(0.07)
    )
    torch.testing.assert_close(
        log["safe_target_score_max"], torch.tensor(0.10)
    )
    torch.testing.assert_close(
        log["safe_target_nominal_inside_ellipse_fraction"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        log["safe_target_nominal_obstacle_safe_fraction"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        log["safe_target_nominal_valid_fraction"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["safe_target_candidate_count_mean"], torch.tensor(32.0)
    )
    torch.testing.assert_close(
        log["safe_target_candidate_inside_ellipse_count_mean"],
        torch.tensor(24.0),
    )
    torch.testing.assert_close(
        log["safe_target_candidate_obstacle_safe_count_mean"],
        torch.tensor(10.0),
    )
    torch.testing.assert_close(
        log["safe_target_candidate_valid_count_mean"], torch.tensor(3.0)
    )


def test_safe_target_diagnostics_ignore_non_search_steps():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.safe_target_search_performed[:] = False
    data.safe_target_final_valid[:] = False
    data.safe_target_used_fallback[:] = True
    data.safe_target_score[:] = 0.20
    data.safe_target_nominal_inside_ellipse[:] = False
    data.safe_target_nominal_obstacle_safe[:] = False
    data.safe_target_nominal_valid[:] = False
    data.safe_target_candidate_count[:] = 32.0
    data.safe_target_candidate_inside_ellipse_count[:] = 24.0
    data.safe_target_candidate_obstacle_safe_count[:] = 12.0
    data.safe_target_candidate_valid_count[:] = 4.0
    monitor.update(dt=0.02)

    log = monitor.get_log()
    torch.testing.assert_close(
        log["safe_target_search_rate"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["safe_target_final_valid_fraction"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["safe_target_fallback_fraction"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["safe_target_score_mean"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["safe_target_nominal_valid_fraction"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["safe_target_candidate_valid_count_mean"], torch.tensor(0.0)
    )


def test_learned_foothold_metrics_separate_evaluations_from_route_events():
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    # A HOLD evaluation is a learning sample, but not an execution route.
    data.learned_foothold_evaluated[:] = torch.tensor([True, True])
    data.learned_foothold_geometric_valid[:] = torch.tensor([True, False])
    data.learned_foothold_safety_valid[:] = torch.tensor([True, False])
    data.learned_foothold_safety_score[:] = torch.tensor([0.8, -0.6])
    data.learned_foothold_penetrating_point_ratio[:] = torch.tensor([0.0, 0.5])
    data.learned_foothold_total_penetration_depth[:] = torch.tensor([0.0, 0.04])
    monitor.update(dt=0.02)

    # A later new-SWING route uses the learned target for env 0 and rejects
    # env 1.  Stale evaluation values must not be counted a second time.
    data.learned_foothold_evaluated[:] = False
    data.learned_foothold_route_event[:] = True
    data.learned_foothold_route_use_nominal[:] = False
    data.learned_foothold_route_use_learned[:] = torch.tensor([True, False])
    data.learned_foothold_route_initial_executable[:] = torch.tensor(
        [True, False]
    )
    data.learned_foothold_used[:] = torch.tensor([True, False])
    data.learned_foothold_safety_valid[:] = torch.tensor([True, False])
    data.planner_valid[:] = torch.tensor([True, False])
    monitor.update(dt=0.02)

    log = monitor.get_log()
    torch.testing.assert_close(
        log["learned_foothold_evaluation_rate"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["learned_foothold_geometric_valid_fraction"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        log["learned_foothold_safety_valid_fraction"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["learned_foothold_safety_score_mean"], torch.tensor(0.1)
    )
    torch.testing.assert_close(
        log["learned_foothold_penetrating_point_ratio_mean"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        log["learned_foothold_total_penetration_depth_mean"],
        torch.tensor(0.02),
    )
    torch.testing.assert_close(
        log["learned_foothold_route_rate"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["learned_foothold_route_nominal_fraction"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["learned_foothold_route_learned_fraction"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["learned_foothold_route_invalid_fraction"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["learned_foothold_route_postcheck_invalid_fraction"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        log["learned_foothold_routed_safe_fraction"], torch.tensor(1.0)
    )


def test_learned_route_postcheck_failure_does_not_change_route_choice():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.learned_foothold_route_event[:] = True
    data.learned_foothold_route_use_nominal[:] = True
    data.learned_foothold_route_initial_executable[:] = True
    # A later terrain/trajectory check invalidated an initially selected
    # nominal route.
    data.planner_valid[:] = False
    monitor.update(dt=0.02)

    log = monitor.get_log()
    torch.testing.assert_close(
        log["learned_foothold_route_nominal_fraction"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["learned_foothold_route_invalid_fraction"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["learned_foothold_route_postcheck_invalid_fraction"],
        torch.tensor(1.0),
    )


def test_safe_target_event_ratios_use_total_search_events_not_env_average():
    module = _load_monitor_module()
    data = _make_data(num_envs=3)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.safe_target_search_performed[:] = torch.tensor([True, False, False])
    data.safe_target_final_valid[:] = torch.tensor([True, False, False])
    data.safe_target_used_fallback[:] = torch.tensor([True, False, False])
    data.safe_target_score[:] = torch.tensor([0.05, 99.0, 99.0])
    data.safe_target_nominal_inside_ellipse[:] = torch.tensor([True, False, False])
    data.safe_target_nominal_obstacle_safe[:] = torch.tensor([True, False, False])
    data.safe_target_nominal_valid[:] = torch.tensor([True, False, False])
    data.safe_target_candidate_count[:] = torch.tensor([32.0, 99.0, 99.0])
    data.safe_target_candidate_inside_ellipse_count[:] = torch.tensor([24.0, 99.0, 99.0])
    data.safe_target_candidate_obstacle_safe_count[:] = torch.tensor([16.0, 99.0, 99.0])
    data.safe_target_candidate_valid_count[:] = torch.tensor([8.0, 99.0, 99.0])
    monitor.update(dt=0.02)

    log = monitor.get_log()
    torch.testing.assert_close(
        log["safe_target_final_valid_fraction"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["safe_target_fallback_fraction"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["safe_target_score_mean"], torch.tensor(0.05)
    )
    torch.testing.assert_close(
        log["safe_target_candidate_count_mean"], torch.tensor(32.0)
    )
    torch.testing.assert_close(
        log["safe_target_candidate_valid_count_mean"], torch.tensor(8.0)
    )


def test_debug_dump_records_limited_invalid_safe_target_events(tmp_path):
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    data.raw_unclipped_foothold_f = torch.zeros(2, 3)
    data.target_foothold_f = torch.zeros(2, 3)
    data.desired_velocity_f = torch.zeros(2, 3)
    data.feasible_velocity_f = torch.zeros(2, 3)
    path = tmp_path / "foothold_debug_events.jsonl"
    monitor = module.FootholdPlannerMonitorTerm(
        _make_cfg(
            debug_event_path=str(path),
            debug_event_max_count=1,
        ),
        _make_env(data),
    )

    data.safe_target_search_performed[:] = torch.tensor([True, True])
    data.safe_target_final_valid[:] = torch.tensor([False, False])
    data.planner_valid[:] = torch.tensor([False, False])
    data.safe_target_nominal_inside_ellipse[:] = torch.tensor([True, False])
    data.safe_target_nominal_obstacle_safe[:] = torch.tensor([False, True])
    data.safe_target_nominal_valid[:] = torch.tensor([False, False])
    data.safe_target_candidate_count[:] = torch.tensor([32.0, 32.0])
    data.safe_target_candidate_inside_ellipse_count[:] = torch.tensor([30.0, 2.0])
    data.safe_target_candidate_obstacle_safe_count[:] = torch.tensor([0.0, 30.0])
    data.safe_target_candidate_valid_count[:] = torch.tensor([0.0, 0.0])
    data.raw_unclipped_foothold_f[:] = torch.tensor(
        [[0.10, 0.02, 0.0], [0.20, -0.03, 0.0]]
    )
    data.target_foothold_f[:] = torch.tensor(
        [[0.12, 0.01, 0.0], [0.18, -0.02, 0.0]]
    )
    data.desired_velocity_f[:] = torch.tensor(
        [[0.50, 0.0, 0.0], [0.40, 0.0, 0.0]]
    )
    monitor.update(dt=0.02)
    monitor.update(dt=0.02)

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "safe_target_invalid"
    assert event["env_id"] == 0
    assert event["reason"] == "candidate_obstacle_blocked"
    assert event["safe_target_nominal_inside_ellipse"] is True
    assert event["safe_target_nominal_obstacle_safe"] is False
    assert event["safe_target_candidate_valid_count"] == 0.0
    assert event["raw_unclipped_foothold_f"] == [0.1, 0.02, 0.0]


def test_touchdown_confirm_counts_only_mode_entry():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.gait_mode[:] = 3
    monitor.update(dt=0.02)
    monitor.update(dt=0.02)
    data.gait_mode[:] = 0
    monitor.update(dt=0.02)
    data.gait_mode[:] = 3
    monitor.update(dt=0.02)

    torch.testing.assert_close(
        monitor._touchdown_confirm_count, torch.tensor([2.0])
    )


def test_monitor_reports_full_gait_mode_distribution_and_swing_entries():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    for mode in [0, 1, 1, 3, 2, 2, 6, 9]:
        data.gait_mode[:] = mode
        monitor.update(dt=0.02)

    torch.testing.assert_close(monitor._hold_step_count, torch.tensor([1.0]))
    torch.testing.assert_close(monitor._left_swing_step_count, torch.tensor([2.0]))
    torch.testing.assert_close(monitor._right_swing_step_count, torch.tensor([2.0]))
    torch.testing.assert_close(
        monitor._touchdown_confirm_step_count, torch.tensor([1.0])
    )
    torch.testing.assert_close(monitor._stance_lost_count, torch.tensor([1.0]))
    torch.testing.assert_close(
        monitor._hold_contact_lost_step_count, torch.tensor([1.0])
    )
    torch.testing.assert_close(
        monitor._hold_contact_lost_entry_count, torch.tensor([1.0])
    )
    torch.testing.assert_close(monitor._swing_entry_count, torch.tensor([2.0]))
    torch.testing.assert_close(monitor._left_swing_entry_count, torch.tensor([1.0]))
    torch.testing.assert_close(monitor._right_swing_entry_count, torch.tensor([1.0]))
    torch.testing.assert_close(monitor._swing_duration_step_sum, torch.tensor([4.0]))

    log = monitor.get_log()
    torch.testing.assert_close(log["hold_fraction"], torch.tensor(1.0 / 8.0))
    torch.testing.assert_close(log["left_swing_fraction"], torch.tensor(2.0 / 8.0))
    torch.testing.assert_close(log["right_swing_fraction"], torch.tensor(2.0 / 8.0))
    torch.testing.assert_close(
        log["touchdown_confirm_fraction"], torch.tensor(1.0 / 8.0)
    )
    torch.testing.assert_close(
        log["hold_contact_lost_fraction"], torch.tensor(1.0 / 8.0)
    )
    torch.testing.assert_close(
        log["hold_contact_lost_entry_step_rate"], torch.tensor(1.0 / 8.0)
    )
    torch.testing.assert_close(log["swing_entry_step_rate"], torch.tensor(2.0 / 8.0))
    torch.testing.assert_close(log["left_swing_entry_step_rate"], torch.tensor(1.0 / 8.0))
    torch.testing.assert_close(log["right_swing_entry_step_rate"], torch.tensor(1.0 / 8.0))
    torch.testing.assert_close(log["mean_swing_duration_steps"], torch.tensor(2.0))


def test_monitor_reports_recovery_step_active_fraction_and_entries():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    for recovery_step_active in [False, True, True, False, True]:
        data.recovery_step_active[:] = recovery_step_active
        monitor.update(dt=0.02)

    log = monitor.get_log()
    torch.testing.assert_close(
        monitor._recovery_step_active_count, torch.tensor([3.0])
    )
    torch.testing.assert_close(
        monitor._recovery_step_entry_count, torch.tensor([2.0])
    )
    torch.testing.assert_close(
        log["recovery_step_fraction"], torch.tensor(3.0 / 5.0)
    )
    torch.testing.assert_close(
        log["recovery_step_entry_step_rate"], torch.tensor(2.0 / 5.0)
    )


def test_monitor_reports_side_specific_touchdown_and_failure_entries():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.gait_mode[:] = 1
    data.swing_side[:] = 0
    data.touchdown_accepted[:] = False
    monitor.update(dt=0.02)

    data.touchdown_accepted[:] = True
    monitor.update(dt=0.02)

    data.gait_mode[:] = 3
    monitor.update(dt=0.02)

    data.touchdown_accepted[:] = False
    data.gait_mode[:] = 2
    data.swing_side[:] = 1
    monitor.update(dt=0.02)

    data.touchdown_accepted[:] = True
    monitor.update(dt=0.02)

    data.gait_mode[:] = 6
    monitor.update(dt=0.02)

    data.gait_mode[:] = 8
    monitor.update(dt=0.02)

    torch.testing.assert_close(
        monitor._left_touchdown_accepted_count, torch.tensor([1.0])
    )
    torch.testing.assert_close(
        monitor._right_touchdown_accepted_count, torch.tensor([1.0])
    )
    torch.testing.assert_close(
        monitor._left_touchdown_confirm_count, torch.tensor([1.0])
    )
    torch.testing.assert_close(
        monitor._right_touchdown_confirm_count, torch.tensor([0.0])
    )
    torch.testing.assert_close(
        monitor._stance_lost_entry_count, torch.tensor([1.0])
    )
    torch.testing.assert_close(
        monitor._recovery_entry_count, torch.tensor([1.0])
    )

    log = monitor.get_log()
    torch.testing.assert_close(
        log["left_touchdown_accepted_step_rate"], torch.tensor(1.0 / 7.0)
    )
    torch.testing.assert_close(
        log["right_touchdown_accepted_step_rate"], torch.tensor(1.0 / 7.0)
    )
    torch.testing.assert_close(
        log["left_touchdown_confirm_step_rate"], torch.tensor(1.0 / 7.0)
    )
    torch.testing.assert_close(
        log["right_touchdown_confirm_step_rate"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["stance_lost_entry_step_rate"], torch.tensor(1.0 / 7.0)
    )
    torch.testing.assert_close(
        log["recovery_entry_step_rate"], torch.tensor(1.0 / 7.0)
    )


def test_monitor_reports_failure_entries_per_swing_entry_by_side():
    module = _load_monitor_module()
    data = _make_data(num_envs=4)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.gait_mode[:] = torch.tensor([1, 2, 1, 2])
    data.swing_side[:] = torch.tensor([0, 1, 0, 1])
    monitor.update(dt=0.02)

    data.gait_mode[:] = torch.tensor([6, 6, 4, 5])
    monitor.update(dt=0.02)

    data.gait_mode[:] = torch.tensor([8, 8, 8, 8])
    monitor.update(dt=0.02)

    log = monitor.get_log()
    torch.testing.assert_close(
        log["stance_lost_per_swing_entry"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["early_contact_per_swing_entry"], torch.tensor(0.25)
    )
    torch.testing.assert_close(
        log["overdue_per_swing_entry"], torch.tensor(0.25)
    )
    torch.testing.assert_close(
        log["recovery_per_swing_entry"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["left_swing_stance_lost_per_swing_entry"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["right_swing_stance_lost_per_swing_entry"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["left_swing_early_contact_per_swing_entry"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["right_swing_early_contact_per_swing_entry"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["left_swing_overdue_per_swing_entry"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        log["right_swing_overdue_per_swing_entry"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        log["left_swing_recovery_per_swing_entry"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["right_swing_recovery_per_swing_entry"], torch.tensor(1.0)
    )


def test_monitor_reports_hold_contact_lost_per_swing_entry():
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.gait_mode[:] = torch.tensor([1, 2])
    data.swing_side[:] = torch.tensor([0, 1])
    monitor.update(dt=0.02)

    data.gait_mode[:] = torch.tensor([9, 0])
    monitor.update(dt=0.02)

    log = monitor.get_log()
    torch.testing.assert_close(
        log["hold_contact_lost_per_swing_entry"], torch.tensor(0.5)
    )


def test_monitor_logs_actual_planner_curriculum_scale():
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    monitor = module.FootholdPlannerMonitorTerm(
        _make_cfg(),
        _make_env(
            data,
            episode_length_buf=torch.tensor([10, 999]),
            curriculum_scale=torch.tensor([0.25, 0.75]),
        ),
    )

    monitor.update(dt=0.02)

    torch.testing.assert_close(
        monitor.get_log()["reward_curriculum_scale"], torch.tensor(0.5)
    )


def test_monitor_curriculum_scale_survives_episode_reset_without_recomputation():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    env = _make_env(
        data,
        episode_length_buf=torch.tensor([950]),
        curriculum_scale=torch.tensor([0.8]),
    )
    monitor = module.FootholdPlannerMonitorTerm(
        _make_cfg(),
        env,
    )

    monitor.update(dt=0.02)
    env.episode_length_buf = torch.tensor([3])
    monitor.update(dt=0.02)

    torch.testing.assert_close(
        monitor.get_log()["reward_curriculum_scale"], torch.tensor(0.8)
    )


def test_partial_reset_reports_completed_env_and_preserves_other_env():
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.gait_mode[:] = torch.tensor([1, 1])
    data.swing_clearance_safe[:] = torch.tensor([True, False])
    data.swing_clearance_penetration[:] = torch.tensor([0.0, 0.04])
    data.swing_apex_height[:] = torch.tensor([0.10, 0.16])
    monitor.update(dt=0.02)
    monitor.reset_idx(torch.tensor([0]))
    episode = monitor.get_log(is_episode=True)

    assert episode["swing_fraction"].item() == 1.0
    assert episode["clearance_safe_fraction"].item() == 1.0
    assert episode["penetration_mean"].item() == 0.0
    assert monitor._step_count[0].item() == 0.0
    assert monitor._step_count[1].item() == 1.0
    torch.testing.assert_close(
        monitor._penetration_sum[1], torch.tensor(0.04)
    )


def test_nonfinite_and_empty_samples_log_finite_zero():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.swing_clearance_penetration[:] = float("nan")
    data.swing_apex_height[:] = float("inf")
    monitor.update(dt=0.02)
    monitor.reset_idx(torch.tensor([0]))
    episode = monitor.get_log(is_episode=True)

    assert episode["nonfinite_fraction"].item() == 1.0
    assert episode["penetration_mean"].item() == 0.0
    assert episode["apex_delta_mean"].item() == 0.0
    assert all(torch.isfinite(value) for value in episode.values())
