from __future__ import annotations

import importlib.util
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


def _make_env(data):
    planner = SimpleNamespace(data=data)
    return SimpleNamespace(
        num_envs=data.gait_mode.shape[0],
        device=torch.device("cpu"),
        scene=SimpleNamespace(sensors={"foothold_planner": planner}),
    )


def _make_cfg():
    return SimpleNamespace(params={"sensor_name": "foothold_planner"})


def _make_data(num_envs=2):
    return SimpleNamespace(
        gait_mode=torch.zeros(num_envs, dtype=torch.long),
        touchdown_accepted=torch.zeros(num_envs, dtype=torch.bool),
        swing_clearance_safe=torch.ones(num_envs, dtype=torch.bool),
        swing_clearance_penetration=torch.zeros(num_envs),
        default_swing_apex_height=torch.full((num_envs,), 0.08),
        swing_apex_height=torch.full((num_envs,), 0.08),
        planner_valid=torch.ones(num_envs, dtype=torch.bool),
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