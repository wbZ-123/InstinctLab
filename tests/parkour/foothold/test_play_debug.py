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
    def __init__(self, command: torch.Tensor):
        self._command = command

    def get_command(self, name: str) -> torch.Tensor:
        assert name == "base_velocity"
        return self._command


def test_build_foothold_debug_payload_reads_command_and_planner_data():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([1]),
        swing_side=torch.tensor([1]),
        phase=torch.tensor([0.25]),
        foot_contact=torch.tensor([[True, False]]),
        planner_valid=torch.tensor([True]),
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
        target_foothold_f=torch.tensor([[0.2, -0.1, 0.0]]),
        target_foothold_w=torch.tensor([[1.0, 2.0, 0.3]]),
        actual_swing_foot_pos_w=torch.tensor([[1.03, 1.96, 0.35]]),
        feasible_velocity_f=torch.tensor([[0.5, 0.0, 0.0]]),
    )
    env = SimpleNamespace(
        command_manager=FakeCommandManager(torch.tensor([[0.5, 0.0, -0.2]])),
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=data)}
        ),
    )

    payload = module.build_foothold_debug_payload(env)
    line = module.format_foothold_debug_line(12, payload)

    assert payload["command"] == [0.5, 0.0, -0.2]
    assert payload["gait_mode"] == "LEFT_SWING"
    assert payload["foot_contact"] == [True, False]
    assert payload["touchdown_swing_contact"] is False
    assert payload["touchdown_xy_ok"] is True
    assert payload["touchdown_z_ok"] is True
    assert payload["touchdown_within_tolerance"] is True
    assert payload["swing_has_lifted"] is True
    assert payload["recovery_step_active"] is True
    assert payload["actual_swing_w"] == [1.03, 1.96, 0.35]
    assert payload["target_w"] == [1.0, 2.0, 0.3]
    assert payload["touchdown_xy_error"] == 0.05
    assert payload["touchdown_z_error"] == 0.05
    assert "step=12" in line
    assert "mode=LEFT_SWING" in line
    assert "command=[0.5, 0.0, -0.2]" in line
    assert "td_contact=False" in line
    assert "td_xy_ok=True" in line
    assert "td_z_ok=True" in line
    assert "td_within_tol=True" in line
    assert "lifted=True" in line
    assert "recovery_step=True" in line
    assert "actual_swing_w=[1.03, 1.96, 0.35]" in line
    assert "target_w=[1.0, 2.0, 0.3]" in line
    assert "td_xy_err=0.05" in line
    assert "td_z_err=0.05" in line


def test_build_foothold_debug_payload_hides_touchdown_error_in_hold():
    module = _load_play_debug_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([0]),
        swing_side=torch.tensor([0]),
        phase=torch.tensor([0.0]),
        foot_contact=torch.tensor([[True, True]]),
        planner_valid=torch.tensor([True]),
        touchdown_accepted=torch.tensor([False]),
        target_foothold_w=torch.tensor([[1.0, 2.0, 0.3]]),
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
    assert payload["touchdown_xy_error"] is None
    assert payload["touchdown_z_error"] is None
    assert "td_xy_err=None" in line
    assert "td_z_err=None" in line
