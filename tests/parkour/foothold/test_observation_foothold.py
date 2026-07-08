from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command

    def get_command(self, command_name: str) -> torch.Tensor:
        assert command_name == "base_velocity"
        return self.command


class _FakePlanner:
    def __init__(self, data):
        self._data = data
        self.desired_velocity = None

    def set_desired_velocity(self, desired_velocity_f: torch.Tensor) -> None:
        self.desired_velocity = desired_velocity_f.clone()

    @property
    def data(self):
        assert self.desired_velocity is not None
        return self._data


def _load_foothold_observation_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "envs"
        / "mdp"
        / "observations"
        / "foothold.py"
    )
    spec = importlib.util.spec_from_file_location(
        "foothold_observation_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_foothold_planner_observation_exposes_target_velocity_phase_and_side():
    foothold = _load_foothold_observation_module()

    planner_data = SimpleNamespace(
        target_foothold_f=torch.tensor(
            [
                [0.10, 0.18, 0.00],
                [0.08, -0.18, 0.00],
            ]
        ),
        feasible_velocity_f=torch.tensor(
            [
                [0.50, 0.00, 0.10],
                [0.40, -0.05, -0.10],
            ]
        ),
        phase=torch.tensor([0.25, 0.75]),
        swing_side=torch.tensor([0, 1]),
        default_swing_apex_height=torch.tensor([0.08, 0.08]),
        swing_apex_height=torch.tensor([0.14, 0.08]),
        swing_clearance_safe=torch.tensor([False, True]),
        swing_clearance_penetration=torch.tensor([0.02, 0.00]),
    )
    command = torch.tensor(
        [
            [0.50, 0.00, 0.10],
            [0.40, -0.05, -0.10],
        ]
    )
    planner = _FakePlanner(planner_data)
    env = SimpleNamespace(
        command_manager=_FakeCommandManager(command),
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": planner,
            }
        ),
    )

    observation = foothold.foothold_planner_observation(env)

    expected = torch.tensor(
        [
            [0.10, 0.18, 0.00, 0.50, 0.00, 0.10, 0.25, -1.00, 0.14, 0.06, 0.00, 0.02,],
            [0.08, -0.18, 0.00, 0.40, -0.05, -0.10, 0.75, 1.00, 0.08, 0.00, 1.00, 0.00,],
        ]
    )
    torch.testing.assert_close(planner.desired_velocity, command)
    torch.testing.assert_close(observation, expected)
