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


def _load_foothold_reward_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "envs"
        / "mdp"
        / "rewards"
        / "foothold.py"
    )
    spec = importlib.util.spec_from_file_location(
        "foothold_reward_under_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_swing_tracking_rewards_reference_match_and_masks_non_swing():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 0]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.3],
                [1.0, 2.0, 0.3],
            ]
        ),
        swing_reference_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.3],
                [2.0, 2.0, 0.3],
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_swing_tracking_exp(env, std=0.2)

    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0]))


def test_swing_tracking_syncs_base_velocity_before_reading_planner_data():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1]),
        actual_swing_foot_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
        swing_reference_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
    )
    planner = _FakePlanner(planner_data)
    command = torch.tensor([[0.5, -0.1, 0.2]])
    env = SimpleNamespace(
        command_manager=_FakeCommandManager(command),
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": planner,
            }
        ),
    )

    reward = foothold.foothold_swing_tracking_exp(env, std=0.2)

    torch.testing.assert_close(planner.desired_velocity, command)
    torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_touchdown_tracking_rewards_accepted_touchdown_only():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([3, 1, 3]),
        touchdown_accepted=torch.tensor([True, True, False]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [0.4, -0.2, 0.1],
                [0.4, -0.2, 0.1],
                [0.4, -0.2, 0.1],
            ]
        ),
        target_foothold_w=torch.tensor(
            [
                [0.4, -0.2, 0.1],
                [0.4, -0.2, 0.1],
                [0.4, -0.2, 0.1],
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_touchdown_tracking_exp(env, std=0.2)

    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, 0.0]))


def test_foothold_diagnostic_indicators_expose_gait_state_events():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        touchdown_accepted=torch.tensor(
            [False, False, False, True, False, False, False, False]
        ),
        planner_valid=torch.tensor(
            [True, True, True, True, True, True, True, False]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    torch.testing.assert_close(
        foothold.foothold_swing_mode_indicator(env),
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_reset_mode_indicator(env),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_left_swing_mode_indicator(env),
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_right_swing_mode_indicator(env),
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_touchdown_confirm_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_early_contact_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_overdue_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_stance_lost_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_touchdown_accepted_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_plan_invalid_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )

def test_foothold_clearance_safe_indicator_returns_float_mask():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        swing_clearance_safe=torch.tensor([True, False, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_clearance_safe_indicator(env)

    expected = torch.tensor([1.0, 0.0, 1.0])
    torch.testing.assert_close(reward, expected)

def test_foothold_clearance_penetration_l1_returns_penetration_depth():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        swing_clearance_penetration=torch.tensor([0.0, 0.02, 0.15]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_clearance_penetration_l1(env)

    expected = torch.tensor([0.0, 0.02, 0.15])
    torch.testing.assert_close(reward, expected)
