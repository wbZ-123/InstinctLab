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


def test_touchdown_tracking_rewards_touchdown_confirm_with_continuous_error():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([3, 1, 3]),
        touchdown_accepted=torch.tensor([False, True, False]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [0.4, -0.2, 0.1],
                [0.4, -0.2, 0.1],
                [0.5, -0.2, 0.1],
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

    expected_error_reward = torch.exp(torch.tensor(-0.01 / 0.04))
    torch.testing.assert_close(
        reward,
        torch.tensor([1.0, 0.0, expected_error_reward.item()]),
    )


def test_swing_contact_indicator_flags_contact_after_liftoff_grace():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 2, 2, 0]),
        swing_side=torch.tensor([0, 0, 1, 1, 0]),
        phase=torch.tensor([0.05, 0.20, 0.20, 0.20, 0.20]),
        foot_contact=torch.tensor(
            [
                [True, True],   # left swing, still in grace window
                [True, True],   # left swing, late enough: bad
                [True, True],   # right swing, late enough: bad
                [True, False],  # right swing, lifted: ok
                [True, True],   # not swing mode
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

    indicator = foothold.foothold_swing_contact_indicator(
        env,
        min_phase=0.20,
    )

    torch.testing.assert_close(
        indicator,
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]),
    )


def test_no_liftoff_indicator_flags_swing_without_confirmed_liftoff():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 2, 2, 0]),
        phase=torch.tensor([0.20, 0.35, 0.35, 0.50, 0.50]),
        swing_has_lifted=torch.tensor([False, False, True, False, False]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    indicator = foothold.foothold_no_liftoff_indicator(
        env,
        min_phase=0.35,
    )

    torch.testing.assert_close(
        indicator,
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
    )


def test_swing_height_under_error_penalizes_only_below_reference_height():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 2, 1, 0]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [0.0, 0.0, 0.10],  # below reference by 0.05
                [0.0, 0.0, 0.16],  # above reference: no penalty
                [0.0, 0.0, 0.15],  # exactly reference: no penalty
                [0.0, 0.0, 0.05],  # non-swing: masked
            ]
        ),
        swing_reference_pos_w=torch.tensor(
            [
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.15],
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

    error = foothold.foothold_swing_height_under_error_l1(
        env,
        max_error_m=0.04,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.04, 0.0, 0.0, 0.0]),
    )


def test_swing_xy_error_l2_penalizes_planar_reference_distance():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 2, 0]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.10],
                [1.0, 2.0, 0.20],
                [1.0, 2.0, 0.30],
            ]
        ),
        swing_reference_pos_w=torch.tensor(
            [
                [1.3, 2.4, 0.50],  # xy distance 0.5
                [1.0, 2.0, 0.00],  # xy distance 0.0
                [2.0, 2.0, 0.30],  # non-swing: masked
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

    error = foothold.foothold_swing_xy_error_l2(
        env,
        max_error_m=0.30,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.30, 0.0, 0.0]),
    )


def test_touchdown_xy_error_l2_penalizes_late_swing_and_overdue_only():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 2, 5, 0]),
        phase=torch.tensor([0.40, 0.70, 0.80, 1.0, 1.0]),
        touchdown_xy_error=torch.tensor([0.50, 0.20, 0.10, 0.30, 0.40]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    error = foothold.foothold_touchdown_xy_error_l2(
        env,
        min_phase=0.65,
        max_error_m=0.25,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.0, 0.20, 0.10, 0.25, 0.0]),
    )


def test_touchdown_z_error_l1_penalizes_late_swing_and_overdue_only():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 2, 5, 3]),
        phase=torch.tensor([0.40, 0.70, 0.80, 1.0, 1.0]),
        touchdown_z_error=torch.tensor([0.50, 0.20, 0.10, 0.30, 0.40]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    error = foothold.foothold_touchdown_z_error_l1(
        env,
        min_phase=0.65,
        max_error_m=0.20,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.0, 0.20, 0.10, 0.20, 0.0]),
    )


def test_parkour_reward_cfg_enables_no_liftoff_penalty():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_no_liftoff_indicator = RewTerm(" in cfg_text
    assert "func=mdp.foothold_no_liftoff_indicator" in cfg_text
    assert '"min_phase": 0.35' in cfg_text


def test_parkour_reward_cfg_enables_swing_height_under_error_penalty():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_swing_height_under_error_l1 = RewTerm(" in cfg_text
    assert "func=mdp.foothold_swing_height_under_error_l1" in cfg_text
    assert "weight=-2.0" in cfg_text
    assert '"max_error_m": 0.25' in cfg_text


def test_parkour_reward_cfg_enables_swing_xy_error_penalty():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_swing_xy_error_l2 = RewTerm(" in cfg_text
    assert "func=mdp.foothold_swing_xy_error_l2" in cfg_text
    assert "weight=-1.0" in cfg_text
    assert '"max_error_m": 0.30' in cfg_text


def test_parkour_reward_cfg_enables_touchdown_continuous_error_penalties():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_touchdown_xy_error_l2 = RewTerm(" in cfg_text
    assert "func=mdp.foothold_touchdown_xy_error_l2" in cfg_text
    assert "foothold_touchdown_z_error_l1 = RewTerm(" in cfg_text
    assert "func=mdp.foothold_touchdown_z_error_l1" in cfg_text
    assert '"min_phase": 0.65' in cfg_text
    assert '"max_error_m": 0.30' in cfg_text
    assert '"max_error_m": 0.20' in cfg_text


def test_parkour_reward_cfg_penalizes_clearance_penetration_without_safe_bonus():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_clearance_safe_indicator = RewTerm(" in cfg_text
    assert "func=instinct_mdp.foothold_clearance_safe_indicator" in cfg_text
    assert "foothold_clearance_penetration_l1 = RewTerm(" in cfg_text
    assert "func=instinct_mdp.foothold_clearance_penetration_l1" in cfg_text
    assert "foothold_clearance_safe_indicator = RewTerm(\n        func=instinct_mdp.foothold_clearance_safe_indicator,\n        weight=0.0" in cfg_text
    assert "foothold_clearance_penetration_l1 = RewTerm(\n        func=instinct_mdp.foothold_clearance_penetration_l1,\n        weight=-4.0" in cfg_text
    assert '"max_penetration_m": 0.15' in cfg_text


def test_parkour_reward_cfg_uses_unified_anomaly_penalty_and_keeps_diagnostics():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_gait_anomaly_indicator = RewTerm(" in cfg_text
    assert "func=mdp.foothold_gait_anomaly_indicator" in cfg_text
    assert "foothold_gait_anomaly_indicator = RewTerm(\n        func=mdp.foothold_gait_anomaly_indicator,\n        weight=-1.0" in cfg_text
    assert "foothold_early_contact_indicator = RewTerm(" in cfg_text
    assert "foothold_overdue_indicator = RewTerm(" in cfg_text
    assert "foothold_stance_lost_indicator = RewTerm(" in cfg_text
    assert "foothold_recovery_indicator = RewTerm(" in cfg_text
    assert "foothold_plan_invalid_indicator = RewTerm(" in cfg_text


def test_foothold_diagnostic_indicators_expose_gait_state_events():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        touchdown_accepted=torch.tensor(
            [False, False, False, True, False, False, False, False, False]
        ),
        planner_valid=torch.tensor(
            [True, True, True, True, True, True, True, False, True]
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
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_reset_mode_indicator(env),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_left_swing_mode_indicator(env),
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_right_swing_mode_indicator(env),
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_touchdown_confirm_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_early_contact_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_overdue_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_stance_lost_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_recovery_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_touchdown_accepted_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_plan_invalid_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )


def test_foothold_gait_anomaly_indicator_max_pools_failure_modes():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([0, 1, 4, 5, 6, 7, 8, 3]),
        planner_valid=torch.tensor([True, False, True, True, True, True, True, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    torch.testing.assert_close(
        foothold.foothold_gait_anomaly_indicator(env),
        torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
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

    reward = foothold.foothold_clearance_penetration_l1(
        env,
        max_penetration_m=0.10,
    )

    expected = torch.tensor([0.0, 0.02, 0.10])
    torch.testing.assert_close(reward, expected)
