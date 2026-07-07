from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


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
