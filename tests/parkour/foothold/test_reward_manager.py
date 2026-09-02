import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


def _load_manager_class():
    manager_module = ModuleType("isaaclab.managers")
    manager_module.ManagerTermBase = object
    manager_module.RewardManager = object
    manager_module.RewardTermCfg = object
    sys.modules.setdefault("isaaclab.managers", manager_module)
    path = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/managers/reward_manager.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_reward_manager_impl",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiRewardManager


def _fake_manager():
    manager_class = _load_manager_class()
    manager = object.__new__(manager_class)
    manager._MultiRewardManager__group_term_names = {
        "foothold_planning": ["learned_foothold_planning"],
    }
    manager._termwise_reward_buf = {
        "foothold_planning": {
            "learned_foothold_planning": torch.tensor(
                [0.75, -0.25],
                requires_grad=True,
            ),
        },
    }
    return manager


def test_get_termwise_reward_returns_unscaled_detached_snapshot():
    manager = _fake_manager()

    result = manager.get_termwise_reward(
        "learned_foothold_planning",
        group_name="foothold_planning",
    )

    torch.testing.assert_close(result, torch.tensor([0.75, -0.25]))
    assert result.requires_grad is False
    assert result.data_ptr() != manager._termwise_reward_buf[
        "foothold_planning"
    ]["learned_foothold_planning"].data_ptr()


def test_get_termwise_reward_rejects_unknown_group_or_term():
    manager = _fake_manager()

    with pytest.raises(ValueError, match="Term 'missing' not found"):
        manager.get_termwise_reward("missing", group_name="foothold_planning")

    with pytest.raises(ValueError, match="Group 'missing' not found"):
        manager.get_termwise_reward(
            "learned_foothold_planning",
            group_name="missing",
        )
