import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_foothold_actions_module(monkeypatch):
    class FakeActionTerm:
        pass

    isaaclab_module = types.ModuleType("isaaclab")
    managers_module = types.ModuleType("isaaclab.managers")
    action_manager_module = types.ModuleType("isaaclab.managers.action_manager")
    managers_module.ActionTerm = FakeActionTerm
    action_manager_module.ActionTerm = FakeActionTerm
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers_module)
    monkeypatch.setitem(sys.modules, "isaaclab.managers.action_manager", action_manager_module)

    path = (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "envs"
        / "mdp"
        / "actions"
        / "foothold_actions.py"
    )
    spec = importlib.util.spec_from_file_location("foothold_actions_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scale_foothold_action_maps_unit_action_to_local_range(monkeypatch):
    module = _load_foothold_actions_module(monkeypatch)
    raw = torch.tensor([[-1.0, 0.0], [1.0, 1.0]])

    scaled = module.scale_foothold_action(
        raw,
        x_range=(-0.05, 0.35),
        y_range=(-0.22, 0.22),
    )

    expected = torch.tensor([[-0.05, 0.0], [0.35, 0.22]])
    torch.testing.assert_close(scaled, expected)


def test_scale_foothold_action_clamps_out_of_range_policy_output(monkeypatch):
    module = _load_foothold_actions_module(monkeypatch)
    raw = torch.tensor([[-2.0, 2.0]])

    scaled = module.scale_foothold_action(
        raw,
        x_range=(-0.05, 0.35),
        y_range=(-0.22, 0.22),
    )
    torch.testing.assert_close(scaled, torch.tensor([[-0.05, 0.22]]))
