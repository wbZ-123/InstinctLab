import importlib.util
import ast
import sys
import types
from pathlib import Path

import torch


def _load_foothold_actions_module(monkeypatch):
    class FakeActionTerm:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

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


def test_environment_action_order_keeps_motors_before_foothold():
    source_path = (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "tasks"
        / "parkour"
        / "config"
        / "parkour_env_cfg.py"
    )
    tree = ast.parse(source_path.read_text())
    actions_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActionsCfg"
    )
    annotated_fields = [
        node.target.id
        for node in actions_cfg.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    ]

    assert annotated_fields.index("joint_pos") < annotated_fields.index(
        "learned_foothold"
    )


def test_normalize_foothold_action_preserves_values_inside_unit_range(monkeypatch):
    module = _load_foothold_actions_module(monkeypatch)
    raw = torch.tensor([[-1.0, 0.25], [0.5, 1.0]])

    normalized = module.normalize_foothold_action(raw)

    torch.testing.assert_close(normalized, raw)


def test_normalize_foothold_action_clamps_without_meter_scaling(monkeypatch):
    module = _load_foothold_actions_module(monkeypatch)
    raw = torch.tensor([[-2.0, 0.25], [0.5, 2.0]])

    normalized = module.normalize_foothold_action(raw)

    torch.testing.assert_close(
        normalized,
        torch.tensor([[-1.0, 0.25], [0.5, 1.0]]),
    )


def test_action_term_shares_processed_action_with_planner_buffer(monkeypatch):
    module = _load_foothold_actions_module(monkeypatch)
    planner_data = types.SimpleNamespace(
        learned_foothold_action_normalized=torch.zeros(2, 2),
    )
    env = types.SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={
            "foothold_planner": types.SimpleNamespace(data=planner_data),
        },
    )
    cfg = types.SimpleNamespace(sensor_name="foothold_planner")

    term = module.LearnedFootholdAction(cfg, env)
    term.process_actions(torch.tensor([[2.0, 0.25], [-0.5, -2.0]]))

    assert (
        planner_data.learned_foothold_action_normalized
        is term.processed_actions
    )
    torch.testing.assert_close(
        planner_data.learned_foothold_action_normalized,
        torch.tensor([[1.0, 0.25], [-0.5, -1.0]]),
    )
