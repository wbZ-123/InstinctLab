import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_wrapper_module():
    previous = {
        name: sys.modules.get(name)
        for name in (
            "isaaclab",
            "isaaclab.envs",
            "instinct_rl",
            "instinct_rl.env",
        )
    }
    isaaclab = ModuleType("isaaclab")
    isaaclab_envs = ModuleType("isaaclab.envs")
    isaaclab_envs.DirectRLEnv = type("DirectRLEnv", (), {})
    isaaclab_envs.ManagerBasedRLEnv = type("ManagerBasedRLEnv", (), {})
    isaaclab.envs = isaaclab_envs
    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.envs"] = isaaclab_envs

    instinct_rl = ModuleType("instinct_rl")
    instinct_rl_env = ModuleType("instinct_rl.env")
    instinct_rl_env.VecEnv = type("VecEnv", (), {})
    instinct_rl.env = instinct_rl_env
    sys.modules["instinct_rl"] = instinct_rl
    sys.modules["instinct_rl.env"] = instinct_rl_env

    path = (
        Path(__file__).resolve().parents[3]
        / "source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_vecenv_wrapper_impl",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


def test_read_foothold_event_reward_uses_raw_term_snapshot():
    module = _load_wrapper_module()

    class Manager:
        def get_termwise_reward(self, term_name, group_name=None):
            assert term_name == "learned_foothold_planning"
            assert group_name == "foothold_planning"
            return torch.tensor([1.0, -0.5], requires_grad=True)

    result = module.read_foothold_event_reward(Manager())

    torch.testing.assert_close(result, torch.tensor([1.0, -0.5]))
    assert result.requires_grad is False


def test_read_foothold_event_reward_fails_without_raw_term_accessor():
    module = _load_wrapper_module()

    with pytest.raises(RuntimeError, match="get_termwise_reward"):
        module.read_foothold_event_reward(SimpleNamespace())
