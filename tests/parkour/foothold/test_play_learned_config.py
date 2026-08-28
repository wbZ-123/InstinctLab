from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "instinct_rl"
        / "play_learned_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "play_learned_config_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_event_gated_checkpoint_enables_matching_play_environment():
    module = _load_module()
    calls: list[str] = []
    env_cfg = SimpleNamespace(
        enable_learned_foothold_planner=lambda: calls.append("env")
    )
    agent_cfg = {
        "algorithm": {"class_name": "EventGatedWasabiPPO"},
    }

    enabled = module.configure_learned_foothold_play(
        env_cfg,
        agent_cfg,
        register_algorithm=lambda: calls.append("algorithm"),
    )

    assert enabled is True
    assert calls == ["env", "algorithm"]


def test_event_gated_sac_checkpoint_enables_matching_play_environment():
    module = _load_module()
    calls: list[str] = []
    env_cfg = SimpleNamespace(
        enable_learned_foothold_planner=lambda: calls.append("env")
    )

    enabled = module.configure_learned_foothold_play(
        env_cfg,
        {"algorithm": {"class_name": "EventGatedWasabiSAC"}},
        register_algorithm=lambda: calls.append("algorithm"),
    )

    assert enabled is True
    assert calls == ["env", "algorithm"]


def test_legacy_checkpoint_preserves_legacy_play_environment():
    module = _load_module()
    calls: list[str] = []
    env_cfg = SimpleNamespace(
        enable_learned_foothold_planner=lambda: calls.append("env")
    )
    agent_cfg = {"algorithm": {"class_name": "WasabiPPO"}}

    enabled = module.configure_learned_foothold_play(
        env_cfg,
        agent_cfg,
        register_algorithm=lambda: calls.append("algorithm"),
    )

    assert enabled is False
    assert calls == []


def test_event_gated_checkpoint_rejects_incompatible_task():
    module = _load_module()

    with pytest.raises(RuntimeError, match="does not support"):
        module.configure_learned_foothold_play(
            SimpleNamespace(),
            {"algorithm": {"class_name": "EventGatedWasabiPPO"}},
            register_algorithm=lambda: None,
        )


def test_play_script_auto_loads_saved_event_gated_config_before_gym_make():
    play_text = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "instinct_rl"
        / "play.py"
    ).read_text()

    configure_index = play_text.index("configure_learned_foothold_play(")
    gym_make_index = play_text.index("env = gym.make(")

    assert configure_index < gym_make_index
    assert "agent_cfg_dict = saved_agent_cfg" in play_text


def test_learned_training_selects_independent_policy_and_sourced_bounds():
    config_text = (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "tasks"
        / "parkour"
        / "config"
        / "g1"
        / "agents"
        / "instinct_rl_amp_cfg.py"
    ).read_text()

    assert "IndependentFootholdEncoderMoEActorCritic" in config_text
    assert "self.policy.motor_action_dim = 29" in config_text
    assert "self.policy.foothold_depth_output_size = 64" in config_text
    assert "self.policy.foothold_depth_hidden_channels = 8" in config_text
    assert "self.algorithm.foothold_min_std_m = (0.02, 0.02)" in config_text
    assert "self.algorithm.foothold_max_std_m = (0.05, 0.05)" in config_text
    assert "self.algorithm.foothold_learning_rate = 1.0e-5" in config_text
    assert (
        "self.algorithm.foothold_entropy_coef = self.algorithm.entropy_coef"
        in config_text
    )
