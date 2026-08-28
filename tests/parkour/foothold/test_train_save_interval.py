from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_cli_args_module():
    path = REPO_ROOT / "scripts" / "instinct_rl" / "cli_args.py"
    spec = importlib.util.spec_from_file_location("instinct_cli_args_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_instinct_rl_cli_overrides_save_interval():
    cli_args = _load_cli_args_module()
    parser = argparse.ArgumentParser()
    cli_args.add_instinct_rl_args(parser)
    args = parser.parse_args(["--save_interval", "2000"])
    agent_cfg = SimpleNamespace(
        seed=None,
        resume=False,
        load_run="",
        load_checkpoint=None,
        run_name="default",
        save_interval=5000,
    )

    cli_args.update_instinct_rl_cfg(agent_cfg, args)

    assert agent_cfg.save_interval == 2000


def test_foothold_train_script_passes_save_interval_from_environment():
    script = (REPO_ROOT / "scripts" / "foothold_train.sh").read_text()

    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"' in script
    assert '--save_interval "${SAVE_INTERVAL}"' in script
    assert 'echo "[foothold_train] save_interval: ${SAVE_INTERVAL}"' in script


@pytest.mark.parametrize(
    "script_name",
    ["foothold_train.sh", "foothold_play_step.sh"],
)
def test_foothold_wrappers_default_to_vendored_submodules(script_name):
    script = (REPO_ROOT / "scripts" / script_name).read_text()

    assert '${REPO_ROOT}/third_party/IsaacLab' in script
    assert '${REPO_ROOT}/third_party/instinct_rl' in script
    assert 'ISAACLAB_ROOT="${ISAACLAB_ROOT:-' in script
    assert 'INSTINCT_RL_ROOT="${INSTINCT_RL_ROOT:-' in script
    assert '${INSTINCT_RL_ROOT}:${PYTHONPATH:-}' in script


def test_training_has_explicit_learned_foothold_planner_opt_in():
    train_script = (
        REPO_ROOT / "scripts" / "instinct_rl" / "train.py"
    ).read_text()
    wrapper = (REPO_ROOT / "scripts" / "foothold_train.sh").read_text()

    assert '"--enable_learned_foothold_planner"' in train_script
    assert "env_cfg.enable_learned_foothold_planner()" in train_script
    assert 'ENABLE_LEARNED_FOOTHOLD_PLANNER="${' in wrapper
    assert "--enable_learned_foothold_planner" in wrapper


def test_learned_planner_opt_in_selects_algorithm_in_causal_order():
    train_script = (
        REPO_ROOT / "scripts" / "instinct_rl" / "train.py"
    ).read_text()

    env_enable = train_script.index(
        "env_cfg.enable_learned_foothold_planner()"
    )
    algorithm_enable = train_script.index(
        "agent_cfg.enable_event_gated_foothold_ppo("
    )
    registration = train_script.index(
        "register_event_gated_foothold_algorithm()"
    )

    assert env_enable < algorithm_enable < registration
    assert "if env.num_actions != 31:" in train_script
    assert "if env.num_rewards != 2:" in train_script


def test_foothold_train_reports_selected_algorithm():
    wrapper = (REPO_ROOT / "scripts" / "foothold_train.sh").read_text()

    assert 'LEARNED_FOOTHOLD_ALGORITHM="EventGatedWasabiPPO"' in wrapper
    assert (
        'echo "[foothold_train] learned_foothold_algorithm: '
        '${LEARNED_FOOTHOLD_ALGORITHM}"'
        in wrapper
    )


def test_resume_syncs_adaptive_learning_rate_from_loaded_optimizer():
    cli_args = _load_cli_args_module()
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            learning_rate=1.0e-3,
            optimizer=SimpleNamespace(
                param_groups=[{"lr": 7.59375e-5}]
            ),
        )
    )

    loaded_lr = cli_args.sync_runner_learning_rate_after_resume(runner)

    assert loaded_lr == 7.59375e-5
    assert runner.alg.learning_rate == 7.59375e-5


def test_resume_and_legacy_initialization_are_mutually_exclusive():
    cli_args = _load_cli_args_module()
    parser = argparse.ArgumentParser()
    cli_args.add_instinct_rl_args(parser)
    args = parser.parse_args(
        [
            "--resume",
            "--initialize_learned_foothold_from",
            "/tmp/model_30000.pt",
        ]
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        cli_args.validate_checkpoint_modes(args)


def test_legacy_initialization_requires_absolute_checkpoint_path():
    cli_args = _load_cli_args_module()
    parser = argparse.ArgumentParser()
    cli_args.add_instinct_rl_args(parser)
    args = parser.parse_args(
        [
            "--initialize_learned_foothold_from",
            "model_30000.pt",
        ]
    )

    with pytest.raises(ValueError, match="absolute"):
        cli_args.validate_checkpoint_modes(args)


def test_train_wires_audited_legacy_initialization():
    train_script = (
        REPO_ROOT / "scripts" / "instinct_rl" / "train.py"
    ).read_text()
    wrapper = (REPO_ROOT / "scripts" / "foothold_train.sh").read_text()

    assert "cli_args.validate_checkpoint_modes(args_cli)" in train_script
    assert "initialize_runner_from_legacy_checkpoint(" in train_script
    assert "learned_foothold_policy_input_expansion(" in train_script
    assert "appended_observation_dim=3" not in train_script
    assert "LEARNED_FOOTHOLD_BASE_CHECKPOINT" in wrapper
    assert "--initialize_learned_foothold_from" in wrapper
