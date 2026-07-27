from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace


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
