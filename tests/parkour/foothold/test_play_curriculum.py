from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "instinct_rl"
        / "play_curriculum.py"
    )
    spec = importlib.util.spec_from_file_location(
        "play_curriculum_under_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_foothold_curriculum_scale_from_matching_report(tmp_path):
    module = _load_module()
    report_dir = tmp_path / "logs" / "foothold_reports"
    report_dir.mkdir(parents=True)
    report = {
        "run": "logs/instinct_rl/g1_parkour/20260721_150741_example",
        "rows": [
            {
                "metric": "foothold_planner_reward_curriculum_scale",
                "summary": {"last": 0.73},
            }
        ],
    }
    (report_dir / "20260721_150741_example.json").write_text(json.dumps(report))

    scale = module.load_recorded_foothold_curriculum_scale(
        "logs/instinct_rl/g1_parkour/20260721_150741_example",
        report_dir=report_dir,
    )

    assert scale == 0.73


def test_load_foothold_curriculum_scale_returns_none_when_report_is_missing(tmp_path):
    module = _load_module()

    scale = module.load_recorded_foothold_curriculum_scale(
        "logs/instinct_rl/g1_parkour/missing_run",
        report_dir=tmp_path,
    )

    assert scale is None
