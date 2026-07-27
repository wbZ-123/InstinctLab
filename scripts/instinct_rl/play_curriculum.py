from __future__ import annotations

import json
from pathlib import Path


_CURRICULUM_METRIC = "foothold_planner_reward_curriculum_scale"


def load_recorded_foothold_curriculum_scale(
    run_path: str | Path,
    *,
    report_dir: str | Path = "logs/foothold_reports",
) -> float | None:
    """Load the last inspected foothold curriculum scale for a training run.

    The play environment starts with a fresh step counter, so it cannot infer
    the curriculum state that was used near the end of training.  The lightweight
    TensorBoard inspector writes machine-readable reports under
    ``logs/foothold_reports``; this helper reads the matching report when it is
    available and returns the last logged foothold curriculum scale.
    """

    run_name = Path(run_path).name
    report_path = Path(report_dir) / f"{run_name}.json"
    if not report_path.exists():
        return None

    try:
        payload = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    for row in payload.get("rows", []):
        if row.get("metric") != _CURRICULUM_METRIC:
            continue
        try:
            scale = float(row["summary"]["last"])
        except (KeyError, TypeError, ValueError):
            return None
        return max(0.0, min(scale, 1.0))

    return None
