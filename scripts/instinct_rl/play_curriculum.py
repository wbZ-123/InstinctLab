from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

import torch


_CURRICULUM_METRIC = "foothold_planner_reward_curriculum_scale"
FOOTHOLD_CURRICULUM_SCALE_KEY = "foothold_curriculum_scale"


def _runtime_foothold_curriculum_scale(env) -> float | None:
    """Return the finite mean curriculum scale currently used by the planner."""

    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    sensors = getattr(scene, "sensors", None)
    if sensors is None or "foothold_planner" not in sensors:
        return None

    planner = sensors["foothold_planner"]
    scale = getattr(planner, "flat_target_curriculum_scale", None)
    if scale is None:
        return None

    scale_tensor = torch.as_tensor(scale).detach().to(dtype=torch.float32)
    finite = torch.isfinite(scale_tensor)
    if scale_tensor.numel() == 0 or not torch.any(finite).item():
        return None
    return float(scale_tensor[finite].mean().clamp(0.0, 1.0).item())


def attach_foothold_curriculum_checkpoint_metadata(runner, env) -> None:
    """Attach the planner curriculum scale to every checkpoint saved by a runner."""

    original_save = runner.save

    @wraps(original_save)
    def save_with_foothold_curriculum(path, infos=None):
        checkpoint_infos = dict(infos or {})
        scale = _runtime_foothold_curriculum_scale(env)
        if scale is not None:
            checkpoint_infos[FOOTHOLD_CURRICULUM_SCALE_KEY] = scale
        return original_save(path, infos=checkpoint_infos or None)

    runner.save = save_with_foothold_curriculum


def load_checkpoint_foothold_curriculum_scale(
    checkpoint_path: str | Path,
) -> float | None:
    """Load the planner curriculum scale embedded in a training checkpoint."""

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        scale = float(checkpoint["infos"][FOOTHOLD_CURRICULUM_SCALE_KEY])
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if not torch.isfinite(torch.tensor(scale)).item():
        return None
    return max(0.0, min(scale, 1.0))


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
