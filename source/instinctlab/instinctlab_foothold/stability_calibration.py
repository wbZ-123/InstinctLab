from __future__ import annotations

import json
from pathlib import Path

import torch

from .contact_adaptation import StabilityBounds


_BOUND_NAMES = (
    "max_tilt_rad",
    "max_angular_speed_rad_s",
    "max_horizontal_speed_m_s",
    "max_support_slip_m_s",
)


def calibrate_stability_bounds(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    quantile: float = 0.99,
    dwell_s: float = 0.10,
) -> dict[str, float]:
    """Calibrate recovery-exit bounds from successful normal-HOLD samples."""
    if values.ndim != 2 or values.shape[-1] != len(_BOUND_NAMES):
        raise ValueError("values must have shape (num_samples, 4)")
    if valid.shape != (values.shape[0],):
        raise ValueError("valid must have shape (num_samples,)")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    if dwell_s <= 0.0:
        raise ValueError("dwell_s must be positive")

    selected = values[valid.bool()]
    if selected.shape[0] == 0:
        raise ValueError("at least one valid calibration sample is required")
    if not torch.isfinite(selected).all().item():
        raise ValueError("calibration samples must be finite")
    if (selected < 0.0).any().item():
        raise ValueError("calibration samples must be non-negative")

    quantiles = torch.quantile(selected.to(dtype=torch.float32), quantile, dim=0)
    return {
        name: float(value.item())
        for name, value in zip(_BOUND_NAMES, quantiles, strict=True)
    } | {"dwell_s": float(dwell_s)}


def load_stability_bounds(path: str | Path) -> StabilityBounds:
    """Load calibrated bounds and reject missing, unknown, or unsafe values."""
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise ValueError(f"stability calibration file does not exist: {path}")
    try:
        payload = json.loads(calibration_path.read_text())
        bounds = StabilityBounds(
            max_tilt_rad=float(payload["max_tilt_rad"]),
            max_angular_speed_rad_s=float(
                payload["max_angular_speed_rad_s"]
            ),
            max_horizontal_speed_m_s=float(
                payload["max_horizontal_speed_m_s"]
            ),
            max_support_slip_m_s=float(payload["max_support_slip_m_s"]),
            dwell_s=float(payload["dwell_s"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid stability calibration file: {path}"
        ) from exc
    return bounds
