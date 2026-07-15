from __future__ import annotations

import torch


def make_recovery_foothold_target(
    *,
    swing_side: torch.Tensor,
    desired_velocity_f: torch.Tensor,
    step_length_m: float,
    velocity_lookahead_s: float,
    max_step_length_m: float,
    step_width_m: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a conservative foothold target for a recovery step.

    The target is expressed in the planner/support-foot frame. Side index
    follows the rest of the foothold stack: ``0`` is left swing and ``1`` is
    right swing.
    """
    side_sign = torch.where(
        swing_side == 0,
        torch.ones_like(swing_side),
        -torch.ones_like(swing_side),
    ).to(device=device, dtype=dtype)
    desired_velocity_f = desired_velocity_f.to(device=device, dtype=dtype)
    base_step_x = torch.full(
        swing_side.shape,
        step_length_m,
        device=device,
        dtype=dtype,
    )
    velocity_extension_x = (
        desired_velocity_f[:, 0] * velocity_lookahead_s
    ).clamp_min(0.0)
    step_x = (base_step_x + velocity_extension_x).clamp_max(
        max_step_length_m
    )
    step_y = side_sign * step_width_m
    step_z = torch.zeros_like(step_x)
    return torch.stack((step_x, step_y, step_z), dim=-1)
