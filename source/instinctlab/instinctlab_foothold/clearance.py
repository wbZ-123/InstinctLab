from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .trajectory import quintic_swing_reference


class PenetrationObstacle(Protocol):
    """Obstacle interface compatible with terrain virtual obstacles."""

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        """Return penetration offset for each query point."""
        ...

@dataclass(frozen=True)
class SwingCenterlinePenetration:
    """Penetration summary for a sampled swing-foot centerline."""

    path_points_w: torch.Tensor
    phases: torch.Tensor
    penetration_offset: torch.Tensor
    penetration_depth: torch.Tensor
    collides: torch.Tensor
    max_penetration_depth: torch.Tensor
    deepest_phase: torch.Tensor

@dataclass(frozen=True)
class ApexAdjustmentResult:
    apex_height: torch.Tensor
    is_safe: torch.Tensor
    num_iterations: torch.Tensor
    penetration: SwingCenterlinePenetration


def _num_centerline_samples(
    start: torch.Tensor,
    goal: torch.Tensor,
    sample_spacing: float,
    min_samples: int,
    max_samples: int,
) -> int:
    if sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive.")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")
    if max_samples < min_samples:
        raise ValueError("max_samples must be greater than or equal to min_samples.")

    path_length = torch.linalg.norm(goal - start, dim=-1).max()
    # Subtract a tiny epsilon before ceil to avoid over-counting exact multiples
    # such as 0.30 / 0.03 due to floating-point roundoff.
    sample_count = int(torch.ceil(path_length / sample_spacing - 1.0e-6).item()) + 1
    return min(max(sample_count, min_samples), max_samples)


def sample_swing_centerline(
    start: torch.Tensor,
    goal: torch.Tensor,
    apex_height: torch.Tensor,
    swing_duration_s: torch.Tensor | float,
    sample_spacing: float = 0.03,
    min_samples: int = 9,
    max_samples: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the swing-foot center trajectory at spacing-bounded phases.

    The number of samples is selected from the largest batch path length:

        ceil(max_path_length / sample_spacing) + 1

    and then clamped into ``[min_samples, max_samples]``.
    """
    num_envs = start.shape[0]
    num_samples = _num_centerline_samples(
        start=start,
        goal=goal,
        sample_spacing=sample_spacing,
        min_samples=min_samples,
        max_samples=max_samples,
    )
    phases = torch.linspace(
        0.0,
        1.0,
        num_samples,
        device=start.device,
        dtype=start.dtype,
    )

    start_batch = start.unsqueeze(1).expand(num_envs, num_samples, 3).reshape(-1, 3)
    goal_batch = goal.unsqueeze(1).expand(num_envs, num_samples, 3).reshape(-1, 3)
    phase_batch = phases.unsqueeze(0).expand(num_envs, num_samples).reshape(-1)
    apex_batch = apex_height.unsqueeze(1).expand(num_envs, num_samples).reshape(-1)

    reference = quintic_swing_reference(
        start=start_batch,
        goal=goal_batch,
        phase=phase_batch,
        apex_height=apex_batch,
        swing_duration_s=swing_duration_s,
    )
    return reference.position.reshape(num_envs, num_samples, 3), phases


def check_swing_centerline_penetration(
    obstacle: PenetrationObstacle,
    start: torch.Tensor,
    goal: torch.Tensor,
    apex_height: torch.Tensor,
    swing_duration_s: torch.Tensor | float,
    sample_spacing: float = 0.03,
    min_samples: int = 9,
    max_samples: int = 25,
    penetration_tolerance: float = 0.0,
) -> SwingCenterlinePenetration:
    """Check whether a swing-foot centerline penetrates a virtual obstacle."""
    path_points, phases = sample_swing_centerline(
        start=start,
        goal=goal,
        apex_height=apex_height,
        swing_duration_s=swing_duration_s,
        sample_spacing=sample_spacing,
        min_samples=min_samples,
        max_samples=max_samples,
    )
    num_envs, num_samples, _ = path_points.shape

    penetration_offset = obstacle.get_points_penetration_offset(
        path_points.reshape(-1, 3)
    ).reshape(num_envs, num_samples, 3)

    penetration_offset = torch.nan_to_num(
        penetration_offset,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    penetration_depth = torch.linalg.norm(penetration_offset, dim=-1)
    max_penetration_depth, deepest_indices = torch.max(penetration_depth, dim=-1)
    collides = max_penetration_depth > penetration_tolerance
    deepest_phase = phases[deepest_indices]

    return SwingCenterlinePenetration(
        path_points_w=path_points,
        phases=phases,
        penetration_offset=penetration_offset,
        penetration_depth=penetration_depth,
        collides=collides,
        max_penetration_depth=max_penetration_depth,
        deepest_phase=deepest_phase,
    )

def adjust_apex_for_edge_clearance(
    obstacle: PenetrationObstacle,
    start: torch.Tensor,
    goal: torch.Tensor,
    default_apex_height: torch.Tensor,
    swing_duration_s: torch.Tensor | float,
    apex_step: float = 0.03,
    max_apex_height: torch.Tensor | float = 0.30,
    sample_spacing: float = 0.03,
    min_samples: int = 9,
    max_samples: int = 25,
    penetration_tolerance: float = 0.0,
) -> ApexAdjustmentResult:
    """Increase swing apex until the centerline clears edge obstacles."""
    if apex_step <= 0.0:
        raise ValueError("apex_step must be positive.")

    apex_height = torch.as_tensor(
        default_apex_height,
        device=start.device,
        dtype=start.dtype,
    ).clone()
    max_apex_height_tensor: torch.Tensor = torch.as_tensor(
        max_apex_height,
        device=start.device,
        dtype=start.dtype,
    )
    if max_apex_height_tensor.ndim == 0:
        max_apex_height_tensor = max_apex_height_tensor.expand_as(apex_height)

    num_iterations = torch.zeros(
        start.shape[0],
        device=start.device,
        dtype=torch.long,
    )

    max_iterations = int(torch.ceil(torch.max((max_apex_height_tensor - apex_height).clamp_min(0.0)) / apex_step).item()) + 1

    penetration = check_swing_centerline_penetration(
        obstacle=obstacle,
        start=start,
        goal=goal,
        apex_height=apex_height,
        swing_duration_s=swing_duration_s,
        sample_spacing=sample_spacing,
        min_samples=min_samples,
        max_samples=max_samples,
        penetration_tolerance=penetration_tolerance,
    )

    for iteration in range(max_iterations):
        penetration = check_swing_centerline_penetration(
            obstacle=obstacle,
            start=start,
            goal=goal,
            apex_height=apex_height,
            swing_duration_s=swing_duration_s,
            sample_spacing=sample_spacing,
            min_samples=min_samples,
            max_samples=max_samples,
            penetration_tolerance=penetration_tolerance,
        )

        unsafe = penetration.collides
        if not torch.any(unsafe).item():
            break

        can_increase = apex_height < max_apex_height_tensor
        update = unsafe & can_increase
        if not torch.any(update).item():
            break

        apex_height = torch.where(
            update,
            torch.minimum(apex_height + apex_step, max_apex_height_tensor),
            apex_height,
        )
        num_iterations = torch.where(
            update,
            num_iterations + 1,
            num_iterations,
        )

    is_safe = ~penetration.collides

    return ApexAdjustmentResult(
        apex_height=apex_height,
        is_safe=is_safe,
        num_iterations=num_iterations,
        penetration=penetration,
    )
