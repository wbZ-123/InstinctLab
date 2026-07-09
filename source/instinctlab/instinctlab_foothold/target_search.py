from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class PenetrationObstacle(Protocol):
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class SafeFootholdSearchResult:
    target_f: torch.Tensor
    valid: torch.Tensor
    used_fallback: torch.Tensor
    selected_score: torch.Tensor


def _expand_foot_points(
    center_f: torch.Tensor,
    foot_points_xy: torch.Tensor,
) -> torch.Tensor:
    num_envs = center_f.shape[0]
    num_points = foot_points_xy.shape[0]

    points = center_f[:, None, :].expand(num_envs, num_points, 3).clone()
    points[:, :, 0:2] += foot_points_xy[None, :, :]
    return points


def _is_target_safe(
    target_f: torch.Tensor,
    foot_points_xy: torch.Tensor,
    obstacle: PenetrationObstacle,
    safety_margin: float,
) -> torch.Tensor:
    foot_points_f = _expand_foot_points(target_f, foot_points_xy)
    penetration = obstacle.get_points_penetration_offset(foot_points_f)
    return torch.all(penetration <= -safety_margin, dim=-1)


def _is_inside_reachable_ellipse(
    target_f: torch.Tensor,
    support_foot_f: torch.Tensor,
    ellipse_half_length: float,
    ellipse_half_width: float,
) -> torch.Tensor:
    delta_xy = target_f[:, :2] - support_foot_f[:, :2]
    normalized = torch.square(delta_xy[:, 0] / ellipse_half_length) + torch.square(
        delta_xy[:, 1] / ellipse_half_width
    )
    return normalized <= 1.0


def _build_candidate_targets(
    nominal_target_f: torch.Tensor,
    candidate_radii: torch.Tensor,
    candidate_directions: torch.Tensor,
) -> torch.Tensor:
    # candidate offsets: (num_candidates, 2)
    offsets_xy = (
        candidate_radii[:, None, None] * candidate_directions[None, :, :]
    ).reshape(-1, 2)

    num_envs = nominal_target_f.shape[0]
    num_candidates = offsets_xy.shape[0]

    candidates = nominal_target_f[:, None, :].expand(
        num_envs, num_candidates, 3
    ).clone()
    candidates[:, :, :2] += offsets_xy[None, :, :]
    return candidates


def search_safe_foothold_target(
    *,
    nominal_target_f: torch.Tensor,
    raw_target_f: torch.Tensor,
    support_foot_f: torch.Tensor,
    desired_velocity_f: torch.Tensor,
    obstacle: PenetrationObstacle,
    ellipse_half_length: float,
    ellipse_half_width: float,
    foot_points_xy: torch.Tensor,
    candidate_radii: torch.Tensor,
    candidate_directions: torch.Tensor,
    safety_margin: float,
) -> SafeFootholdSearchResult:
    del raw_target_f
    del desired_velocity_f

    foot_points_xy = foot_points_xy.to(
        device=nominal_target_f.device,
        dtype=nominal_target_f.dtype,
    )
    candidate_radii = candidate_radii.to(
        device=nominal_target_f.device,
        dtype=nominal_target_f.dtype,
    )
    candidate_directions = candidate_directions.to(
        device=nominal_target_f.device,
        dtype=nominal_target_f.dtype,
    )

    safe = _is_target_safe(
        target_f=nominal_target_f,
        foot_points_xy=foot_points_xy,
        obstacle=obstacle,
        safety_margin=safety_margin,
    )
    inside_ellipse = _is_inside_reachable_ellipse(
        target_f=nominal_target_f,
        support_foot_f=support_foot_f,
        ellipse_half_length=ellipse_half_length,
        ellipse_half_width=ellipse_half_width,
    )
    nominal_valid = safe & inside_ellipse

    target_f = nominal_target_f.clone()
    valid = nominal_valid.clone()
    used_fallback = torch.zeros_like(valid, dtype=torch.bool)
    selected_score = torch.zeros(
        nominal_target_f.shape[0],
        device=nominal_target_f.device,
        dtype=nominal_target_f.dtype,
    )

    if torch.any(~nominal_valid):
        candidates = _build_candidate_targets(
            nominal_target_f=nominal_target_f,
            candidate_radii=candidate_radii,
            candidate_directions=candidate_directions,
        )

        num_envs = candidates.shape[0]
        num_candidates = candidates.shape[1]

        flat_candidates = candidates.reshape(num_envs * num_candidates, 3)
        flat_support = support_foot_f[:, None, :].expand_as(candidates).reshape(
            num_envs * num_candidates, 3
        )

        candidate_inside = _is_inside_reachable_ellipse(
            target_f=flat_candidates,
            support_foot_f=flat_support,
            ellipse_half_length=ellipse_half_length,
            ellipse_half_width=ellipse_half_width,
        ).reshape(num_envs, num_candidates)

        candidate_safe = _is_target_safe(
            target_f=flat_candidates,
            foot_points_xy=foot_points_xy,
            obstacle=obstacle,
            safety_margin=safety_margin,
        ).reshape(num_envs, num_candidates)

        candidate_valid = candidate_inside & candidate_safe
        has_candidate = torch.any(candidate_valid, dim=1)

        candidate_distance = torch.linalg.norm(
            candidates[:, :, :2] - nominal_target_f[:, None, :2],
            dim=-1,
        )
        tie_break = (
            torch.arange(
                num_candidates,
                device=nominal_target_f.device,
                dtype=nominal_target_f.dtype,
            )
            * torch.finfo(nominal_target_f.dtype).eps
        )
        candidate_score = (candidate_distance + tie_break).masked_fill(
            ~candidate_valid, float("inf")
        )
        best_idx = torch.argmin(candidate_score, dim=1)

        env_ids = torch.arange(
            num_envs,
            device=nominal_target_f.device,
        )
        fallback_target = candidates[env_ids, best_idx]
        fallback_score = candidate_distance[env_ids, best_idx]

        replace = (~nominal_valid) & has_candidate
        target_f[replace] = fallback_target[replace]
        valid[replace] = True
        used_fallback[replace] = True
        selected_score[replace] = fallback_score[replace]

    return SafeFootholdSearchResult(
        target_f=target_f,
        valid=valid,
        used_fallback=used_fallback,
        selected_score=selected_score,
    )


def make_sole_perimeter_points_xy(
    *,
    foot_length: float,
    foot_width: float,
    num_x: int = 10,
    num_y: int = 5,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_x < 2:
        raise ValueError("num_x must be at least 2.")
    if num_y < 2:
        raise ValueError("num_y must be at least 2.")

    xs = torch.linspace(
        -0.5 * foot_length,
        0.5 * foot_length,
        num_x,
        device=device,
        dtype=dtype,
    )
    ys = torch.linspace(
        -0.5 * foot_width,
        0.5 * foot_width,
        num_y,
        device=device,
        dtype=dtype,
    )

    bottom = torch.stack(
        [xs, torch.full_like(xs, ys[0])],
        dim=-1,
    )
    top = torch.stack(
        [xs, torch.full_like(xs, ys[-1])],
        dim=-1,
    )

    # 去掉左右边的角点，避免和 top/bottom 重复
    side_ys = ys[1:-1]
    left = torch.stack(
        [torch.full_like(side_ys, xs[0]), side_ys],
        dim=-1,
    )
    right = torch.stack(
        [torch.full_like(side_ys, xs[-1]), side_ys],
        dim=-1,
    )

    return torch.cat([bottom, top, left, right], dim=0)
