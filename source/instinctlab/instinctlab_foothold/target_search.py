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
    nominal_inside_ellipse: torch.Tensor
    nominal_obstacle_safe: torch.Tensor
    nominal_valid: torch.Tensor
    candidate_count: torch.Tensor
    candidate_inside_ellipse_count: torch.Tensor
    candidate_obstacle_safe_count: torch.Tensor
    candidate_valid_count: torch.Tensor


@dataclass
class SafeFootholdCandidateDebug:
    candidates_f: torch.Tensor
    nominal_inside_ellipse: torch.Tensor
    nominal_obstacle_safe: torch.Tensor
    nominal_valid: torch.Tensor
    candidate_inside_ellipse: torch.Tensor
    candidate_obstacle_safe: torch.Tensor
    candidate_valid: torch.Tensor


def _expand_foot_points(
    center: torch.Tensor,
    foot_points_xy: torch.Tensor,
) -> torch.Tensor:
    num_envs = center.shape[0]
    num_points = foot_points_xy.shape[0]

    points = center[:, None, :].expand(num_envs, num_points, 3).clone()
    points[:, :, 0:2] += foot_points_xy[None, :, :]
    return points


def _rotate_points_yaw(
    points: torch.Tensor,
    yaw_w: torch.Tensor,
) -> torch.Tensor:
    cos_yaw = torch.cos(yaw_w)
    sin_yaw = torch.sin(yaw_w)

    rotated = points.clone()
    rotated[..., 0] = cos_yaw * points[..., 0] - sin_yaw * points[..., 1]
    rotated[..., 1] = sin_yaw * points[..., 0] + cos_yaw * points[..., 1]
    return rotated


def _is_target_safe(
    target_f: torch.Tensor,
    target_origin_w: torch.Tensor,
    target_yaw_w: torch.Tensor,
    foot_points_xy: torch.Tensor,
    obstacle: PenetrationObstacle,
    safety_margin: float,
) -> torch.Tensor:
    foot_points_f = _expand_foot_points(target_f, foot_points_xy)
    foot_points_w = (
        target_origin_w[:, None, :]
        + _rotate_points_yaw(foot_points_f, target_yaw_w[:, None])
    )
    num_targets = foot_points_w.shape[0]
    num_foot_points = foot_points_w.shape[1]

    flat_foot_points_w = foot_points_w.reshape(num_targets * num_foot_points, 3)
    penetration_offset = obstacle.get_points_penetration_offset(flat_foot_points_w)
    if penetration_offset.ndim == 2:
        penetration = torch.linalg.norm(penetration_offset, dim=-1)
    else:
        penetration = penetration_offset

    penetration = penetration.reshape(
        num_targets,
        num_foot_points,
    )
    return torch.all(penetration <= safety_margin, dim=-1)


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


def _score_candidate_targets(
    *,
    candidates: torch.Tensor,
    nominal_target_f: torch.Tensor,
    desired_velocity_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score candidates by distance first, then velocity-direction alignment."""
    candidate_offsets_xy = candidates[:, :, :2] - nominal_target_f[:, None, :2]
    candidate_distance = torch.linalg.norm(candidate_offsets_xy, dim=-1)
    distance_bucket_size = torch.tensor(
        1.0e-5,
        device=nominal_target_f.device,
        dtype=nominal_target_f.dtype,
    )
    candidate_distance_bucket = (
        torch.round(candidate_distance / distance_bucket_size)
        * distance_bucket_size
    )

    desired_velocity_xy = desired_velocity_f[:, :2].to(
        device=nominal_target_f.device,
        dtype=nominal_target_f.dtype,
    )
    velocity_norm = torch.linalg.norm(desired_velocity_xy, dim=-1)
    candidate_direction = candidate_offsets_xy / candidate_distance.clamp_min(
        1.0e-6
    )[:, :, None]
    velocity_direction = desired_velocity_xy / velocity_norm.clamp_min(
        1.0e-6
    )[:, None]
    velocity_alignment = torch.sum(
        candidate_direction * velocity_direction[:, None, :],
        dim=-1,
    )
    velocity_alignment = torch.where(
        velocity_norm[:, None] > 1.0e-6,
        velocity_alignment,
        torch.zeros_like(velocity_alignment),
    )

    eps = torch.finfo(nominal_target_f.dtype).eps
    direction_tie_break = (1.0 - velocity_alignment) * eps
    order_tie_break = (
        torch.arange(
            candidates.shape[1],
            device=nominal_target_f.device,
            dtype=nominal_target_f.dtype,
        )
        * eps
        * 0.01
    )
    return (
        candidate_distance_bucket + direction_tie_break + order_tie_break,
        candidate_distance,
    )


def search_safe_foothold_target(
    *,
    nominal_target_f: torch.Tensor,
    raw_target_f: torch.Tensor,
    support_foot_f: torch.Tensor,
    target_origin_w: torch.Tensor | None = None,
    target_yaw_w: torch.Tensor | None = None,
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
    if target_origin_w is None:
        target_origin_w = torch.zeros_like(nominal_target_f)
    else:
        target_origin_w = target_origin_w.to(
            device=nominal_target_f.device,
            dtype=nominal_target_f.dtype,
        )
    if target_yaw_w is None:
        target_yaw_w = torch.zeros(
            nominal_target_f.shape[0],
            device=nominal_target_f.device,
            dtype=nominal_target_f.dtype,
        )
    else:
        target_yaw_w = target_yaw_w.to(
            device=nominal_target_f.device,
            dtype=nominal_target_f.dtype,
        )

    debug = debug_safe_foothold_candidates(
        nominal_target_f=nominal_target_f,
        support_foot_f=support_foot_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        obstacle=obstacle,
        ellipse_half_length=ellipse_half_length,
        ellipse_half_width=ellipse_half_width,
        foot_points_xy=foot_points_xy,
        candidate_radii=candidate_radii,
        candidate_directions=candidate_directions,
        safety_margin=safety_margin,
    )
    nominal_inside_ellipse = debug.nominal_inside_ellipse
    nominal_obstacle_safe = debug.nominal_obstacle_safe
    nominal_valid = debug.nominal_valid

    target_f = nominal_target_f.clone()
    valid = nominal_valid.clone()
    used_fallback = torch.zeros_like(valid, dtype=torch.bool)
    candidate_count = torch.zeros(
        nominal_target_f.shape[0],
        device=nominal_target_f.device,
        dtype=nominal_target_f.dtype,
    )
    candidate_inside_ellipse_count = torch.zeros_like(candidate_count)
    candidate_obstacle_safe_count = torch.zeros_like(candidate_count)
    candidate_valid_count = torch.zeros_like(candidate_count)
    selected_score = torch.zeros(
        nominal_target_f.shape[0],
        device=nominal_target_f.device,
        dtype=nominal_target_f.dtype,
    )

    if torch.any(~nominal_valid):
        candidates = debug.candidates_f
        candidate_inside = debug.candidate_inside_ellipse
        candidate_safe = debug.candidate_obstacle_safe
        num_envs = candidates.shape[0]
        num_candidates = candidates.shape[1]
        candidate_valid = candidate_inside & candidate_safe
        fallback_needed = ~nominal_valid
        candidate_count[fallback_needed] = float(num_candidates)
        candidate_inside_ellipse_count[fallback_needed] = (
            candidate_inside.sum(dim=1).to(dtype=nominal_target_f.dtype)[
                fallback_needed
            ]
        )
        candidate_obstacle_safe_count[fallback_needed] = (
            candidate_safe.sum(dim=1).to(dtype=nominal_target_f.dtype)[
                fallback_needed
            ]
        )
        candidate_valid_count[fallback_needed] = (
            candidate_valid.sum(dim=1).to(dtype=nominal_target_f.dtype)[
                fallback_needed
            ]
        )
        has_candidate = torch.any(candidate_valid, dim=1)

        candidate_score, candidate_distance = _score_candidate_targets(
            candidates=candidates,
            nominal_target_f=nominal_target_f,
            desired_velocity_f=desired_velocity_f,
        )
        candidate_score = candidate_score.masked_fill(~candidate_valid, float("inf"))
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
        nominal_inside_ellipse=nominal_inside_ellipse,
        nominal_obstacle_safe=nominal_obstacle_safe,
        nominal_valid=nominal_valid,
        candidate_count=candidate_count,
        candidate_inside_ellipse_count=candidate_inside_ellipse_count,
        candidate_obstacle_safe_count=candidate_obstacle_safe_count,
        candidate_valid_count=candidate_valid_count,
    )


def debug_safe_foothold_candidates(
    *,
    nominal_target_f: torch.Tensor,
    support_foot_f: torch.Tensor,
    target_origin_w: torch.Tensor | None = None,
    target_yaw_w: torch.Tensor | None = None,
    obstacle: PenetrationObstacle,
    ellipse_half_length: float,
    ellipse_half_width: float,
    foot_points_xy: torch.Tensor,
    candidate_radii: torch.Tensor,
    candidate_directions: torch.Tensor,
    safety_margin: float,
) -> SafeFootholdCandidateDebug:
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
    if target_origin_w is None:
        target_origin_w = torch.zeros_like(nominal_target_f)
    else:
        target_origin_w = target_origin_w.to(
            device=nominal_target_f.device,
            dtype=nominal_target_f.dtype,
        )
    if target_yaw_w is None:
        target_yaw_w = torch.zeros(
            nominal_target_f.shape[0],
            device=nominal_target_f.device,
            dtype=nominal_target_f.dtype,
        )
    else:
        target_yaw_w = target_yaw_w.to(
            device=nominal_target_f.device,
            dtype=nominal_target_f.dtype,
        )

    nominal_obstacle_safe = _is_target_safe(
        target_f=nominal_target_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        foot_points_xy=foot_points_xy,
        obstacle=obstacle,
        safety_margin=safety_margin,
    )
    nominal_inside_ellipse = _is_inside_reachable_ellipse(
        target_f=nominal_target_f,
        support_foot_f=support_foot_f,
        ellipse_half_length=ellipse_half_length,
        ellipse_half_width=ellipse_half_width,
    )
    nominal_valid = nominal_obstacle_safe & nominal_inside_ellipse

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
    flat_target_origin_w = target_origin_w[:, None, :].expand_as(candidates).reshape(
        num_envs * num_candidates, 3
    )
    flat_target_yaw_w = target_yaw_w[:, None].expand(
        num_envs,
        num_candidates,
    ).reshape(num_envs * num_candidates)

    candidate_inside = _is_inside_reachable_ellipse(
        target_f=flat_candidates,
        support_foot_f=flat_support,
        ellipse_half_length=ellipse_half_length,
        ellipse_half_width=ellipse_half_width,
    ).reshape(num_envs, num_candidates)

    candidate_safe = _is_target_safe(
        target_f=flat_candidates,
        target_origin_w=flat_target_origin_w,
        target_yaw_w=flat_target_yaw_w,
        foot_points_xy=foot_points_xy,
        obstacle=obstacle,
        safety_margin=safety_margin,
    ).reshape(num_envs, num_candidates)

    return SafeFootholdCandidateDebug(
        candidates_f=candidates,
        nominal_inside_ellipse=nominal_inside_ellipse,
        nominal_obstacle_safe=nominal_obstacle_safe,
        nominal_valid=nominal_valid,
        candidate_inside_ellipse=candidate_inside,
        candidate_obstacle_safe=candidate_safe,
        candidate_valid=candidate_inside & candidate_safe,
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
