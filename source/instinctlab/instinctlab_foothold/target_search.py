from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


class PenetrationObstacle(Protocol):
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        ...


TerrainHeightQuery = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


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
    final_max_penetration_depth: torch.Tensor


@dataclass
class SafeFootholdCandidateDebug:
    nominal_target_f: torch.Tensor
    candidates_f: torch.Tensor
    nominal_inside_ellipse: torch.Tensor
    nominal_obstacle_safe: torch.Tensor
    nominal_valid: torch.Tensor
    candidate_inside_ellipse: torch.Tensor
    candidate_obstacle_safe: torch.Tensor
    candidate_valid: torch.Tensor


@dataclass
class SafeFootholdTargetEvaluation:
    target_f: torch.Tensor
    height_valid: torch.Tensor
    inside_ellipse: torch.Tensor
    obstacle_safe: torch.Tensor
    valid: torch.Tensor
    max_penetration_depth: torch.Tensor
    mean_penetration_depth: torch.Tensor
    total_penetration_depth: torch.Tensor
    penetrating_point_count: torch.Tensor
    penetrating_point_ratio: torch.Tensor
    safety_score: torch.Tensor


@dataclass
class SolePerimeterPenetrationScore:
    score: torch.Tensor
    penetrating_point_count: torch.Tensor
    penetrating_point_ratio: torch.Tensor
    total_penetration_depth: torch.Tensor


def score_sole_perimeter_penetration(
    penetration_depths: torch.Tensor,
    *,
    full_penalty_depth_m: float = 0.02,
) -> SolePerimeterPenetrationScore:
    """Return a bounded score from sole intrusion count and total depth.

    A completely clear sole scores ``+1``.  Any positive penetration scores
    strictly below zero. Penetrating-point ratio and normalized total depth
    contribute equally to the bounded negative magnitude.
    """

    if penetration_depths.ndim < 2 or penetration_depths.shape[-1] < 1:
        raise ValueError(
            "penetration_depths must contain at least one sole point."
        )
    if full_penalty_depth_m <= 0.0:
        raise ValueError("full_penalty_depth_m must be positive.")

    # The cylinder penetration direction is undefined exactly on a cylinder
    # centerline.  The Warp kernel can therefore return NaN even though the
    # point is maximally unsafe.  Preserve the conservative safety meaning:
    # non-finite positive/undefined penetration receives the existing full
    # penalty depth, while negative infinity means no positive penetration.
    finite_depth = torch.nan_to_num(
        penetration_depths,
        nan=full_penalty_depth_m,
        posinf=full_penalty_depth_m,
        neginf=0.0,
    )
    positive_depth = torch.clamp(finite_depth, min=0.0)
    penetrating = positive_depth > 0.0
    point_count = penetrating.sum(dim=-1).to(
        dtype=penetration_depths.dtype
    )
    point_ratio = penetrating.to(
        dtype=penetration_depths.dtype
    ).mean(dim=-1)
    total_depth = positive_depth.sum(dim=-1)
    full_total_depth = (
        penetration_depths.shape[-1] * full_penalty_depth_m
    )
    normalized_total_depth = torch.clamp(
        total_depth / full_total_depth,
        min=0.0,
        max=1.0,
    )
    unsafe_penalty = torch.clamp(
        point_ratio + normalized_total_depth,
        min=0.0,
        max=1.0,
    )
    score = torch.where(
        penetrating.any(dim=-1),
        -unsafe_penalty,
        torch.ones_like(unsafe_penalty),
    )
    return SolePerimeterPenetrationScore(
        score=score,
        penetrating_point_count=point_count,
        penetrating_point_ratio=point_ratio,
        total_penetration_depth=total_depth,
    )


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


def _compose_world_from_frame(
    origin_w: torch.Tensor,
    vector_f: torch.Tensor,
    yaw_w: torch.Tensor,
) -> torch.Tensor:
    return origin_w + _rotate_points_yaw(vector_f, yaw_w)


def _lift_targets_to_terrain_height(
    *,
    target_f: torch.Tensor,
    target_origin_w: torch.Tensor,
    target_yaw_w: torch.Tensor,
    terrain_height_query_w: TerrainHeightQuery | None,
    max_step_height_m: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift target centers onto terrain before footprint obstacle checks."""
    if terrain_height_query_w is None:
        return target_f, torch.ones(
            target_f.shape[0],
            device=target_f.device,
            dtype=torch.bool,
        )

    target_w = _compose_world_from_frame(
        target_origin_w,
        target_f,
        target_yaw_w,
    )
    terrain_height_w, terrain_valid = terrain_height_query_w(target_w[:, :2])
    terrain_height_w = terrain_height_w.to(
        device=target_f.device,
        dtype=target_f.dtype,
    )
    terrain_valid = terrain_valid.to(device=target_f.device, dtype=torch.bool)

    finite_height = torch.isfinite(terrain_height_w)
    height_valid = terrain_valid & finite_height
    if max_step_height_m is not None:
        height_delta = torch.abs(terrain_height_w - target_origin_w[:, 2])
        height_valid = height_valid & (height_delta <= max_step_height_m)

    lifted = target_f.clone()
    lifted[:, 2] = torch.where(
        finite_height,
        terrain_height_w - target_origin_w[:, 2],
        lifted[:, 2],
    )
    return lifted, height_valid


def _target_penetration_depths(
    target_f: torch.Tensor,
    target_origin_w: torch.Tensor,
    target_yaw_w: torch.Tensor,
    foot_points_xy: torch.Tensor,
    obstacle: PenetrationObstacle,
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
    return penetration


def _target_max_penetration_depth(
    target_f: torch.Tensor,
    target_origin_w: torch.Tensor,
    target_yaw_w: torch.Tensor,
    foot_points_xy: torch.Tensor,
    obstacle: PenetrationObstacle,
) -> torch.Tensor:
    return torch.max(
        _target_penetration_depths(
            target_f=target_f,
            target_origin_w=target_origin_w,
            target_yaw_w=target_yaw_w,
            foot_points_xy=foot_points_xy,
            obstacle=obstacle,
        ),
        dim=-1,
    ).values


def _is_target_safe(
    target_f: torch.Tensor,
    target_origin_w: torch.Tensor,
    target_yaw_w: torch.Tensor,
    foot_points_xy: torch.Tensor,
    obstacle: PenetrationObstacle,
    safety_margin: float,
) -> torch.Tensor:
    return _target_max_penetration_depth(
        target_f=target_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        foot_points_xy=foot_points_xy,
        obstacle=obstacle,
    ) <= safety_margin


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


def _evaluate_nominal_target(
    *,
    nominal_target_f: torch.Tensor,
    support_foot_f: torch.Tensor,
    target_origin_w: torch.Tensor,
    target_yaw_w: torch.Tensor,
    obstacle: PenetrationObstacle,
    ellipse_half_length: float,
    ellipse_half_width: float,
    foot_points_xy: torch.Tensor,
    safety_margin: float,
    terrain_height_query_w: TerrainHeightQuery | None,
    max_step_height_m: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nominal_target_f, nominal_height_valid = _lift_targets_to_terrain_height(
        target_f=nominal_target_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        terrain_height_query_w=terrain_height_query_w,
        max_step_height_m=max_step_height_m,
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
    nominal_valid = nominal_height_valid & nominal_obstacle_safe & nominal_inside_ellipse
    return (
        nominal_target_f,
        nominal_inside_ellipse,
        nominal_obstacle_safe,
        nominal_valid,
    )


def _evaluate_candidate_targets(
    *,
    nominal_target_f: torch.Tensor,
    support_foot_f: torch.Tensor,
    target_origin_w: torch.Tensor,
    target_yaw_w: torch.Tensor,
    obstacle: PenetrationObstacle,
    ellipse_half_length: float,
    ellipse_half_width: float,
    foot_points_xy: torch.Tensor,
    candidate_radii: torch.Tensor,
    candidate_directions: torch.Tensor,
    safety_margin: float,
    terrain_height_query_w: TerrainHeightQuery | None,
    max_step_height_m: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    flat_candidates, candidate_height_valid = _lift_targets_to_terrain_height(
        target_f=flat_candidates,
        target_origin_w=flat_target_origin_w,
        target_yaw_w=flat_target_yaw_w,
        terrain_height_query_w=terrain_height_query_w,
        max_step_height_m=max_step_height_m,
    )
    candidates = flat_candidates.reshape(num_envs, num_candidates, 3)
    candidate_height_valid = candidate_height_valid.reshape(num_envs, num_candidates)

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

    return (
        candidates,
        candidate_inside,
        candidate_safe,
        candidate_height_valid & candidate_inside & candidate_safe,
    )


def evaluate_safe_foothold_target(
    *,
    target_f: torch.Tensor,
    support_foot_f: torch.Tensor,
    target_origin_w: torch.Tensor | None = None,
    target_yaw_w: torch.Tensor | None = None,
    obstacle: PenetrationObstacle,
    ellipse_half_length: float,
    ellipse_half_width: float,
    foot_points_xy: torch.Tensor,
    safety_margin: float,
    terrain_height_query_w: TerrainHeightQuery | None = None,
    max_step_height_m: float | None = None,
) -> SafeFootholdTargetEvaluation:
    """Evaluate one terrain-aware foothold target without fallback search.

    This is the public single-target safety contract used by planner-side
    scoring and diagnostics.  It applies the same height, reachability, and
    obstacle-footprint checks as the planner search without running fallback
    candidate selection.
    """

    foot_points_xy = foot_points_xy.to(device=target_f.device, dtype=target_f.dtype)
    if target_origin_w is None:
        target_origin_w = torch.zeros_like(target_f)
    else:
        target_origin_w = target_origin_w.to(device=target_f.device, dtype=target_f.dtype)
    if target_yaw_w is None:
        target_yaw_w = torch.zeros(
            target_f.shape[0],
            device=target_f.device,
            dtype=target_f.dtype,
        )
    else:
        target_yaw_w = target_yaw_w.to(device=target_f.device, dtype=target_f.dtype)

    lifted_target_f, height_valid = _lift_targets_to_terrain_height(
        target_f=target_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        terrain_height_query_w=terrain_height_query_w,
        max_step_height_m=max_step_height_m,
    )
    inside_ellipse = _is_inside_reachable_ellipse(
        target_f=lifted_target_f,
        support_foot_f=support_foot_f,
        ellipse_half_length=ellipse_half_length,
        ellipse_half_width=ellipse_half_width,
    )
    penetration_depths = _target_penetration_depths(
        target_f=lifted_target_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        foot_points_xy=foot_points_xy,
        obstacle=obstacle,
    )
    max_penetration_depth = torch.max(penetration_depths, dim=-1).values
    positive_penetration_depths = torch.clamp(penetration_depths, min=0.0)
    penetration_score = score_sole_perimeter_penetration(
        positive_penetration_depths,
    )
    mean_penetration_depth = torch.mean(positive_penetration_depths, dim=-1)
    obstacle_safe = max_penetration_depth <= safety_margin
    valid = height_valid & inside_ellipse & obstacle_safe
    return SafeFootholdTargetEvaluation(
        target_f=lifted_target_f,
        height_valid=height_valid,
        inside_ellipse=inside_ellipse,
        obstacle_safe=obstacle_safe,
        valid=valid,
        max_penetration_depth=max_penetration_depth,
        mean_penetration_depth=mean_penetration_depth,
        total_penetration_depth=(
            penetration_score.total_penetration_depth
        ),
        penetrating_point_count=(
            penetration_score.penetrating_point_count
        ),
        penetrating_point_ratio=(
            penetration_score.penetrating_point_ratio
        ),
        safety_score=penetration_score.score,
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
    terrain_height_query_w: TerrainHeightQuery | None = None,
    max_step_height_m: float | None = None,
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

    (
        nominal_target_f,
        nominal_inside_ellipse,
        nominal_obstacle_safe,
        nominal_valid,
    ) = _evaluate_nominal_target(
        nominal_target_f=nominal_target_f,
        support_foot_f=support_foot_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        obstacle=obstacle,
        ellipse_half_length=ellipse_half_length,
        ellipse_half_width=ellipse_half_width,
        foot_points_xy=foot_points_xy,
        safety_margin=safety_margin,
        terrain_height_query_w=terrain_height_query_w,
        max_step_height_m=max_step_height_m,
    )

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
        fallback_needed = ~nominal_valid
        fallback_env_ids = torch.nonzero(fallback_needed, as_tuple=False).squeeze(-1)
        (
            candidates,
            candidate_inside,
            candidate_safe,
            candidate_valid,
        ) = _evaluate_candidate_targets(
            nominal_target_f=nominal_target_f[fallback_env_ids],
            support_foot_f=support_foot_f[fallback_env_ids],
            target_origin_w=target_origin_w[fallback_env_ids],
            target_yaw_w=target_yaw_w[fallback_env_ids],
            obstacle=obstacle,
            ellipse_half_length=ellipse_half_length,
            ellipse_half_width=ellipse_half_width,
            foot_points_xy=foot_points_xy,
            candidate_radii=candidate_radii,
            candidate_directions=candidate_directions,
            safety_margin=safety_margin,
            terrain_height_query_w=terrain_height_query_w,
            max_step_height_m=max_step_height_m,
        )
        num_fallback_envs = candidates.shape[0]
        num_candidates = candidates.shape[1]
        candidate_count[fallback_needed] = float(num_candidates)
        candidate_inside_ellipse_count[fallback_needed] = (
            candidate_inside.sum(dim=1).to(dtype=nominal_target_f.dtype)
        )
        candidate_obstacle_safe_count[fallback_needed] = (
            candidate_safe.sum(dim=1).to(dtype=nominal_target_f.dtype)
        )
        candidate_valid_count[fallback_needed] = (
            candidate_valid.sum(dim=1).to(dtype=nominal_target_f.dtype)
        )
        has_candidate = torch.any(candidate_valid, dim=1)

        candidate_score, candidate_distance = _score_candidate_targets(
            candidates=candidates,
            nominal_target_f=nominal_target_f[fallback_env_ids],
            desired_velocity_f=desired_velocity_f[fallback_env_ids],
        )
        candidate_score = candidate_score.masked_fill(~candidate_valid, float("inf"))
        best_idx = torch.argmin(candidate_score, dim=1)

        env_ids = torch.arange(
            num_fallback_envs,
            device=nominal_target_f.device,
        )
        fallback_target = candidates[env_ids, best_idx]
        fallback_score = candidate_distance[env_ids, best_idx]

        replace_env_ids = fallback_env_ids[has_candidate]
        target_f[replace_env_ids] = fallback_target[has_candidate]
        valid[replace_env_ids] = True
        used_fallback[replace_env_ids] = True
        selected_score[replace_env_ids] = fallback_score[has_candidate]

    final_max_penetration_depth = _target_max_penetration_depth(
        target_f=target_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        foot_points_xy=foot_points_xy,
        obstacle=obstacle,
    )

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
        final_max_penetration_depth=final_max_penetration_depth,
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
    terrain_height_query_w: TerrainHeightQuery | None = None,
    max_step_height_m: float | None = None,
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

    nominal_target_f, nominal_height_valid = _lift_targets_to_terrain_height(
        target_f=nominal_target_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        terrain_height_query_w=terrain_height_query_w,
        max_step_height_m=max_step_height_m,
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
    nominal_valid = nominal_height_valid & nominal_obstacle_safe & nominal_inside_ellipse

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

    flat_candidates, candidate_height_valid = _lift_targets_to_terrain_height(
        target_f=flat_candidates,
        target_origin_w=flat_target_origin_w,
        target_yaw_w=flat_target_yaw_w,
        terrain_height_query_w=terrain_height_query_w,
        max_step_height_m=max_step_height_m,
    )
    candidates = flat_candidates.reshape(num_envs, num_candidates, 3)
    candidate_height_valid = candidate_height_valid.reshape(num_envs, num_candidates)

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
        nominal_target_f=nominal_target_f,
        candidates_f=candidates,
        nominal_inside_ellipse=nominal_inside_ellipse,
        nominal_obstacle_safe=nominal_obstacle_safe,
        nominal_valid=nominal_valid,
        candidate_inside_ellipse=candidate_inside,
        candidate_obstacle_safe=candidate_safe,
        candidate_valid=candidate_height_valid & candidate_inside & candidate_safe,
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
