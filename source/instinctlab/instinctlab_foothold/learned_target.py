from __future__ import annotations

from dataclasses import dataclass

import torch

from .frame_transform import (
    TerrainHeightQuery,
    apply_world_height_to_planner_target,
)
from .geometry import make_frozen_stance_frame, world_to_frozen


@dataclass
class LearnedFootholdPreparation:
    decoded_f: torch.Tensor
    target_f: torch.Tensor
    target_w: torch.Tensor
    height_valid: torch.Tensor
    geometric_valid: torch.Tensor


@dataclass
class LearnedFootholdRoute:
    use_nominal: torch.Tensor
    use_learned: torch.Tensor
    executable: torch.Tensor


def reachable_ellipse_usage(
    target_xy_f: torch.Tensor,
    *,
    radius_x: float,
    radius_y: float,
) -> torch.Tensor:
    """Return normalized XY distance from the center of a reachability ellipse."""

    if target_xy_f.shape[-1] != 2:
        raise ValueError("target_xy_f must have two coordinates.")
    if radius_x <= 0.0 or radius_y <= 0.0:
        raise ValueError("reachability ellipse radii must be positive.")
    return torch.sqrt(
        (target_xy_f[..., 0] / radius_x).square()
        + (target_xy_f[..., 1] / radius_y).square()
    )


def reframe_cached_world_foothold(
    *,
    target_w: torch.Tensor,
    current_origin_w: torch.Tensor,
    current_yaw_w: torch.Tensor,
    radius_x: float,
    radius_y: float,
    max_step_height_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a frozen world target in the current support frame and validate it."""

    if target_w.shape != current_origin_w.shape or target_w.shape[-1] != 3:
        raise ValueError("target_w and current_origin_w must share shape (N, 3).")
    if current_yaw_w.shape != target_w.shape[:-1]:
        raise ValueError("current_yaw_w must match the target batch shape.")
    if radius_x <= 0.0 or radius_y <= 0.0:
        raise ValueError("reachability radii must be positive.")
    if max_step_height_m <= 0.0:
        raise ValueError("max_step_height_m must be positive.")

    current_frame = make_frozen_stance_frame(
        current_origin_w,
        current_yaw_w,
    )
    target_f = world_to_frozen(target_w, current_frame)
    normalized_radius = reachable_ellipse_usage(
        target_f[:, :2],
        radius_x=radius_x,
        radius_y=radius_y,
    ).square()
    finite = torch.isfinite(target_f).all(dim=-1)
    geometric_valid = (
        finite
        & (normalized_radius <= 1.0 + 1.0e-6)
        & (torch.abs(target_f[:, 2]) <= max_step_height_m)
    )
    return target_f, geometric_valid


def route_nominal_and_learned_footholds(
    *,
    nominal_geometric_valid: torch.Tensor,
    nominal_safety_valid: torch.Tensor,
    learned_prepared: torch.Tensor,
    learned_geometric_valid: torch.Tensor,
    recovery_step: torch.Tensor | None = None,
) -> LearnedFootholdRoute:
    """Route privileged training targets without invoking candidate search.

    During normal walking, a prepared and geometrically valid learned proposal
    is executed even when the nominal target is safe.  This closes the PPO
    control loop while keeping danger-cylinder safety as a soft learning signal.
    A safe nominal target is only the fallback when the learned proposal is not
    executable.  Recovery steps remain analytic and nominal-only.
    """

    if not (
        nominal_geometric_valid.shape
        == nominal_safety_valid.shape
        == learned_prepared.shape
        == learned_geometric_valid.shape
    ):
        raise ValueError("foothold route masks must share one shape.")

    if recovery_step is None:
        recovery_mask = torch.zeros_like(nominal_geometric_valid, dtype=torch.bool)
    else:
        recovery_mask = recovery_step.to(
            device=nominal_geometric_valid.device,
            dtype=torch.bool,
        )
        if recovery_mask.shape != nominal_geometric_valid.shape:
            raise ValueError("recovery_step must match the route mask shape.")

    learned_available = learned_prepared.bool() & learned_geometric_valid.bool()
    use_learned = ~recovery_mask & learned_available

    # During normal walking the nominal target is a safe fallback only.  A
    # recovery step instead uses its analytic target whenever geometry permits,
    # even when the virtual danger-cylinder score is negative; otherwise the
    # state machine could reject the very step intended to restore contact.
    safe_nominal = nominal_geometric_valid.bool() & nominal_safety_valid.bool()
    use_nominal = (recovery_mask & nominal_geometric_valid.bool()) | (
        ~recovery_mask & ~use_learned & safe_nominal
    )
    return LearnedFootholdRoute(
        use_nominal=use_nominal,
        use_learned=use_learned,
        executable=use_nominal | use_learned,
    )


def store_learned_foothold_preparation(
    *,
    data: object,
    env_ids: torch.Tensor,
    preparation: LearnedFootholdPreparation,
    safety_valid: torch.Tensor,
    safety_score: torch.Tensor,
    penetrating_point_count: torch.Tensor,
    penetrating_point_ratio: torch.Tensor,
    total_penetration_depth: torch.Tensor,
) -> None:
    """Store a proposal and its score without silently replacing unsafe data."""

    data.learned_foothold_decoded_f[env_ids] = preparation.decoded_f
    data.learned_foothold_prepared_f[env_ids] = preparation.target_f
    data.learned_foothold_prepared_w[env_ids] = preparation.target_w
    data.learned_foothold_height_valid[env_ids] = preparation.height_valid
    data.learned_foothold_geometric_valid[env_ids] = (
        preparation.geometric_valid
    )
    data.learned_foothold_safety_valid[env_ids] = safety_valid
    data.learned_foothold_evaluated[env_ids] = True
    # ``prepared_valid`` means executable geometry. Danger-cylinder safety is
    # deliberately retained as a separate soft PPO signal; folding it into
    # this flag would prevent the policy from experiencing the proposal whose
    # safety score it must learn to improve.
    data.learned_foothold_prepared_valid[env_ids] = (
        preparation.geometric_valid
    )
    data.learned_foothold_safety_score[env_ids] = safety_score
    data.learned_foothold_penetrating_point_count[env_ids] = (
        penetrating_point_count
    )
    data.learned_foothold_penetrating_point_ratio[env_ids] = (
        penetrating_point_ratio
    )
    data.learned_foothold_total_penetration_depth[env_ids] = (
        total_penetration_depth
    )


def lock_prepared_learned_foothold(
    *,
    data: object,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Lock geometrically valid prepared targets and return their use mask."""

    use = data.learned_foothold_prepared_valid[env_ids].clone()
    data.learned_foothold_locked[env_ids] = False
    data.learned_foothold_used[env_ids] = False
    if torch.any(use):
        used_env_ids = env_ids[use]
        data.learned_foothold_target_f[used_env_ids] = (
            data.learned_foothold_prepared_f[used_env_ids]
        )
        data.learned_foothold_target_w[used_env_ids] = (
            data.learned_foothold_prepared_w[used_env_ids]
        )
        data.learned_foothold_locked[used_env_ids] = True
        data.learned_foothold_used[used_env_ids] = True
    return use


def clear_learned_foothold_buffers(
    data: object,
    env_ids,
) -> None:
    """Clear event-prepared and swing-locked learned foothold state."""

    vector_fields = (
        "learned_foothold_action_normalized",
        "learned_foothold_decoded_f",
        "learned_foothold_prepared_f",
        "learned_foothold_prepared_w",
        "learned_foothold_target_f",
        "learned_foothold_target_w",
        "learned_foothold_safety_score",
        "learned_foothold_penetrating_point_count",
        "learned_foothold_penetrating_point_ratio",
        "learned_foothold_total_penetration_depth",
    )
    flag_fields = (
        "learned_foothold_prepared_valid",
        "learned_foothold_locked",
        "learned_foothold_used",
        "learned_foothold_height_valid",
        "learned_foothold_geometric_valid",
        "learned_foothold_safety_valid",
        "learned_foothold_evaluated",
        "learned_foothold_route_event",
        "learned_foothold_route_use_nominal",
        "learned_foothold_route_use_learned",
        "learned_foothold_route_initial_executable",
    )

    for name in vector_fields:
        value = getattr(data, name, None)
        if value is not None:
            value[env_ids] = 0.0
    for name in flag_fields:
        value = getattr(data, name, None)
        if value is not None:
            value[env_ids] = False


def decode_normalized_foothold(
    normalized_action: torch.Tensor,
    *,
    radius_x: float,
    radius_y: float,
) -> torch.Tensor:
    """Decode a normalized 2D action through the shared reachability ellipse."""

    if normalized_action.shape[-1] != 2:
        raise ValueError("normalized_action must have two coordinates.")
    if radius_x <= 0.0 or radius_y <= 0.0:
        raise ValueError("reachability ellipse radii must be positive.")

    normalized = torch.clamp(normalized_action, -1.0, 1.0)
    radial_scale = torch.linalg.vector_norm(
        normalized,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0)
    projected = normalized / radial_scale
    radii = normalized.new_tensor((radius_x, radius_y))
    return projected * radii


def prepare_learned_foothold_target(
    *,
    normalized_action: torch.Tensor,
    origin_w: torch.Tensor,
    yaw_w: torch.Tensor,
    radius_x: float,
    radius_y: float,
    max_step_height_m: float,
    terrain_height_query_w: TerrainHeightQuery,
) -> LearnedFootholdPreparation:
    """Decode XY, query world-frame terrain height, and apply hard geometry."""

    if max_step_height_m <= 0.0:
        raise ValueError("max_step_height_m must be positive.")

    decoded_xy_f = decode_normalized_foothold(
        normalized_action,
        radius_x=radius_x,
        radius_y=radius_y,
    )
    decoded_f = torch.cat(
        (
            decoded_xy_f,
            torch.zeros_like(decoded_xy_f[:, :1]),
        ),
        dim=-1,
    )
    target_f, target_w, height_valid = (
        apply_world_height_to_planner_target(
            origin_w=origin_w,
            target_xy_f=decoded_xy_f,
            yaw_w=yaw_w,
            terrain_height_query_w=terrain_height_query_w,
        )
    )
    geometric_valid = height_valid & (
        torch.abs(target_f[:, 2]) <= max_step_height_m
    )
    return LearnedFootholdPreparation(
        decoded_f=decoded_f,
        target_f=target_f,
        target_w=target_w,
        height_valid=height_valid,
        geometric_valid=geometric_valid,
    )


def learned_foothold_event_masks(
    *,
    hold: torch.Tensor,
    hold_contact_ready: torch.Tensor,
    nominal_ready: torch.Tensor,
    new_swing: torch.Tensor,
    enable: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks for HOLD preparation and new-SWING target locking."""

    if not (
        hold.shape
        == hold_contact_ready.shape
        == nominal_ready.shape
        == new_swing.shape
    ):
        raise ValueError("learned foothold event masks must share one shape.")

    if not enable:
        disabled = torch.zeros_like(hold, dtype=torch.bool)
        return disabled, disabled.clone()

    prepare = (
        hold.bool()
        & hold_contact_ready.bool()
        & nominal_ready.bool()
    )
    lock = new_swing.bool()
    return prepare, lock


def learned_foothold_swing_ready(
    *,
    nominal_route_ready: torch.Tensor,
    learned_evaluated: torch.Tensor,
    learned_prepared_valid: torch.Tensor,
    learned_geometric_valid: torch.Tensor,
    recovery_step: torch.Tensor,
) -> torch.Tensor:
    """Gate swing start until the normal learned route has been evaluated.

    Normal walking gets one control cycle to evaluate the learned proposal.
    A valid learned proposal is preferred; an evaluated but geometrically
    invalid proposal may fall back to a safe nominal route.  Recovery uses the
    analytic nominal route immediately and never waits for learned output.
    """

    if not (
        nominal_route_ready.shape
        == learned_evaluated.shape
        == learned_prepared_valid.shape
        == learned_geometric_valid.shape
        == recovery_step.shape
    ):
        raise ValueError("learned foothold swing masks must share one shape.")

    nominal_ready = nominal_route_ready.bool()
    recovery = recovery_step.bool()
    learned_ready = (
        learned_evaluated.bool()
        & learned_prepared_valid.bool()
        & learned_geometric_valid.bool()
    )
    normal_ready = learned_evaluated.bool() & (
        learned_ready | nominal_ready
    )
    return torch.where(recovery, nominal_ready, normal_ready)


def nominal_foothold_prepare_mask(
    *,
    hold: torch.Tensor,
    hold_contact_ready: torch.Tensor,
    nominal_ready: torch.Tensor,
    startup_hold: torch.Tensor,
) -> torch.Tensor:
    """Allow nominal planning only after the support contact is confirmed."""

    if not (
        hold.shape
        == hold_contact_ready.shape
        == nominal_ready.shape
        == startup_hold.shape
    ):
        raise ValueError("nominal foothold masks must share one shape.")
    return (
        hold.bool()
        & hold_contact_ready.bool()
        & ~nominal_ready.bool()
        & ~startup_hold.bool()
    )
