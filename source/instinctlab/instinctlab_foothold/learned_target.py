from __future__ import annotations

from dataclasses import dataclass

import torch

from .frame_transform import (
    TerrainHeightQuery,
    apply_world_height_to_planner_target,
)
from .geometry import make_frozen_stance_frame, world_to_frozen


# Route outcomes are diagnostics only.  They are intentionally integer codes
# so the planner can publish one mutually exclusive reason per committed
# swing without changing any routing decision.
LEARNED_FOOTHOLD_ROUTE_REASON_SUCCESS = 0
LEARNED_FOOTHOLD_ROUTE_REASON_RECOVERY = 1
LEARNED_FOOTHOLD_ROUTE_REASON_TRANSACTION_INVALIDATED = 2
LEARNED_FOOTHOLD_ROUTE_REASON_GEOMETRIC_INVALID = 3
LEARNED_FOOTHOLD_ROUTE_REASON_ENDPOINT_UNSAFE = 4
LEARNED_FOOTHOLD_ROUTE_REASON_PREFLIGHT_UNSAFE = 5
LEARNED_FOOTHOLD_ROUTE_REASON_POSTCHECK_INVALID = 6


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


def select_preflight_target_w(
    *,
    route_use_learned: torch.Tensor,
    learned_prepared_w: torch.Tensor,
    nominal_target_w: torch.Tensor,
) -> torch.Tensor:
    """Select the HOLD-prepared world target for trajectory preflight."""

    if route_use_learned.dtype != torch.bool:
        raise TypeError("route_use_learned must be boolean.")
    if learned_prepared_w.shape != nominal_target_w.shape:
        raise ValueError("prepared learned and nominal targets must share shape.")
    if (
        learned_prepared_w.ndim != 2
        or learned_prepared_w.shape[-1] != 3
        or route_use_learned.shape != learned_prepared_w.shape[:-1]
    ):
        raise ValueError("preflight targets must have shape (N, 3).")
    return torch.where(
        route_use_learned.unsqueeze(-1),
        learned_prepared_w,
        nominal_target_w,
    )


def finalize_learned_foothold_route_outcome(
    *,
    initial_outcome: torch.Tensor,
    route_initial_executable: torch.Tensor,
    final_planner_valid: torch.Tensor,
) -> torch.Tensor:
    """Apply the final lock/terrain validity result to a route diagnostic."""

    if not (
        initial_outcome.shape
        == route_initial_executable.shape
        == final_planner_valid.shape
    ):
        raise ValueError("route outcome tensors must share one shape.")
    if route_initial_executable.dtype != torch.bool or final_planner_valid.dtype != torch.bool:
        raise TypeError("route validity masks must be boolean.")
    outcome = initial_outcome.clone()
    outcome[route_initial_executable & ~final_planner_valid] = (
        LEARNED_FOOTHOLD_ROUTE_REASON_POSTCHECK_INVALID
    )
    return outcome


def classify_learned_foothold_route(
    *,
    recovery_step: torch.Tensor,
    transaction_evaluated: torch.Tensor,
    learned_geometric_valid: torch.Tensor,
    learned_safety_valid: torch.Tensor,
    preflight_ready: torch.Tensor,
    preflight_safe: torch.Tensor,
    route_use_learned: torch.Tensor,
) -> torch.Tensor:
    """Return the first failed learned-route gate for diagnostics.

    This helper deliberately has no side effects and is not used to decide
    which target the planner executes.  It only labels the already-committed
    route so monitor logs can distinguish endpoint failures from trajectory
    preflight failures and transaction invalidation.
    """

    masks = (
        recovery_step,
        transaction_evaluated,
        learned_geometric_valid,
        learned_safety_valid,
        preflight_ready,
        preflight_safe,
        route_use_learned,
    )
    if any(mask.dtype != torch.bool for mask in masks):
        raise TypeError("learned route diagnostic masks must be boolean.")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("learned route diagnostic masks must share a shape.")

    reason = torch.full(
        shape,
        LEARNED_FOOTHOLD_ROUTE_REASON_TRANSACTION_INVALIDATED,
        dtype=torch.long,
        device=recovery_step.device,
    )
    normal = ~recovery_step
    reason[recovery_step] = LEARNED_FOOTHOLD_ROUTE_REASON_RECOVERY
    reason[normal & transaction_evaluated & ~learned_geometric_valid] = (
        LEARNED_FOOTHOLD_ROUTE_REASON_GEOMETRIC_INVALID
    )
    endpoint_valid = normal & transaction_evaluated & learned_geometric_valid
    reason[endpoint_valid & ~learned_safety_valid] = (
        LEARNED_FOOTHOLD_ROUTE_REASON_ENDPOINT_UNSAFE
    )
    preflight_candidate = endpoint_valid & learned_safety_valid
    reason[preflight_candidate & (~preflight_ready | ~preflight_safe)] = (
        LEARNED_FOOTHOLD_ROUTE_REASON_PREFLIGHT_UNSAFE
    )
    route_success = (
        preflight_candidate
        & preflight_ready
        & preflight_safe
        & route_use_learned
    )
    reason[route_success] = LEARNED_FOOTHOLD_ROUTE_REASON_SUCCESS
    return reason


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
        & (torch.abs(target_f[:, 2]) < max_step_height_m)
    )
    return target_f, geometric_valid


def route_nominal_and_learned_footholds(
    *,
    nominal_geometric_valid: torch.Tensor,
    nominal_safety_valid: torch.Tensor,
    learned_prepared: torch.Tensor,
    learned_geometric_valid: torch.Tensor,
    learned_safety_valid: torch.Tensor | None = None,
    recovery_step: torch.Tensor | None = None,
) -> LearnedFootholdRoute:
    """Route privileged training targets without invoking candidate search.

    A learned proposal must pass both geometry and danger-cylinder safety
    before it can be executed, including during contact-adaptive recovery. A
    safe nominal target is the fallback when the learned proposal is not
    executable. Recovery must not introduce a second, weaker definition of a
    safe foothold.
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

    if learned_safety_valid is None:
        learned_safety_valid = torch.ones_like(learned_geometric_valid)
    if learned_safety_valid.shape != learned_geometric_valid.shape:
        raise ValueError("learned_safety_valid must match the route masks.")
    learned_available_safe = (
        learned_prepared.bool()
        & learned_geometric_valid.bool()
        & learned_safety_valid.bool()
    )
    use_learned = learned_available_safe

    # The recovery mask remains part of the API for diagnostics, but it cannot
    # bypass the same endpoint safety gate used during normal walking.
    safe_nominal = nominal_geometric_valid.bool() & nominal_safety_valid.bool()
    use_nominal = (
        ~use_learned
        & safe_nominal
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
    safety_margin_score: torch.Tensor | None = None,
    minimum_signed_clearance: torch.Tensor | None = None,
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
    data.learned_foothold_transaction_evaluated[env_ids] = True
    # ``prepared_valid`` means executable geometry. Danger-cylinder safety is
    # deliberately retained as a separate soft PPO signal; folding it into
    # this flag would prevent the policy from experiencing the proposal whose
    # safety score it must learn to improve.
    data.learned_foothold_prepared_valid[env_ids] = (
        preparation.geometric_valid
    )
    data.learned_foothold_safety_score[env_ids] = safety_score
    if safety_margin_score is not None and getattr(
        data, "learned_foothold_safety_margin_score", None
    ) is not None:
        data.learned_foothold_safety_margin_score[env_ids] = safety_margin_score
    if minimum_signed_clearance is not None and getattr(
        data, "learned_foothold_minimum_signed_clearance", None
    ) is not None:
        data.learned_foothold_minimum_signed_clearance[env_ids] = (
            minimum_signed_clearance
        )
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
    safety_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lock prepared targets that have passed the execution safety gate."""

    use = data.learned_foothold_prepared_valid[env_ids].clone()
    if safety_valid is not None:
        if safety_valid.shape != use.shape:
            raise ValueError("safety_valid must match selected environments.")
        use &= safety_valid.bool()
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
        "learned_foothold_safety_margin_score",
        "learned_foothold_minimum_signed_clearance",
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
        "learned_foothold_transaction_evaluated",
        "learned_foothold_route_event",
        "learned_foothold_route_use_nominal",
        "learned_foothold_route_use_learned",
        "learned_foothold_route_initial_executable",
    )
    integer_fields = ("learned_foothold_route_outcome",)

    for name in vector_fields:
        value = getattr(data, name, None)
        if value is not None:
            value[env_ids] = 0.0
    for name in flag_fields:
        value = getattr(data, name, None)
        if value is not None:
            value[env_ids] = False
    for name in integer_fields:
        value = getattr(data, name, None)
        if value is not None:
            value[env_ids] = 0


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


def decode_residual_foothold(
    normalized_action: torch.Tensor,
    *,
    nominal_xy_f: torch.Tensor,
    max_adjustment_x: float,
    max_adjustment_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a normalized planner action as a residual around a nominal XY.

    The nominal target is the action origin.  Keeping this anchor in the
    decoder makes a zero action an identity adjustment, so the planner cannot
    collapse both swing legs onto the support-foot centerline merely because
    its policy mean is initially near zero.
    """

    if normalized_action.shape[-1] != 2:
        raise ValueError("normalized_action must have two coordinates.")
    if nominal_xy_f.shape != normalized_action.shape:
        raise ValueError("nominal_xy_f must match normalized_action shape.")
    if max_adjustment_x <= 0.0 or max_adjustment_y <= 0.0:
        raise ValueError("planner adjustment bounds must be positive.")
    normalized = torch.clamp(normalized_action, -1.0, 1.0)
    limits = normalized.new_tensor((max_adjustment_x, max_adjustment_y))
    residual_xy = normalized * limits
    return nominal_xy_f + residual_xy, residual_xy


def prepare_learned_foothold_target(
    *,
    normalized_action: torch.Tensor,
    nominal_xy_f: torch.Tensor | None = None,
    origin_w: torch.Tensor,
    yaw_w: torch.Tensor,
    radius_x: float,
    radius_y: float,
    max_adjustment_x: float | None = None,
    max_adjustment_y: float | None = None,
    max_step_height_m: float,
    terrain_height_query_w: TerrainHeightQuery,
) -> LearnedFootholdPreparation:
    """Decode XY, query world-frame terrain height, and apply hard geometry."""

    if max_step_height_m <= 0.0:
        raise ValueError("max_step_height_m must be positive.")

    # ``None`` preserves the source-file utility's old absolute-decoding API
    # for standalone callers.  The simulator planner always supplies the
    # frozen nominal target and explicit residual bounds below.
    if nominal_xy_f is None:
        decoded_xy_f = decode_normalized_foothold(
            normalized_action,
            radius_x=radius_x,
            radius_y=radius_y,
        )
    else:
        if max_adjustment_x is None:
            max_adjustment_x = radius_x
        if max_adjustment_y is None:
            max_adjustment_y = radius_y
        decoded_xy_f, _ = decode_residual_foothold(
            normalized_action,
            nominal_xy_f=nominal_xy_f,
            max_adjustment_x=max_adjustment_x,
            max_adjustment_y=max_adjustment_y,
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
        torch.abs(target_f[:, 2]) < max_step_height_m
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
    transaction_evaluated: torch.Tensor,
    learned_prepared_valid: torch.Tensor,
    learned_geometric_valid: torch.Tensor,
    learned_safety_valid: torch.Tensor,
    recovery_step: torch.Tensor,
) -> torch.Tensor:
    """Gate swing start until the normal learned route has been evaluated.

    Normal walking gets one control cycle to evaluate the learned proposal.
    A cached learned proposal that has passed the execution gate remains ready
    for the rest of the HOLD transaction. An unsafe proposal may fall back to a
    safe nominal route only after it has been evaluated. Recovery uses the same
    safety gate and therefore cannot start a swing with an unsafe endpoint.
    """

    if not (
        nominal_route_ready.shape
        == transaction_evaluated.shape
        == learned_prepared_valid.shape
        == learned_geometric_valid.shape
        == learned_safety_valid.shape
        == recovery_step.shape
    ):
        raise ValueError("learned foothold swing masks must share one shape.")

    nominal_ready = nominal_route_ready.bool()
    learned_ready_safe = (
        learned_prepared_valid.bool()
        & learned_geometric_valid.bool()
        & learned_safety_valid.bool()
    )
    normal_ready = transaction_evaluated.bool() & (
        learned_ready_safe | nominal_ready
    )
    return normal_ready


def learned_foothold_transaction_ready(
    *,
    nominal_route_ready: torch.Tensor,
    transaction_evaluated: torch.Tensor,
    learned_prepared_valid: torch.Tensor,
    learned_geometric_valid: torch.Tensor,
    learned_safety_valid: torch.Tensor,
) -> torch.Tensor:
    """Return whether one HOLD proposal can be committed without resampling.

    A safe nominal fallback is an executable result of the current learned
    action, even if that action is unsafe.  The unsafe action remains latched
    for its PPO penalty; sampling another one in the same HOLD would make the
    policy's action-to-result relationship non-causal.
    """
    masks = (
        nominal_route_ready,
        transaction_evaluated,
        learned_prepared_valid,
        learned_geometric_valid,
        learned_safety_valid,
    )
    if any(mask.shape != nominal_route_ready.shape for mask in masks[1:]):
        raise ValueError("learned foothold transaction masks must share one shape.")
    learned_safe = (
        learned_prepared_valid.bool()
        & learned_geometric_valid.bool()
        & learned_safety_valid.bool()
    )
    return transaction_evaluated.bool() & (
        learned_safe | nominal_route_ready.bool()
    )


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
