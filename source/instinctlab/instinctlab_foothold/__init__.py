"""Simulator-independent foothold planning primitives."""

from .frame_transform import apply_world_height_to_planner_target, planner_frame_to_world_xy
from .learned_target import (
    LearnedFootholdPreparation,
    clear_learned_foothold_buffers,
    decode_normalized_foothold,
    learned_foothold_event_masks,
    learned_foothold_swing_ready,
    nominal_foothold_prepare_mask,
    lock_prepared_learned_foothold,
    prepare_learned_foothold_target,
    reachable_ellipse_usage,
    reframe_cached_world_foothold,
    route_nominal_and_learned_footholds,
    store_learned_foothold_preparation,
)
from .geometry import FrozenFrame, SoleGeometry, frozen_to_world, make_frozen_stance_frame, world_to_frozen
from .types import FOOTHOLD_OBSERVATION_DIM, GaitState, ObservationSlice
from .flat_provider import (
    FlatProviderConfig,
    FlatTargetBatch,
    TerrainCorridor,
    sample_flat_targets,
)
from .recovery_target import make_recovery_foothold_target

from .state_machine import (
    GaitMachineConfig,
    GaitMachineState,
    advance_gait,
    gait_phase,
    initial_gait_state,
)

from .contact_adaptation import (
    ContactEvent,
    EventResponse,
    StabilityBounds,
    StabilitySignals,
    response_for_event,
    stability_ready,
    support_roles_from_contacts,
)
from .stability_calibration import (
    calibrate_stability_bounds,
    load_stability_bounds,
)

from .terrain_query import (
    FlatTerrainQuery,
    StepTerrainQuery,
    TerrainQueryResult,
)

from .trajectory import SwingReference, quintic_swing_reference

from .terrain_provider import lift_flat_targets_to_terrain

from .clearance import (
    ApexAdjustmentResult,
    SwingCenterlinePenetration,
    adjust_apex_for_edge_clearance,
    check_swing_centerline_penetration,
    sample_swing_centerline,
)
from .target_search import (
    SafeFootholdCandidateDebug,
    SafeFootholdTargetEvaluation,
    SolePerimeterPenetrationScore,
    debug_safe_foothold_candidates,
    evaluate_safe_foothold_target,
    make_sole_perimeter_points_xy,
    score_sole_perimeter_penetration,
    search_safe_foothold_target,
)

__all__ = [
    "FOOTHOLD_OBSERVATION_DIM",
    "FrozenFrame",
    "GaitState",
    "ObservationSlice",
    "apply_world_height_to_planner_target",
    "LearnedFootholdPreparation",
    "clear_learned_foothold_buffers",
    "decode_normalized_foothold",
    "learned_foothold_event_masks",
    "learned_foothold_swing_ready",
    "nominal_foothold_prepare_mask",
    "lock_prepared_learned_foothold",
    "prepare_learned_foothold_target",
    "reachable_ellipse_usage",
    "reframe_cached_world_foothold",
    "route_nominal_and_learned_footholds",
    "store_learned_foothold_preparation",
    "planner_frame_to_world_xy",
    "SoleGeometry",
    "frozen_to_world",
    "make_frozen_stance_frame",
    "world_to_frozen",
    "FlatProviderConfig",
    "FlatTargetBatch",
    "TerrainCorridor",
    "sample_flat_targets",
    "make_recovery_foothold_target",
    "GaitMachineConfig",
    "GaitMachineState",
    "SwingReference",
    "advance_gait",
    "gait_phase",
    "initial_gait_state",
    "ContactEvent",
    "EventResponse",
    "StabilityBounds",
    "StabilitySignals",
    "response_for_event",
    "stability_ready",
    "support_roles_from_contacts",
    "calibrate_stability_bounds",
    "load_stability_bounds",
    "quintic_swing_reference",
    "FlatTerrainQuery",
    "StepTerrainQuery",
    "TerrainQueryResult",
    "lift_flat_targets_to_terrain",
    "SwingCenterlinePenetration",
    "check_swing_centerline_penetration",
    "sample_swing_centerline",
    "ApexAdjustmentResult",
    "adjust_apex_for_edge_clearance",
    "SafeFootholdCandidateDebug",
    "SafeFootholdTargetEvaluation",
    "SolePerimeterPenetrationScore",
    "debug_safe_foothold_candidates",
    "evaluate_safe_foothold_target",
    "make_sole_perimeter_points_xy",
    "score_sole_perimeter_penetration",
    "search_safe_foothold_target",
]
