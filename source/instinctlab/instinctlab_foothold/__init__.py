"""Simulator-independent foothold planning primitives."""

from .geometry import FrozenFrame, SoleGeometry, frozen_to_world, make_frozen_stance_frame, world_to_frozen
from .types import FOOTHOLD_OBSERVATION_DIM, GaitState, ObservationSlice
from .flat_provider import (
    FlatProviderConfig,
    FlatTargetBatch,
    TerrainCorridor,
    sample_flat_targets,
)

from .state_machine import (
    GaitMachineConfig,
    GaitMachineState,
    advance_gait,
    gait_phase,
    initial_gait_state,
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
    debug_safe_foothold_candidates,
    make_sole_perimeter_points_xy,
    search_safe_foothold_target,
)

__all__ = [
    "FOOTHOLD_OBSERVATION_DIM",
    "FrozenFrame",
    "GaitState",
    "ObservationSlice",
    "SoleGeometry",
    "frozen_to_world",
    "make_frozen_stance_frame",
    "world_to_frozen",
    "FlatProviderConfig",
    "FlatTargetBatch",
    "TerrainCorridor",
    "sample_flat_targets",
    "GaitMachineConfig",
    "GaitMachineState",
    "SwingReference",
    "advance_gait",
    "gait_phase",
    "initial_gait_state",
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
    "debug_safe_foothold_candidates",
    "make_sole_perimeter_points_xy",
    "search_safe_foothold_target",
]
