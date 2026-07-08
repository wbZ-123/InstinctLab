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
]
