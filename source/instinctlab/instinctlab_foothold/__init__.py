"""Simulator-independent foothold planning primitives."""

from .geometry import FrozenFrame, SoleGeometry, frozen_to_world, make_frozen_stance_frame, world_to_frozen
from .types import FOOTHOLD_OBSERVATION_DIM, GaitState, ObservationSlice
from .flat_provider import (
    FlatProviderConfig,
    FlatTargetBatch,
    TerrainCorridor,
    sample_flat_targets,
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
]
