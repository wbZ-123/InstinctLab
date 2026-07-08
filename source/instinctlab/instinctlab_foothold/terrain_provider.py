from __future__ import annotations

from .flat_provider import FlatTargetBatch, TerrainCorridor


def lift_flat_targets_to_terrain(
    flat_targets: FlatTargetBatch,
    terrain_query,
) -> FlatTargetBatch:
    """Lift flat foothold targets onto queried terrain heights.

    This keeps the flat planner's x/y target, yaw, normal, feasible velocity,
    and validity unchanged. It only replaces target z with queried terrain
    height and stores terrain metadata for later reward/planning use.
    """
    terrain = terrain_query.query(flat_targets.position_f[:, :2])

    position_f = flat_targets.position_f.clone()
    position_f[:, 2] = terrain.height

    corridor_width = flat_targets.terrain.heights.shape[1]
    terrain_corridor = TerrainCorridor(
        heights=terrain.height.unsqueeze(-1).repeat(1, corridor_width),
        confidences=terrain.confidence.unsqueeze(-1).repeat(1, corridor_width),
        support_margin=terrain.support_margin,
        edge_risk=terrain.edge_risk,
        unknown_fraction=1.0 - terrain.confidence,
    )

    return FlatTargetBatch(
        position_f=position_f,
        yaw_f=flat_targets.yaw_f,
        normal_f=flat_targets.normal_f,
        feasible_velocity_f=flat_targets.feasible_velocity_f,
        valid=flat_targets.valid,
        terrain=terrain_corridor,
    )