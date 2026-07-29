from __future__ import annotations

from collections.abc import Callable

import torch


TerrainHeightQuery = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def planner_frame_to_world_xy(
    origin_w: torch.Tensor,
    target_xy_f: torch.Tensor,
    yaw_w: torch.Tensor,
) -> torch.Tensor:
    """Transform foothold planner-frame XY coordinates into world-frame XY coordinates."""

    cos_yaw = torch.cos(yaw_w)
    sin_yaw = torch.sin(yaw_w)
    x_w = origin_w[:, 0] + cos_yaw * target_xy_f[:, 0] - sin_yaw * target_xy_f[:, 1]
    y_w = origin_w[:, 1] + sin_yaw * target_xy_f[:, 0] + cos_yaw * target_xy_f[:, 1]
    return torch.stack([x_w, y_w], dim=-1)


def apply_world_height_to_planner_target(
    *,
    origin_w: torch.Tensor,
    target_xy_f: torch.Tensor,
    yaw_w: torch.Tensor,
    terrain_height_query_w: TerrainHeightQuery,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attach terrain height queried in world coordinates to a planner-frame foothold target."""

    target_xy_w = planner_frame_to_world_xy(origin_w, target_xy_f, yaw_w)
    terrain_z_w, valid = terrain_height_query_w(target_xy_w)
    terrain_z_w = terrain_z_w.to(device=target_xy_f.device, dtype=target_xy_f.dtype)
    valid = valid.to(device=target_xy_f.device, dtype=torch.bool) & torch.isfinite(terrain_z_w)
    target_w = torch.cat([target_xy_w, terrain_z_w[:, None]], dim=-1)
    target_f = torch.cat([target_xy_f, (terrain_z_w - origin_w[:, 2])[:, None]], dim=-1)
    return target_f, target_w, valid
