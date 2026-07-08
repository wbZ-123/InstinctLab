from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TerrainQueryResult:
    """Terrain information queried at candidate foothold points."""

    height: torch.Tensor
    confidence: torch.Tensor
    support_margin: torch.Tensor
    edge_risk: torch.Tensor


@dataclass(frozen=True)
class FlatTerrainQuery:
    """Analytic flat terrain query for tests and simple visualization."""

    height_m: float = 0.0
    confidence: float = 1.0
    support_margin: float = 1.0
    edge_risk: float = 0.0

    def query(self, points_xy: torch.Tensor) -> TerrainQueryResult:
        num_points = points_xy.shape[0]
        device = points_xy.device
        dtype = points_xy.dtype

        return TerrainQueryResult(
            height=torch.full(
                (num_points,),
                self.height_m,
                device=device,
                dtype=dtype,
            ),
            confidence=torch.full(
                (num_points,),
                self.confidence,
                device=device,
                dtype=dtype,
            ),
            support_margin=torch.full(
                (num_points,),
                self.support_margin,
                device=device,
                dtype=dtype,
            ),
            edge_risk=torch.full(
                (num_points,),
                self.edge_risk,
                device=device,
                dtype=dtype,
            ),
        )


@dataclass(frozen=True)
class StepTerrainQuery:
    """Analytic one-step terrain query.

    Points with x >= step_x_m are assigned the upper height.
    Points before the step remain at lower height.

    The edge risk is high near the step boundary so later foothold planners can
    avoid placing the sole too close to the edge.
    """

    step_x_m: float = 0.5
    lower_height_m: float = 0.0
    upper_height_m: float = 0.2
    edge_half_width_m: float = 0.05
    confidence: float = 1.0
    support_margin: float = 1.0
    edge_risk_at_boundary: float = 1.0

    def query(self, points_xy: torch.Tensor) -> TerrainQueryResult:
        x = points_xy[:, 0]
        device = points_xy.device
        dtype = points_xy.dtype

        height = torch.where(
            x >= self.step_x_m,
            torch.full_like(x, self.upper_height_m),
            torch.full_like(x, self.lower_height_m),
        )

        distance_to_edge = torch.abs(x - self.step_x_m)
        edge_risk = torch.clamp(
            1.0 - distance_to_edge / self.edge_half_width_m,
            min=0.0,
            max=1.0,
        ) * self.edge_risk_at_boundary

        confidence = torch.full(
            x.shape,
            self.confidence,
            device=device,
            dtype=dtype,
        )
        support_margin = torch.full(
            x.shape,
            self.support_margin,
            device=device,
            dtype=dtype,
        )

        return TerrainQueryResult(
            height=height,
            confidence=confidence,
            support_margin=support_margin,
            edge_risk=edge_risk,
        )