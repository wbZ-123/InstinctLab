import torch

from instinctlab_foothold.terrain_query import (
    FlatTerrainQuery,
    StepTerrainQuery,
)


def test_flat_terrain_query_returns_constant_fields():
    query = FlatTerrainQuery(
        height_m=0.15,
        confidence=0.8,
        support_margin=0.9,
        edge_risk=0.1,
    )

    points_xy = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -0.5],
            [-1.0, 0.5],
        ]
    )

    result = query.query(points_xy)

    torch.testing.assert_close(
        result.height,
        torch.tensor([0.15, 0.15, 0.15]),
    )
    torch.testing.assert_close(
        result.confidence,
        torch.tensor([0.8, 0.8, 0.8]),
    )
    torch.testing.assert_close(
        result.support_margin,
        torch.tensor([0.9, 0.9, 0.9]),
    )
    torch.testing.assert_close(
        result.edge_risk,
        torch.tensor([0.1, 0.1, 0.1]),
    )


def test_step_terrain_query_changes_height_across_step():
    query = StepTerrainQuery(
        step_x_m=0.5,
        lower_height_m=0.0,
        upper_height_m=0.2,
        edge_half_width_m=0.05,
    )

    points_xy = torch.tensor(
        [
            [0.0, 0.0],
            [0.49, 0.0],
            [0.50, 0.0],
            [0.75, 0.0],
        ]
    )

    result = query.query(points_xy)

    torch.testing.assert_close(
        result.height,
        torch.tensor([0.0, 0.0, 0.2, 0.2]),
    )


def test_step_terrain_query_reports_edge_risk_near_step_boundary():
    query = StepTerrainQuery(
        step_x_m=0.5,
        edge_half_width_m=0.1,
    )

    points_xy = torch.tensor(
        [
            [0.50, 0.0],
            [0.55, 0.0],
            [0.60, 0.0],
            [0.80, 0.0],
        ]
    )

    result = query.query(points_xy)

    torch.testing.assert_close(
        result.edge_risk,
        torch.tensor([1.0, 0.5, 0.0, 0.0]),
    )


def test_step_terrain_query_preserves_device_and_dtype():
    query = StepTerrainQuery()
    points_xy = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    result = query.query(points_xy)

    assert result.height.dtype == torch.float64
    assert result.height.device == points_xy.device