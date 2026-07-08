import torch

from instinctlab_foothold.flat_provider import FlatTargetBatch, TerrainCorridor
from instinctlab_foothold.terrain_provider import lift_flat_targets_to_terrain
from instinctlab_foothold.terrain_query import StepTerrainQuery


def _make_flat_targets() -> FlatTargetBatch:
    return FlatTargetBatch(
        position_f=torch.tensor(
            [
                [0.25, 0.10, 0.0],
                [0.75, -0.10, 0.0],
            ],
            dtype=torch.float32,
        ),
        yaw_f=torch.tensor([0.1, -0.2], dtype=torch.float32),
        normal_f=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        feasible_velocity_f=torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [0.4, 0.1, 0.0],
            ],
            dtype=torch.float32,
        ),
        valid=torch.tensor([True, True]),
        terrain=TerrainCorridor(
            heights=torch.zeros((2, 8), dtype=torch.float32),
            confidences=torch.ones((2, 8), dtype=torch.float32),
            support_margin=torch.ones(2, dtype=torch.float32),
            edge_risk=torch.zeros(2, dtype=torch.float32),
            unknown_fraction=torch.zeros(2, dtype=torch.float32),
        ),
    )


def test_lift_flat_targets_sets_z_from_terrain_without_changing_xy():
    flat_targets = _make_flat_targets()
    terrain_query = StepTerrainQuery(
        step_x_m=0.5,
        lower_height_m=0.0,
        upper_height_m=0.2,
    )

    result = lift_flat_targets_to_terrain(flat_targets, terrain_query)

    torch.testing.assert_close(result.position_f[:, :2], flat_targets.position_f[:, :2])
    torch.testing.assert_close(result.position_f[:, 2], torch.tensor([0.0, 0.2]))
    torch.testing.assert_close(result.yaw_f, flat_targets.yaw_f)
    torch.testing.assert_close(result.normal_f, flat_targets.normal_f)
    torch.testing.assert_close(result.feasible_velocity_f, flat_targets.feasible_velocity_f)
    torch.testing.assert_close(result.valid, flat_targets.valid)


def test_lift_flat_targets_records_terrain_query_fields():
    flat_targets = _make_flat_targets()
    terrain_query = StepTerrainQuery(
        step_x_m=0.5,
        lower_height_m=0.0,
        upper_height_m=0.2,
        confidence=0.75,
        support_margin=0.6,
    )

    result = lift_flat_targets_to_terrain(flat_targets, terrain_query)

    torch.testing.assert_close(result.terrain.heights[:, 0], torch.tensor([0.0, 0.2]))
    torch.testing.assert_close(result.terrain.confidences[:, 0], torch.tensor([0.75, 0.75]))
    torch.testing.assert_close(result.terrain.support_margin, torch.tensor([0.6, 0.6]))
    torch.testing.assert_close(result.terrain.edge_risk, torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(result.terrain.unknown_fraction, torch.tensor([0.25, 0.25]))