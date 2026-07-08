import torch

from instinctlab_foothold.clearance import (
    adjust_apex_for_edge_clearance,
    check_swing_centerline_penetration,
    sample_swing_centerline,
)


class FakeVerticalCylinderObstacle:
    def __init__(
        self,
        center_xy: tuple[float, float],
        z_min: float,
        z_max: float,
        radius: float,
    ):
        self.center_xy = torch.tensor(center_xy, dtype=torch.float32)
        self.z_min = z_min
        self.z_max = z_max
        self.radius = radius

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        center_xy = self.center_xy.to(device=points.device, dtype=points.dtype)
        delta_xy = points[:, :2] - center_xy
        radial_distance = torch.linalg.norm(delta_xy, dim=-1)
        inside_xy = radial_distance < self.radius
        inside_z = (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        inside = inside_xy & inside_z

        penetration_depth = self.radius - radial_distance
        direction_xy = torch.zeros_like(delta_xy)
        nonzero = radial_distance > 1.0e-8
        direction_xy[nonzero] = delta_xy[nonzero] / radial_distance[nonzero].unsqueeze(-1)
        direction_xy[~nonzero, 0] = 1.0

        offset = torch.zeros_like(points)
        offset[inside, :2] = direction_xy[inside] * penetration_depth[inside].unsqueeze(-1)
        return offset


def test_sample_swing_centerline_uses_spacing_with_bounds():
    start = torch.tensor([[0.0, 0.0, 0.0]])
    goal = torch.tensor([[0.30, 0.0, 0.0]])

    path, phases = sample_swing_centerline(
        start=start,
        goal=goal,
        apex_height=torch.tensor([0.08]),
        swing_duration_s=0.8,
        sample_spacing=0.03,
        min_samples=9,
        max_samples=25,
    )

    assert path.shape == (1, 11, 3)
    assert phases.shape == (11,)
    torch.testing.assert_close(path[:, 0], start)
    torch.testing.assert_close(path[:, -1], goal)


def test_swing_centerline_penetration_is_false_when_obstacle_is_far_from_path():
    start = torch.tensor([[0.0, 0.0, 0.10]])
    goal = torch.tensor([[0.40, 0.0, 0.10]])
    obstacle = FakeVerticalCylinderObstacle(
        center_xy=(0.20, 0.30),
        z_min=0.0,
        z_max=0.20,
        radius=0.05,
    )

    result = check_swing_centerline_penetration(
        obstacle=obstacle,
        start=start,
        goal=goal,
        apex_height=torch.tensor([0.0]),
        swing_duration_s=0.8,
        sample_spacing=0.03,
    )

    assert result.collides.tolist() == [False]
    torch.testing.assert_close(result.max_penetration_depth, torch.tensor([0.0]))


def test_swing_centerline_penetration_detects_centerline_entering_edge_cylinder():
    start = torch.tensor([[0.0, 0.0, 0.10]])
    goal = torch.tensor([[0.40, 0.0, 0.10]])
    obstacle = FakeVerticalCylinderObstacle(
        center_xy=(0.20, 0.0),
        z_min=0.0,
        z_max=0.20,
        radius=0.05,
    )

    result = check_swing_centerline_penetration(
        obstacle=obstacle,
        start=start,
        goal=goal,
        apex_height=torch.tensor([0.0]),
        swing_duration_s=0.8,
        sample_spacing=0.03,
    )

    assert result.collides.tolist() == [True]
    assert result.max_penetration_depth.item() > 0.0
    torch.testing.assert_close(result.deepest_phase, torch.tensor([0.5]))


def test_swing_centerline_penetration_can_be_removed_by_higher_apex():
    start = torch.tensor([[0.0, 0.0, 0.10]])
    goal = torch.tensor([[0.40, 0.0, 0.10]])
    obstacle = FakeVerticalCylinderObstacle(
        center_xy=(0.20, 0.0),
        z_min=0.0,
        z_max=0.20,
        radius=0.05,
    )

    result = check_swing_centerline_penetration(
        obstacle=obstacle,
        start=start,
        goal=goal,
        apex_height=torch.tensor([0.20]),
        swing_duration_s=0.8,
        sample_spacing=0.03,
    )

    assert result.collides.tolist() == [False]
    torch.testing.assert_close(result.max_penetration_depth, torch.tensor([0.0]))

def test_adjust_apex_for_edge_clearance_increases_until_centerline_is_safe():
    start = torch.tensor([[0.0, 0.0, 0.10]])
    goal = torch.tensor([[0.40, 0.0, 0.10]])
    obstacle = FakeVerticalCylinderObstacle(
        center_xy=(0.20, 0.0),
        z_min=0.0,
        z_max=0.20,
        radius=0.05,
    )

    result = adjust_apex_for_edge_clearance(
        obstacle=obstacle,
        start=start,
        goal=goal,
        default_apex_height=torch.tensor([0.0]),
        swing_duration_s=0.8,
        apex_step=0.05,
        max_apex_height=torch.tensor([0.30]),
        sample_spacing=0.03,
    )

    assert result.is_safe.tolist() == [True]
    assert result.apex_height.item() > 0.0
    assert result.apex_height.item() <= 0.30
    torch.testing.assert_close(
        result.penetration.max_penetration_depth,
        torch.tensor([0.0]),
    )


def test_adjust_apex_for_edge_clearance_reports_unsafe_at_max_apex():
    start = torch.tensor([[0.0, 0.0, 0.10]])
    goal = torch.tensor([[0.40, 0.0, 0.10]])
    obstacle = FakeVerticalCylinderObstacle(
        center_xy=(0.20, 0.0),
        z_min=0.0,
        z_max=1.00,
        radius=0.05,
    )

    result = adjust_apex_for_edge_clearance(
        obstacle=obstacle,
        start=start,
        goal=goal,
        default_apex_height=torch.tensor([0.0]),
        swing_duration_s=0.8,
        apex_step=0.05,
        max_apex_height=torch.tensor([0.20]),
        sample_spacing=0.03,
    )

    assert result.is_safe.tolist() == [False]
    torch.testing.assert_close(result.apex_height, torch.tensor([0.20]))
    assert result.penetration.max_penetration_depth.item() > 0.0
