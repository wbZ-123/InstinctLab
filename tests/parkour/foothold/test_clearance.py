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


class FakeNonFiniteObstacle:
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        return torch.full_like(points, float("nan"))


class GoalSolePointObstacle:
    def __init__(self, max_penetrating_y: float):
        self.max_penetrating_y = max_penetrating_y

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        offset = torch.zeros_like(points)
        at_goal = points[:, 0] >= 0.399
        selected = at_goal & (points[:, 1] <= self.max_penetrating_y)
        offset[selected, 0] = 0.01
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


def test_swing_clearance_checks_sole_perimeter_not_only_centerline():
    start = torch.tensor([[0.0, 0.0, 0.10]])
    goal = torch.tensor([[0.40, 0.0, 0.10]])
    obstacle = FakeVerticalCylinderObstacle(
        center_xy=(0.20, 0.04),
        z_min=0.0,
        z_max=0.20,
        radius=0.02,
    )

    center_only = check_swing_centerline_penetration(
        obstacle=obstacle,
        start=start,
        goal=goal,
        apex_height=torch.tensor([0.0]),
        swing_duration_s=0.8,
        sample_spacing=0.05,
    )
    sole_sweep = check_swing_centerline_penetration(
        obstacle=obstacle,
        start=start,
        goal=goal,
        apex_height=torch.tensor([0.0]),
        swing_duration_s=0.8,
        sample_spacing=0.05,
        foot_points_xy=torch.tensor([[0.0, 0.04]]),
        foot_yaw_w=torch.tensor([0.0]),
    )

    assert center_only.collides.tolist() == [False]
    assert sole_sweep.collides.tolist() == [True]


def test_swing_clearance_allows_two_goal_points_but_not_three():
    common = dict(
        start=torch.tensor([[0.0, 0.0, 0.10]]),
        goal=torch.tensor([[0.40, 0.0, 0.10]]),
        apex_height=torch.tensor([0.20]),
        swing_duration_s=0.8,
        foot_points_xy=torch.tensor([[0.0, 0.0], [0.0, 0.02], [0.0, 0.04]]),
        foot_yaw_w=torch.tensor([0.0]),
        goal_max_penetrating_points=2,
    )

    two_points = check_swing_centerline_penetration(
        obstacle=GoalSolePointObstacle(0.02),
        **common,
    )
    three_points = check_swing_centerline_penetration(
        obstacle=GoalSolePointObstacle(0.04),
        **common,
    )

    assert two_points.goal_penetrating_point_count.tolist() == [2]
    assert two_points.collides.tolist() == [False]
    assert three_points.goal_penetrating_point_count.tolist() == [3]
    assert three_points.collides.tolist() == [True]


def test_swing_clearance_treats_nonfinite_penetration_as_unsafe():
    result = check_swing_centerline_penetration(
        obstacle=FakeNonFiniteObstacle(),
        start=torch.tensor([[0.0, 0.0, 0.10]]),
        goal=torch.tensor([[0.40, 0.0, 0.10]]),
        apex_height=torch.tensor([0.0]),
        swing_duration_s=0.8,
    )

    assert result.collides.tolist() == [True]
    assert torch.isinf(result.max_penetration_depth).tolist() == [True]


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


def test_clearance_allows_an_existing_start_overlap_when_the_foot_exits():
    start = torch.tensor([[0.20, 0.0, 0.10]])
    goal = torch.tensor([[0.50, 0.0, 0.10]])
    obstacle = FakeVerticalCylinderObstacle(
        center_xy=(0.20, 0.0),
        z_min=0.0,
        z_max=0.20,
        radius=0.08,
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
        allow_start_penetration_escape=True,
    )

    assert result.penetration.start_penetration_depth.item() > 0.0
    assert result.penetration.goal_penetration_depth.item() == 0.0
    assert result.is_safe.tolist() == [True]


def test_clearance_does_not_hide_a_penetrating_goal():
    start = torch.tensor([[0.0, 0.0, 0.10]])
    goal = torch.tensor([[0.20, 0.0, 0.10]])
    obstacle = FakeVerticalCylinderObstacle(
        center_xy=(0.20, 0.0),
        z_min=0.0,
        z_max=0.20,
        radius=0.08,
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
        allow_start_penetration_escape=True,
    )

    assert result.penetration.goal_penetration_depth.item() > 0.0
    assert result.is_safe.tolist() == [False]
