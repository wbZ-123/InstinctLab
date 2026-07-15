import torch

from instinctlab_foothold.target_search import (
    debug_safe_foothold_candidates,
    make_sole_perimeter_points_xy,
    search_safe_foothold_target,
)


class FakeCylinderObstacle:
    def __init__(self, centers_xy: torch.Tensor, radius: float):
        if centers_xy.ndim == 1:
            centers_xy = centers_xy[None, :]
        self.centers_xy = centers_xy
        self.radius = radius

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        # Match the real edge-cylinder obstacle API: points must be flat (N, 3).
        assert points.ndim == 2
        assert points.shape[1] == 3

        delta = points[:, None, :2] - self.centers_xy
        distance_xy = torch.linalg.norm(delta, dim=-1)
        penetration = self.radius - distance_xy
        max_penetration, nearest_idx = torch.max(penetration, dim=-1)

        offset = torch.zeros_like(points)
        penetrated = max_penetration > 0.0
        if torch.any(penetrated):
            nearest_center = self.centers_xy[nearest_idx[penetrated]]
            direction_xy = points[penetrated, :2] - nearest_center
            direction_norm = torch.linalg.norm(direction_xy, dim=-1, keepdim=True)
            safe_direction_xy = torch.where(
                direction_norm > 1.0e-6,
                direction_xy / direction_norm.clamp_min(1.0e-6),
                torch.tensor(
                    [1.0, 0.0],
                    device=points.device,
                    dtype=points.dtype,
                ).expand_as(direction_xy),
            )
            offset[penetrated, :2] = (
                safe_direction_xy * max_penetration[penetrated, None]
            )

        return offset


def test_returns_nominal_target_when_sole_points_are_safe():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    desired_velocity_f = torch.tensor([[0.5, 0.0, 0.0]])

    # 先用很小的鞋底点阵，后面再换成 26 个外圈点
    foot_points_xy = torch.tensor(
        [
            [0.00, 0.00],
            [0.05, 0.02],
            [0.05, -0.02],
            [-0.05, 0.02],
            [-0.05, -0.02],
        ]
    )

    # 障碍放远一点，确保 nominal 落足点安全
    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([1.0, 1.0]),
        radius=0.05,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.025, 0.05, 0.075, 0.10]),
        candidate_directions=torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        ),
        safety_margin=0.0,
    )

    assert result.valid
    assert not result.used_fallback
    torch.testing.assert_close(result.target_f, nominal_target_f)
    assert result.nominal_inside_ellipse
    assert result.nominal_obstacle_safe
    assert result.nominal_valid
    torch.testing.assert_close(result.candidate_count, torch.tensor([0.0]))
    torch.testing.assert_close(
        result.candidate_valid_count, torch.tensor([0.0])
    )

def test_selects_safe_candidate_when_nominal_target_is_unsafe():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    desired_velocity_f = torch.tensor([[0.5, 0.0, 0.0]])

    # 为了测试简单，这里先只用脚掌中心点。
    # 后面再加 26 个鞋底点阵测试。
    foot_points_xy = torch.tensor([[0.0, 0.0]])

    # 圆柱正好覆盖 nominal_target_f=(0.10, 0.00)
    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([0.10, 0.00]),
        radius=0.04,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor(
            [
                [1.0, 0.0],   # 往前：0.15, 0.00，安全
                [-1.0, 0.0],  # 往后：0.05, 0.00，也安全
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        ),
        safety_margin=0.0,
    )

    assert result.valid
    assert result.used_fallback
    assert result.nominal_inside_ellipse
    assert not result.nominal_obstacle_safe
    assert not result.nominal_valid
    torch.testing.assert_close(result.candidate_count, torch.tensor([4.0]))
    torch.testing.assert_close(
        result.candidate_inside_ellipse_count, torch.tensor([4.0])
    )
    torch.testing.assert_close(
        result.candidate_obstacle_safe_count, torch.tensor([4.0])
    )
    torch.testing.assert_close(
        result.candidate_valid_count, torch.tensor([4.0])
    )

    # 我们希望优先沿速度方向往前找，所以应该选 0.15, 0.00
    expected = torch.tensor([[0.15, 0.00, 0.0]])
    torch.testing.assert_close(result.target_f, expected)


def test_queries_obstacle_in_world_frame_while_returning_target_in_foot_frame():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    target_origin_w = torch.tensor([[10.0, 20.0, 0.0]])
    desired_velocity_f = torch.tensor([[0.5, 0.0, 0.0]])
    foot_points_xy = torch.tensor([[0.0, 0.0]])

    # The obstacle lives in world frame.  It blocks world x=10.10 but not
    # local x=0.10.  This catches accidental foot-frame obstacle queries.
    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([10.10, 20.00]),
        radius=0.03,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        target_origin_w=target_origin_w,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor([[1.0, 0.0]]),
        safety_margin=0.0,
    )

    assert result.valid
    assert result.used_fallback
    assert result.nominal_inside_ellipse
    assert not result.nominal_obstacle_safe
    expected_target_f = torch.tensor([[0.15, 0.00, 0.0]])
    torch.testing.assert_close(result.target_f, expected_target_f)


def test_queries_obstacle_after_rotating_target_frame_by_yaw():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    target_origin_w = torch.tensor([[10.0, 20.0, 0.0]])
    target_yaw_w = torch.tensor([torch.pi / 2.0])
    desired_velocity_f = torch.tensor([[0.5, 0.0, 0.0]])
    foot_points_xy = torch.tensor([[0.0, 0.0]])

    # The nominal body-frame target is +x.  With yaw=90deg it lands at
    # world +y, i.e. (10.0, 20.1), and must be considered blocked.
    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([10.00, 20.10]),
        radius=0.03,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        target_origin_w=target_origin_w,
        target_yaw_w=target_yaw_w,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor([[1.0, 0.0]]),
        safety_margin=0.0,
    )

    assert result.valid
    assert result.used_fallback
    assert not result.nominal_obstacle_safe
    torch.testing.assert_close(result.target_f, torch.tensor([[0.15, 0.0, 0.0]]))


def test_debug_safe_foothold_candidates_returns_candidate_masks():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    foot_points_xy = torch.tensor([[0.0, 0.0]])
    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor(
            [
                [0.10, 0.00],
                [0.15, 0.00],
            ]
        ),
        radius=0.03,
    )

    debug = debug_safe_foothold_candidates(
        nominal_target_f=nominal_target_f,
        support_foot_f=support_foot_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
            ]
        ),
        safety_margin=0.0,
    )

    torch.testing.assert_close(
        debug.candidates_f,
        torch.tensor([[[0.15, 0.00, 0.0], [0.05, 0.00, 0.0]]]),
    )
    assert debug.nominal_inside_ellipse
    assert not debug.nominal_obstacle_safe
    assert not debug.nominal_valid
    torch.testing.assert_close(
        debug.candidate_inside_ellipse, torch.tensor([[True, True]])
    )
    torch.testing.assert_close(
        debug.candidate_obstacle_safe, torch.tensor([[False, True]])
    )
    torch.testing.assert_close(
        debug.candidate_valid, torch.tensor([[False, True]])
    )

def test_candidate_counts_only_apply_when_nominal_target_needs_fallback():
    nominal_target_f = torch.tensor(
        [
            [0.10, 0.00, 0.0],
            [0.20, 0.00, 0.0],
        ]
    )
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.zeros_like(nominal_target_f)
    desired_velocity_f = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    foot_points_xy = torch.tensor([[0.0, 0.0]])
    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([0.20, 0.00]),
        radius=0.03,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
            ]
        ),
        safety_margin=0.0,
    )

    torch.testing.assert_close(
        result.nominal_valid, torch.tensor([True, False])
    )
    torch.testing.assert_close(
        result.candidate_count, torch.tensor([0.0, 2.0])
    )
    torch.testing.assert_close(
        result.candidate_valid_count, torch.tensor([0.0, 2.0])
    )

def test_rejects_safe_candidates_outside_reachable_ellipse():
    nominal_target_f = torch.tensor([[0.28, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    desired_velocity_f = torch.tensor([[0.5, 0.0, 0.0]])
    foot_points_xy = torch.tensor([[0.0, 0.0]])

    # nominal 被障碍覆盖，所以必须找 fallback
    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([0.28, 0.00]),
        radius=0.04,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        # 往前会到 x=0.33，安全但是椭圆外；应该被拒绝
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor([[1.0, 0.0]]),
        safety_margin=0.0,
    )

    assert not result.valid
    assert not result.used_fallback
    torch.testing.assert_close(result.target_f, nominal_target_f)

def test_rejects_candidate_when_any_sole_point_enters_obstacle():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    desired_velocity_f = torch.tensor([[0.5, 0.0, 0.0]])

    # 这里模拟鞋底有一个前缘点，中心在候选点处安全，
    # 但前缘点会进入危险圆柱。
    foot_points_xy = torch.tensor(
        [
            [0.00, 0.00],
            [0.05, 0.00],
        ]
    )

    # nominal 被挡住，迫使它找 fallback。
    # fallback 往前会到 center=(0.15, 0.00)。
    # center 距离 obstacle center=(0.20, 0.00) 是 0.05，半径 0.02，所以中心安全；
    # 但前缘点在 (0.20, 0.00)，正中障碍，所以候选必须无效。
    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor(
            [
                [0.10, 0.00],  # 挡住 nominal
                [0.20, 0.00],  # 挡住 fallback 的前缘点
            ]
        ),
        radius=0.02,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor([[1.0, 0.0]]),
        safety_margin=0.0,
    )

    assert not result.valid
    assert not result.used_fallback
    torch.testing.assert_close(result.target_f, nominal_target_f)

def test_make_sole_perimeter_points_xy_returns_10_by_5_outer_ring():
    points = make_sole_perimeter_points_xy(
        foot_length=0.20,
        foot_width=0.10,
        num_x=10,
        num_y=5,
    )

    assert points.shape == (26, 2)

    x_min = torch.tensor(-0.10)
    x_max = torch.tensor(0.10)
    y_min = torch.tensor(-0.05)
    y_max = torch.tensor(0.05)

    on_outer_ring = (
        torch.isclose(points[:, 0], x_min)
        | torch.isclose(points[:, 0], x_max)
        | torch.isclose(points[:, 1], y_min)
        | torch.isclose(points[:, 1], y_max)
    )
    assert torch.all(on_outer_ring)

    # 四个角都必须存在
    expected_corners = torch.tensor(
        [
            [-0.10, -0.05],
            [-0.10, 0.05],
            [0.10, -0.05],
            [0.10, 0.05],
        ]
    )
    for corner in expected_corners:
        assert torch.any(torch.all(torch.isclose(points, corner), dim=1))

def test_selects_nearest_valid_candidate_to_nominal_target():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    desired_velocity_f = torch.tensor([[0.5, 0.0, 0.0]])
    foot_points_xy = torch.tensor([[0.0, 0.0]])

    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([0.10, 0.00]),
        radius=0.03,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        # 0.08 更远，0.05 更近；虽然 0.08 在前面，也不应该选它。
        candidate_radii=torch.tensor([0.08, 0.05]),
        candidate_directions=torch.tensor([[1.0, 0.0]]),
        safety_margin=0.0,
    )

    assert result.valid
    assert result.used_fallback
    expected = torch.tensor([[0.15, 0.00, 0.0]])
    torch.testing.assert_close(result.target_f, expected)
    torch.testing.assert_close(result.selected_score, torch.tensor([0.05]))


def test_breaks_equal_distance_ties_by_velocity_direction():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    desired_velocity_f = torch.tensor([[0.0, 0.5, 0.0]])
    foot_points_xy = torch.tensor([[0.0, 0.0]])

    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([0.10, 0.00]),
        radius=0.03,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor(
            [
                [1.0, 0.0],  # 前，和左一样近，但和速度方向不一致。
                [0.0, 1.0],  # 左，和速度方向一致。
            ]
        ),
        safety_margin=0.0,
    )

    assert result.valid
    assert result.used_fallback
    expected = torch.tensor([[0.10, 0.05, 0.0]])
    torch.testing.assert_close(result.target_f, expected)
    torch.testing.assert_close(result.selected_score, torch.tensor([0.05]))


def test_breaks_equal_distance_ties_by_candidate_direction_order_when_velocity_is_zero():
    nominal_target_f = torch.tensor([[0.10, 0.00, 0.0]])
    raw_target_f = nominal_target_f.clone()
    support_foot_f = torch.tensor([[0.0, 0.0, 0.0]])
    desired_velocity_f = torch.tensor([[0.0, 0.0, 0.0]])
    foot_points_xy = torch.tensor([[0.0, 0.0]])

    obstacle = FakeCylinderObstacle(
        centers_xy=torch.tensor([0.10, 0.00]),
        radius=0.03,
    )

    result = search_safe_foothold_target(
        nominal_target_f=nominal_target_f,
        raw_target_f=raw_target_f,
        support_foot_f=support_foot_f,
        desired_velocity_f=desired_velocity_f,
        obstacle=obstacle,
        ellipse_half_length=0.30,
        ellipse_half_width=0.16,
        foot_points_xy=foot_points_xy,
        candidate_radii=torch.tensor([0.05]),
        candidate_directions=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        safety_margin=0.0,
    )

    assert result.valid
    assert result.used_fallback
    expected = torch.tensor([[0.15, 0.00, 0.0]])
    torch.testing.assert_close(result.target_f, expected)
    torch.testing.assert_close(result.selected_score, torch.tensor([0.05]))
