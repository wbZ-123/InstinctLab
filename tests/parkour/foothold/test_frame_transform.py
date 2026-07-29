import math

import torch

from instinctlab_foothold import (
    apply_world_height_to_planner_target,
    planner_frame_to_world_xy,
)


def test_planner_frame_to_world_xy_uses_origin_and_yaw():
    origin_w = torch.tensor([[1.0, 2.0, 0.3]])
    target_xy_f = torch.tensor([[0.2, 0.0]])
    yaw_w = torch.tensor([math.pi / 2.0])

    xy_w = planner_frame_to_world_xy(origin_w, target_xy_f, yaw_w)

    torch.testing.assert_close(xy_w, torch.tensor([[1.0, 2.2]]), atol=1.0e-6, rtol=0.0)


def test_apply_world_height_queries_world_xy_and_returns_local_z():
    queried = []

    def terrain_query(points_xy_w: torch.Tensor):
        queried.append(points_xy_w.clone())
        return torch.tensor([0.75]), torch.tensor([True])

    origin_w = torch.tensor([[1.0, 2.0, 0.25]])
    target_xy_f = torch.tensor([[0.2, 0.0]])
    yaw_w = torch.tensor([math.pi / 2.0])

    target_f, target_w, valid = apply_world_height_to_planner_target(
        origin_w=origin_w,
        target_xy_f=target_xy_f,
        yaw_w=yaw_w,
        terrain_height_query_w=terrain_query,
    )

    torch.testing.assert_close(queried[0], torch.tensor([[1.0, 2.2]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(target_w, torch.tensor([[1.0, 2.2, 0.75]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(target_f, torch.tensor([[0.2, 0.0, 0.5]]), atol=1.0e-6, rtol=0.0)
    assert valid.tolist() == [True]
