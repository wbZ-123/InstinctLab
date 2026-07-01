import torch

from instinctlab_foothold.geometry import (
    SoleGeometry,
    frozen_to_world,
    make_frozen_stance_frame,
    world_to_frozen,
)


def test_frozen_stance_frame_round_trip():
    origin_w = torch.tensor([[1.0, 2.0, 0.3]])
    yaw_w = torch.tensor([torch.pi / 2])
    point_w = torch.tensor([[1.2, 2.1, 0.35]])

    frame = make_frozen_stance_frame(origin_w, yaw_w)
    point_f = world_to_frozen(point_w, frame)
    restored_w = frozen_to_world(point_f, frame)

    torch.testing.assert_close(restored_w, point_w)


def test_sole_center_uses_ankle_offset():
    geometry = SoleGeometry(
        center_offset_b=torch.tensor([0.02, 0.0, -0.058]),
        half_length=0.12,
        half_width=0.055,
    )
    ankle_pos_w = torch.zeros((1, 3))
    ankle_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    sole_center_w = geometry.center_world(ankle_pos_w, ankle_quat_w)

    torch.testing.assert_close(
        sole_center_w,
        torch.tensor([[0.02, 0.0, -0.058]]),
    )


def test_sole_center_offset_rotates_with_ankle():
    geometry = SoleGeometry(
        center_offset_b=torch.tensor([0.02, 0.0, -0.058]),
        half_length=0.12,
        half_width=0.055,
    )
    ankle_pos_w = torch.zeros((1, 3))
    half_sqrt_two = 2.0**-0.5
    ankle_quat_w = torch.tensor([[half_sqrt_two, 0.0, 0.0, half_sqrt_two]])

    sole_center_w = geometry.center_world(ankle_pos_w, ankle_quat_w)

    torch.testing.assert_close(
        sole_center_w,
        torch.tensor([[0.0, 0.02, -0.058]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_sole_corners_form_expected_rectangle():
    geometry = SoleGeometry(
        center_offset_b=torch.tensor([0.02, 0.0, -0.058]),
        half_length=0.12,
        half_width=0.055,
    )
    ankle_pos_w = torch.zeros((1, 3))
    ankle_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    corners_w = geometry.corners_world(ankle_pos_w, ankle_quat_w)

    expected_w = torch.tensor(
        [
            [
                [0.14, 0.055, -0.058],
                [0.14, -0.055, -0.058],
                [-0.10, -0.055, -0.058],
                [-0.10, 0.055, -0.058],
            ]
        ]
    )
    torch.testing.assert_close(corners_w, expected_w)
