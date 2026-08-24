from dataclasses import dataclass

import torch


def _rotate_vector_by_quaternion(
    quaternion_w: torch.Tensor,
    vector_b: torch.Tensor,
) -> torch.Tensor:
    quaternion_xyz = quaternion_w[:, 1:]
    quaternion_scalar = quaternion_w[:, :1]

    cross_product = 2.0 * torch.cross(
        quaternion_xyz,
        vector_b,
        dim=-1,
    )
    return (
        vector_b
        + quaternion_scalar * cross_product
        + torch.cross(quaternion_xyz, cross_product, dim=-1)
    )


@dataclass(frozen=True)
class SoleGeometry:
    center_offset_b: torch.Tensor
    half_length: float
    half_width: float

    def center_world(
        self,
        ankle_pos_w: torch.Tensor,
        ankle_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        center_offset_b = self.center_offset_b.to(
            device=ankle_pos_w.device,
            dtype=ankle_pos_w.dtype,
        )
        center_offset_b = center_offset_b.unsqueeze(0).expand_as(ankle_pos_w)

        center_offset_w = _rotate_vector_by_quaternion(
            ankle_quat_w,
            center_offset_b,
        )
        return ankle_pos_w + center_offset_w

    def center_velocity_world(
        self,
        ankle_linear_vel_w: torch.Tensor,
        ankle_angular_vel_w: torch.Tensor,
        ankle_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return the sole-center world velocity from rigid-body kinematics."""
        center_offset_b = self.center_offset_b.to(
            device=ankle_linear_vel_w.device,
            dtype=ankle_linear_vel_w.dtype,
        )
        center_offset_b = center_offset_b.unsqueeze(0).expand_as(ankle_linear_vel_w)
        center_offset_w = _rotate_vector_by_quaternion(
            ankle_quat_w,
            center_offset_b,
        )
        return ankle_linear_vel_w + torch.cross(
            ankle_angular_vel_w,
            center_offset_w,
            dim=-1,
        )

    def corners_world(
        self,
        ankle_pos_w: torch.Tensor,
        ankle_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        corner_offsets = ankle_pos_w.new_tensor(
            [
                [self.half_length, self.half_width, 0.0],
                [self.half_length, -self.half_width, 0.0],
                [-self.half_length, -self.half_width, 0.0],
                [-self.half_length, self.half_width, 0.0],
            ]
        )
        center_offset_b = self.center_offset_b.to(
            device=ankle_pos_w.device,
            dtype=ankle_pos_w.dtype,
        )
        corners_b = corner_offsets + center_offset_b
        corners_b = corners_b.unsqueeze(0).expand(ankle_pos_w.shape[0], -1, -1)

        num_envs, num_corners, _ = corners_b.shape
        ankle_quat_expanded = ankle_quat_w[:, None, :].expand(-1, num_corners, -1)
        corners_w = _rotate_vector_by_quaternion(
            ankle_quat_expanded.reshape(-1, 4),
            corners_b.reshape(-1, 3),
        ).reshape(num_envs, num_corners, 3)
        return corners_w + ankle_pos_w[:, None, :]


@dataclass(frozen=True)
class FrozenFrame:
    origin_w: torch.Tensor
    cos_yaw: torch.Tensor
    sin_yaw: torch.Tensor


def make_frozen_stance_frame(
    origin_w: torch.Tensor,
    yaw_w: torch.Tensor,
) -> FrozenFrame:
    return FrozenFrame(
        origin_w=origin_w.clone(),
        cos_yaw=torch.cos(yaw_w),
        sin_yaw=torch.sin(yaw_w),
    )


def world_to_frozen(
    points_w: torch.Tensor,
    frame: FrozenFrame,
) -> torch.Tensor:
    delta_w = points_w - frame.origin_w

    x_f = frame.cos_yaw * delta_w[:, 0] + frame.sin_yaw * delta_w[:, 1]
    y_f = -frame.sin_yaw * delta_w[:, 0] + frame.cos_yaw * delta_w[:, 1]
    z_f = delta_w[:, 2]

    return torch.stack((x_f, y_f, z_f), dim=-1)


def frozen_to_world(
    points_f: torch.Tensor,
    frame: FrozenFrame,
) -> torch.Tensor:
    x_w = frame.cos_yaw * points_f[:, 0] - frame.sin_yaw * points_f[:, 1]
    y_w = frame.sin_yaw * points_f[:, 0] + frame.cos_yaw * points_f[:, 1]
    z_w = points_f[:, 2]

    points_w = torch.stack((x_w, y_w, z_w), dim=-1)
    return points_w + frame.origin_w
