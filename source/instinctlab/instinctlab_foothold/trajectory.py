from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SwingReference:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor


def quintic_swing_reference(
    start: torch.Tensor,
    goal: torch.Tensor,
    phase: torch.Tensor,
    apex_height: torch.Tensor,
    swing_duration_s: torch.Tensor | float,
) -> SwingReference:
    def _quintic(
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        first = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        second = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

        return blend, first, second
    duration = torch.as_tensor(
        swing_duration_s,
        device=start.device,
        dtype=start.dtype,
    )
    if duration.ndim == 0:
        duration = duration.expand(start.shape[0])

    if torch.any(duration <= 0.0).item():
        raise ValueError("swing_duration_s must be positive.")

    u = torch.clamp(phase, min=0.0, max=1.0)
    blend, first, second = _quintic(u)

    delta_xy = goal[:, :2] - start[:, :2]
    position_xy = start[:, :2] + delta_xy * blend.unsqueeze(-1)
    velocity_xy = delta_xy * (first / duration).unsqueeze(-1)
    acceleration_xy = delta_xy * (second / duration.square()).unsqueeze(-1)

    first_half = u <= 0.5
    local_u = torch.where(first_half, 2.0 * u, 2.0 * u - 1.0)
    z_blend, z_first, z_second = _quintic(local_u)

    apex_z = torch.maximum(start[:, 2], goal[:, 2]) + apex_height
    segment_start_z = torch.where(first_half, start[:, 2], apex_z)
    segment_goal_z = torch.where(first_half, apex_z, goal[:, 2])
    delta_z = segment_goal_z - segment_start_z

    position_z = segment_start_z + delta_z * z_blend
    velocity_z = delta_z * z_first * 2.0 / duration
    acceleration_z = delta_z * z_second * 4.0 / duration.square()

    return SwingReference(
        position=torch.cat(
            (position_xy, position_z.unsqueeze(-1)),
            dim=-1,
        ),
        velocity=torch.cat(
            (velocity_xy, velocity_z.unsqueeze(-1)),
            dim=-1,
        ),
        acceleration=torch.cat(
            (acceleration_xy, acceleration_z.unsqueeze(-1)),
            dim=-1,
        ),
    )