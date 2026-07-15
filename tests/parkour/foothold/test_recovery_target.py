from __future__ import annotations

import torch

from instinctlab_foothold.recovery_target import make_recovery_foothold_target


def test_recovery_target_keeps_normal_step_width_and_small_forward_step():
    swing_side = torch.tensor([0, 1])

    target = make_recovery_foothold_target(
        swing_side=swing_side,
        desired_velocity_f=torch.zeros(2, 3),
        step_length_m=0.04,
        velocity_lookahead_s=0.0,
        max_step_length_m=0.12,
        step_width_m=0.18,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        target,
        torch.tensor(
            [
                [0.04, 0.18, 0.0],
                [0.04, -0.18, 0.0],
            ]
        ),
    )


def test_recovery_target_extends_forward_step_from_positive_velocity():
    target = make_recovery_foothold_target(
        swing_side=torch.tensor([0, 1]),
        desired_velocity_f=torch.tensor(
            [
                [0.6, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ]
        ),
        step_length_m=0.04,
        velocity_lookahead_s=0.10,
        max_step_length_m=0.12,
        step_width_m=0.18,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        target,
        torch.tensor(
            [
                [0.10, 0.18, 0.0],
                [0.04, -0.18, 0.0],
            ]
        ),
    )


def test_recovery_target_clamps_velocity_extension_to_reachable_limit():
    target = make_recovery_foothold_target(
        swing_side=torch.tensor([0]),
        desired_velocity_f=torch.tensor([[2.0, 0.0, 0.0]]),
        step_length_m=0.04,
        velocity_lookahead_s=0.20,
        max_step_length_m=0.12,
        step_width_m=0.18,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        target,
        torch.tensor([[0.12, 0.18, 0.0]]),
    )
