import pytest
import torch

from instinctlab_foothold.trajectory import quintic_swing_reference


def test_reference_matches_endpoints_with_zero_derivatives():
    start = torch.tensor(
        [
            [0.0, 0.10, 0.02],
            [0.1, -0.10, 0.03],
        ]
    )
    goal = torch.tensor(
        [
            [0.3, 0.12, 0.00],
            [0.4, -0.08, 0.01],
        ]
    )
    phase = torch.tensor([0.0, 1.0])

    reference = quintic_swing_reference(
        start=start,
        goal=goal,
        phase=phase,
        apex_height=torch.tensor([0.10, 0.10]),
        swing_duration_s=0.32,
    )

    torch.testing.assert_close(
        reference.position,
        torch.stack((start[0], goal[1])),
    )
    torch.testing.assert_close(reference.velocity, torch.zeros_like(start))
    torch.testing.assert_close(
        reference.acceleration,
        torch.zeros_like(start),
    )



def test_reference_reaches_apex_at_half_phase():
    start = torch.tensor([[0.0, 0.10, 0.02]])
    goal = torch.tensor([[0.3, 0.12, 0.00]])

    reference = quintic_swing_reference(
        start,
        goal,
        phase=torch.tensor([0.5]),
        apex_height=torch.tensor([0.10]),
        swing_duration_s=0.32,
    )

    torch.testing.assert_close(
        reference.position[:, 2],
        torch.tensor([0.12]),
    )
    torch.testing.assert_close(
        reference.velocity[:, 2],
        torch.zeros(1),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        reference.acceleration[:, 2],
        torch.zeros(1),
        atol=1.0e-5,
        rtol=0.0,
    )



def test_duration_scales_velocity_and_acceleration():
    start = torch.tensor(
        [
            [0.0, 0.10, 0.0],
            [0.0, 0.10, 0.0],
        ]
    )
    goal = torch.tensor(
        [
            [0.3, 0.12, 0.0],
            [0.3, 0.12, 0.0],
        ]
    )

    reference = quintic_swing_reference(
        start,
        goal,
        phase=torch.tensor([0.25, 0.25]),
        apex_height=torch.tensor([0.10, 0.10]),
        swing_duration_s=torch.tensor([0.32, 0.64]),
    )

    assert torch.linalg.norm(reference.velocity[0]) > 0.0
    assert torch.linalg.norm(reference.acceleration[0]) > 0.0
    torch.testing.assert_close(reference.position[0], reference.position[1])
    torch.testing.assert_close(reference.velocity[0], 2.0 * reference.velocity[1])
    torch.testing.assert_close(
        reference.acceleration[0],
        4.0 * reference.acceleration[1],
    )



def test_reference_is_continuous_across_apex():
    start = torch.tensor([[0.0, 0.10, 0.02]]).repeat(3, 1)
    goal = torch.tensor([[0.3, 0.12, 0.00]]).repeat(3, 1)

    reference = quintic_swing_reference(
        start,
        goal,
        phase=torch.tensor([0.49999, 0.5, 0.50001]),
        apex_height=torch.tensor([0.10, 0.10, 0.10]),
        swing_duration_s=0.32,
    )

    torch.testing.assert_close(
        reference.position[0],
        reference.position[2],
        atol=2.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        reference.velocity[0],
        reference.velocity[2],
        atol=2.0e-4,
        rtol=0.0,
    )
    torch.testing.assert_close(
        reference.acceleration[0],
        reference.acceleration[2],
        atol=2.0e-2,
        rtol=0.0,
    )

def test_non_positive_duration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        quintic_swing_reference(
            torch.zeros((1, 3)),
            torch.ones((1, 3)),
            phase=torch.tensor([0.5]),
            apex_height=torch.tensor([0.1]),
            swing_duration_s=0.0,
        )



def test_phase_is_clamped_outside_unit_interval():
    start = torch.zeros((2, 3))
    goal = torch.tensor(
        [
            [0.3, 0.1, 0.0],
            [0.3, 0.1, 0.0],
        ]
    )

    reference = quintic_swing_reference(
        start,
        goal,
        phase=torch.tensor([-0.2, 1.2]),
        apex_height=torch.tensor([0.1, 0.1]),
        swing_duration_s=0.32,
    )

    torch.testing.assert_close(
        reference.position,
        torch.stack((start[0], goal[1])),
    )
    torch.testing.assert_close(reference.velocity, torch.zeros_like(start))
    torch.testing.assert_close(reference.acceleration, torch.zeros_like(start))