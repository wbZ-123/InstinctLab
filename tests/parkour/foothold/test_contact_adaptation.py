import torch

from instinctlab_foothold.contact_adaptation import (
    ContactEvent,
    EventResponse,
    StabilityBounds,
    StabilitySignals,
    response_for_event,
    stability_ready,
    support_roles_from_contacts,
)


def test_support_roles_follow_confirmed_physical_contacts():
    support, swing = support_roles_from_contacts(
        confirmed_contact=torch.tensor(
            [[True, False], [False, True], [True, True], [False, False]]
        ),
        previous_swing_side=torch.tensor([0, 1, 0, 1]),
    )

    torch.testing.assert_close(support, torch.tensor([0, 1, 0, -1]))
    torch.testing.assert_close(swing, torch.tensor([1, 0, 1, -1]))


def test_stability_requires_all_bounds_for_full_dwell():
    bounds = StabilityBounds(
        max_tilt_rad=0.20,
        max_angular_speed_rad_s=0.80,
        max_horizontal_speed_m_s=0.25,
        max_support_slip_m_s=0.05,
        dwell_s=0.10,
    )
    elapsed = torch.tensor([0.08, 0.08])
    signals = StabilitySignals(
        confirmed_contact=torch.tensor([[True, False], [True, False]]),
        body_tilt_rad=torch.tensor([0.10, 0.10]),
        body_angular_speed_rad_s=torch.tensor([0.20, 1.20]),
        body_horizontal_speed_m_s=torch.tensor([0.10, 0.10]),
        support_slip_m_s=torch.tensor([0.01, 0.01]),
    )

    ready, next_elapsed = stability_ready(
        signals,
        bounds,
        elapsed,
        dt=0.02,
    )

    torch.testing.assert_close(ready, torch.tensor([True, False]))
    torch.testing.assert_close(next_elapsed, torch.tensor([0.10, 0.0]))


def test_stability_exit_gate_does_not_block_on_support_slip():
    """Support slip remains observable but is not a recovery exit gate."""
    bounds = StabilityBounds(
        max_tilt_rad=0.35,
        max_angular_speed_rad_s=1.5,
        max_horizontal_speed_m_s=0.35,
        max_support_slip_m_s=0.05,
        dwell_s=0.04,
    )
    signals = StabilitySignals(
        confirmed_contact=torch.tensor([[True, False]]),
        body_tilt_rad=torch.tensor([0.10]),
        body_angular_speed_rad_s=torch.tensor([0.20]),
        body_horizontal_speed_m_s=torch.tensor([0.10]),
        # Large slip is kept in the signal for diagnostics, but does not
        # prevent the autonomous recovery policy from returning to HOLD.
        support_slip_m_s=torch.tensor([0.50]),
    )

    ready, elapsed = stability_ready(
        signals,
        bounds,
        torch.zeros(1),
        dt=0.02,
    )
    torch.testing.assert_close(ready, torch.tensor([False]))
    torch.testing.assert_close(elapsed, torch.tensor([0.02]))

    ready, elapsed = stability_ready(signals, bounds, elapsed, dt=0.02)
    torch.testing.assert_close(ready, torch.tensor([True]))
    torch.testing.assert_close(elapsed, torch.tensor([0.04]))


def test_event_response_keeps_invalid_plan_in_hold_and_searches_late_contact():
    retry = response_for_event(
        event=torch.tensor([ContactEvent.PLAN_INVALID]),
        support_stable=torch.tensor([True]),
        late_search_available=torch.tensor([False]),
    )
    search = response_for_event(
        event=torch.tensor([ContactEvent.LATE_CONTACT]),
        support_stable=torch.tensor([True]),
        late_search_available=torch.tensor([True]),
    )

    assert retry.item() == EventResponse.RETRY_PLAN
    assert search.item() == EventResponse.SEARCH_DOWN


def test_confirmed_contact_during_late_search_finishes_the_locked_swing():
    response = response_for_event(
        event=torch.tensor([ContactEvent.LATE_CONTACT]),
        support_stable=torch.tensor([True]),
        late_search_available=torch.tensor([True]),
        late_touchdown_confirmed=torch.tensor([True]),
    )

    assert response.item() == EventResponse.ACCEPT_TOUCHDOWN


def test_unstable_contact_event_enters_autonomous_stabilization():
    response = response_for_event(
        event=torch.tensor([ContactEvent.SUPPORT_LOST]),
        support_stable=torch.tensor([False]),
        late_search_available=torch.tensor([False]),
    )

    assert response.item() == EventResponse.STABILIZE
