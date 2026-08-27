from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class ContactEvent(IntEnum):
    NONE = 0
    EARLY_CONTACT = 1
    LATE_CONTACT = 2
    SUPPORT_LOST = 3
    PLAN_INVALID = 4


class EventResponse(IntEnum):
    NONE = 0
    ACCEPT_TOUCHDOWN = 1
    SEARCH_DOWN = 2
    REASSIGN_SUPPORT = 3
    RETRY_PLAN = 4
    STABILIZE = 5


@dataclass(frozen=True)
class StabilityBounds:
    max_tilt_rad: float
    max_angular_speed_rad_s: float
    max_horizontal_speed_m_s: float
    max_support_slip_m_s: float
    dwell_s: float

    def __post_init__(self) -> None:
        if any(
            value <= 0.0
            for value in (
                self.max_tilt_rad,
                self.max_angular_speed_rad_s,
                self.max_horizontal_speed_m_s,
                self.max_support_slip_m_s,
                self.dwell_s,
            )
        ):
            raise ValueError("stability bounds must be positive")


@dataclass(frozen=True)
class StabilitySignals:
    confirmed_contact: torch.Tensor
    body_tilt_rad: torch.Tensor
    body_angular_speed_rad_s: torch.Tensor
    body_horizontal_speed_m_s: torch.Tensor
    support_slip_m_s: torch.Tensor


def _require_batch_shape(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def support_roles_from_contacts(
    confirmed_contact: torch.Tensor,
    previous_swing_side: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive support and next swing sides from confirmed physical contacts."""
    if confirmed_contact.ndim != 2 or confirmed_contact.shape[-1] != 2:
        raise ValueError("confirmed_contact must have shape (num_envs, 2)")
    _require_batch_shape(
        previous_swing_side,
        (confirmed_contact.shape[0],),
        "previous_swing_side",
    )
    if torch.any((previous_swing_side < 0) | (previous_swing_side > 1)):
        raise ValueError("previous_swing_side must contain only 0 or 1")

    contact = confirmed_contact.bool()
    left, right = contact[:, 0], contact[:, 1]
    support = torch.full_like(previous_swing_side, -1)
    swing = torch.full_like(previous_swing_side, -1)

    left_only = left & ~right
    right_only = right & ~left
    both = left & right
    support[left_only], swing[left_only] = 0, 1
    support[right_only], swing[right_only] = 1, 0
    swing[both] = 1 - previous_swing_side[both]
    support[both] = 1 - swing[both]
    return support, swing


def stability_ready(
    signals: StabilitySignals,
    bounds: StabilityBounds,
    elapsed_s: torch.Tensor,
    dt: float,
    require_both_contact: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate a continuous contact dwell and return readiness plus timer.

    Recovery uses this as a gait-resynchronization handshake: contact (one or
    both feet, selected by ``require_both_contact``) must remain confirmed for
    the calibrated dwell.  Motion magnitudes and support slip are retained in
    ``StabilitySignals`` for diagnostics/calibration, but they are not exit
    gates.  Requiring the body to nearly stop would prevent a moving command
    from handing control back to the normal HOLD/planner transaction.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if signals.confirmed_contact.ndim != 2 or signals.confirmed_contact.shape[-1] != 2:
        raise ValueError("confirmed_contact must have shape (num_envs, 2)")
    num_envs = signals.confirmed_contact.shape[0]
    for name in (
        "body_tilt_rad",
        "body_angular_speed_rad_s",
        "body_horizontal_speed_m_s",
        "support_slip_m_s",
    ):
        _require_batch_shape(getattr(signals, name), (num_envs,), name)
    _require_batch_shape(elapsed_s, (num_envs,), "elapsed_s")

    # Motion and slip remain sampled for diagnostics and calibration, but are
    # intentionally not hard gates for the Recovery handoff.
    confirmed_contact = signals.confirmed_contact.bool()
    contact_ready = (
        torch.all(confirmed_contact, dim=-1)
        if require_both_contact
        else torch.any(confirmed_contact, dim=-1)
    )
    stable = (
        contact_ready
        & torch.isfinite(signals.body_tilt_rad)
        & torch.isfinite(signals.body_angular_speed_rad_s)
        & torch.isfinite(signals.body_horizontal_speed_m_s)
    )
    next_elapsed_s = torch.where(
        stable,
        elapsed_s + dt,
        torch.zeros_like(elapsed_s),
    )
    ready = next_elapsed_s >= bounds.dwell_s - 1.0e-6
    return ready, next_elapsed_s


def response_for_event(
    event: torch.Tensor,
    support_stable: torch.Tensor,
    late_search_available: torch.Tensor,
    late_touchdown_confirmed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map a contact event to a bounded response without changing frames."""
    if event.ndim != 1:
        raise ValueError("event must have shape (num_envs,)")
    for name, value in (
        ("support_stable", support_stable),
        ("late_search_available", late_search_available),
    ):
        _require_batch_shape(value, tuple(event.shape), name)
    if late_touchdown_confirmed is None:
        late_touchdown_confirmed = torch.zeros_like(
            late_search_available,
            dtype=torch.bool,
        )
    else:
        _require_batch_shape(
            late_touchdown_confirmed,
            tuple(event.shape),
            "late_touchdown_confirmed",
        )
    if torch.any((event < ContactEvent.NONE) | (event > ContactEvent.PLAN_INVALID)):
        raise ValueError("event contains an unknown ContactEvent value")

    response = torch.full_like(event, EventResponse.NONE)
    plan_invalid = event == ContactEvent.PLAN_INVALID
    early = event == ContactEvent.EARLY_CONTACT
    late = event == ContactEvent.LATE_CONTACT
    support_lost = event == ContactEvent.SUPPORT_LOST

    response[plan_invalid] = EventResponse.RETRY_PLAN
    response[early & support_stable.bool()] = EventResponse.ACCEPT_TOUCHDOWN
    response[early & ~support_stable.bool()] = EventResponse.STABILIZE
    response[late & late_touchdown_confirmed.bool()] = (
        EventResponse.ACCEPT_TOUCHDOWN
    )
    response[
        late & ~late_touchdown_confirmed.bool() & late_search_available.bool()
    ] = EventResponse.SEARCH_DOWN
    response[
        late & ~late_touchdown_confirmed.bool() & ~late_search_available.bool()
    ] = EventResponse.STABILIZE
    # A lost stance is a physical gait failure.  Recovery first belongs to the
    # motor policy; if one foot is later confirmed, the state machine may open
    # one single-support learned-planner transaction.
    response[support_lost] = EventResponse.STABILIZE
    return response
