from dataclasses import dataclass

import torch

from .contact_adaptation import EventResponse, support_roles_from_contacts
from .types import GaitState


@dataclass(frozen=True)
class GaitMachineConfig:
    reset_hold_s: float = 0.40
    swing_s: float = 0.32
    contact_confirm_s: float = 0.04
    stance_lost_confirm_s: float = 0.10
    hold_contact_lost_confirm_s: float = 0.10
    early_contact_phase: float = 0.65
    overdue_s: float = 0.12
    recovery_hold_s: float = 0.20
    step_hold_s: float = 0.04

    def __post_init__(self) -> None:
        timing_values = (
            self.reset_hold_s,
            self.swing_s,
            self.contact_confirm_s,
            self.stance_lost_confirm_s,
            self.hold_contact_lost_confirm_s,
            self.overdue_s,
            self.recovery_hold_s,
            self.step_hold_s,
        )

        if any(
            value <= 0.0
            for value in timing_values
        ):
            raise ValueError(
                "Gait timing values must be positive."
            )

        if not (
            0.0
            < self.early_contact_phase
            < 1.0
        ):
            raise ValueError(
                "early_contact_phase must be "
                "between zero and one."
            )


@dataclass(frozen=True)
class GaitMachineState:
    mode: torch.Tensor
    swing_side: torch.Tensor
    elapsed_s: torch.Tensor
    hold_elapsed_s: torch.Tensor
    hold_required_s: torch.Tensor
    contact_elapsed_s: torch.Tensor
    no_contact_elapsed_s: torch.Tensor
    swing_has_lifted: torch.Tensor
    recovery_step_pending: torch.Tensor
    recovery_step_active: torch.Tensor
    stabilization_elapsed_s: torch.Tensor | None = None
    late_search_elapsed_s: torch.Tensor | None = None
    planning_failure: torch.Tensor | None = None


def initial_gait_state(
    num_envs: int,
    device: torch.device | str,
    env_ids: torch.Tensor | None = None,
) -> GaitMachineState:
    if env_ids is None:
        swing_side_source = torch.arange(
            num_envs,
            device=device,
            dtype=torch.long,
        )
    else:
        swing_side_source = env_ids.to(device=device, dtype=torch.long)
        if swing_side_source.shape != (num_envs,):
            raise ValueError("env_ids must match the number of environments.")

    return GaitMachineState(
        mode=torch.full(
            (num_envs,),
            GaitState.HOLD,
            device=device,
            dtype=torch.long,
        ),
        swing_side=swing_side_source.remainder(2),
        elapsed_s=torch.zeros(num_envs, device=device),
        hold_elapsed_s=torch.zeros(num_envs, device=device),
        hold_required_s=torch.full((num_envs,), -1.0, device=device),
        contact_elapsed_s=torch.zeros(
            (num_envs, 2),
            device=device,
        ),
        no_contact_elapsed_s=torch.zeros(
            (num_envs, 2),
            device=device,
        ),
        swing_has_lifted=torch.zeros(
            num_envs,
            device=device,
            dtype=torch.bool,
        ),
        recovery_step_pending=torch.zeros(
            num_envs,
            device=device,
            dtype=torch.bool,
        ),
        recovery_step_active=torch.zeros(
            num_envs,
            device=device,
            dtype=torch.bool,
        ),
        stabilization_elapsed_s=torch.zeros(num_envs, device=device),
        late_search_elapsed_s=torch.zeros(num_envs, device=device),
        planning_failure=torch.zeros(
            num_envs,
            device=device,
            dtype=torch.bool,
        ),
    )


def gait_phase(
    state: GaitMachineState,
    cfg: GaitMachineConfig,
) -> torch.Tensor:
    return torch.clamp(
        state.elapsed_s / cfg.swing_s,
        min=0.0,
        max=1.0,
    )


def advance_gait(
    state: GaitMachineState,
    contact: torch.Tensor,
    touchdown_accepted: torch.Tensor,
    planner_valid: torch.Tensor,
    dt: float,
    cfg: GaitMachineConfig,
    step_hold_s: torch.Tensor | None = None,
    swing_ready: torch.Tensor | None = None,
    hold_contact_ready: torch.Tensor | None = None,
    hold_contact_lost: torch.Tensor | None = None,
    plan_wait_expired: torch.Tensor | None = None,
    event_response: torch.Tensor | None = None,
    stabilization_ready: torch.Tensor | None = None,
    stability_current: torch.Tensor | None = None,
    late_search_exhausted: torch.Tensor | None = None,
    planning_failure: torch.Tensor | None = None,
) -> GaitMachineState:
    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )
    contact_adaptive = event_response is not None
    if event_response is not None:
        event_response = event_response.to(
            device=state.mode.device,
            dtype=torch.long,
        )
        if event_response.shape != state.mode.shape:
            raise ValueError(
                "event_response must match the number of environments."
            )
    if stabilization_ready is None:
        stabilization_ready = torch.zeros_like(
            state.mode,
            dtype=torch.bool,
        )
    else:
        stabilization_ready = stabilization_ready.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if stabilization_ready.shape != state.mode.shape:
            raise ValueError(
                "stabilization_ready must match the number of environments."
            )
    if stability_current is None:
        stability_current = stabilization_ready.clone()
    else:
        stability_current = stability_current.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if stability_current.shape != state.mode.shape:
            raise ValueError(
                "stability_current must match the number of environments."
            )
    if late_search_exhausted is None:
        late_search_exhausted = torch.zeros_like(
            state.mode,
            dtype=torch.bool,
        )
    else:
        late_search_exhausted = late_search_exhausted.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if late_search_exhausted.shape != state.mode.shape:
            raise ValueError(
                "late_search_exhausted must match the number of environments."
            )
    if planning_failure is None:
        planning_failure = torch.zeros_like(
            state.mode,
            dtype=torch.bool,
        )
    else:
        planning_failure = planning_failure.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if planning_failure.shape != state.mode.shape:
            raise ValueError(
                "planning_failure must match the number of environments."
            )
    if step_hold_s is None:
        step_hold_s = torch.full_like(
            state.hold_required_s,
            cfg.step_hold_s,
        )
    else:
        step_hold_s = step_hold_s.to(
            device=state.hold_required_s.device,
            dtype=state.hold_required_s.dtype,
        )
        if step_hold_s.shape != state.hold_required_s.shape:
            raise ValueError(
                "step_hold_s must match the number of environments."
            )
        if torch.any(step_hold_s < 0.0):
            raise ValueError(
                "step_hold_s must be non-negative."
            )
    if swing_ready is None:
        swing_ready = torch.ones_like(state.mode, dtype=torch.bool)
    else:
        swing_ready = swing_ready.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if swing_ready.shape != state.mode.shape:
            raise ValueError(
                "swing_ready must match the number of environments."
            )
    if plan_wait_expired is None:
        plan_wait_expired = torch.zeros_like(state.mode, dtype=torch.bool)
    else:
        plan_wait_expired = plan_wait_expired.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if plan_wait_expired.shape != state.mode.shape:
            raise ValueError(
                "plan_wait_expired must match the number of environments."
            )

    mode = state.mode.clone()
    swing_side = state.swing_side.clone()
    elapsed_s = state.elapsed_s.clone()
    hold_required_s = state.hold_required_s.clone()
    contact_elapsed_s = torch.where(
        contact,
        state.contact_elapsed_s + dt,
        torch.zeros_like(state.contact_elapsed_s),
    )
    no_contact_elapsed_s = torch.where(
        contact,
        torch.zeros_like(state.no_contact_elapsed_s),
        state.no_contact_elapsed_s + dt,
    )
    swing_has_lifted = state.swing_has_lifted.clone()
    recovery_step_pending = state.recovery_step_pending.clone()
    recovery_step_active = state.recovery_step_active.clone()
    stabilization_elapsed_s = (
        state.stabilization_elapsed_s
        if state.stabilization_elapsed_s is not None
        else torch.zeros_like(state.elapsed_s)
    ).clone()
    late_search_elapsed_s = (
        state.late_search_elapsed_s
        if state.late_search_elapsed_s is not None
        else torch.zeros_like(state.elapsed_s)
    ).clone()
    planning_failure_state = (
        state.planning_failure
        if state.planning_failure is not None
        else torch.zeros_like(state.mode, dtype=torch.bool)
    ).clone()

    confirmed_contact = contact_elapsed_s >= cfg.contact_confirm_s - 1.0e-6
    any_confirmed_contact = torch.any(confirmed_contact, dim=-1)
    both_contacts_confirmed = torch.all(confirmed_contact, dim=-1)

    was_touchdown_confirm = (
        (state.mode == GaitState.TOUCHDOWN_CONFIRM)
        & planner_valid
    )
    next_swing_side = 1 - swing_side[was_touchdown_confirm]

    swing_side[was_touchdown_confirm] = next_swing_side
    mode[was_touchdown_confirm] = GaitState.HOLD

    elapsed_s[was_touchdown_confirm] = 0.0
    hold_required_s[was_touchdown_confirm] = step_hold_s[
        was_touchdown_confirm
    ]
    swing_has_lifted[was_touchdown_confirm] = False
    recovery_step_pending[was_touchdown_confirm] = False
    recovery_step_active[was_touchdown_confirm] = False

    accepted_early_touchdown = (
        (state.mode == GaitState.EARLY_CONTACT)
        & touchdown_accepted
        & planner_valid
    )
    mode[accepted_early_touchdown] = GaitState.TOUCHDOWN_CONFIRM

    if contact_adaptive:
        # In the event-driven path PLAN_INVALID and OVERDUE are transient
        # planner events.  RETRY_PLAN/SEARCH_DOWN below owns their routing;
        # they must not first become a long-lived RECOVERY state.  Recovery
        # is reserved for a physical contact failure (unstable early contact,
        # confirmed stance loss, or loss of the HOLD support contact).
        was_failure_reason = (
            (
                (state.mode == GaitState.EARLY_CONTACT)
                & ~accepted_early_touchdown
            )
            | (state.mode == GaitState.STANCE_LOST)
            | (state.mode == GaitState.HOLD_CONTACT_LOST)
        )
    else:
        # Preserve the legacy non-adaptive state-machine behavior for old
        # callers and checkpoints that do not provide event responses.
        was_failure_reason = (
            (state.mode == GaitState.PLAN_INVALID)
            | (
                (state.mode == GaitState.EARLY_CONTACT)
                & ~accepted_early_touchdown
            )
            | (state.mode == GaitState.OVERDUE)
            | (state.mode == GaitState.STANCE_LOST)
            | (state.mode == GaitState.HOLD_CONTACT_LOST)
        )
    mode[was_failure_reason] = GaitState.RECOVERY

    was_recovery = state.mode == GaitState.RECOVERY
    # Recovery is a physical-contact phase.  Its dwell timer must continue
    # independently of planner_valid; a recovery plan is prepared only after
    # a support foot has been identified.
    recovery_counting = was_recovery
    recovery_ready = recovery_counting & (
        any_confirmed_contact
        & (
            state.hold_elapsed_s + dt >= cfg.recovery_hold_s - 1.0e-6
        )
    )
    mode[was_recovery] = GaitState.RECOVERY
    mode[recovery_ready] = GaitState.HOLD

    # Select the next swing from the actual confirmed support.  A single
    # confirmed contact is sufficient for a recovery step; when both feet are
    # down, preserve the normal alternating gait relation.
    only_left_support = confirmed_contact[:, 0] & ~confirmed_contact[:, 1]
    only_right_support = confirmed_contact[:, 1] & ~confirmed_contact[:, 0]
    both_support = both_contacts_confirmed
    swing_side[recovery_ready & only_left_support] = 1
    swing_side[recovery_ready & only_right_support] = 0
    swing_side[recovery_ready & both_support] = 1 - state.swing_side[
        recovery_ready & both_support
    ]
    elapsed_s[recovery_ready] = 0.0
    hold_required_s[recovery_ready] = cfg.reset_hold_s
    swing_has_lifted[recovery_ready] = False
    recovery_step_pending[recovery_ready] = True
    recovery_step_active[recovery_ready] = False

    # A recovery HOLD is allowed to wait for the newly prepared planner
    # proposal.  A normal HOLD with an invalid plan remains a failure.
    invalid_during_hold = (
        (state.mode == GaitState.HOLD)
        & ~planner_valid
        & ~state.recovery_step_pending
    )
    mode[invalid_during_hold] = GaitState.PLAN_INVALID

    valid_hold = (
        (state.mode == GaitState.HOLD)
        & planner_valid
    )

    hold_elapsed_s = torch.where(
        valid_hold | recovery_counting,
        state.hold_elapsed_s + dt,
        torch.zeros_like(state.hold_elapsed_s),
    )
    hold_elapsed_s[recovery_ready] = 0.0
    hold_required_s = torch.where(
        valid_hold & (state.hold_required_s >= 0.0),
        hold_required_s,
        torch.full_like(hold_required_s, cfg.reset_hold_s),
    )
    hold_required_s[was_touchdown_confirm] = step_hold_s[
        was_touchdown_confirm
    ]
    hold_required_s[recovery_ready] = cfg.reset_hold_s

    if hold_contact_ready is None:
        hold_contact_ready = torch.where(
            state.recovery_step_pending,
            any_confirmed_contact,
            both_contacts_confirmed,
        )
    else:
        hold_contact_ready = hold_contact_ready.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if hold_contact_ready.shape != state.mode.shape:
            raise ValueError(
                "hold_contact_ready must match the number of environments."
            )
    if hold_contact_lost is None:
        hold_contact_lost = (
            ~hold_contact_ready
            & (
                hold_elapsed_s
                >= hold_required_s
                + cfg.hold_contact_lost_confirm_s
                - 1.0e-6
            )
        )
    else:
        hold_contact_lost = hold_contact_lost.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if hold_contact_lost.shape != state.mode.shape:
            raise ValueError(
                "hold_contact_lost must match the number of environments."
            )

    start_swing = (
        valid_hold
        & hold_contact_ready
        & (hold_elapsed_s >= hold_required_s - 1.0e-6)
        & swing_ready
    )

    # Do not start a swing before the complete plan transaction is ready.  An
    # unsafe proposal is retained as a PPO event, but it never becomes an
    # executable swing trajectory.
    hold_plan_timeout = (
        valid_hold
        & hold_contact_ready
        & (hold_elapsed_s >= hold_required_s - 1.0e-6)
        & ~swing_ready
        & plan_wait_expired
    )

    hold_contact_lost = valid_hold & hold_contact_lost

    mode[hold_contact_lost] = GaitState.HOLD_CONTACT_LOST

    mode[start_swing] = torch.where(
        swing_side[start_swing] == 0,
        torch.full_like(
            swing_side[start_swing],
            GaitState.LEFT_SWING,
        ),
        torch.full_like(
            swing_side[start_swing],
            GaitState.RIGHT_SWING,
        ),
    )
    # Missing a planning deadline is not a physical failure.  Keep the robot
    # in stable HOLD and let the next planning event replace the proposal;
    # entering RECOVERY here used to create a PLAN_INVALID -> RECOVERY loop.
    mode[hold_plan_timeout] = GaitState.HOLD
    elapsed_s[hold_plan_timeout] = 0.0
    hold_elapsed_s[hold_plan_timeout] = 0.0
    recovery_step_pending[hold_plan_timeout] = False
    recovery_step_active[hold_plan_timeout] = False
    elapsed_s[start_swing] = 0.0
    hold_required_s[start_swing] = cfg.reset_hold_s
    recovery_step_active[start_swing] = recovery_step_pending[start_swing]

    active_swing = (
        (state.mode == GaitState.LEFT_SWING)
        | (state.mode == GaitState.RIGHT_SWING)
    )
    elapsed_s[active_swing] += dt

    rows = torch.arange(
        state.mode.shape[0],
        device=state.mode.device,
    )
    swing_no_contact_s = no_contact_elapsed_s[
        rows,
        swing_side,
    ]
    confirmed_liftoff = active_swing & (
        swing_no_contact_s
        >= cfg.contact_confirm_s - 1.0e-6
    )
    swing_has_lifted[confirmed_liftoff] = True

    swing_contact_s = contact_elapsed_s[
        rows,
        swing_side,
    ]
    phase = torch.clamp(
        elapsed_s / cfg.swing_s,
        min=0.0,
        max=1.0,
    )

    overdue = active_swing & (
        elapsed_s
        >= cfg.swing_s + cfg.overdue_s - 1.0e-6
    )

    mode[overdue] = GaitState.OVERDUE

    early_contact = (
        active_swing
        & swing_has_lifted
        & (
            swing_contact_s
            >= cfg.contact_confirm_s - 1.0e-6
        )
        & (phase < cfg.early_contact_phase)
    )

    mode[early_contact] = GaitState.EARLY_CONTACT

    confirmed_swing_contact = (
        swing_contact_s
        >= cfg.contact_confirm_s - 1.0e-6
    )

    accepted_late_touchdown = (
        active_swing
        & swing_has_lifted
        & confirmed_swing_contact
        & (phase >= cfg.early_contact_phase)
    )

    mode[accepted_late_touchdown] = GaitState.TOUCHDOWN_CONFIRM

    stance_side = 1 - swing_side
    stance_no_contact_s = no_contact_elapsed_s[
        rows,
        stance_side,
    ]

    stance_lost = active_swing & ~accepted_late_touchdown & (
        stance_no_contact_s
        >= cfg.stance_lost_confirm_s - 1.0e-6
    )

    mode[stance_lost] = GaitState.STANCE_LOST

    invalid_during_active_step = (
        active_swing | (state.mode == GaitState.TOUCHDOWN_CONFIRM)
    ) & ~planner_valid

    # A plan is fully preflighted before SWING starts.  In the adaptive path
    # the target/frame are locked for the whole motion, so a later planner
    # validity flag cannot rewrite an already-running swing.  Keep the legacy
    # behavior for non-adaptive callers that still use PLAN_INVALID directly.
    if not contact_adaptive:
        mode[invalid_during_active_step] = GaitState.PLAN_INVALID

    if contact_adaptive:
        # Event responses are applied after the legacy timer arithmetic so
        # older callers remain source-compatible while the planner migrates.
        response = event_response
        assert response is not None
        retry_plan = response == EventResponse.RETRY_PLAN
        accept_touchdown = response == EventResponse.ACCEPT_TOUCHDOWN
        search_down = response == EventResponse.SEARCH_DOWN
        reassign_support = response == EventResponse.REASSIGN_SUPPORT
        stabilize = response == EventResponse.STABILIZE

        mode[retry_plan] = GaitState.HOLD
        elapsed_s[retry_plan] = 0.0
        hold_elapsed_s[retry_plan] = 0.0
        recovery_step_pending[retry_plan] = False
        recovery_step_active[retry_plan] = False

        mode[accept_touchdown] = GaitState.TOUCHDOWN_CONFIRM
        mode[search_down & ~late_search_exhausted] = GaitState.OVERDUE
        late_search_elapsed_s[search_down] += dt
        mode[search_down & late_search_exhausted] = GaitState.RECOVERY
        mode[reassign_support] = GaitState.HOLD

        if torch.any(reassign_support).item():
            _, next_swing = support_roles_from_contacts(
                confirmed_contact,
                swing_side,
            )
            valid_next_swing = reassign_support & (next_swing >= 0)
            swing_side[valid_next_swing] = next_swing[valid_next_swing]
            elapsed_s[reassign_support] = 0.0
            hold_elapsed_s[reassign_support] = 0.0

        mode[stabilize] = GaitState.RECOVERY
        stabilization_elapsed_s = torch.where(
            stabilize & stability_current,
            stabilization_elapsed_s + dt,
            torch.zeros_like(stabilization_elapsed_s),
        )
        mode[state.mode == GaitState.RECOVERY] = GaitState.RECOVERY
        # Recovery is owned by the motor policy in contact-adaptive mode.
        # Legacy recovery-step flags must not leak into the next frame and
        # cause the planner to synthesize an analytic foothold concurrently.
        recovery_step_pending[state.mode == GaitState.RECOVERY] = False
        recovery_step_active[state.mode == GaitState.RECOVERY] = False
        exited_recovery = (
            (state.mode == GaitState.RECOVERY)
            & stabilization_ready
        )
        if torch.any(exited_recovery).item():
            _, next_swing = support_roles_from_contacts(
                confirmed_contact,
                swing_side,
            )
            valid_next_swing = exited_recovery & (next_swing >= 0)
            mode[exited_recovery] = GaitState.HOLD
            swing_side[valid_next_swing] = next_swing[valid_next_swing]
            elapsed_s[exited_recovery] = 0.0
            hold_elapsed_s[exited_recovery] = 0.0
            stabilization_elapsed_s[exited_recovery] = 0.0
            recovery_step_pending[exited_recovery] = False
            recovery_step_active[exited_recovery] = False

    planning_failure_state |= planning_failure
    # A failed preflight is a planning transaction failure, not a physical
    # contact failure.  Stay in HOLD and let the planner propose again.
    mode[planning_failure] = GaitState.HOLD
    elapsed_s[planning_failure] = 0.0
    hold_elapsed_s[planning_failure] = 0.0
    recovery_step_pending[planning_failure] = False
    recovery_step_active[planning_failure] = False

    # The late-contact timer belongs to one locked swing only.  It must never
    # leak through touchdown, recovery, or a fresh HOLD transaction.
    late_search_elapsed_s = torch.where(
        mode == GaitState.OVERDUE,
        late_search_elapsed_s,
        torch.zeros_like(late_search_elapsed_s),
    )

    return GaitMachineState(
        mode=mode,
        swing_side=swing_side,
        elapsed_s=elapsed_s,
        hold_elapsed_s=hold_elapsed_s,
        hold_required_s=hold_required_s,
        contact_elapsed_s=contact_elapsed_s,
        no_contact_elapsed_s=no_contact_elapsed_s,
        swing_has_lifted=swing_has_lifted,
        recovery_step_pending=recovery_step_pending,
        recovery_step_active=recovery_step_active,
        stabilization_elapsed_s=stabilization_elapsed_s,
        late_search_elapsed_s=late_search_elapsed_s,
        planning_failure=planning_failure_state,
    )
