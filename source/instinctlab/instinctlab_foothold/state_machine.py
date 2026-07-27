from dataclasses import dataclass

import torch

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
) -> GaitMachineState:
    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
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
    contact_elapsed_s[was_touchdown_confirm] = 0.0
    no_contact_elapsed_s[was_touchdown_confirm] = 0.0
    swing_has_lifted[was_touchdown_confirm] = False
    recovery_step_pending[was_touchdown_confirm] = False
    recovery_step_active[was_touchdown_confirm] = False

    was_failure_reason = (
        (state.mode == GaitState.PLAN_INVALID)
        | (state.mode == GaitState.EARLY_CONTACT)
        | (state.mode == GaitState.OVERDUE)
        | (state.mode == GaitState.STANCE_LOST)
        | (state.mode == GaitState.HOLD_CONTACT_LOST)
    )
    mode[was_failure_reason] = GaitState.RECOVERY

    was_recovery = state.mode == GaitState.RECOVERY
    recovery_counting = (
        was_recovery
        & planner_valid
    )
    recovery_ready = recovery_counting & (
        state.hold_elapsed_s + dt >= cfg.recovery_hold_s - 1.0e-6
    )
    mode[was_recovery] = GaitState.RECOVERY
    mode[recovery_ready] = GaitState.HOLD
    swing_side[recovery_ready] = 1 - swing_side[recovery_ready]
    elapsed_s[recovery_ready] = 0.0
    hold_required_s[recovery_ready] = cfg.reset_hold_s
    swing_has_lifted[recovery_ready] = False
    recovery_step_pending[recovery_ready] = True
    recovery_step_active[recovery_ready] = False

    invalid_during_hold = (state.mode == GaitState.HOLD) & ~planner_valid
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

    confirmed_hold_contact = torch.all(
        contact_elapsed_s >= cfg.contact_confirm_s - 1.0e-6,
        dim=-1,
    )

    start_swing = (
        valid_hold
        & confirmed_hold_contact
        & (hold_elapsed_s >= hold_required_s - 1.0e-6)
    )

    hold_contact_lost = (
        valid_hold
        & ~confirmed_hold_contact
        & (
            hold_elapsed_s
            >= hold_required_s + cfg.hold_contact_lost_confirm_s - 1.0e-6
        )
    )

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

    mode[invalid_during_active_step] = GaitState.PLAN_INVALID

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
    )
