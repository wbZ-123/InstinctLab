from dataclasses import dataclass

import torch

from .types import GaitState


@dataclass(frozen=True)
class GaitMachineConfig:
    reset_hold_s: float = 0.40
    swing_s: float = 0.32
    contact_confirm_s: float = 0.04
    early_contact_phase: float = 0.65
    overdue_s: float = 0.12

    def __post_init__(self) -> None:
        timing_values = (
            self.reset_hold_s,
            self.swing_s,
            self.contact_confirm_s,
            self.overdue_s,
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
    contact_elapsed_s: torch.Tensor
    no_contact_elapsed_s: torch.Tensor
    swing_has_lifted: torch.Tensor


def initial_gait_state(
    num_envs: int,
    device: torch.device | str,
) -> GaitMachineState:
    return GaitMachineState(
        mode=torch.full(
            (num_envs,),
            GaitState.HOLD,
            device=device,
            dtype=torch.long,
        ),
        swing_side=torch.zeros(
            num_envs,
            device=device,
            dtype=torch.long,
        ),
        elapsed_s=torch.zeros(num_envs, device=device),
        hold_elapsed_s=torch.zeros(num_envs, device=device),
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
) -> GaitMachineState:
    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )
    mode = state.mode.clone()
    swing_side = state.swing_side.clone()
    elapsed_s = state.elapsed_s.clone()
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

    was_touchdown_confirm = (
        (state.mode == GaitState.TOUCHDOWN_CONFIRM)
        & planner_valid
    )
    next_swing_side = 1 - swing_side[was_touchdown_confirm]

    swing_side[was_touchdown_confirm] = next_swing_side
    mode[was_touchdown_confirm] = torch.where(
        next_swing_side == 0,
        torch.full_like(
            next_swing_side,
            GaitState.LEFT_SWING,
        ),
        torch.full_like(
            next_swing_side,
            GaitState.RIGHT_SWING,
        ),
    )

    elapsed_s[was_touchdown_confirm] = 0.0
    contact_elapsed_s[was_touchdown_confirm] = 0.0
    no_contact_elapsed_s[was_touchdown_confirm] = 0.0
    swing_has_lifted[was_touchdown_confirm] = False

    was_failure_reason = (
        (state.mode == GaitState.PLAN_INVALID)
        | (state.mode == GaitState.EARLY_CONTACT)
        | (state.mode == GaitState.OVERDUE)
        | (state.mode == GaitState.STANCE_LOST)
    )
    mode[was_failure_reason] = GaitState.RECOVERY

    was_recovery = state.mode == GaitState.RECOVERY
    mode[was_recovery] = GaitState.RECOVERY

    invalid_during_hold = (state.mode == GaitState.HOLD) & ~planner_valid
    mode[invalid_during_hold] = GaitState.PLAN_INVALID

    stable_hold = (
        (state.mode == GaitState.HOLD)
        & planner_valid
        & torch.all(contact, dim=-1)
    )

    hold_elapsed_s = torch.where(
        stable_hold,
        state.hold_elapsed_s + dt,
        torch.zeros_like(state.hold_elapsed_s),
    )

    start_left_swing = stable_hold & (
        hold_elapsed_s >= cfg.reset_hold_s - 1.0e-6
    )

    mode[start_left_swing] = GaitState.LEFT_SWING
    swing_side[start_left_swing] = 0
    elapsed_s[start_left_swing] = 0.0

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
        & touchdown_accepted
    )

    mode[accepted_late_touchdown] = GaitState.TOUCHDOWN_CONFIRM

    stance_side = 1 - swing_side
    stance_no_contact_s = no_contact_elapsed_s[
        rows,
        stance_side,
    ]

    stance_lost = active_swing & (
        stance_no_contact_s
        >= cfg.contact_confirm_s - 1.0e-6
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
        contact_elapsed_s=contact_elapsed_s,
        no_contact_elapsed_s=no_contact_elapsed_s,
        swing_has_lifted=swing_has_lifted,
    )
