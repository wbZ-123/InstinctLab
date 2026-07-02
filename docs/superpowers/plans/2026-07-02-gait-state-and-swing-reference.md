# Gait State Machine and Swing Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simulator-independent, vectorized gait state machine and a time-scaled quintic swing-foot reference for the foothold planner.

**Architecture:** `state_machine.py` owns discrete gait modes, timers, contact debouncing, liftoff gating, and failure reasons. `trajectory.py` independently maps start/goal poses and normalized phase to translational position, physical velocity, and physical acceleration. Task 4 will compose both modules inside `FootholdPlannerSensor`.

**Tech Stack:** Python 3.11, PyTorch, dataclasses, pytest.

## Global Constraints

- Work only in `/home/zhangweibo/InstinctLab-foothold` on `feat/foothold-01-flat-tracking`.
- Keep `instinctlab_foothold` importable without Isaac Lab, `pxr`, Omniverse, or `SimulationApp`.
- Treat the state machine as an opt-in auxiliary estimator for the existing
  training pipeline, not as a replacement controller.
- Do not reset environments, override policy actions, write robot state, issue
  joint commands, or add hard terminations from Task 3 state transitions.
- Keep existing task configurations and behavior unchanged; only the new
  foothold configuration may opt in.
- In the initial integration, expose recovery and failure states for
  diagnostics, metrics, visualization, reward masking, or soft penalties only.
- When a plan is invalid or recovery is latched, preserve the existing
  locomotion action path rather than freezing or replacing it.
- Use side index `0 = left`, `1 = right`.
- Preserve stable `GaitState` integer IDs from `instinctlab_foothold.types`.
- Use time in seconds for state-machine timers and m, m/s, m/s² for trajectory outputs.
- Write and observe each failing test before adding the production behavior.
- Run tests with `PYTHONPATH="$PWD/source/instinctlab"` and disable pytest cache.

---

### Task 1: Add gait state storage and reset hold

**Files:**
- Create: `source/instinctlab/instinctlab_foothold/state_machine.py`
- Create: `tests/parkour/foothold/test_state_machine.py`

**Interfaces:**
- Consumes: `GaitState` from `instinctlab_foothold.types`.
- Produces: `GaitMachineConfig`, `GaitMachineState`, `initial_gait_state`, `gait_phase`, and `advance_gait`.

- [ ] **Step 1: Write the reset-hold and invalid-plan tests**

Create `tests/parkour/foothold/test_state_machine.py`:

```python
import pytest
import torch

from instinctlab_foothold.state_machine import (
    GaitMachineConfig,
    advance_gait,
    gait_phase,
    initial_gait_state,
)
from instinctlab_foothold.types import GaitState


def _advance(state, contact, planner_valid, steps, cfg):
    for _ in range(steps):
        state = advance_gait(
            state=state,
            contact=contact,
            touchdown_accepted=torch.zeros(contact.shape[0], dtype=torch.bool),
            planner_valid=planner_valid,
            dt=0.02,
            cfg=cfg,
        )
    return state


def test_reset_holds_for_exactly_point_four_seconds():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")
    contact = torch.tensor([[True, True]])
    valid = torch.tensor([True])

    state = _advance(state, contact, valid, steps=19, cfg=cfg)
    assert state.mode.item() == GaitState.HOLD

    state = _advance(state, contact, valid, steps=1, cfg=cfg)
    assert state.mode.item() == GaitState.LEFT_SWING
    torch.testing.assert_close(gait_phase(state, cfg), torch.zeros(1))


def test_invalid_plan_is_visible_then_enters_recovery():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")

    state = advance_gait(
        state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([False]),
        dt=0.02,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.PLAN_INVALID

    state = advance_gait(
        state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.RECOVERY


def test_vectorized_environments_advance_independently():
    cfg = GaitMachineConfig()
    state = initial_gait_state(2, device="cpu")
    state = _advance(
        state,
        contact=torch.tensor([[True, True], [True, False]]),
        planner_valid=torch.tensor([True, True]),
        steps=20,
        cfg=cfg,
    )
    torch.testing.assert_close(
        state.mode,
        torch.tensor([GaitState.LEFT_SWING, GaitState.HOLD]),
    )


def test_non_positive_timing_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        GaitMachineConfig(swing_s=0.0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab" conda run -n hiking python -m pytest \
  -p no:cacheprovider tests/parkour/foothold/test_state_machine.py -q
```

Expected: collection fails because `instinctlab_foothold.state_machine` does not exist.

- [ ] **Step 3: Add the public data structures and initializer**

Create `source/instinctlab/instinctlab_foothold/state_machine.py` with:

```python
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

    def __post_init__(self):
        positive = (
            self.reset_hold_s,
            self.swing_s,
            self.contact_confirm_s,
            self.overdue_s,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Gait timing values must be positive.")
        if not 0.0 < self.early_contact_phase < 1.0:
            raise ValueError("early_contact_phase must be between zero and one.")


@dataclass(frozen=True)
class GaitMachineState:
    mode: torch.Tensor
    swing_side: torch.Tensor
    elapsed_s: torch.Tensor
    hold_elapsed_s: torch.Tensor
    contact_elapsed_s: torch.Tensor
    no_contact_elapsed_s: torch.Tensor
    swing_has_lifted: torch.Tensor


def initial_gait_state(num_envs: int, device: torch.device | str) -> GaitMachineState:
    return GaitMachineState(
        mode=torch.full((num_envs,), GaitState.HOLD, device=device, dtype=torch.long),
        swing_side=torch.zeros(num_envs, device=device, dtype=torch.long),
        elapsed_s=torch.zeros(num_envs, device=device),
        hold_elapsed_s=torch.zeros(num_envs, device=device),
        contact_elapsed_s=torch.zeros((num_envs, 2), device=device),
        no_contact_elapsed_s=torch.zeros((num_envs, 2), device=device),
        swing_has_lifted=torch.zeros(num_envs, device=device, dtype=torch.bool),
    )


def gait_phase(state: GaitMachineState, cfg: GaitMachineConfig) -> torch.Tensor:
    return torch.clamp(state.elapsed_s / cfg.swing_s, min=0.0, max=1.0)
```

- [ ] **Step 4: Implement reset hold and reason-to-recovery transitions**

Add `advance_gait` with these exact update rules:

```python
def advance_gait(
    state: GaitMachineState,
    contact: torch.Tensor,
    touchdown_accepted: torch.Tensor,
    planner_valid: torch.Tensor,
    dt: float,
    cfg: GaitMachineConfig,
) -> GaitMachineState:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    mode = state.mode.clone()
    swing_side = state.swing_side.clone()
    elapsed_s = state.elapsed_s.clone()
    hold_elapsed_s = state.hold_elapsed_s.clone()
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

    reason_state = (
        (state.mode == GaitState.EARLY_CONTACT)
        | (state.mode == GaitState.OVERDUE)
        | (state.mode == GaitState.STANCE_LOST)
        | (state.mode == GaitState.PLAN_INVALID)
    )
    mode[reason_state] = GaitState.RECOVERY

    recovery = state.mode == GaitState.RECOVERY
    mode[recovery] = GaitState.RECOVERY

    hold = state.mode == GaitState.HOLD
    invalid_hold = hold & ~planner_valid
    mode[invalid_hold] = GaitState.PLAN_INVALID

    stable_hold = hold & planner_valid & torch.all(contact, dim=-1)
    hold_elapsed_s = torch.where(
        stable_hold,
        hold_elapsed_s + dt,
        torch.zeros_like(hold_elapsed_s),
    )
    start_left = stable_hold & (hold_elapsed_s >= cfg.reset_hold_s)
    mode[start_left] = GaitState.LEFT_SWING
    swing_side[start_left] = 0
    elapsed_s[start_left] = 0.0
    swing_has_lifted[start_left] = False

    return GaitMachineState(
        mode=mode,
        swing_side=swing_side,
        elapsed_s=elapsed_s,
        hold_elapsed_s=hold_elapsed_s,
        contact_elapsed_s=contact_elapsed_s,
        no_contact_elapsed_s=no_contact_elapsed_s,
        swing_has_lifted=swing_has_lifted,
    )
```

- [ ] **Step 5: Run the state-machine tests**

Run the command from Step 2.

Expected: `4 passed`.

---

### Task 2: Add liftoff, touchdown, alternation, and failure transitions

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/state_machine.py`
- Modify: `tests/parkour/foothold/test_state_machine.py`

**Interfaces:**
- Consumes: Task 1 state/config and `[N, 2]` contact tensors.
- Produces: vectorized normal and exceptional swing transitions.

- [ ] **Step 1: Add transition tests**

Append tests that construct a left-swing state by running the reset hold, then verify:

```python
def _start_left_swing(cfg):
    state = initial_gait_state(1, device="cpu")
    return _advance(
        state,
        contact=torch.tensor([[True, True]]),
        planner_valid=torch.tensor([True]),
        steps=20,
        cfg=cfg,
    )


def test_old_contact_is_ignored_until_liftoff_is_confirmed():
    cfg = GaitMachineConfig()
    state = _start_left_swing(cfg)
    state = _advance(
        state,
        contact=torch.tensor([[True, True]]),
        planner_valid=torch.tensor([True]),
        steps=2,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.LEFT_SWING
    assert not state.swing_has_lifted.item()


def test_accepted_late_touchdown_swaps_to_right_swing():
    cfg = GaitMachineConfig()
    state = _start_left_swing(cfg)

    state = _advance(
        state,
        contact=torch.tensor([[False, True]]),
        planner_valid=torch.tensor([True]),
        steps=2,
        cfg=cfg,
    )
    assert state.swing_has_lifted.item()

    state = _advance(
        state,
        contact=torch.tensor([[False, True]]),
        planner_valid=torch.tensor([True]),
        steps=9,
        cfg=cfg,
    )
    for _ in range(2):
        state = advance_gait(
            state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([True]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
        )
    assert state.mode.item() == GaitState.TOUCHDOWN_CONFIRM

    state = advance_gait(
        state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.RIGHT_SWING
    assert state.swing_side.item() == 1
```

Add separate tests with the same two-sample confirmation pattern for:

```python
def test_confirmed_contact_before_phase_threshold_is_early_contact():
    cfg = GaitMachineConfig()
    state = _start_left_swing(cfg)
    state = _advance(
        state,
        contact=torch.tensor([[False, True]]),
        planner_valid=torch.tensor([True]),
        steps=2,
        cfg=cfg,
    )
    state = _advance(
        state,
        contact=torch.tensor([[True, True]]),
        planner_valid=torch.tensor([True]),
        steps=2,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.EARLY_CONTACT


def test_confirmed_stance_loss_is_reported():
    cfg = GaitMachineConfig()
    state = _start_left_swing(cfg)
    state = _advance(
        state,
        contact=torch.tensor([[False, False]]),
        planner_valid=torch.tensor([True]),
        steps=2,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.STANCE_LOST


def test_rejected_late_contact_eventually_becomes_overdue():
    cfg = GaitMachineConfig()
    state = _start_left_swing(cfg)
    state = _advance(
        state,
        contact=torch.tensor([[False, True]]),
        planner_valid=torch.tensor([True]),
        steps=11,
        cfg=cfg,
    )
    assert state.swing_has_lifted.item()

    state = _advance(
        state,
        contact=torch.tensor([[True, True]]),
        planner_valid=torch.tensor([True]),
        steps=2,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.LEFT_SWING

    state = _advance(
        state,
        contact=torch.tensor([[True, True]]),
        planner_valid=torch.tensor([True]),
        steps=9,
        cfg=cfg,
    )
    assert state.mode.item() == GaitState.OVERDUE
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab" conda run -n hiking python -m pytest \
  -p no:cacheprovider tests/parkour/foothold/test_state_machine.py -q
```

Expected: the new transition assertions fail while the two Task 1 tests pass.

- [ ] **Step 3: Implement active-swing transitions**

Extend `advance_gait` before the return:

```python
    touchdown_confirm = state.mode == GaitState.TOUCHDOWN_CONFIRM
    next_swing_side = 1 - swing_side[touchdown_confirm]
    swing_side[touchdown_confirm] = next_swing_side
    mode[touchdown_confirm] = torch.where(
        next_swing_side == 0,
        torch.full_like(next_swing_side, GaitState.LEFT_SWING),
        torch.full_like(next_swing_side, GaitState.RIGHT_SWING),
    )
    elapsed_s[touchdown_confirm] = 0.0
    contact_elapsed_s[touchdown_confirm] = 0.0
    no_contact_elapsed_s[touchdown_confirm] = 0.0
    swing_has_lifted[touchdown_confirm] = False

    active = (state.mode == GaitState.LEFT_SWING) | (
        state.mode == GaitState.RIGHT_SWING
    )
    elapsed_s[active] += dt

    rows = torch.arange(mode.shape[0], device=mode.device)
    stance_side = 1 - swing_side
    swing_no_contact = no_contact_elapsed_s[rows, swing_side]
    swing_has_lifted |= active & (
        swing_no_contact >= cfg.contact_confirm_s
    )

    stance_lost = active & (
        no_contact_elapsed_s[rows, stance_side]
        >= cfg.contact_confirm_s
    )
    swing_contact_confirmed = (
        active
        & swing_has_lifted
        & (
            contact_elapsed_s[rows, swing_side]
            >= cfg.contact_confirm_s
        )
    )
    phase = torch.clamp(
        elapsed_s / cfg.swing_s,
        min=0.0,
        max=1.0,
    )

    invalid = active & ~planner_valid
    early = swing_contact_confirmed & (
        phase < cfg.early_contact_phase
    )
    touchdown = (
        swing_contact_confirmed
        & (phase >= cfg.early_contact_phase)
        & touchdown_accepted
    )
    overdue = active & (
        elapsed_s >= cfg.swing_s + cfg.overdue_s
    )

    mode = torch.where(
        overdue,
        torch.full_like(mode, GaitState.OVERDUE),
        mode,
    )
    mode = torch.where(
        touchdown,
        torch.full_like(mode, GaitState.TOUCHDOWN_CONFIRM),
        mode,
    )
    mode = torch.where(
        early,
        torch.full_like(mode, GaitState.EARLY_CONTACT),
        mode,
    )
    mode = torch.where(
        stance_lost,
        torch.full_like(mode, GaitState.STANCE_LOST),
        mode,
    )
    mode = torch.where(
        invalid,
        torch.full_like(mode, GaitState.PLAN_INVALID),
        mode,
    )
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the command from Step 2.

Expected: all state-machine tests pass.

- [ ] **Step 5: Commit the state machine**

```bash
git add \
  source/instinctlab/instinctlab_foothold/state_machine.py \
  tests/parkour/foothold/test_state_machine.py
git diff --cached --check
git commit -m "feat(foothold): add gait state machine"
```

---

### Task 3: Add the time-scaled quintic swing reference

**Files:**
- Create: `source/instinctlab/instinctlab_foothold/trajectory.py`
- Create: `tests/parkour/foothold/test_trajectory.py`

**Interfaces:**
- Consumes: `[N, 3]` start/goal positions, normalized phase, apex height, and duration in seconds.
- Produces: `SwingReference(position, velocity, acceleration)`.

- [ ] **Step 1: Write trajectory boundary and scaling tests**

Create `tests/parkour/foothold/test_trajectory.py` with:

```python
import torch

from instinctlab_foothold.trajectory import quintic_swing_reference


def test_reference_matches_endpoints_with_zero_derivatives():
    start = torch.tensor([[0.0, 0.10, 0.02], [0.1, -0.10, 0.03]])
    goal = torch.tensor([[0.3, 0.12, 0.00], [0.4, -0.08, 0.01]])
    phase = torch.tensor([0.0, 1.0])

    reference = quintic_swing_reference(
        start=start,
        goal=goal,
        phase=phase,
        apex_height=torch.tensor([0.10, 0.10]),
        swing_duration_s=0.32,
    )

    torch.testing.assert_close(reference.position, torch.stack((start[0], goal[1])))
    torch.testing.assert_close(reference.velocity, torch.zeros_like(start))
    torch.testing.assert_close(reference.acceleration, torch.zeros_like(start))


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
    torch.testing.assert_close(reference.position[:, 2], torch.tensor([0.12]))
    torch.testing.assert_close(reference.velocity[:, 2], torch.zeros(1), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(reference.acceleration[:, 2], torch.zeros(1), atol=1e-5, rtol=0.0)


def test_duration_scales_velocity_and_acceleration():
    start = torch.tensor([[0.0, 0.10, 0.0], [0.0, 0.10, 0.0]])
    goal = torch.tensor([[0.3, 0.12, 0.0], [0.3, 0.12, 0.0]])
    reference = quintic_swing_reference(
        start,
        goal,
        phase=torch.tensor([0.25, 0.25]),
        apex_height=torch.tensor([0.10, 0.10]),
        swing_duration_s=torch.tensor([0.32, 0.64]),
    )
    torch.testing.assert_close(reference.position[0], reference.position[1])
    torch.testing.assert_close(reference.velocity[0], 2.0 * reference.velocity[1])
    torch.testing.assert_close(reference.acceleration[0], 4.0 * reference.acceleration[1])


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
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab" conda run -n hiking python -m pytest \
  -p no:cacheprovider tests/parkour/foothold/test_trajectory.py -q
```

Expected: collection fails because `instinctlab_foothold.trajectory` does not exist.

- [ ] **Step 3: Implement the trajectory**

Create `source/instinctlab/instinctlab_foothold/trajectory.py`:

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SwingReference:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor


def _quintic(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    first = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    second = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    return blend, first, second


def quintic_swing_reference(
    start: torch.Tensor,
    goal: torch.Tensor,
    phase: torch.Tensor,
    apex_height: torch.Tensor,
    swing_duration_s: torch.Tensor | float,
) -> SwingReference:
    duration = torch.as_tensor(
        swing_duration_s,
        device=start.device,
        dtype=start.dtype,
    )
    if duration.ndim == 0:
        duration = duration.expand(start.shape[0])
    if duration.shape != phase.shape:
        raise ValueError("swing_duration_s must be scalar or match phase.")
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
        position=torch.cat((position_xy, position_z.unsqueeze(-1)), dim=-1),
        velocity=torch.cat((velocity_xy, velocity_z.unsqueeze(-1)), dim=-1),
        acceleration=torch.cat(
            (acceleration_xy, acceleration_z.unsqueeze(-1)),
            dim=-1,
        ),
    )
```

- [ ] **Step 4: Add phase-clamping and invalid-duration tests**

Append:

```python
def test_phase_is_clamped_outside_unit_interval():
    start = torch.zeros((2, 3))
    goal = torch.tensor([[0.3, 0.1, 0.0], [0.3, 0.1, 0.0]])
    reference = quintic_swing_reference(
        start,
        goal,
        phase=torch.tensor([-0.2, 1.2]),
        apex_height=torch.tensor([0.1, 0.1]),
        swing_duration_s=0.32,
    )
    torch.testing.assert_close(reference.position, torch.stack((start[0], goal[1])))
    torch.testing.assert_close(reference.velocity, torch.zeros_like(start))
    torch.testing.assert_close(reference.acceleration, torch.zeros_like(start))


def test_non_positive_duration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        quintic_swing_reference(
            torch.zeros((1, 3)),
            torch.ones((1, 3)),
            phase=torch.tensor([0.5]),
            apex_height=torch.tensor([0.1]),
            swing_duration_s=0.0,
        )
```

Add `import pytest` at the top of the test file.

- [ ] **Step 5: Run trajectory tests and verify GREEN**

Run the command from Step 2.

Expected: `6 passed`.

- [ ] **Step 6: Commit the trajectory**

```bash
git add \
  source/instinctlab/instinctlab_foothold/trajectory.py \
  tests/parkour/foothold/test_trajectory.py
git diff --cached --check
git commit -m "feat(foothold): add swing reference trajectory"
```

---

### Task 4: Export Task 3 and run the regression suite

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/__init__.py`
- Modify: `tests/parkour/foothold/test_state_machine.py`

**Interfaces:**
- Produces: stable top-level imports for all Task 3 public classes and functions.

- [ ] **Step 1: Write the failing public-API test**

Append to `test_state_machine.py`:

```python
import instinctlab_foothold


def test_task3_types_are_public_package_api():
    expected_names = (
        "GaitMachineConfig",
        "GaitMachineState",
        "SwingReference",
        "advance_gait",
        "gait_phase",
        "initial_gait_state",
        "quintic_swing_reference",
    )
    for name in expected_names:
        assert hasattr(instinctlab_foothold, name)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab" conda run -n hiking python -m pytest \
  -p no:cacheprovider tests/parkour/foothold/test_state_machine.py \
  tests/parkour/foothold/test_trajectory.py -q
```

Expected: only the public-API test fails.

- [ ] **Step 3: Export the public API**

Add imports from `.state_machine` and `.trajectory` to `instinctlab_foothold/__init__.py`, and add these exact strings to `__all__`:

```python
"GaitMachineConfig",
"GaitMachineState",
"SwingReference",
"advance_gait",
"gait_phase",
"initial_gait_state",
"quintic_swing_reference",
```

- [ ] **Step 4: Run complete verification**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab" conda run -n hiking python -m pytest \
  -p no:cacheprovider tests/parkour/foothold -q
git diff --check
git status --short --branch
```

Expected: all Task 1–3 tests pass, `git diff --check` has no output, and only intended Task 3 files are modified.

- [ ] **Step 5: Commit and push the integration**

```bash
git add \
  source/instinctlab/instinctlab_foothold/__init__.py \
  tests/parkour/foothold/test_state_machine.py
git diff --cached --check
git commit -m "feat(foothold): export gait and trajectory API"
git push
git status --short --branch
```

Expected: the branch is clean and synchronized with `fork/feat/foothold-01-flat-tracking`.
