# Gait State Machine and Swing Reference Design

**Date:** 2026-07-02

**Status:** Approved for implementation

## 1. Purpose and scope

Task 3 connects the flat foothold target from Task 2 to a time-based stepping process. It adds two simulator-independent PyTorch modules:

- A vectorized gait state machine that decides which foot is swinging, advances phase, debounces contact, and exposes failure reasons.
- A vectorized translational swing reference that produces position, velocity in m/s, and acceleration in m/s².

Task 3 does not read Isaac Lab sensors, judge geometric touchdown error, interpolate foot orientation, compute rewards, or train a policy. Those responsibilities remain in the later Sensor and environment tasks.

## 2. Selected architecture

The state machine and trajectory remain separate:

- `state_machine.py` owns discrete modes, timers, contact confirmation, and transitions.
- `trajectory.py` owns continuous reference mathematics.

Both modules import PyTorch and the lightweight foothold types only. Neither imports `instinctlab`, Isaac Lab, `pxr`, or Omniverse.

This separation lets the Sensor use the same state machine with flat, oracle-map, and depth-map target providers, while trajectory mathematics can be tested without contact logic.

## 3. Gait state machine

### 3.1 Public interface

```python
@dataclass(frozen=True)
class GaitMachineConfig:
    reset_hold_s: float = 0.40
    swing_s: float = 0.32
    contact_confirm_s: float = 0.04
    early_contact_phase: float = 0.65
    overdue_s: float = 0.12


@dataclass
class GaitMachineState:
    mode: torch.Tensor
    swing_side: torch.Tensor
    elapsed_s: torch.Tensor
    hold_elapsed_s: torch.Tensor
    contact_elapsed_s: torch.Tensor
    no_contact_elapsed_s: torch.Tensor
    swing_has_lifted: torch.Tensor


initial_gait_state(num_envs: int, device: torch.device | str) -> GaitMachineState


def advance_gait(
    state: GaitMachineState,
    contact: torch.Tensor,
    touchdown_accepted: torch.Tensor,
    planner_valid: torch.Tensor,
    dt: float,
    cfg: GaitMachineConfig,
) -> GaitMachineState
```

`contact` has shape `[N, 2]` with side index `0` for left and `1` for right. `touchdown_accepted` has shape `[N]` and is computed by the future Sensor from target error and kinematic evidence. This keeps geometric acceptance outside the state machine while making rejected contact explicit.

`mode` stores stable `GaitState` integer values. `swing_side` uses the same `0 = left`, `1 = right` convention. `elapsed_s` is time in the current swing. The normalized phase is `clamp(elapsed_s / swing_s, 0, 1)`.

### 3.2 Normal transitions

Every environment starts in `HOLD`. It must remain in double support with a valid plan for `reset_hold_s`. The first swing foot is deterministically the left foot.

```text
HOLD
  -> LEFT_SWING
  -> TOUCHDOWN_CONFIRM
  -> RIGHT_SWING
  -> TOUCHDOWN_CONFIRM
  -> LEFT_SWING
```

`TOUCHDOWN_CONFIRM` is observable for one control update. On the next update, the machine swaps the swing side, resets swing time and contact timers, and starts the opposite swing.

A touchdown succeeds only when all conditions hold:

- The selected swing foot has first completed a confirmed liftoff.
- The swing contact has remained true for `contact_confirm_s`.
- The phase is not earlier than `early_contact_phase`.
- `touchdown_accepted` is true.

Late contact that is not geometrically accepted does not swap the feet. It remains unresolved until a valid touchdown or the overdue deadline.

At the start of each swing, the new swing foot is normally still touching the ground. `swing_has_lifted` therefore starts false and becomes true only after that foot has remained out of contact for `contact_confirm_s`. Swing-foot contact is not classified as early contact or touchdown before confirmed liftoff. Without this gate, the old stance contact would be misclassified as an immediate collision after the feet swap.

### 3.3 Failure transitions and priority

Transition priority is:

1. Invalid plan.
2. Lost stance contact after contact confirmation time.
3. Confirmed early swing-foot contact.
4. Confirmed accepted touchdown.
5. Overdue swing.

Failures are observable for one control update:

```text
active state -> PLAN_INVALID  -> RECOVERY
active swing -> STANCE_LOST   -> RECOVERY
active swing -> EARLY_CONTACT -> RECOVERY
active swing -> OVERDUE       -> RECOVERY
```

`RECOVERY` is latched in Task 3. A later Sensor or environment reset reinitializes it. This avoids silently resuming normal stepping without a recovery planner.

Contact confirmation is time-based rather than sample-count based. A true contact accumulates `contact_elapsed_s` and resets `no_contact_elapsed_s`; a false contact does the opposite. This supports both touchdown confirmation and stance-loss/liftoff confirmation while preserving behavior if the control period changes.

`dt`, `reset_hold_s`, `swing_s`, `contact_confirm_s`, and `overdue_s` must be positive. Invalid timing configuration raises `ValueError`.

## 4. Swing reference

### 4.1 Public interface

```python
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
) -> SwingReference
```

Inputs are vectorized over environments. `start` and `goal` have shape `[N, 3]`; `phase` and `apex_height` have shape `[N]`. Duration may be one positive scalar or one value per environment.

The function clamps phase to `[0, 1]`. At and outside the boundaries it returns the exact endpoint with zero velocity and acceleration.

Every duration must be positive. A non-positive scalar or tensor entry raises `ValueError`.

### 4.2 Quintic mathematics

The blend and its phase derivatives are:

```text
s(u)   = 10u³ - 15u⁴ + 6u⁵
s'(u)  = 30u² - 60u³ + 30u⁴
s''(u) = 60u - 180u² + 120u³
```

Horizontal X/Y motion uses one blend over the full phase. Vertical motion uses two blends:

```text
start_z -> apex_z -> goal_z
apex_z = max(start_z, goal_z) + apex_height
```

The first vertical segment uses local phase `2u`; the second uses `2u - 1`. Both meet the apex with zero vertical velocity and acceleration. Horizontal motion continues through the apex rather than stopping mid-step.

Derivatives are converted to physical units inside the function:

```text
velocity     = d(position)/du / swing_duration_s
acceleration = d²(position)/du² / swing_duration_s²
```

This prevents the Sensor and reward code from accidentally treating normalized phase derivatives as m/s.

Task 3 covers translational reference only. Target yaw and surface-normal interpolation are added when the Sensor owns the full target pose.

## 5. Testing strategy

State-machine tests cover:

- The exact reset hold boundary.
- Independent vectorized environments.
- Invalid-plan reason visibility followed by latched recovery.
- Contact debouncing in seconds.
- Liftoff confirmation before interpreting swing-foot contact.
- Early contact classification.
- Rejected late contact.
- Accepted touchdown and left/right alternation.
- Stance loss.
- Overdue classification.

Trajectory tests cover:

- Exact start and goal position.
- Zero endpoint velocity and acceleration.
- Apex height at phase `0.5`.
- Position, velocity, and acceleration continuity at the apex.
- Physical velocity scaling when duration changes.
- Phase clamping.

All Task 3 tests run on CPU without `SimulationApp`.

## 6. Integration boundary

Task 2 supplies the accepted target position and feasible navigation velocity. Task 3 supplies gait mode, swing side, normalized phase, and translational swing reference. Task 4 will combine them in `FootholdPlannerSensor`, compute `touchdown_accepted` from measured sole state, and expose the authoritative structured data used by observations, rewards, metrics, and visualization.
