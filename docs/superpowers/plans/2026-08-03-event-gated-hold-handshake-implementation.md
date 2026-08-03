# Event-Gated HOLD Handshake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace normal-step double-support HOLD gating with a one-support-foot planner handshake while retaining strict startup and recovery stabilization.

**Architecture:** `advance_gait` will consume an explicit per-environment HOLD contact-ready mask instead of always deriving double support internally. The foothold sensor will build that mask from the newly landed support foot during normal stepping and from both feet during startup/recovery, and the learned-planner event gate will consume the same mask. Touchdown role changes will preserve physical contact debounce history so the new support foot is not incorrectly treated as newly unconfirmed.

**Tech Stack:** Python 3.11, PyTorch tensor state machine, Isaac Lab manager-based sensor, pytest.

## Global Constraints

- Normal stepping requires only the newly landed support foot to remain contact-confirmed.
- Startup and recovery stabilization still require both feet contact-confirmed before execution can continue.
- Normal HOLD has no additional fixed double-support delay; it only waits for the planner handshake.
- Learned foothold evaluation must complete before a new swing begins.
- Preserve left/right symmetry.
- Do not change `feet_air_time`, swing duration, early-contact threshold, learned-planner reward, safety scoring, observation dimensions, or trajectory generation.
- Do not add per-step GPU allocations or CPU synchronization to the runtime path.

---

### Task 1: Make HOLD Contact Readiness Explicit

**Files:**
- Modify: `tests/parkour/foothold/test_state_machine.py`
- Modify: `source/instinctlab/instinctlab_foothold/state_machine.py`

**Interfaces:**
- Consumes: existing `GaitMachineState.contact_elapsed_s`, `GaitMachineState.no_contact_elapsed_s`, and `advance_gait(...)` inputs.
- Produces: optional `hold_contact_ready: torch.Tensor | None` argument on `advance_gait`; `None` preserves the legacy double-contact default for callers that do not select a scenario.

- [ ] **Step 1: Add failing normal-HOLD behavior tests**

Add tests that directly construct touchdown/HOLD states and assert:

```python
def test_touchdown_preserves_new_support_contact_confirmation():
    cfg = GaitMachineConfig(contact_confirm_s=0.04)
    state = initial_gait_state(1, device="cpu")
    state.mode[:] = GaitState.TOUCHDOWN_CONFIRM
    state.swing_side[:] = 0
    state.contact_elapsed_s[:] = 0.04

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, True]]),
        touchdown_accepted=torch.tensor([True]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
        step_hold_s=torch.tensor([0.0]),
    )

    assert state.swing_side.item() == 1
    assert state.contact_elapsed_s[0, 0] >= cfg.contact_confirm_s


def test_normal_hold_starts_with_only_new_support_contact_ready():
    cfg = GaitMachineConfig()
    state = initial_gait_state(1, device="cpu")
    state.swing_side[:] = 1
    state.hold_required_s[:] = 0.0

    state = advance_gait(
        state=state,
        contact=torch.tensor([[True, False]]),
        touchdown_accepted=torch.tensor([False]),
        planner_valid=torch.tensor([True]),
        dt=0.02,
        cfg=cfg,
        step_hold_s=torch.tensor([0.0]),
        swing_ready=torch.tensor([True]),
        hold_contact_ready=torch.tensor([True]),
    )
    assert state.mode.item() == GaitState.RIGHT_SWING


def test_normal_hold_reports_loss_when_new_support_is_not_ready():
    cfg = GaitMachineConfig(hold_contact_lost_confirm_s=0.04)
    state = initial_gait_state(1, device="cpu")
    state.swing_side[:] = 0
    state.hold_required_s[:] = 0.0

    for _ in range(2):
        state = advance_gait(
            state=state,
            contact=torch.tensor([[True, True]]),
            touchdown_accepted=torch.tensor([False]),
            planner_valid=torch.tensor([True]),
            dt=0.02,
            cfg=cfg,
            step_hold_s=torch.tensor([0.0]),
            swing_ready=torch.tensor([True]),
            hold_contact_ready=torch.tensor([False]),
        )

    assert state.mode.item() == GaitState.HOLD_CONTACT_LOST
```

Add a mirrored left/right parametrization so the same assertions hold for both swing-side values.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_state_machine.py \
-k "touchdown_preserves_new_support or normal_hold"
```

Expected: failures because `advance_gait` does not accept `hold_contact_ready` and clears both debounce histories at touchdown.

- [ ] **Step 3: Implement the minimal state-machine interface**

In `advance_gait`:

```python
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
) -> GaitMachineState:
    confirmed_double_contact = torch.all(
        contact_elapsed_s >= cfg.contact_confirm_s - 1.0e-6,
        dim=-1,
    )
    if hold_contact_ready is None:
        hold_contact_ready = confirmed_double_contact
    else:
        hold_contact_ready = hold_contact_ready.to(
            device=state.mode.device,
            dtype=torch.bool,
        )
        if hold_contact_ready.shape != state.mode.shape:
            raise ValueError(
                "hold_contact_ready must match the number of environments."
            )
```

Use `hold_contact_ready` for `start_swing` and `hold_contact_lost`. Do not clear `contact_elapsed_s` or `no_contact_elapsed_s` on `TOUCHDOWN_CONFIRM -> HOLD`; these tensors describe current physical debounce state and remain valid after foot-role exchange.

- [ ] **Step 4: Run the focused state-machine tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Run the complete state-machine suite**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_state_machine.py
```

Expected: all tests pass after existing fixed-double-support assertions are updated only where they describe normal touchdown HOLD; startup/recovery assertions remain strict.

### Task 2: Wire Scenario-Specific Contact Readiness into the Planner

**Files:**
- Modify: `tests/parkour/foothold/test_learned_foothold_planner.py`
- Modify: `tests/parkour/foothold/test_foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`

**Interfaces:**
- Consumes: Task 1 `advance_gait(..., hold_contact_ready=...)`.
- Produces: `learned_foothold_event_masks(..., hold_contact_ready=...)`; runtime masks for support-only normal HOLD and double-contact startup/recovery HOLD.

- [ ] **Step 1: Add failing event-mask and planner-wiring tests**

Update the pure event-mask test to show that the learned proposal is prepared when the scenario-specific contact mask is true, independent of the future swing foot:

```python
prepare, lock = learned_foothold_event_masks(
    hold=torch.tensor([True, True, False]),
    hold_contact_ready=torch.tensor([True, False, True]),
    nominal_ready=torch.tensor([True, True, True]),
    new_swing=torch.tensor([False, False, True]),
    enable=True,
)
assert prepare.tolist() == [True, False, False]
assert lock.tolist() == [False, False, True]
```

Add source-level wiring assertions that the sensor computes the confirmed new-support contact using `stance_side = 1 - swing_side`, selects double contact for startup or `recovery_step_pending`, passes `hold_contact_ready` to the event mask and state machine, and passes a zero normal `step_hold_s`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_learned_foothold_planner.py \
tests/parkour/foothold/test_foothold_planner_data.py \
-k "event_masks or hold_contact_ready or event_gated_hold"
```

Expected: failures because the event mask still requires `both_contacts_confirmed` and the sensor does not pass scenario-specific readiness.

- [ ] **Step 3: Implement one shared tensor mask in the sensor**

After computing the next contact debounce state:

```python
confirmed_contact = (
    next_contact_elapsed_s >= self.cfg.contact_confirm_s - 1.0e-6
)
both_contacts_confirmed = torch.all(confirmed_contact, dim=-1)
rows = torch.arange(contact.shape[0], device=self._device)
stance_side = 1 - previous_gait_state.swing_side
new_support_confirmed = confirmed_contact[rows, stance_side]
strict_double_support = (
    startup_hold_mask
    | previous_gait_state.recovery_step_pending
)
hold_contact_ready = torch.where(
    strict_double_support,
    both_contacts_confirmed,
    new_support_confirmed,
)
```

Pass that same mask into both event preparation and `advance_gait`. Set the normal touchdown hold-duration tensor to zero; startup continues through `startup_hold_s`, and recovery continues through `recovery_step_pending` plus existing recovery/reset timing. Rename `both_contacts_confirmed` to `hold_contact_ready` only in the pure learned event-mask API.

- [ ] **Step 4: Run the focused planner tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Run state-machine and learned-planner regressions**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_state_machine.py \
tests/parkour/foothold/test_learned_foothold_planner.py \
tests/parkour/foothold/test_foothold_planner_data.py
```

Expected: all tests pass.

### Task 3: Verify Scope and Full Foothold Regression

**Files:**
- Verify only; no new production behavior.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: evidence that the HOLD change did not alter observations, reward dimensions, trajectory generation, or safety scoring.

- [ ] **Step 1: Review the exact diff for scope**

Run:

```bash
git diff -- \
source/instinctlab/instinctlab_foothold/state_machine.py \
source/instinctlab/instinctlab_foothold/learned_target.py \
source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py \
tests/parkour/foothold/test_state_machine.py \
tests/parkour/foothold/test_learned_foothold_planner.py \
tests/parkour/foothold/test_foothold_planner_data.py
```

Expected: only contact gating, touchdown debounce preservation, normal zero-delay HOLD, and their tests changed.

- [ ] **Step 2: Run the full foothold test suite**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold
```

Expected: all tests pass with no collection errors.

- [ ] **Step 3: Check formatting and accidental files**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the six scoped files plus the plan are newly changed by this task, while all pre-existing dirty files remain preserved.

- [ ] **Step 4: Record runtime validation command without starting a long train**

After unit verification, use:

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=event_gated_hold_handshake_4096env_100it \
NUM_ENVS=4096 \
MAX_ITERATIONS=100 \
SAVE_INTERVAL=100 \
./scripts/foothold_train.sh
```

Compare a matched late window against the prior run. `HOLD_CONTACT_LOST` entry rate must fall without increasing planner invalidity, non-finite values, or left/right imbalance.
