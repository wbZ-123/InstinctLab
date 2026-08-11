# Contact-Adaptive Foothold Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed analytic recovery step with contact-event adaptation and a planner-free autonomous stabilization mode driven by the existing motor policy.

**Architecture:** Keep normal HOLD/planning/locked-SWING behavior unchanged. Route early contact, late contact, support loss, and pre-liftoff planning failure through explicit event responses; enter `RECOVERY` only for physical instability, with planning and motion-command tracking disabled until measured stability persists. Recovery exit reconstructs support roles from confirmed contacts and always starts a fresh normal HOLD transaction.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab manager-based environments, Instinct-RL/WASABI, pytest.

## Global Constraints

- Do not add a second motor policy or recovery network.
- Do not change learned-planner architecture or PPO update rules.
- Do not execute a hand-authored analytic recovery foothold.
- Do not modify unrelated locomotion or parkour task behavior.
- Do not invent stability thresholds: derive them from successful normal-HOLD samples produced by the existing trained checkpoint.
- Do not reinterpret a locked world-frame foothold in a changing support frame.
- Keep all runtime paths vectorized; no per-environment Python loops or GPU-to-CPU synchronization in the control loop.
- Preserve the dirty worktree and commit only files changed by each task.

## File Map

- Create `source/instinctlab/instinctlab_foothold/contact_adaptation.py`: pure, vectorized contact-event and stabilization decisions.
- Create `tests/parkour/foothold/test_contact_adaptation.py`: focused tests for physical stability and event routing.
- Modify `source/instinctlab/instinctlab_foothold/state_machine.py`: consume contact-adaptation decisions and remove analytic-recovery transaction state.
- Modify `source/instinctlab/instinctlab_foothold/types.py`: retain anomaly reason states while using `RECOVERY` only for autonomous stabilization.
- Modify `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`: compute signals, implement downward touchdown search, freeze/clear transactions, and reconstruct support roles.
- Modify `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py`: remove analytic recovery target parameters and add calibrated stabilization/search bounds.
- Modify `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`: publish confirmed contacts, event response, stabilization state, and planning-failure state.
- Modify `scripts/instinct_rl/play.py`: optionally capture successful-HOLD stability samples for calibration.
- Modify `scripts/foothold_train.sh`: validate and forward the recovery-calibration JSON path.
- Modify `source/instinctlab/instinctlab_foothold/learned_target.py`: remove geometry-only recovery routing.
- Delete `source/instinctlab/instinctlab_foothold/recovery_target.py`: no fixed recovery foothold remains.
- Modify `source/instinctlab/instinctlab/envs/mdp/observations/foothold.py`: expose confirmed contacts and zero the effective command/reference channels during stabilization.
- Modify `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`: provide recovery masks for command and planner rewards.
- Modify `source/instinctlab/instinctlab/envs/mdp/terminations/general.py`: expose the distinct planner-wait failure termination.
- Modify `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`: connect recovery-gated observations/rewards and distinct planning-failure termination.
- Modify `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`: mask AMP auxiliary reward for stabilization samples without changing PPO optimization.
- Modify `source/instinctlab/instinctlab/monitors/foothold.py`: replace recovery-step metrics with event-response and stabilization metrics.
- Modify play/log analysis tests and documentation to reflect the new semantics.

---

### Task 1: Pure Contact-Adaptation Decisions

**Files:**
- Create: `source/instinctlab/instinctlab_foothold/contact_adaptation.py`
- Create: `tests/parkour/foothold/test_contact_adaptation.py`
- Modify: `source/instinctlab/instinctlab_foothold/__init__.py`

**Interfaces:**
- Produces `StabilitySignals`, `StabilityBounds`, `EventResponse`, `support_roles_from_contacts()`, `stability_ready()`, and `response_for_event()`.
- All tensor outputs have shape `(num_envs,)`; support and swing sides use `0=left`, `1=right`, and `-1=unknown`.

- [ ] **Step 1: Write failing unit tests for contact roles and continuous stability**

```python
def test_support_roles_follow_confirmed_physical_contacts():
    support, swing = support_roles_from_contacts(
        confirmed_contact=torch.tensor([[True, False], [False, True], [True, True], [False, False]]),
        previous_swing_side=torch.tensor([0, 1, 0, 1]),
    )
    torch.testing.assert_close(support, torch.tensor([0, 1, 1, -1]))
    torch.testing.assert_close(swing, torch.tensor([1, 0, 0, -1]))


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
    ready, next_elapsed = stability_ready(signals, bounds, elapsed, dt=0.02)
    torch.testing.assert_close(ready, torch.tensor([True, False]))
    torch.testing.assert_close(next_elapsed, torch.tensor([0.10, 0.0]))
```

- [ ] **Step 2: Run the focused tests and verify import failures**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_contact_adaptation.py
```

Expected: FAIL because `instinctlab_foothold.contact_adaptation` does not exist.

- [ ] **Step 3: Implement the pure vectorized decision module**

```python
from dataclasses import dataclass
from enum import IntEnum
import torch


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


@dataclass(frozen=True)
class StabilitySignals:
    confirmed_contact: torch.Tensor
    body_tilt_rad: torch.Tensor
    body_angular_speed_rad_s: torch.Tensor
    body_horizontal_speed_m_s: torch.Tensor
    support_slip_m_s: torch.Tensor


def support_roles_from_contacts(confirmed_contact, previous_swing_side):
    left, right = confirmed_contact[:, 0], confirmed_contact[:, 1]
    support = torch.full_like(previous_swing_side, -1)
    swing = torch.full_like(previous_swing_side, -1)
    support[left & ~right], swing[left & ~right] = 0, 1
    support[right & ~left], swing[right & ~left] = 1, 0
    both = left & right
    swing[both] = 1 - previous_swing_side[both]
    support[both] = 1 - swing[both]
    return support, swing


def stability_ready(signals, bounds, elapsed_s, dt):
    stable = (
        torch.any(signals.confirmed_contact, dim=-1)
        & (signals.body_tilt_rad <= bounds.max_tilt_rad)
        & (signals.body_angular_speed_rad_s <= bounds.max_angular_speed_rad_s)
        & (signals.body_horizontal_speed_m_s <= bounds.max_horizontal_speed_m_s)
        & (signals.support_slip_m_s <= bounds.max_support_slip_m_s)
    )
    next_elapsed = torch.where(stable, elapsed_s + dt, torch.zeros_like(elapsed_s))
    return next_elapsed >= bounds.dwell_s - 1.0e-6, next_elapsed
```

Implement `response_for_event()` as a vectorized priority table: pre-liftoff invalid plan → `RETRY_PLAN`; confirmed early contact with stable support → `ACCEPT_TOUCHDOWN`; nominal swing timeout with remaining vertical reach → `SEARCH_DOWN`; support loss with stable opposite support → `REASSIGN_SUPPORT`; otherwise physical instability → `STABILIZE`.

- [ ] **Step 4: Run the new test module**

Expected: all tests PASS.

- [ ] **Step 5: Commit the pure decision layer**

```bash
git add source/instinctlab/instinctlab_foothold/contact_adaptation.py source/instinctlab/instinctlab_foothold/__init__.py tests/parkour/foothold/test_contact_adaptation.py
git commit -m "Add contact-adaptation decisions"
```

### Task 2: State-Machine Semantics Without Analytic Recovery Steps

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/state_machine.py`
- Modify: `tests/parkour/foothold/test_state_machine.py`
- Modify: `tests/parkour/foothold/test_types.py`

**Interfaces:**
- Consumes `EventResponse` and `stability_ready` outputs from Task 1.
- Changes `advance_gait()` inputs by adding `event_response`, `stabilization_ready`, `late_search_exhausted`, and `planning_failure`; removes `recovery_step_pending` and `recovery_step_active` state.

- [ ] **Step 1: Replace legacy recovery-step tests with event-response tests**

Add seven tests named `test_confirmed_early_contact_accepts_touchdown_without_recovery`, `test_late_contact_search_preserves_swing_side_until_contact`, `test_invalid_plan_before_liftoff_stays_hold_and_retries`, `test_stable_single_support_reassigns_support_without_recovery_step`, `test_unstable_or_no_support_enters_recovery`, `test_recovery_requires_continuous_stability_before_hold`, and `test_plan_wait_timeout_sets_distinct_planning_failure`. Build each from `initial_gait_state(1, "cpu")`, set the required source mode and contact timers directly, then call `advance_gait()` once with a `0.02 s` step.

The key expectations are:

```python
assert early.mode.item() == GaitState.TOUCHDOWN_CONFIRM
assert late.mode.item() == GaitState.OVERDUE
assert retry.mode.item() == GaitState.HOLD
assert unstable.mode.item() == GaitState.RECOVERY
assert recovered.mode.item() == GaitState.HOLD
assert recovered.swing_side.item() == expected_swing_from_contacts
```

- [ ] **Step 2: Run only the state-machine tests and verify legacy expectations fail**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_state_machine.py tests/parkour/foothold/test_types.py
```

Expected: FAIL on old recovery-step behavior and missing new inputs.

- [ ] **Step 3: Refactor state and transitions**

`GaitMachineState` must contain:

```python
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
    stabilization_elapsed_s: torch.Tensor
    late_search_elapsed_s: torch.Tensor
    planning_failure: torch.Tensor
```

Apply transitions in this priority order:

```python
# 1. Existing touchdown confirmation completes a normal step.
# 2. ACCEPT_TOUCHDOWN truncates early swing into TOUCHDOWN_CONFIRM.
# 3. SEARCH_DOWN remains OVERDUE and preserves swing_side.
# 4. REASSIGN_SUPPORT returns to HOLD using contact-derived swing_side.
# 5. RETRY_PLAN remains HOLD, clears the failed transaction externally, and never starts swing.
# 6. STABILIZE enters RECOVERY and clears swing/liftoff timers.
# 7. RECOVERY exits only when stabilization_ready is true, then returns to fresh HOLD.
# 8. Planning timeout marks planning_failure; it does not enter RECOVERY.
```

Delete all `recovery_step_pending`, `recovery_step_active`, and fixed `recovery_hold_s` behavior.

- [ ] **Step 4: Run state-machine tests until all pass**

Expected: all focused tests PASS with no recovery-step assertions remaining.

- [ ] **Step 5: Commit state-machine behavior**

```bash
git add source/instinctlab/instinctlab_foothold/state_machine.py tests/parkour/foothold/test_state_machine.py tests/parkour/foothold/test_types.py
git commit -m "Replace analytic recovery transitions with contact adaptation"
```

### Task 3: Calibrate and Publish Physical Stability Signals

**Files:**
- Create: `tests/parkour/foothold/calibrate_recovery_stability.py`
- Create: `tests/parkour/foothold/test_calibrate_recovery_stability.py`
- Modify: `scripts/instinct_rl/play.py`
- Modify: `scripts/foothold_train.sh`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`

**Interfaces:**
- Publishes `confirmed_foot_contact`, `body_tilt_rad`, `body_angular_speed_rad_s`, `body_horizontal_speed_m_s`, `support_slip_m_s`, `stabilization_active`, and `stabilization_ready`.
- Calibration JSON keys exactly match `StabilityBounds` fields.

- [ ] **Step 1: Add failing tests for metric calculation and deterministic calibration**

```python
def test_calibration_uses_successful_hold_quantiles_only():
    result = calibrate(samples, quantile=0.99, dwell_s=0.10)
    assert result == {
        "max_tilt_rad": 0.198,
        "max_angular_speed_rad_s": 0.396,
        "max_horizontal_speed_m_s": 0.297,
        "max_support_slip_m_s": 0.0396,
        "dwell_s": 0.10,
    }
```

Use a synthetic tensor whose 0.99 quantiles are exactly the values above; include rejected rows marked non-HOLD, unstable touchdown, and reset so the test proves they are excluded.

- [ ] **Step 2: Run calibration and data tests; verify failure**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_calibrate_recovery_stability.py tests/parkour/foothold/test_foothold_planner_data.py
```

- [ ] **Step 3: Implement vectorized signal calculation and JSON calibration**

Compute:

```python
body_tilt_rad = torch.acos(torch.clamp(-projected_gravity_b[:, 2], -1.0, 1.0))
body_angular_speed_rad_s = torch.linalg.vector_norm(root_ang_vel_b, dim=-1)
body_horizontal_speed_m_s = torch.linalg.vector_norm(root_lin_vel_b[:, :2], dim=-1)
support_slip_m_s = torch.linalg.vector_norm(support_foot_lin_vel_w[:, :2], dim=-1)
```

The calibration tool must select only samples with normal `HOLD`, at least one confirmed contact, no reset, no anomaly mode in the previous dwell, and episode length at least 80% of the configured maximum. It writes the empirical 0.99 quantile for each metric and the fixed contact-confirmation-aligned dwell `0.10 s` to JSON. Add `--recovery_stability_sample_path` to `play.py`; when supplied, append only those accepted rows to device buffers, concatenate once when play exits, and write one `.pt` file. `foothold_train.sh` reads `FOOTHOLD_RECOVERY_CALIBRATION_FILE`, validates that the path exists, and forwards it to the environment configuration. The runtime config loads that JSON; if contact-adaptive recovery is enabled and the file is missing or any bound is non-positive, configuration validation raises a clear `ValueError` rather than silently using guessed values.

- [ ] **Step 4: Capture calibration data from the existing successful checkpoint**

Run the existing play diagnostic for 4,000 control steps with the new stability-sample output enabled, then run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python tests/parkour/foothold/calibrate_recovery_stability.py --input logs/foothold_stability_samples.pt --output logs/foothold_recovery_stability_g1.json --quantile 0.99 --dwell-s 0.10
```

Expected output contains five finite positive keys and reports the accepted sample count; accepted sample count must exceed 1,000 before proceeding.

- [ ] **Step 5: Run focused tests and commit**

```bash
git add tests/parkour/foothold/calibrate_recovery_stability.py tests/parkour/foothold/test_calibrate_recovery_stability.py scripts/instinct_rl/play.py scripts/foothold_train.sh source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py
git commit -m "Calibrate foothold recovery stability signals"
```

### Task 4: Planner Integration and Downward Contact Search

**Files:**
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Delete: `source/instinctlab/instinctlab_foothold/recovery_target.py`
- Test: `tests/parkour/foothold/test_foothold_planner_data.py`
- Test: `tests/parkour/foothold/test_learned_foothold_planner.py`
- Test: `tests/parkour/foothold/test_clearance.py`

**Interfaces:**
- Consumes Task 1 responses and Task 3 signals.
- Produces a locked late-contact search reference with invariant world XY and bounded Z descent.

- [ ] **Step 1: Add failing planner tests**

Add tests named `test_late_search_keeps_locked_world_xy_and_descends_only_z`, `test_late_search_never_swaps_support_before_confirmed_contact`, `test_recovery_clears_nominal_and_learned_transactions`, `test_recovery_emits_no_learned_planner_event`, `test_recovery_exit_creates_fresh_world_frame_plan`, and `test_pre_liftoff_invalid_plan_retries_without_swing`. Reuse the existing fake planner-data fixtures and mixed environment batches so each cache/event assertion is checked without starting Isaac Sim.

For the search test, two consecutive references must satisfy:

```python
torch.testing.assert_close(second[:, :2], first[:, :2])
torch.testing.assert_close(second[:, 2], first[:, 2] - cfg.late_contact_search_speed_m_s * cfg.control_dt_s)
```

- [ ] **Step 2: Run focused planner tests and verify failures**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_foothold_planner_data.py tests/parkour/foothold/test_learned_foothold_planner.py tests/parkour/foothold/test_clearance.py
```

- [ ] **Step 3: Implement event wiring and delete legacy routing**

Add late-search behavior as derived properties of existing geometry/timing rather than introducing independently tuned recovery parameters:

```python
@property
def late_contact_search_speed_m_s(self) -> float:
    return self.touchdown_z_tolerance_m / self.overdue_s

@property
def late_contact_search_max_distance_m(self) -> float:
    return self.max_foothold_step_height_m
```

During `RECOVERY`, set planner cache-valid masks, locked target masks, learned proposal masks, and learned event masks false. During late search, preserve locked target world XY and support/body-yaw snapshots; descend the reference along the queried terrain normal until confirmed contact, maximum distance, or kinematic reach. Remove `recovery_step_*` parameters, branches, cache fields, route masks, and the import/file for `recovery_target.py`.

- [ ] **Step 4: Ensure active SWING never revalidates a locked target in a new frame**

Move all height, reachability, danger-cylinder, and clearance validation before the transition into SWING. After lift-off, only contact-event handling may truncate or vertically extend the transaction; ordinary tracking error must not overwrite `target_foothold_w`, support world position, body-yaw snapshot, or trajectory start.

- [ ] **Step 5: Run planner tests and commit**

```bash
git add source/instinctlab/instinctlab/sensors/foothold_planner source/instinctlab/instinctlab_foothold/learned_target.py source/instinctlab/instinctlab_foothold/recovery_target.py tests/parkour/foothold/test_foothold_planner_data.py tests/parkour/foothold/test_learned_foothold_planner.py tests/parkour/foothold/test_clearance.py
git commit -m "Integrate contact-adaptive foothold execution"
```

### Task 5: Stabilization Observations and Reward/AMP Gating

**Files:**
- Modify: `source/instinctlab/instinctlab/envs/mdp/observations/foothold.py`
- Modify: `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Modify: `tests/parkour/foothold/test_observation_foothold.py`
- Modify: `tests/parkour/foothold/test_reward_foothold.py`
- Modify: `tests/parkour/foothold/test_event_gated_foothold_ppo.py`

**Interfaces:**
- Policy foothold observation replaces the obsolete recovery-step bit with `stabilization_active` and appends confirmed left/right contacts.
- `recovery_mask_observation()` returns `(num_envs, 1)` float mask for auxiliary reward gating.

- [ ] **Step 1: Add failing observation and reward-mask tests**

Add tests named `test_stabilization_observation_zeros_effective_command_and_reference_errors`, `test_foothold_observation_exposes_both_confirmed_contacts`, `test_command_tracking_reward_is_zero_during_stabilization`, `test_amp_auxiliary_reward_is_zero_during_stabilization_only`, and `test_no_planner_reward_event_is_emitted_during_stabilization`.

Each test must use a mixed two-environment batch so the same call proves the normal environment remains numerically identical while the stabilization environment is masked.

- [ ] **Step 2: Run focused tests and verify failures**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_observation_foothold.py tests/parkour/foothold/test_reward_foothold.py tests/parkour/foothold/test_event_gated_foothold_ppo.py
```

- [ ] **Step 3: Implement mask-preserving wrappers**

Implement named Isaac-Lab reward-term wrappers with the same parameter
signatures as the original terms:

```python
def stabilization_active(env, sensor_name="foothold_planner"):
    return env.scene.sensors[sensor_name].data.stabilization_active.bool()


def recovery_gated_track_lin_vel_xy_exp(
    env, command_name, std, asset_cfg, sensor_name="foothold_planner"
):
    reward = track_lin_vel_xy_exp(env, command_name, std, asset_cfg)
    return torch.where(stabilization_active(env, sensor_name), 0.0, reward)


def recovery_gated_track_ang_vel_z_exp(
    env, command_name, std, asset_cfg, sensor_name="foothold_planner"
):
    reward = track_ang_vel_z_exp(env, command_name, std, asset_cfg)
    return torch.where(stabilization_active(env, sensor_name), 0.0, reward)
```

Use wrappers only for command velocity, foothold, swing, and touchdown terms in the foothold-enabled parkour config. Leave body orientation, angular velocity, legal-contact, action regularization, termination, and the existing bounded recovery cost unchanged. In `EventGatedWasabiPPO.compute_auxiliary_reward()`, call the parent implementation and then apply the per-environment recovery mask; do not change discriminator optimization, PPO minibatches, KL, or learning rates.

- [ ] **Step 4: Update policy observation dimensions through configuration, not hard-coded indexes**

Normal observation channels remain unchanged. Stabilization exposes zero effective command and zero target/reference errors while preserving proprioception, `stabilization_active`, and confirmed contacts. Update tests that assert observation width to account for the two contact booleans and renamed recovery bit.

- [ ] **Step 5: Run focused tests and commit**

```bash
git add source/instinctlab/instinctlab/envs/mdp/observations/foothold.py source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py tests/parkour/foothold/test_observation_foothold.py tests/parkour/foothold/test_reward_foothold.py tests/parkour/foothold/test_event_gated_foothold_ppo.py
git commit -m "Gate policy objectives during autonomous stabilization"
```

### Task 6: Planning-Failure Termination and Diagnostics

**Files:**
- Modify: `source/instinctlab/instinctlab/envs/mdp/terminations/general.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Modify: `source/instinctlab/instinctlab/monitors/foothold.py`
- Modify: `scripts/instinct_rl/play_debug.py`
- Modify: `tests/parkour/foothold/test_foothold_monitor.py`
- Modify: `tests/parkour/foothold/test_play_debug.py`
- Modify: `tests/parkour/foothold/inspect_foothold_tensorboard.py`
- Modify: `tests/parkour/foothold/test_inspect_foothold_tensorboard.py`

**Interfaces:**
- Publishes event response counts, stabilization entry/success/duration, post-exit re-entry, late-search success, planning-retry success, and planning-failure termination.

- [ ] **Step 1: Add failing monitor and play-debug tests**

Add tests named `test_monitor_reports_each_contact_event_response`, `test_monitor_reports_stabilization_success_and_duration`, `test_monitor_reports_post_exit_reentry_rate`, `test_monitor_reports_late_search_and_plan_retry_success`, `test_play_debug_keeps_original_anomaly_and_selected_response`, and `test_planning_timeout_terminates_with_planning_failure_reason`. Use deterministic five-step sequences and assert exact counter ratios; for play debug, parse the emitted JSON payload and assert `anomaly_reason == "STANCE_LOST"` and `event_response == "STABILIZE"` coexist in the same record.

- [ ] **Step 2: Run focused diagnostics tests and verify failures**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_foothold_monitor.py tests/parkour/foothold/test_play_debug.py tests/parkour/foothold/test_inspect_foothold_tensorboard.py
```

- [ ] **Step 3: Implement counters without control-loop synchronization**

All counters remain device tensors and are reduced only through the existing monitor logging interval. Replace `recovery_step_fraction` and `recovery_step_entry_step_rate` with:

```text
early_contact_accept_rate
late_contact_search_entry_rate
late_contact_search_success_rate
support_reassignment_rate
planning_retry_entry_rate
planning_retry_success_rate
stabilization_entry_rate
stabilization_success_rate
stabilization_mean_duration_s
stabilization_post_exit_reentry_rate
planning_failure_termination_rate
```

The play payload must contain both `anomaly_reason` and `event_response`; retaining an old marker on screen remains allowed because it visibly indicates no new valid plan.

- [ ] **Step 4: Run focused diagnostics tests and commit**

```bash
git add source/instinctlab/instinctlab/envs/mdp/terminations/general.py source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py source/instinctlab/instinctlab/monitors/foothold.py scripts/instinct_rl/play_debug.py tests/parkour/foothold/test_foothold_monitor.py tests/parkour/foothold/test_play_debug.py tests/parkour/foothold/inspect_foothold_tensorboard.py tests/parkour/foothold/test_inspect_foothold_tensorboard.py
git commit -m "Add contact-adaptation recovery diagnostics"
```

### Task 7: Remove Legacy Surface Area and Update Documentation

**Files:**
- Modify: `tests/parkour/foothold/test_foothold_planner_data.py`
- Modify: `tests/parkour/foothold/test_analyze_foothold_play_log.py`
- Modify: `tests/parkour/foothold/analyze_foothold_play_log.py`
- Modify: `docs/foothold_planner_implementation.md`
- Modify: `docs/foothold_parameter_audit.md`

**Interfaces:**
- No `recovery_step_*` runtime/config/logging name remains.

- [ ] **Step 1: Add a static regression test for removed legacy behavior**

```python
def test_analytic_recovery_surface_is_removed():
    source = "\n".join(path.read_text() for path in SOURCE_PATHS)
    assert "recovery_step_length_m" not in source
    assert "recovery_step_velocity_lookahead_s" not in source
    assert "recovery_step_max_length_m" not in source
    assert "recovery_step_width_m" not in source
    assert "compute_recovery_target" not in source
```

- [ ] **Step 2: Run the static test and remove remaining references**

Run:

```bash
rg -n "recovery_step|recovery_target" source/instinctlab tests/parkour/foothold docs scripts
```

Expected after cleanup: no runtime/config references; only migration wording in the design/implementation history may mention the removed names.

- [ ] **Step 3: Document final runtime flow and calibration provenance**

Update the implementation document with this exact sequence:

```text
normal HOLD → plan/score/lock → SWING
early contact → accepted touchdown or stabilization
late contact → vertical search → touchdown or stabilization
support loss → support reassignment or stabilization
invalid pre-liftoff plan → HOLD retry → distinct planning failure on timeout
stabilization success → contact-derived roles → fresh normal HOLD plan
```

Record the calibration checkpoint, accepted sample count, quantile `0.99`, generated JSON path, and resulting five bound values in the parameter audit.

- [ ] **Step 4: Run affected log-analysis tests and commit**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_analyze_foothold_play_log.py tests/parkour/foothold/test_foothold_planner_data.py
git add docs/foothold_planner_implementation.md docs/foothold_parameter_audit.md source/instinctlab/instinctlab_foothold/learned_target.py source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py tests/parkour/foothold/analyze_foothold_play_log.py tests/parkour/foothold/test_analyze_foothold_play_log.py tests/parkour/foothold/test_foothold_planner_data.py
git commit -m "Remove analytic recovery configuration"
```

### Task 8: Full Verification and Short Vectorized Acceptance Run

**Files:**
- No new production files.
- Update only tests or documentation if verification reveals a behavior mismatch; do not tune unrelated task rewards.

**Interfaces:**
- Verifies normal behavior, recovery behavior, planner-event isolation, and runtime performance together.

- [ ] **Step 1: Run the complete foothold unit suite**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold
```

Expected: all tests PASS.

- [ ] **Step 2: Run a 64-environment, 100-iteration diagnostic from scratch**

```bash
FOOTHOLD_RECOVERY_CALIBRATION_FILE="$PWD/logs/foothold_recovery_stability_g1.json" ENABLE_LEARNED_FOOTHOLD_PLANNER=1 RUN_NAME=contact_adaptive_recovery_64env_100it NUM_ENVS=64 MAX_ITERATIONS=100 SAVE_INTERVAL=100 ./scripts/foothold_train.sh 2>&1 | tee logs/contact_adaptive_recovery_64env_100it.txt
```

Expected: finite losses/rewards, nonzero normal planner events, zero learned planner events while stabilization is active, and no configuration or tensor-shape exceptions.

- [ ] **Step 3: Inspect event metrics before a long run**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python tests/parkour/foothold/inspect_foothold_tensorboard.py --latest-pattern contact_adaptive_recovery_64env_100it
```

Acceptance requirements:

```text
planning_failure_termination_rate is finite
late_contact_search_success_rate is finite when searches occur
stabilization_success_rate is finite when entries occur
stabilization_post_exit_reentry_rate is reported
learned planner event count during stabilization is exactly zero
all normal planner/cache invariants pass monitor assertions
```

- [ ] **Step 4: Compare performance at 4,096 environments for 100 iterations**

```bash
FOOTHOLD_RECOVERY_CALIBRATION_FILE="$PWD/logs/foothold_recovery_stability_g1.json" ENABLE_LEARNED_FOOTHOLD_PLANNER=1 RUN_NAME=contact_adaptive_recovery_perf_4096env_100it NUM_ENVS=4096 MAX_ITERATIONS=100 SAVE_INTERVAL=100 ./scripts/foothold_train.sh 2>&1 | tee logs/contact_adaptive_recovery_perf_4096env_100it.txt
grep "collection:" logs/contact_adaptive_recovery_perf_4096env_100it.txt | tail -20
```

Acceptance: after warm-up, collection time must not regress by more than 5% versus the latest event-gated 4,096-environment baseline under the same machine load. A larger regression blocks the 30,000-iteration run and must be profiled before changing behavior.

- [ ] **Step 5: Finish with a clean verification report**

```bash
git status --short
```

Verification itself must not change files. If a mismatch is found, return to
the task that owns that behavior, add a failing regression test there, fix it,
rerun that task, and commit it under that task before repeating Task 8.
