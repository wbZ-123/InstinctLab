# Learned Foothold Nominal-Distance Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore a continuous nominal-foothold learning signal and remove the uncalibrated 6 cm lateral hard gate from learned foothold geometry.

**Architecture:** The analytic planner remains the nominal prior and keeps its existing 18 cm step-width behavior. The learned planner continues to output absolute XY in the frozen stance frame. Learned geometry validates finite values, terrain height, step height, and the reachability ellipse; lateral side/separation is no longer a learned geometric gate. The event reward gives an exact nominal match `+1`, decreases linearly to `0` at 2 cm, then decreases continuously to `-1` at the directional ellipse boundary. The runtime lookahead stored by the planner is used instead of a stale fixed fallback whenever it is valid.

**Tech Stack:** Python, PyTorch, pytest, IsaacLab manager-based environment.

## Global Constraints

- Do not modify AMP, MoE, motor PPO, Recovery, swing trajectory generation, or the analytic nominal target generator.
- Keep world-frame terrain height queries and frozen support-frame target locking unchanged.
- Keep the existing danger-cylinder penetration score and execution safety gate.
- Do not introduce a new meter threshold or a new lateral separation parameter.
- Run tests with `PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH"`.

---

### Task 1: Add failing tests for learned geometry and nominal-distance reward

**Files:**
- Modify: `tests/parkour/foothold/test_learned_foothold_planner.py`
- Modify: `tests/parkour/foothold/test_reward_foothold.py`

**Interfaces:**
- `prepare_learned_foothold_target` and `reframe_cached_world_foothold` no longer accept learned-side hard-gate arguments.
- `learned_foothold_planning_event_reward` returns a bounded event reward using normalized nominal deviation.

- [x] **Step 1: Replace the learned-side rejection test with a geometry-only test**

Replace `test_learned_target_rejects_crossed_or_too_narrow_swing_side` with a test that passes the same XY values without side arguments and asserts that valid height and ellipse points remain geometrically valid.

- [x] **Step 2: Add a cached-target test proving no lateral gate is required**

Call `reframe_cached_world_foothold` with a finite, reachable cached target and no side arguments; assert it remains valid. Keep the existing height and reachability rejection test.

- [x] **Step 3: Add reward tests for continuous nominal deviation**

Construct safe nominal planner data with exact, 9 cm, and 18 cm lateral deviations. Assert the rewards are ordered monotonically and are not all `-1`; assert exact match has the highest reward and the farthest case has the lowest reward.

- [x] **Step 4: Add reward tests for invalid geometry and obstacle penetration**

Assert true height/ellipse invalidity still returns `-1`, while a geometrically valid penetrating target returns the existing negative safety score rather than a geometry penalty.

- [x] **Step 5: Run the focused tests and verify they fail for the intended reason**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
  /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_learned_foothold_planner.py \
  tests/parkour/foothold/test_reward_foothold.py
```

Expected: failures show the old lateral hard gate and old reward precedence, not import or fixture errors.

### Task 2: Remove the learned lateral hard gate

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`

**Interfaces:**
- `prepare_learned_foothold_target(..., normalized_action, origin_w, yaw_w, radius_x, radius_y, max_step_height_m, terrain_height_query_w)` returns the same preparation object and validates only finite/height/reachability geometry.
- `reframe_cached_world_foothold(..., target_w, current_origin_w, current_yaw_w, radius_x, radius_y, max_step_height_m)` performs the same geometry-only recheck.

- [x] **Step 1: Remove `_swing_side_valid` and the two optional side arguments**

Delete the helper and remove `swing_side`/`min_lateral_separation` from both learned-target function signatures and their geometric-valid expressions.

- [x] **Step 2: Stop passing the analytic provider’s 6 cm value into learned-target preparation**

Remove the two `min_lateral_separation` keyword arguments in `FootholdPlannerSensor` learned-target preparation and cached-target reframing. Do not change the analytic `sample_flat_targets` implementation or its 18 cm nominal width.

- [x] **Step 3: Run the Task 1 focused tests**

Expected: all replacement geometry tests pass; no learned target is rejected solely for lateral side/separation.

### Task 3: Replace nominal-closeness reward semantics

**Files:**
- Modify: `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`
- Modify: `tests/parkour/foothold/test_reward_foothold.py`

**Interfaces:**
- Add private helpers `_nominal_deviation_reward` and `_normalized_nominal_deviation_cost`; the former is the public semantic reference and returns a tensor in `[-1, 1]`.

- [x] **Step 1: Add the helper test first**

Test zero delta gives `0`, increasing delta gives a strictly increasing cost, and opposite points in the normalized ellipse are capped at `1`.

- [x] **Step 2: Implement the helper**

Use existing reachability radii and the fixed 2 cm tolerance only:

```python
distance <= 0.02 m: 1 - distance / 0.02
distance > 0.02 m: -(distance - 0.02) / (directional_ellipse_radius - 0.02)
```

- [x] **Step 3: Rewrite the event reward branch**

Keep the existing command-consistency computation and continuous safety score. Use this precedence:

```python
if not execution_safe:
    score = -1
elif not learned_geometric_valid:
    score = -1
elif not learned_safety_valid:
    score = learned_safety
elif nominal_safe:
    score = nominal_deviation_reward
else:
    score = command_consistency - nominal_deviation_cost
```

Mask non-events to zero exactly as before. Do not add a left/right reward term.

- [x] **Step 4: Run reward tests**

Expected: exact nominal is the best safe-nominal result, every nonzero deviation has a negative cost, and invalid/penetrating cases retain their safety semantics.

### Task 4: Align diagnostics and regression tests

**Files:**
- Modify: `scripts/instinct_rl/play_debug.py`
- Modify: `scripts/instinct_rl/play.py`
- Modify: `tests/parkour/foothold/test_play_debug.py`
- Modify: `docs/foothold_parameter_audit.md`

**Interfaces:**
- Event diagnostics report nominal XY deviation and normalized deviation cost instead of treating 6 cm as a learned-side validity signal.

- [x] **Step 1: Replace play diagnostic fields**

Remove learned-side hard-gate fields from the event payload and add `nominal_delta_f`, `nominal_deviation_cost`, and `reward_branch` when the learned event is printed.

- [x] **Step 2: Update play diagnostic tests**

Assert the new fields are present and the removed 6 cm learned-gate fields are absent.

- [x] **Step 3: Document parameter scope**

Clarify in the audit that `min_lateral_separation=0.06` is not used to reject learned proposals; the analytic nominal provider remains unchanged and 18 cm remains a nominal prior awaiting calibration.

- [x] **Step 4: Run the focused diagnostic tests**

Expected: all play-debug and planner-data tests pass.

### Task 5: Full verification and short performance check

**Files:**
- No additional source changes unless a test exposes a regression in the scoped behavior.

- [x] **Step 1: Run all foothold tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
  /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold
```

- [x] **Step 2: Run static checks**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
  /home/zhangweibo/miniconda3/envs/hiking/bin/python -m py_compile \
  source/instinctlab/instinctlab_foothold/learned_target.py \
  source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py \
  source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py \
  scripts/instinct_rl/play_debug.py
git diff --check
```

- [ ] **Step 3: Run a 4096-environment 30-iteration smoke test** (blocked here: no CUDA device)

Confirm the environment starts, learned evaluations occur, the planner no longer reports every centerline proposal as lateral-invalid, and collection/learning timing does not materially change.

- [ ] **Step 4: Inspect the first short-run metrics** (blocked here: no CUDA device)

Check `learned_foothold_geometric_valid_fraction`, `learned_foothold_route_learned_fraction`, nominal deviation cost, and `reward_1` branch diagnostics before deciding on a 30000-iteration run.
