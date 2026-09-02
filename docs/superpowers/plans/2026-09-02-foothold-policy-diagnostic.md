# Foothold Policy Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Play diagnostic that locates whether fixed centerline foothold outputs originate in observation propagation, the planner actor, or the SAC critics.

**Architecture:** A focused helper module will create counterfactual copies of a real policy observation, change only the `nominal_foothold` lateral coordinate, run the existing planner feature extractor and deterministic actor, and sweep critic Q values over valid lateral actions. `play.py` will invoke it once per selected environment on the first learned foothold event, using the exact pre-step observation that produced the action.

**Tech Stack:** Python 3.11, PyTorch, IsaacLab Play integration, pytest.

## Global Constraints

- The diagnostic is disabled by default and must not alter actions, planner state, rewards, replay, or training parameters.
- Use the existing checkpoint actor, SAC critics, radial action transform, observation segments, and runtime reachability radii.
- Emit `[FOOTHOLD_POLICY_SENSITIVITY]` and `[FOOTHOLD_Q_SWEEP]` records.
- Diagnose each selected environment at most once.
- A diagnostic exception prints one error and does not stop Play.

---

### Task 1: Pure counterfactual and Q-sweep diagnostic helpers

**Files:**
- Create: `scripts/instinct_rl/foothold_policy_diagnostics.py`
- Create: `tests/parkour/foothold/test_foothold_policy_diagnostics.py`

**Interfaces:**
- Produces: `replace_nominal_lateral(observations, obs_segments, nominal_y_m) -> torch.Tensor`
- Produces: `diagnose_foothold_policy(actor_critic, sac, observation, radius_x_m, radius_y_m, nominal_y_values_m=(-0.18, 0.18), q_grid_size=51) -> dict`
- Produces: `format_policy_sensitivity(result) -> str`
- Produces: `format_q_sweep(result, nominal_y_m) -> str`

- [ ] **Step 1: Write failing tests for counterfactual observation replacement**

Create a small ordered observation layout containing `proprioception`, `nominal_foothold`, and `depth_image`. Verify that replacing the nominal lateral value changes only the second element of `nominal_foothold`, preserves the input tensor, and raises a clear error if the component is absent or not three-dimensional.

- [ ] **Step 2: Run the replacement tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PWD/third_party/instinct_rl:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_foothold_policy_diagnostics.py -k replace
```

Expected: collection/import failure because `foothold_policy_diagnostics.py` does not exist.

- [ ] **Step 3: Implement component slicing and immutable replacement**

Implement a local ordered-segment slice helper using the product of preceding component shapes. Clone the observation, replace exactly the nominal component's lateral column, validate shapes, and return the clone.

- [ ] **Step 4: Run replacement tests and verify GREEN**

Run the Step 2 command. Expected: all replacement tests pass.

- [ ] **Step 5: Write failing tests for actor sensitivity and critic sweep**

Use deterministic fake actor/critic modules where features retain the raw observation, actor lateral mean depends on nominal lateral position, and critic Q is maximal when action lateral position matches the nominal position. Verify the returned result contains the original nominal value, feature delta norm, normalized actor action, decoded lateral foothold, best-Q lateral foothold, center Q, and nominal Q. Verify Q scanning respects the unit-disk lateral limit for the fixed actor longitudinal action.

- [ ] **Step 6: Run diagnostic tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PWD/third_party/instinct_rl:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_foothold_policy_diagnostics.py -k 'diagnose or format'
```

Expected: failure because the diagnostic and formatters are absent.

- [ ] **Step 7: Implement the read-only diagnostic**

For original and counterfactual observations, call `actor_critic.planner_features`, `actor_critic.planner_distribution_from_features`, and `radial_squash`. Keep the normalized longitudinal action fixed, scan legal lateral actions within the unit disk, and evaluate the minimum of `sac.critic_1` and `sac.critic_2`. Convert normalized actions to physical coordinates with runtime radii. Run under `torch.no_grad()` and return CPU scalar/list data only.

- [ ] **Step 8: Implement stable one-line formatters**

Use the exact prefixes from Global Constraints, explicit nominal signs, and centimeter-resolving precision.

- [ ] **Step 9: Run helper tests and commit**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PWD/third_party/instinct_rl:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_foothold_policy_diagnostics.py
```

Expected: PASS.

Commit `scripts/instinct_rl/foothold_policy_diagnostics.py` and its test as `Add foothold policy sensitivity diagnostics`.

### Task 2: Read-only Play integration

**Files:**
- Modify: `scripts/instinct_rl/play.py`
- Modify: `tests/parkour/foothold/test_play_debug.py`

**Interfaces:**
- Consumes: the Task 1 helper functions.
- Produces: CLI flag `--diagnose_foothold_policy`.

- [ ] **Step 1: Write failing source-level Play integration tests**

Verify `play.py` defines `--diagnose_foothold_policy`, preserves the pre-step observation in a distinct variable before calling the policy, and invokes the helper only for a learned planning event whose environment has not already been diagnosed.

- [ ] **Step 2: Run Play integration tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PWD/third_party/instinct_rl:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_play_debug.py
```

Expected: the new integration test fails because the CLI and call are absent.

- [ ] **Step 3: Add the CLI flag and imports**

Add the disabled-by-default boolean flag and import the Task 1 helpers without changing existing debug flags.

- [ ] **Step 4: Preserve the exact policy observation**

Assign `policy_observation = obs` before policy inference and use it for diagnostics after `env.step`. Do not clone or alter the observation used for normal inference.

- [ ] **Step 5: Invoke once per environment on learned events**

Maintain `diagnosed_foothold_policy_env_ids: set[int]`. On an enabled learned event, obtain `ppo_runner.alg.actor_critic`, `ppo_runner.alg.sac`, runtime `outer_radius_x/y`, and the corresponding row of `policy_observation`. Print the helper results and mark success once. On failure, print `[FOOTHOLD_POLICY_DIAGNOSTIC_ERROR]` once without stopping Play.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PWD/third_party/instinct_rl:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_foothold_policy_diagnostics.py \
tests/parkour/foothold/test_play_debug.py
```

Expected: PASS.

- [ ] **Step 7: Run the full foothold test suite**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PWD/third_party/instinct_rl:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold
```

Expected: PASS with no regression.

- [ ] **Step 8: Commit the Play integration**

Commit `scripts/instinct_rl/play.py` and `tests/parkour/foothold/test_play_debug.py` as `Expose foothold policy diagnostics in play`.

### Task 3: Runtime evidence collection

**Files:**
- No source changes.
- Runtime output: `logs/foothold_policy_diagnostic_5000.txt`

**Interfaces:**
- Consumes: `--diagnose_foothold_policy` and the current `model_5000.pt` checkpoint.
- Produces: sensitivity and Q-sweep evidence for root-cause classification.

- [ ] **Step 1: Run the existing stair Play command with diagnostics enabled**

Use the exact run directory containing the current `model_5000.pt`, retain the existing two-environment stair configuration, and add `--diagnose_foothold_policy`. Capture output to `logs/foothold_policy_diagnostic_5000.txt`.

- [ ] **Step 2: Extract diagnostic records**

Run:

```bash
grep -E 'FOOTHOLD_POLICY_SENSITIVITY|FOOTHOLD_Q_SWEEP|FOOTHOLD_POLICY_DIAGNOSTIC_ERROR' \
  logs/foothold_policy_diagnostic_5000.txt
```

Expected: one sensitivity record and two Q-sweep records per selected environment, with no diagnostic error.

- [ ] **Step 3: Classify the root cause before any behavior change**

Use this decision table: zero feature delta means observation/encoder wiring; nonzero feature delta with a fixed actor and correct Q peak means actor/entropy optimization; a center Q peak means critic/reward/replay pipeline; correct diagnostics with fixed normal Play output means inference checkpoint/action routing. Do not modify rewards, constraints, or SAC hyperparameters before the evidence selects a branch.
