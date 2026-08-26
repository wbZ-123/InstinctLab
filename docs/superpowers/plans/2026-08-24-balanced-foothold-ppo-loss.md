# Balanced Learned Foothold PPO Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Balance the learned foothold actor surrogate loss between nominal-safe and nominal-unsafe planning events without changing reward semantics, motor PPO, recovery, or trajectory execution.

**Architecture:** The environment will expose two mutually exclusive branch masks alongside the existing planner-event mask. Rollout storage will preserve both masks, and PPO will compute branch-specific masked means before combining them with equal weights when both branches are present. Planner value and entropy terms remain masked over all planner events.

**Tech Stack:** Python 3.11, PyTorch, pytest, existing IsaacLab/InstinctRL rollout and PPO interfaces.

## Global Constraints

- Keep `learned_foothold_planning_event_reward` unchanged.
- Keep motor PPO, AMP, MoE, Recovery, swing tracking, and depth encoder behavior unchanged.
- Do not add new distance thresholds or reverse the PPO surrogate sign for unsafe events.
- Non-planner transitions must have both branch masks false.
- Empty branch masks must return a finite zero tensor with a gradient path.

---

### Task 1: Add branch-mask storage and PPO helper tests

**Files:**
- Modify: `tests/parkour/foothold/test_foothold_rollout_storage.py`
- Modify: `tests/parkour/foothold/test_event_gated_foothold_ppo.py`

**Interfaces:**
- `FootholdTransition.foothold_nominal_safe_event` and `foothold_nominal_unsafe_event` are boolean tensors shaped `(num_envs,)`.
- `FootholdMiniBatch` exposes the same two flattened boolean tensors.
- `balanced_event_masked_mean(values, event_mask, nominal_safe_event, nominal_unsafe_event)` returns the balanced planner loss plus the two branch means; the existing `grouped_clipped_surrogates` return shape stays backward-compatible.

- [ ] **Step 1: Write failing storage tests**

Add transition setup with safe/unsafe masks and assert storage and selected minibatches retain both masks. Add rejection tests for missing, non-boolean, mismatched, overlapping, and event-incomplete masks.

- [ ] **Step 2: Write failing PPO balancing tests**

Test that both branches produce the arithmetic mean of branch losses; one branch uses only the available branch; empty masks return finite zero; non-event rows cannot affect either branch; and returned diagnostics expose both branch losses.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold/test_foothold_rollout_storage.py tests/parkour/foothold/test_event_gated_foothold_ppo.py
```

Expected: failures identify missing branch fields and the missing balanced-loss interface.

### Task 2: Preserve branch labels from environment to rollout

**Files:**
- Modify: `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py`
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Modify: `source/instinctlab/instinctlab/learning/foothold_rollout_storage.py`

**Interfaces:**
- Wrapper extras include `learned_foothold_nominal_safe_event` and `learned_foothold_nominal_unsafe_event` for every environment step.
- PPO `process_env_step` copies those detached masks into the current transition.

- [ ] **Step 1: Add environment extras**

At the same causal event-generation boundary as `learned_foothold_action_event`, read the planner’s nominal-safe and nominal-unsafe event generations/flags and emit boolean tensors. Require that they are mutually exclusive and that their union equals the planner event.

- [ ] **Step 2: Extend transition and storage**

Allocate two boolean rollout tensors, validate shape/type/completeness in `add_transitions`, and append both masks to `FootholdMiniBatch` in `get_minibatch_from_selection`.

- [ ] **Step 3: Copy masks in PPO process step**

Require both new info keys, detach and clone them, and pass them through the unchanged upstream transition path.

- [ ] **Step 4: Run Task 1 tests and verify GREEN**

Run the focused pytest command from Task 1. Expected: all storage and environment-to-rollout tests pass.

### Task 3: Implement balanced planner actor surrogate loss

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`

**Interfaces:**
- `grouped_clipped_surrogates` computes the existing motor loss and returns balanced planner loss plus per-branch planner losses.
- `compute_losses` adds branch counts and advantage/surrogate diagnostics.

- [ ] **Step 1: Add branch reduction helper**

Implement a private helper that validates shape/type, computes masked means, selects equal weighting when both masks are non-empty, falls back to the present branch when only one exists, and returns a zero-with-gradient tensor when neither exists.

- [ ] **Step 2: Use helper for planner actor loss**

Apply it only to the clipped planner actor surrogate. Keep event-masked value loss, entropy, KL, optimizer separation, and KL gating unchanged.

- [ ] **Step 3: Add diagnostics**

Record safe/unsafe counts, advantage means, per-branch surrogate losses, and balanced surrogate loss. Keep existing event, KL, std, and routing metrics.

- [ ] **Step 4: Run PPO tests and verify GREEN**

Run the focused pytest command. Expected: all balancing and existing PPO tests pass.

### Task 4: Full verification and smoke training

**Files:**
- Modify: `docs/foothold_project_status.md` with measured validation results only.

- [ ] **Step 1: Run all foothold tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
python -m pytest -q tests/parkour/foothold
```

- [ ] **Step 2: Run static checks**

```bash
python -m py_compile \
  source/instinctlab/instinctlab/learning/foothold_rollout_storage.py \
  source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py \
  source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py
git diff --check
```

- [ ] **Step 3: Run a 64-environment smoke training**

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=balanced_foothold_loss_64env_100it \
NUM_ENVS=64 MAX_ITERATIONS=100 SAVE_INTERVAL=50 \
./scripts/foothold_train.sh 2>&1 | tee logs/balanced_foothold_loss_64env_100it.txt
```

Confirm both branch counts are nonzero, planner KL remains finite, and no new route/recovery errors appear.

- [ ] **Step 4: Run a 4096-environment 100-iteration acceptance training**

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=balanced_foothold_loss_4096env_100it \
NUM_ENVS=4096 MAX_ITERATIONS=100 SAVE_INTERVAL=50 \
./scripts/foothold_train.sh 2>&1 | tee logs/balanced_foothold_loss_4096env_100it.txt
```

Compare collection time, planner KL, correction success rate, correction distance, motor reward, and recovery fraction to the pre-change short-run baseline.

- [ ] **Step 5: Update status documentation and report results**

Only record observed metrics; do not claim long-training success until the smoke runs pass.
