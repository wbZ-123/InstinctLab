# Event-SAC Planner Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two-dimensional foothold Event-SAC use causal per-environment rewards, correctly scaled residual exploration, balanced event data, and a small nominal-point anchor without changing motor PPO or gait-state logic.

**Architecture:** Keep the existing external planner actor and dedicated planner depth encoder. Correct the event adapter so only environments that generated a planner event contribute that event score, convert exploration standard deviations in the residual decoder's physical units, and schedule SAC updates from newly completed planner events. The actor receives a small auxiliary identity loss only for safe nominal events; unsafe events remain optimized by SAC safety/progress rewards.

**Tech Stack:** PyTorch, IsaacLab/RSL-RL wrapper, TensorBoard scalar diagnostics, pytest.

## Global Constraints

- Do not modify motor PPO/AMP/MoE, gait state machine, recovery routing, trajectory generation, or locomotion reward terms.
- Keep planner action dimension at two normalized residual coordinates around the frozen nominal foothold.
- Keep physical residual limits at X=0.12 m and Y=0.10 m.
- Keep the existing `[-1, 1]` planner event reward contract and reject non-finite or out-of-range replay rewards.
- Preserve checkpoint compatibility only where the actor shape is unchanged; a changed stochastic actor requires a fresh planner SAC state.

### Task 1: Enforce causal per-environment event rewards

**Files:**
- Modify: `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py:244-254`
- Test: `tests/parkour/foothold/test_event_gated_foothold_sac.py`

**Interfaces:**
- Consumes the existing `foothold_event` boolean tensor and raw termwise reward tensor.
- Produces `learned_foothold_event_reward` with zero for every environment whose event mask is false.

- [x] Mask the read reward with `torch.where(foothold_event, reward, torch.zeros_like(reward))` and validate finite values and `[-1, 1]` range before putting them in `extras`.
- [x] Run the focused test and the existing event-accumulator tests.

### Task 2: Make residual exploration units physically correct

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py:32-52,391-405`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py:77-89`
- Modify: `source/instinctlab/instinctlab/learning/independent_foothold_actor_critic.py:162-177`
- Test: `tests/parkour/foothold/test_event_gated_foothold_sac.py`

**Interfaces:**
- The standard-deviation conversion must accept the two residual decoder limits rather than reachability ellipse radii.
- Planner stochastic actor must expose per-state mean and log-standard-deviation for the two normalized residual actions, with deterministic mean inference unchanged.

- [x] Change the conversion call site to use the decoder residual limits `(0.12, 0.10)` rather than reachability radii.
- [x] Add a separate planner log-standard-deviation head with bounded state-dependent output; deterministic inference remains two-dimensional.
- [x] Configure initial physical standard deviation `(0.025, 0.020)` m, minimum `(0.005, 0.005)` m, maximum `(0.040, 0.040)` m, initial alpha `0.05`, and target entropy `-0.5`.
- [x] Run focused actor/distribution tests and verify deterministic inference has the same two-dimensional output shape.

### Task 3: Correct event-scaled SAC updates and branch sampling

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/foothold_sac.py:275-321,580-625`
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py:1207-1237`
- Modify: `source/instinctlab/instinctlab/learning/foothold_sac_replay.py`
- Test: `tests/parkour/foothold/test_event_gated_foothold_sac.py`

**Interfaces:**
- `FootholdSAC.update(new_event_count)` consumes complete planner events and schedules a bounded number of critic updates from an explicit UTD target.
- Replay entries retain nominal-safe versus nominal-unsafe branch labels for balanced minibatches.

- [x] Extend replay transitions with nominal-safe labels and balanced (approximately 1:1) branch sampling.
- [x] Set baseline `target_sample_ratio=0.5`, `max_updates_per_rollout=24`, `warmup_events=10000`, and require at least 512 nominal-unsafe events in the training config.
- [x] Keep replay capacity 100,000 and batch size 256; set `gamma=0.95`, `tau=0.005`, and actor/critic/alpha learning rates to `1e-4`.
- [x] Update actor and temperature every two critic updates; update target critics every two critic updates with the existing `tau`.
- [x] Log requested, completed, deferred, effective sample ratio, branch counts, and warmup status.
- [x] Run focused SAC tests and verify event credit is carried when a per-rollout cap is reached.

### Task 4: Add the safe-nominal identity anchor

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/foothold_sac.py`
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Test: `tests/parkour/foothold/test_event_gated_foothold_sac.py`

**Interfaces:**
- Actor update receives a nominal-safe mask and applies `0.25 * ||delta_mean / residual_limits||²` only to safe nominal events.
- Unsafe nominal events receive no identity loss and remain governed by safety/progress reward.

- [x] Add the bounded auxiliary loss to the actor update and log its value and sample count.
- [x] Run focused tests and ensure gradients reach only planner actor parameters, not motor PPO parameters.

### Task 5: End-to-end validation and documentation

**Files:**
- Modify: `docs/foothold_parameter_audit.md`
- Test: `tests/parkour/foothold/test_event_gated_foothold_sac.py`

- [x] Run the complete foothold test subset.
- [ ] Run a 4096-environment 100-iteration smoke training without a base planner checkpoint so replay and actor state are fresh.
- [ ] Verify replay rewards stay in `[-1,1]`, effective sample ratio is `0.45–0.55`, dropped updates are zero, alpha does not monotonically diverge, and effective physical standard deviations remain within 0.5–4 cm.
- [x] Record the new parameters and the required fresh-planner-checkpoint rule in the parameter audit.
