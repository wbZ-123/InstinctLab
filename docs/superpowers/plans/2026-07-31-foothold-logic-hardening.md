# Foothold Logic Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed safety-score, cached-frame, startup-timing, command-consistency, and play-reproducibility gaps.

**Architecture:** Preserve the existing event-gated planner and PPO. Make cached world targets authoritative, make unsafe scores strictly negative, invalidate stale HOLD plans on command change, and reconstruct learned play configuration from saved run metadata.

**Tech Stack:** Python, PyTorch, Isaac Lab manager environment, Instinct-RL PPO, pytest.

## Global Constraints

- Do not add external-clearance reward without a real signed-distance interface.
- Keep safety and reward outputs bounded to `[-1, 1]`.
- Preserve legacy 29-action training and play behavior.
- Write and observe a failing regression test before each production change.
- Do not modify unrelated parkour or locomotion parameters.

---

### Task 1: Make any penetration strictly negative

**Files:**
- Modify: `tests/parkour/foothold/test_target_search.py`
- Modify: `source/instinctlab/instinctlab_foothold/target_search.py:69-126`

**Interfaces:**
- Consumes: `penetration_depths: torch.Tensor`
- Produces: `score_sole_perimeter_penetration(...).score` bounded to `[-1, 1]`

- [ ] Add tests asserting a clear sole scores `+1`, any positive intrusion scores below zero, and increasing count/depth never improves the score.
- [ ] Run the focused tests and confirm the sparse-penetration case fails under the old formula.
- [ ] Replace the positive unsafe score with the negative bounded sum of point ratio and normalized total depth.
- [ ] Run `test_target_search.py` and confirm all score tests pass.

### Task 2: Freeze and revalidate HOLD targets

**Files:**
- Modify: `tests/parkour/foothold/test_learned_foothold_planner.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`

**Interfaces:**
- Produces: cached preparation origin, yaw, and authoritative world target
- Consumes: current support pose only for inverse transform and lock-time validation

- [ ] Add a regression test in which support origin and yaw change after preparation and assert that the world target does not move.
- [ ] Confirm the test fails because the old path recomposes from the new frame.
- [ ] Store the preparation frame and route the selected cached world target.
- [ ] Recompute its current-frame representation and reject it if reachability or step-height validation fails.
- [ ] Run learned planner, frame-transform, state-machine, and planner-data tests.

### Task 3: Remove doubled startup HOLD and stale command plans

**Files:**
- Modify: `tests/parkour/foothold/test_state_machine.py`
- Modify: `tests/parkour/foothold/test_observation_foothold.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Modify: `source/instinctlab/instinctlab/envs/mdp/observations/foothold.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`

**Interfaces:**
- Startup HOLD total: 0.15 seconds
- Command update: invalidates a cached HOLD nominal before proposal evaluation

- [ ] Add tests for one 0.15-second startup window and command-change invalidation.
- [ ] Confirm both tests expose the old behavior.
- [ ] Remove the redundant startup/reset accumulation while preserving initial HOLD.
- [ ] Add command-change detection that invalidates only HOLD preparation buffers.
- [ ] Run focused state, observation, and planner tests.

### Task 4: Reconstruct learned play configuration

**Files:**
- Modify: `tests/parkour/foothold/test_train_save_interval.py`
- Modify: `tests/parkour/foothold/test_play_step_terrain.py`
- Modify: `scripts/instinct_rl/play.py`
- Modify: `scripts/foothold_play_step.sh`

**Interfaces:**
- Consumes: `<run>/params/agent.yaml`
- Produces: matching learned/legacy environment and algorithm registration

- [ ] Add tests for automatic `EventGatedWasabiPPO` detection and unchanged legacy behavior.
- [ ] Confirm the learned-run test fails with the current play setup.
- [ ] Load saved metadata before environment creation, enable the learned path, and register the algorithm conditionally.
- [ ] Run focused play/script tests.

### Task 5: Full verification

**Files:**
- No production files beyond Tasks 1-4

- [ ] Run `python -m pytest -q tests/parkour/foothold`.
- [ ] Inspect `git diff --check`.
- [ ] Inspect the final diff for unrelated parameter or architecture changes.
- [ ] Report exact test counts and any remaining runtime-only risks.
