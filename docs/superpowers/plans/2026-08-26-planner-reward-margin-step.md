# Planner Reward Margin and Step-Length Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the learned foothold planner's sparse score with a bounded reward that values signed progress, reasonable step length, safety margin, and conditional nominal-point adherence, without adding terrain-tread sampling.

**Architecture:** Keep the existing event-gated single reward term and execution gates. Extend obstacle queries only enough to expose signed clearance for the sole perimeter, compute the new terms only at planner evaluation events, and compose them in `learned_foothold_planning_event_reward`.

**Tech Stack:** PyTorch tensors, existing Warp cylinder grid, pytest.

## Global Constraints

- Do not modify AMP, MoE, motor-policy PPO, Recovery routing, or swing trajectory execution.
- Do not add terrain-tread sampling or a terrain-tread reward.
- Planner reward is non-event zero and event output is clamped to `[-1, 1]`.
- Forward progress is measured from the frozen support-foot frame; swing-start coordinates are not used.
- Any positive sole penetration must not produce a positive final planner reward.

### Task 1: Lock the reward contract with tests

**Files:**
- Modify: `tests/parkour/foothold/test_reward_foothold.py`

- [x] Add tests for support-frame forward scoring, oversize-step penalty, safe/unsafe nominal deviation weights, safety-margin sign, and final `[-1, 1]` bounds.
- [x] Run the focused tests and confirm they fail against the old expression.

### Task 2: Add signed cylinder clearance

**Files:**
- Modify: `source/instinctlab/instinctlab/utils/warp/kernels.py`
- Modify: `source/instinctlab/instinctlab/utils/warp/cylinder.py`
- Modify: `source/instinctlab/instinctlab/terrains/virtual_obstacle/virtual_obstacle_base.py`
- Modify: `source/instinctlab/instinctlab/terrains/virtual_obstacle/edge_cylinder.py`
- Modify: `source/instinctlab/instinctlab_foothold/target_search.py`
- Test: `tests/parkour/foothold/test_target_search.py`

- [x] Add an optional signed-clearance query returning the nearest cylinder-surface distance for each point; preserve the current penetration API.
- [x] Aggregate the minimum clearance across sole-perimeter points and derive a bounded safety-margin component.
- [x] Keep no-obstacle/unsupported test doubles neutral and do not change existing hard validity gates.
- [x] Run target-search tests.

### Task 3: Implement the bounded planner reward

**Files:**
- Modify: `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`

- [x] Compute all terms only when `learned_foothold_evaluated` is true.
- [x] Use signed support-frame command projection for every event.
- [x] Add the triangular reasonable-step score centered at the command-predicted displacement.
- [x] Use strong nominal adherence for nominal-safe events and weak adherence for nominal-unsafe events.
- [x] Make safety margin dominant and cap any penetrating result at `-0.05` or lower.
- [x] Clamp the final score to `[-1, 1]`.

### Task 4: Configure and document the expression

**Files:**
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Create: `docs/planner_reward_margin_step_design.md`

- [x] Configure the initial weights: safety `0.40`, progress `0.25`, step `0.20`, nominal `0.15` when nominal-safe; safety `0.45`, progress `0.30`, step `0.20`, nominal `0.05` when nominal-unsafe.
- [x] Remove unused terrain-tread parameters from the planner reward configuration.
- [x] Record that reward changes do not change execution routing or motor rewards.

### Task 5: Verify regressions

- [x] Run `pytest -q tests/parkour/foothold/test_reward_foothold.py tests/parkour/foothold/test_target_search.py`.
- [x] Run the broader foothold test suite.
- [x] Inspect `git diff --check` and report any pre-existing unrelated worktree changes separately.
