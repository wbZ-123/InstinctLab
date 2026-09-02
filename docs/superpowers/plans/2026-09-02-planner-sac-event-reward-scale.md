# Planner SAC Event Reward Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Feed the planner SAC the unscaled high-level event score while preserving the existing `dt`-scaled reward groups used by motor PPO and logging.

**Architecture:** `MultiRewardManager` exposes the current weighted term value before time-step scaling. The vector-environment wrapper forwards that value as an explicit event extra, and `EventGatedWasabiSAC` consumes it when closing planner transitions. SAC additionally reports replay-reward and critic-target diagnostics so scale or Q drift is visible in the first smoke run.

**Tech Stack:** Python, PyTorch, Isaac Lab manager-based environments, pytest.

## Global Constraints

- Do not change nominal foothold generation, coordinate transforms, trajectory generation, AMP, MoE, motor PPO, or planner reward semantics.
- Keep ordinary environment reward groups time-step scaled exactly as before.
- The planner SAC replay reward must be the raw weighted event score, bounded by the existing reward function to `[-1, 1]`.
- Missing raw event rewards must fail loudly instead of silently falling back to the `dt`-scaled tensor.
- No new rejection gates or action-space constraints are introduced.

### Task 1: Expose the raw weighted planner term

**Files:**
- Modify: `source/instinctlab/instinctlab/managers/reward_manager.py`
- Test: `tests/parkour/foothold/test_reward_manager.py`

- [ ] Add a focused test with a lightweight manager instance whose termwise buffer contains a known tensor; assert `get_termwise_reward("learned_foothold_planning", "foothold_planning")` returns that tensor and rejects unknown names.
- [ ] Run the focused test and observe the expected missing-method failure.
- [ ] Implement a read-only `get_termwise_reward(term_name, group_name=None)` that validates the group/term through existing configuration names and returns a detached clone of the current pre-`dt` weighted term value.
- [ ] Re-run the focused test and the existing manager tests.

### Task 2: Forward and require raw event rewards

**Files:**
- Modify: `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py`
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Test: `tests/parkour/foothold/test_event_gated_foothold_sac.py`

- [ ] Extend the SAC environment-step test with `learned_foothold_event_reward` and assert the replay transition stores that value, not the reward-group value.
- [ ] Add a test that omits the raw extra and asserts `EventGatedWasabiSAC.process_env_step` raises `KeyError`.
- [ ] Run the tests and observe the expected failure before implementation.
- [ ] In the wrapper, after a foothold event is detected, read the raw weighted term from the unwrapped `MultiRewardManager` and add it to extras under `learned_foothold_event_reward`; raise a clear `RuntimeError` if the event exists but the manager cannot provide the term.
- [ ] In `process_env_step`, read and shape-check `infos["learned_foothold_event_reward"]`; remove the fallback to `rewards[..., foothold_reward_index]`.
- [ ] Re-run focused SAC tests and the complete foothold test suite.

### Task 3: Add SAC scale and Q diagnostics

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/foothold_sac.py`
- Test: `tests/parkour/foothold/test_foothold_sac.py`

- [ ] Add a test asserting `FootholdSAC.update()` returns `sac_replay_reward_mean`, `sac_replay_reward_min`, `sac_replay_reward_max`, `sac_target_q_mean`, and `sac_q_abs_max`.
- [ ] Run the test and observe the expected missing-key failure.
- [ ] Add those keys to zero-update statistics and accumulate them from the sampled batch and Bellman target/current Q during completed updates; keep all existing update behavior unchanged.
- [ ] Re-run SAC tests and the complete foothold suite.

### Task 4: Runtime smoke verification

**Files:**
- No production changes.
- Verify: `logs/` smoke output and checkpoint replay statistics.

- [ ] Run a 4096-environment, 100-iteration planner SAC smoke test from scratch.
- [ ] Confirm replay rewards are order-one rather than order-`dt`, Q and critic loss do not show the prior runaway trend, and the existing policy sensitivity diagnostic distinguishes positive and negative nominal lateral inputs.
- [ ] If these checks fail, stop before any second-stage actor parameterization change.
