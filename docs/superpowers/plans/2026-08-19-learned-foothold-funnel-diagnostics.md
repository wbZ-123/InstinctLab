# Learned Foothold Funnel Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose per-stage learned-foothold proposal and execution counts without changing planner routing or PPO behavior.

**Architecture:** The planner publishes a one-step diagnostic snapshot for each proposal/route event. The existing foothold monitor accumulates mutually exclusive route outcomes and emits normalized fractions using the existing event denominators. Play debug prints the same outcome label when enabled; no new geometry, reward, or state-machine decisions are introduced.

**Tech Stack:** PyTorch tensors, existing `FootholdPlannerData`, `FootholdPlannerMonitorTerm`, pytest.

## Global Constraints

- Do not change learned-foothold routing, Recovery, PPO, reward values, action noise, or trajectory generation.
- Count each proposal and committed route once; do not reuse stale HOLD fields as new events.
- Keep the metrics finite and per-environment like existing monitor buffers.

### Task 1: Add pure funnel classification

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Test: `tests/parkour/foothold/test_learned_foothold_planner.py`

- [x] Add a failing test for classification precedence: Recovery, geometry, endpoint safety, preflight, transaction invalidation, success.
- [x] Implement a pure tensor helper returning an integer reason code and stage booleans.
- [x] Run the focused learned-target tests.

### Task 2: Persist route outcome diagnostics

**Files:**
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Test: `tests/parkour/foothold/test_foothold_planner_data.py`

- [x] Add a per-environment integer route outcome field and reset it with the existing learned buffers.
- [x] Set the outcome only at the committed new-swing route decision; leave routing behavior unchanged.
- [x] Preserve the outcome long enough for the monitor update to consume it.
- [x] Test initialization and reset isolation.

### Task 3: Accumulate and expose funnel metrics

**Files:**
- Modify: `source/instinctlab/instinctlab/monitors/foothold.py`
- Test: `tests/parkour/foothold/test_foothold_monitor.py`

- [x] Add counters for proposal, geometry-valid, endpoint-safe, preflight-safe, transaction-survived, route-used, and each mutually exclusive failure reason.
- [x] Emit event-normalized fractions with stable `0.0` behavior when the denominator is empty.
- [x] Verify counters are not double-counted across consecutive monitor updates.

### Task 4: Play diagnostics and verification

**Files:**
- Modify: `scripts/instinct_rl/play_debug.py`
- Test: `tests/parkour/foothold/test_play_debug.py`

- [x] Include the route outcome and stage booleans in the existing learned-foothold debug payload.
- [x] Keep output opt-in and bounded by the existing debug controls.
- [x] Run the focused foothold test suite and a syntax/import check.
