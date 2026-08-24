# Foothold HOLD Transaction Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the one-step planner reward pulse from the persistent HOLD transaction state so one learned proposal is evaluated at most once per HOLD transaction.

**Architecture:** Add one persistent boolean buffer beside the existing event pulse. Reward and PPO sampling keep consuming the existing pulse; HOLD routing, swing gating, and preflight fallback consume the persistent latch. A failed learned trajectory falls back to the already-safe nominal route without sampling another learned action in the same transaction.

**Tech Stack:** Python 3.11, PyTorch tensors, pytest, IsaacLab manager-based sensor buffers.

## Global Constraints

- Do not change planner positive reward formulas or weights; restore the
  preflight-failure floor so every planner event reward remains in `[-1, 1]`.
- Do not change planner PPO or motor PPO hyperparameters.
- Do not change AMP, MoE, curricula, or swing tracking rewards.
- Never execute an unsafe learned foothold.
- Preserve safe nominal fallback.

---

### Task 1: Specify the two lifetimes with failing tests

**Files:**
- Modify: `tests/parkour/foothold/test_learned_foothold_planner.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`

**Interfaces:**
- Produces: `FootholdPlannerData.learned_foothold_transaction_evaluated: torch.Tensor | None`
- Consumes: existing `learned_foothold_evaluated` as an event pulse.

- [ ] **Step 1: Write failing structure and clear-boundary tests**

Add assertions that the planner data class declares `learned_foothold_transaction_evaluated`, that `clear_learned_foothold_buffers()` clears it only for selected environments, and that the per-step update block clears only `learned_foothold_evaluated`.

```python
assert "learned_foothold_transaction_evaluated" in field_names

data.learned_foothold_transaction_evaluated = torch.ones(3, dtype=torch.bool)
clear_learned_foothold_buffers(data, torch.tensor([0, 2]))
assert data.learned_foothold_transaction_evaluated.tolist() == [False, True, False]

assert "self._data.learned_foothold_evaluated[env_ids] = False" in update_prefix
assert "self._data.learned_foothold_transaction_evaluated[env_ids] = False" not in update_prefix
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_learned_foothold_planner.py \
-k "planner_data_declares or clear_learned or transaction_latch"
```

Expected: FAIL because the persistent field does not exist.

- [ ] **Step 3: Add the persistent field and initialization**

Add the field to the data class and allocate a boolean tensor in `FootholdPlanner._initialize_impl()` next to the existing event pulse.

```python
learned_foothold_evaluated: torch.Tensor | None = None
learned_foothold_transaction_evaluated: torch.Tensor | None = None
```

- [ ] **Step 4: Add it to explicit transaction clearing**

Include the new field in `clear_learned_foothold_buffers()` so reset, new HOLD, support loss, and Recovery invalidation clear the latch through existing lifecycle boundaries.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Use the command from Step 2. Expected: PASS.

---

### Task 2: Latch one proposal and route nominal fallback without resampling

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Modify: `tests/parkour/foothold/test_learned_foothold_planner.py`

**Interfaces:**
- `store_learned_foothold_preparation(...)` sets both the one-step event pulse and the transaction latch.
- `learned_foothold_transaction_ready(..., transaction_evaluated=...)` decides whether the HOLD action has already been consumed.
- `learned_foothold_swing_ready(..., transaction_evaluated=...)` permits safe learned or safe nominal fallback only after one evaluation.

- [ ] **Step 1: Write failing lifecycle tests**

Test these cases:

```python
# Unsafe learned + safe nominal consumes exactly one transaction.
ready = learned_foothold_transaction_ready(
    nominal_route_ready=torch.tensor([True]),
    transaction_evaluated=torch.tensor([True]),
    learned_prepared_valid=torch.tensor([True]),
    learned_geometric_valid=torch.tensor([True]),
    learned_safety_valid=torch.tensor([False]),
)
assert ready.tolist() == [True]

# Storing a proposal emits one reward pulse and latches the transaction.
assert data.learned_foothold_evaluated.tolist() == [True]
assert data.learned_foothold_transaction_evaluated.tolist() == [True]
```

Add a source-level integration assertion that transaction decisions use `learned_foothold_transaction_evaluated`, while rewards continue to use `learned_foothold_evaluated`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_learned_foothold_planner.py \
tests/parkour/foothold/test_reward_foothold.py
```

Expected: FAIL because storing and transaction gating do not yet use the new latch.

- [ ] **Step 3: Set the latch when evaluating a proposal**

In `store_learned_foothold_preparation()`:

```python
data.learned_foothold_evaluated[env_ids] = True
data.learned_foothold_transaction_evaluated[env_ids] = True
```

- [ ] **Step 4: Use the latch for HOLD transaction decisions**

Replace transaction uses of the event pulse in `foothold_planner.py` with the new persistent latch:

- suppressing another `_prepare_learned_footholds()` call;
- `learned_foothold_swing_ready()`;
- `plan_wait_expired`;
- any assertion associated with those transaction decisions.

Keep the per-step reward, monitor event sample, play event debug, and PPO event mask on `learned_foothold_evaluated`.

- [ ] **Step 5: Turn failed learned preflight into nominal fallback**

When a learned trajectory preflight fails, invalidate the learned prepared route and clear `swing_preflight_ready` for those environments. The persistent latch remains true. On the next HOLD update the route therefore selects the safe nominal target and preflights it without sampling or rewarding a second learned action.

```python
self._data.learned_foothold_prepared_valid[failed_preflight] = False
self._data.swing_preflight_ready[failed_preflight] = False
```

If the nominal route also fails, HOLD remains blocked; it must not execute a known-unsafe swing and must not generate another planner reward merely because another control step elapsed.

- [ ] **Step 6: Run focused tests and verify GREEN**

Use the command from Step 2. Expected: PASS.

---

### Task 3: Full regression and lifecycle audit

**Files:**
- Review: all files modified in Tasks 1-2
- Test: `tests/parkour/foothold`

**Interfaces:**
- Verifies the reward event and transaction latch have no accidental consumers.

- [ ] **Step 0: Restore the planner event reward lower bound**

The existing reward regression test already requires a failed swing preflight
to return `-1.0`. Replace the out-of-contract `-2.0` fallback with `-1.0` in
`learned_foothold_planning_event_reward()`; do not change any positive-score
calculation or reward weight.

- [ ] **Step 1: Audit every field consumer**

Run:

```bash
rg -n "learned_foothold_(evaluated|transaction_evaluated)" source scripts tests
```

Expected ownership:

- `learned_foothold_evaluated`: reward, PPO event sampling, event monitors/debug, per-step reset.
- `learned_foothold_transaction_evaluated`: HOLD routing/gating, explicit transaction clearing, initialization.

- [ ] **Step 2: Run the complete foothold test suite**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold
```

Expected: all tests pass.

- [ ] **Step 3: Run static diff checks**

Run:

```bash
git diff --check
git diff --stat
git status --short
```

Expected: no whitespace errors; only the planned source, tests, and documentation files changed.

- [ ] **Step 4: Review state boundaries manually**

Confirm reset, new HOLD, support loss, and Recovery call `clear_learned_foothold_buffers()`; confirm ordinary updates do not clear the transaction latch; confirm SWING consumes a frozen target and does not modify the latch.

- [ ] **Step 5: Commit the verified implementation**

```bash
git add \
  source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py \
  source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py \
  source/instinctlab/instinctlab_foothold/learned_target.py \
  tests/parkour/foothold/test_learned_foothold_planner.py \
  docs/superpowers/plans/2026-08-18-foothold-hold-transaction-lifecycle.md
git commit -m "Fix learned foothold HOLD transaction lifecycle"
```
