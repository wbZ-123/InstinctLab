# Learned Foothold Direct Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a prepared, geometrically valid learned foothold proposal control every normal walking step, while retaining safe nominal fallback and analytic Recovery behavior.

**Architecture:** Change only the pure route-selection function that chooses between nominal and learned footholds. Normal walking becomes learned-first; Recovery remains nominal-only; danger-cylinder safety stays a soft PPO signal for learned proposals, while geometry remains the hard execution gate.

**Tech Stack:** Python 3.11, PyTorch tensors, pytest.

## Global Constraints

- Do not change the foothold observation, reward, state-machine timing, trajectory generation, or Recovery target logic.
- During normal walking, execute a learned proposal only when its action was prepared and its decoded target passed the existing geometric checks.
- If the learned proposal is unavailable or geometrically invalid, execute the nominal target only when the nominal target is geometrically valid and safe.
- During Recovery, never execute the learned proposal; use a geometrically valid analytic target even when its soft danger-cylinder score is negative.
- Preserve the existing PPO safety reward, nominal-closeness reward, and event-gated learning behavior.

---

### Task 1: Specify learned-first route behavior

**Files:**
- Modify: `tests/parkour/foothold/test_learned_foothold_planner.py:345`

**Interfaces:**
- Consumes: `route_nominal_and_learned_footholds(...) -> LearnedFootholdRoute`
- Produces: regression coverage for learned-first normal routing, safe nominal fallback, and unchanged Recovery routing.

- [x] **Step 1: Change the existing normal-route test to require learned execution even when the nominal target is safe**

```python
def test_normal_route_prefers_geometrically_valid_learned_even_when_nominal_safe():
    route = route_nominal_and_learned_footholds(
        nominal_geometric_valid=torch.tensor([True, True, False, True]),
        nominal_safety_valid=torch.tensor([True, False, False, False]),
        learned_prepared=torch.tensor([True, True, True, True]),
        learned_geometric_valid=torch.tensor([True, True, True, False]),
    )

    assert route.use_nominal.tolist() == [False, False, False, False]
    assert route.use_learned.tolist() == [True, True, True, False]
    assert route.executable.tolist() == [True, True, True, False]
```

- [x] **Step 2: Add safe nominal fallback coverage**

```python
def test_normal_route_falls_back_to_safe_nominal_when_learned_is_unavailable():
    route = route_nominal_and_learned_footholds(
        nominal_geometric_valid=torch.tensor([True, True, True]),
        nominal_safety_valid=torch.tensor([True, True, False]),
        learned_prepared=torch.tensor([False, True, False]),
        learned_geometric_valid=torch.tensor([True, False, True]),
    )

    assert route.use_nominal.tolist() == [True, True, False]
    assert route.use_learned.tolist() == [False, False, False]
    assert route.executable.tolist() == [True, True, False]
```

- [x] **Step 3: Run the two normal-route tests and verify the learned-first test fails before implementation**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_learned_foothold_planner.py \
-k "normal_route"
```

Expected: the learned-first test fails because the existing route still chooses a safe nominal point.

### Task 2: Implement learned-first normal routing

**Files:**
- Modify: `source/instinctlab/instinctlab_foothold/learned_target.py:87`
- Test: `tests/parkour/foothold/test_learned_foothold_planner.py`

**Interfaces:**
- Consumes: nominal validity masks, learned preparation/geometric-validity masks, optional Recovery mask.
- Produces: mutually exclusive `use_nominal`, `use_learned`, and their union `executable`.

- [x] **Step 1: Replace nominal-first selection with learned-first normal selection**

```python
learned_available = learned_prepared.bool() & learned_geometric_valid.bool()
use_learned = ~recovery_mask & learned_available

safe_nominal = nominal_geometric_valid.bool() & nominal_safety_valid.bool()
use_nominal = (recovery_mask & nominal_geometric_valid.bool()) | (
    ~recovery_mask & ~use_learned & safe_nominal
)
```

- [x] **Step 2: Update the route docstring and comments to document learned-first normal routing and analytic Recovery**

```python
"""Route privileged training targets without invoking candidate search.

During normal walking, a prepared and geometrically valid learned proposal
is executed even when the nominal target is safe.  This closes the PPO
control loop while keeping danger-cylinder safety as a soft learning signal.
A safe nominal target is only the fallback when the learned proposal is not
executable.  Recovery steps remain analytic and nominal-only.
"""
```

- [x] **Step 3: Run the focused route tests**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_learned_foothold_planner.py \
-k "route"
```

Expected: all route tests pass, including analytic Recovery and geometric-invalid rejection.

### Task 3: Verify no foothold behavior regressed

**Files:**
- Verify: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Verify: `tests/parkour/foothold/`

**Interfaces:**
- Consumes: the complete foothold unit and integration test suite.
- Produces: evidence that the minimal routing change did not alter unrelated planner behavior.

- [x] **Step 1: Run the complete foothold test suite**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold
```

Expected: all existing tests pass; any skipped test remains skipped for its existing reason.

- [x] **Step 2: Check patch formatting**

Run:

```bash
git diff --check -- \
source/instinctlab/instinctlab_foothold/learned_target.py \
tests/parkour/foothold/test_learned_foothold_planner.py \
docs/superpowers/plans/2026-08-07-learned-foothold-direct-execution.md
```

Expected: no output and exit code 0.
