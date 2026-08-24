# Learned Foothold Play Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add event-only play diagnostics that identify why a learned foothold proposal fails geometry without changing planner behavior.

**Architecture:** Reuse the existing foothold sensor data already read by `play_debug.py`. Compute separate height-difference and swing-side validity diagnostics in the debug layer, format one compact `[LEARNED_FOOTHOLD_DEBUG]` line, and emit it only on learned evaluation or route-decision events.

**Tech Stack:** Python, PyTorch, pytest.

## Global Constraints

- Do not modify planner, reward, state-machine, action, or training behavior.
- Do not add tensors to the environment or checkpoint.
- Do not print the new fields on every control step.

---

### Task 1: Event-only learned foothold diagnostics

**Files:**
- Modify: `scripts/instinct_rl/play_debug.py`
- Modify: `scripts/instinct_rl/play.py`
- Test: `tests/parkour/foothold/test_play_debug.py`

**Interfaces:**
- Consumes: existing `FootholdPlannerData` tensors and planner configuration.
- Produces: `is_learned_foothold_debug_event(payload) -> bool` and `format_learned_foothold_debug_line(timestep, payload) -> str`.

- [ ] **Step 1: Write failing tests**

Add tests proving that evaluation and route frames are selected, ordinary frames are ignored, and the formatted event contains raw action, decoded/prepared targets, separate height and side checks, safety metrics, and route selection.

- [ ] **Step 2: Verify tests fail for missing diagnostics**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" python -m pytest -q tests/parkour/foothold/test_play_debug.py -k learned_foothold_debug
```

Expected: failure because the event predicate/formatter and payload fields do not exist.

- [ ] **Step 3: Implement the minimal debug-only data path**

Read existing tensors, compute `abs(relative_z) <= max_step_height_m` and the existing signed minimum lateral separation check, and add a compact event formatter. Call it from `play.py` only when the event predicate is true.

- [ ] **Step 4: Verify focused and related tests pass**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" python -m pytest -q tests/parkour/foothold/test_play_debug.py tests/parkour/foothold/test_play_foothold_viz.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Verify syntax and diff scope**

Run `python -m py_compile scripts/instinct_rl/play.py scripts/instinct_rl/play_debug.py` and inspect `git diff --check` plus the three scoped files.
