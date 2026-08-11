# Foothold Cache and Diagnostics Fix Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with tests after each task.

**Goal:** Freeze foothold targets only after confirmed support contact and correct play diagnostics.

**Architecture:** Keep the learned decoder, safety reward, PPO, curriculum, gait thresholds, reachability radii, and world-frame terrain query unchanged. Gate nominal preparation on `hold_contact_ready`, resolve debug contact names against the full contact sensor, and compute ellipse usage from normalized XY coordinates.

**Global constraints:** No reward, PPO, curriculum, swing-duration, reachability, or action-semantics changes. Preserve the existing hard SWING revalidation and `PLAN_INVALID` path.

## Tasks

### 1. HOLD cache timing

Files: `source/instinctlab/instinctlab_foothold/learned_target.py`, `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`, `tests/parkour/foothold/test_learned_foothold_planner.py`, `tests/parkour/foothold/test_foothold_planner_data.py`.

- Add a failing test for a nominal-preparation mask: HOLD + confirmed contact + not prepared + not startup is the only true case.
- Run the focused tests and confirm failure.
- Add `nominal_foothold_prepare_mask(...)` and use it in the planner.
- Remove only the immediate `_prepare_nominal_footholds(...)` call inside the `entered_hold` block; retain cache clearing and the false prepared flag.
- Add a recovery-HOLD test proving one contact is insufficient.
- Run the focused tests and confirm pass.

### 2. Contact-time diagnostics

Files: `scripts/instinct_rl/play_debug.py`, `tests/parkour/foothold/test_play_debug.py`.

- Add a failing test where full contact sensor ankle indices are 2 and 3 while the planner reduced view uses 0 and 1.
- Resolve configured ankle names against full contact-sensor `body_names`; use those indices for air/contact times and return `None` if names cannot be resolved.
- Run the focused play-debug tests and confirm pass.

### 3. Ellipse diagnostics

Files: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`, `tests/parkour/foothold/test_learned_foothold_planner.py`.

- Add a failing test for a target with `abs(y) > radius_y`; usage must be greater than 1, not 0.
- Replace the current zero-width branch with `sqrt((x/rx)^2 + (y/ry)^2)`; retain `target_ellipse_max_x` as a separate visualization field.
- Run the focused learned-planner tests and confirm pass.

### 4. Verification

- Run `PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold`.
- Run `git diff --check` and compile the changed Python packages.
- Run a short existing model play with both debug environments enabled; verify contact times reset/alternate and recovery HOLD waits for confirmed contact.
- Do not start a 30000-iteration run until this short play is clean.
