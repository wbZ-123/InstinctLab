# Foothold Reward Curriculum Design

Date: 2026-07-07

## Goal

Add foothold tracking to the existing parkour training without letting the new planner dominate or destabilize the original task. The original parkour rewards should remain the main driver until the policy reliably enters swing and touchdown states.

## Current State

The foothold planner is mounted as a sensor and exposes swing references, target footholds, gait mode, feasible velocity, contacts, and touchdown diagnostics. Smoke tests show that the planner runs inside the IsaacLab environment and receives the `base_velocity` command.

The current reward terms are:

- `foothold_swing_tracking`: rewards swing foot center tracking the planner reference.
- `foothold_touchdown_tracking`: rewards accepted touchdown near the target foothold.
- `foothold_swing_mode_indicator`: logs how often the planner enters left/right swing.
- `foothold_touchdown_confirm_indicator`: logs successful touchdown transition.
- `foothold_touchdown_accepted_indicator`: logs whether touchdown is within planner tolerance.
- `foothold_plan_invalid_indicator`: logs planner failure.

Recent short training runs show that swing events can appear, but touchdown-confirmed events are still rare. Termination signals such as base contact, bad orientation, and root height remain important bottlenecks.

## Recommended Strategy

Use manual staged tuning between training runs, not live weight changes during a running process.

This keeps each run reproducible: a checkpoint plus a fixed config fully describes the experiment. If a stage becomes unstable, we can roll back to the previous checkpoint and previous weights.

## Stage 1: Conservative Observation

Purpose: confirm the planner and rewards are active while preserving the original parkour learning behavior.

Recommended weights:

- `foothold_swing_tracking`: `0.1` to `0.2`
- `foothold_touchdown_tracking`: `0.0` to `0.05` if touchdown events are too rare; `0.1` is acceptable only if it does not destabilize training.
- Diagnostic indicators: `0.01`

Success criteria:

- Training does not terminate much earlier than the original baseline.
- `foothold_swing_mode_indicator` is nonzero over short runs.
- `foothold_plan_invalid_indicator` remains near zero.

Do not increase touchdown weight in this stage if touchdown events are mostly absent.

## Stage 2: Swing Tracking Emphasis

Purpose: once swing states appear regularly, teach the policy to follow the swing reference trajectory.

Recommended weights:

- `foothold_swing_tracking`: increase toward `0.3` to `0.5`
- `foothold_touchdown_tracking`: keep small, around `0.05`
- Diagnostic indicators: keep `0.01`

Success criteria:

- Swing tracking reward becomes consistently nonzero.
- Termination rates do not worsen significantly.
- Touchdown accepted/confirm indicators begin to appear more often.

## Stage 3: Touchdown Accuracy

Purpose: only after the policy can swing and land with some regularity, reinforce accurate final foothold placement.

Recommended weights:

- `foothold_swing_tracking`: keep around the best Stage 2 value.
- `foothold_touchdown_tracking`: increase toward `0.1` to `0.3`.
- Diagnostic indicators: keep `0.01` or remove from final production config if logs become noisy.

Success criteria:

- Touchdown tracking reward is nonzero often enough to influence learning.
- Touchdown accepted/confirm indicators are not rare one-off events.
- Locomotion stability remains comparable to the pre-foothold baseline.

## Run Protocol

Each stage should be a separate run name. After each run:

1. Check episode length and termination terms.
2. Check swing and touchdown diagnostic indicators.
3. Check foothold tracking reward magnitudes.
4. Decide whether to keep, increase, or reduce foothold weights.

Example run names:

- `foothold_stage1_observe`
- `foothold_stage2_swing`
- `foothold_stage3_touchdown`

## Why Not Live Weight Editing

Changing weights during a running training process makes results hard to reproduce and debug. If training improves or collapses, it becomes unclear whether the policy learned better behavior or the live reward change caused the shift.

Live or automatic scheduling can be added later, but the first stable version should use explicit staged runs.

## Open Follow-Up

Terrain-aware swing clearance is still a separate future improvement. The current curriculum assumes the existing swing reference is sufficient for flat or simple terrain training.
