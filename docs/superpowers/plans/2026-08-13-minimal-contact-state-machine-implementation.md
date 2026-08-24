# Minimal contact-adaptive foothold state machine

## Goal

Keep the long-lived behavior to three modes—HOLD, LEFT/RIGHT SWING, and
RECOVERY—while preserving legacy enum values, observation dimensions, and
checkpoint compatibility. Planning failure is a retryable planning event;
RECOVERY is reserved for physical instability or loss of all usable support.

## Constraints

- Do not modify the upstream locomotion reward implementations.
- Keep the learned foothold planner output and observation layout unchanged.
- Freeze support/world-frame data and the target once SWING starts.
- Keep the existing 0.04 s recovery dwell and use support slip for diagnostics,
  not as a hard recovery-exit gate.

## Implementation steps

1. State routing: in contact-adaptive mode, route `PLAN_INVALID` and ordinary
   late-touchdown search through HOLD/temporary events; only unstable early
   contact, unrecoverable support loss, or exhausted late-contact search may
   enter RECOVERY.
2. Recovery rewards: keep normal command-tracking/progress rewards active so
   the motor policy retains the walking direction during recovery.  Pause
   only planner-dependent terms (including feet-air-time, which otherwise
   pushes toward single support) while preserving stability, smoothness,
   energy, alive, and termination terms.
3. Regression tests: cover planning retry without Recovery, late-contact
   search, recovery exit dwell, and recovery reward masking.
4. Run targeted state-machine/reward tests, then the complete foothold suite
   and `git diff --check`.

## Acceptance criteria

- A failed foothold proposal cannot create a long-lived RECOVERY state.
- RECOVERY cannot exit without both feet confirmed and 0.04 s continuous
  contact dwell.
- A recovery episode keeps velocity/heading/progress pressure, but does not
  receive planner-dependent foothold pressure.
- Existing observation/checkpoint compatibility tests remain green.
