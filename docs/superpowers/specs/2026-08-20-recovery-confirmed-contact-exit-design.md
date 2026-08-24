# Recovery Confirmed-Contact Exit Design

## Goal

Make contact-adaptive Recovery hand control back to normal HOLD as soon as
both feet are already confirmed in contact, without adding a second dwell
timer or changing the normal locomotion rewards.

## Scope

- Keep the planner paused throughout Recovery.
- Keep velocity, heading, and existing motor-policy rewards unchanged.
- Suppress only `dont_wait` during Recovery. Its `-2.0` penalty for slowing
  down directly conflicts with the requirement to regain two confirmed foot
  contacts; outside Recovery it retains the original implementation and
  weight.
- Keep the existing per-foot contact confirmation interval
  (`contact_confirm_s = 0.04 s`).
- Remove only the additional Recovery stability dwell.
- Do not change swing duration, normal HOLD behavior, foothold routing,
  learned-planner rewards, or terrain safety checks.

## Safety-diagnostic semantics

`learned_foothold_safety_score`, `penetrating_point_count`, and
`total_penetration_depth` describe the proposed final foothold footprint. They
do not describe the swing start. Swing-path clearance is evaluated separately;
an initially penetrating swing start may escape its existing obstacle region
provided penetration does not deepen and the final foothold is safe.

## State transition

```text
RECOVERY + both confirmed contacts
  -> HOLD in the same planner update
  -> clear Recovery timer/state
  -> select the next support/swing roles from physical contact
  -> start a fresh normal planning transaction

RECOVERY + fewer than two confirmed contacts
  -> remain RECOVERY
```

`confirmed contact` remains authoritative: a raw one-frame collision cannot
exit Recovery because the contact sensor/state machine must first accumulate
the existing 0.04 s confirmation interval.

## Verification

- A unit test must fail under the old extra-dwell behavior and pass when one
  update with two *confirmed* contacts makes Recovery ready.
- Existing one-foot/no-contact Recovery tests must continue to pass.
- A source/configuration test must prove that only `dont_wait` uses the
  Recovery-masked wrapper while linear/angular velocity tracking remains on
  the upstream reward functions.
- The complete foothold test suite must pass.
