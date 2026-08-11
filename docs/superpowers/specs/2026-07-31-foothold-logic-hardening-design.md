# Foothold Logic Hardening Design

## Goal

Close the confirmed learned-foothold logic gaps without changing the selected
high-level architecture or adding unrelated planner features.

## Safety score

The sole perimeter score is bounded to `[-1, 1]`.

- No perimeter point penetrates an edge cylinder: score `+1`.
- Any positive penetration: score is strictly negative without introducing a
  second arbitrary minimum-penalty parameter.
- The negative magnitude increases monotonically with both penetrating-point
  ratio and normalized total penetration depth.
- Count and depth contributions are added, never subtracted.
- Non-finite penetration remains conservatively unsafe.
- External clearance is not rewarded in this change because the current
  obstacle interface reports penetration only and cannot measure distance
  outside a cylinder.

## HOLD-to-SWING frame contract

Nominal and learned targets prepared during HOLD are tied to the support-foot
origin and base yaw used for preparation. The selected cached world target is
the source of truth at SWING lock. It must not be recomposed from a later
support pose.

At lock time, the fixed world target is transformed into the current support
frame for reporting and feasible-velocity calculation, then geometry is
revalidated. Support-foot motion therefore cannot silently move a target that
was already scored. A target that is no longer reachable or exceeds the
0.25-metre step-height limit is rejected as `PLAN_INVALID`.

## Startup timing

The parkour task has one 0.15-second initial HOLD requirement. Startup gating
must not clear accumulated HOLD time and add a second 0.15-second wait.

## Command consistency

When the command changes during HOLD, a nominal target prepared for the old
command is invalidated. The new command must produce a new nominal target
before a learned foothold proposal is consumed.

## Play reproducibility

Play detects an event-gated learned-foothold run from the saved agent
configuration, enables the same 31-action environment path, registers
`EventGatedWasabiPPO`, and uses the saved agent configuration. Legacy
29-action runs retain their existing behavior.

## Verification

Regression tests cover:

- clear, shallow, deep, sparse, dense, and non-finite penetration;
- support origin/yaw drift between HOLD preparation and SWING lock;
- a command change while a nominal target is cached;
- effective 0.15-second startup HOLD;
- learned checkpoint play configuration and legacy play compatibility.

The complete `tests/parkour/foothold` suite must pass.
