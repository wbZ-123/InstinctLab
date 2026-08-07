# Learned Foothold Direct Execution Design

## Goal

Make the learned foothold policy the normal-walking foothold executor, while
retaining the nominal foothold as intent/teacher data and retaining the
analytic recovery foothold during abnormal-contact recovery.

## Normal routing

At each committed normal swing:

1. Use the frozen HOLD support frame and nominal foothold already observed by
   the learned policy.
2. If the learned proposal was prepared and remains geometrically valid in
   the frozen frame, execute the learned proposal, regardless of whether the
   nominal foothold is already safe.
3. If the learned proposal is unavailable or geometrically invalid, use a
   geometrically valid and safe nominal foothold as the fallback.
4. If neither route is executable, preserve the existing PLAN_INVALID path.

Danger-cylinder safety remains a soft PPO signal during normal training. The
existing finite-value, reachability, terrain-height, support-height and swing
clearance checks remain hard execution checks.

## Recovery routing

Recovery continues to disable learned foothold execution. It uses the existing
short analytic recovery target selected from confirmed physical support:

- no confirmed support: remain in RECOVERY;
- only left support: recover with the right foot;
- only right support: recover with the left foot;
- both supports: preserve normal alternation after stable contact.

The existing contact confirmation and recovery hold timers remain the
hysteresis/deadband. This change does not add new timing constants.

## Learning signal for safe nominal footholds

The existing event reward is unchanged. When the nominal foothold is safe, the
learned policy receives the normalized closeness reward and therefore learns
an identity mapping around the nominal foothold. After this routing change,
the nearby learned result is also executed, so downstream tracking, contact,
recovery and locomotion rewards become causally dependent on the learned
foothold.

This can temporarily increase early-contact, clearance or recovery events
during exploration. That is expected training behavior rather than a new
state-machine transition. Geometry hard checks and analytic recovery continue
to bound the failure modes.

## Tests

- safe nominal plus geometrically valid learned proposal routes to learned;
- unsafe nominal plus geometrically valid learned proposal routes to learned;
- invalid/unprepared learned proposal falls back to a safe nominal;
- recovery always routes to the analytic nominal target;
- neither executable target produces an invalid route;
- existing reward and full foothold test suites remain green.
