# Contact-Adaptive Foothold Recovery Design

Date: 2026-08-11

## Goal

Replace the current one-size-fits-all analytic recovery step with a small,
contact-event-driven layer. Normal foothold selection remains the
responsibility of the learned foothold planner. The existing motor policy
handles physical stabilization only when contact adaptation can no longer
maintain a reliable support state.

The design must remove the current contradiction where an analytic recovery
target is accepted using geometry-only routing and is then rejected by the
swing-clearance check for danger-cylinder penetration.

## Non-goals

- Do not add a second motor policy or a dedicated recovery network.
- Do not change the learned planner architecture or PPO implementation.
- Do not replace explicit foothold planning with end-to-end locomotion.
- Do not add a family of hand-authored recovery footholds.
- Do not alter the original parkour task outside recovery-specific routing,
  observations, reward gating, and diagnostics.

## Architecture

The runtime is divided into three responsibilities:

1. **Normal locomotion**: HOLD, foothold planning, locked SWING, and touchdown.
2. **Contact adaptation**: small event-driven corrections for early contact,
   late contact, support loss, and invalid plans.
3. **Autonomous stabilization**: the normal motor policy acts without an
   active foothold plan until physical stability is recovered.

Contact adaptation is not a separate controller. It is a compact transition
layer between the existing state machine and planner.

## Event Handling

### Early contact

After the existing contact debounce confirms an early swing-foot contact:

- truncate the remaining swing reference;
- record the measured contact location as the actual touchdown;
- if contact and body support remain stable, enter touchdown confirmation and
  continue to the next normal HOLD;
- enter autonomous stabilization only if the contact is not supportable or the
  body is physically unstable.

An early contact is therefore a contact-timing event, not automatically a
recovery failure.

### Late contact

At the nominal end of swing without confirmed contact:

- keep the locked horizontal foothold coordinates unchanged;
- do not swap the support and swing legs;
- extend the foot downward along the local terrain normal at a bounded rate;
- confirm touchdown using the existing filtered contact signal;
- enter autonomous stabilization only after the contact-search duration or
  kinematic reach is exhausted.

The downward search is a continuation of the existing swing transaction, not
a new foothold plan.

### Support loss

Discard the old support assignment and derive support from confirmed physical
contacts:

- if the opposite foot remains reliable and the body is stable, use it as the
  support foot and return to normal HOLD for a fresh foothold plan;
- if no reliable support exists or body motion is unstable, enter autonomous
  stabilization.

### Invalid foothold or trajectory before lift-off

Planning invalidity is not a physical recovery event:

- remain in HOLD;
- do not start SWING;
- invalidate the failed planning transaction and request a fresh proposal;
- use the existing plan-wait timeout as the bounded failure path;
- if that timeout expires while the body remains physically stable, terminate
  the simulation episode with a distinct planning-failure reason; deployment
  must hold a safe stop and report failure instead of entering stabilization
  or executing an unsafe swing.

A locked plan is fully checked before lift-off. Runtime execution error must
not reinterpret the locked world target in a new support frame.

## Autonomous Stabilization

On entry:

- invalidate the active nominal/learned foothold transaction;
- stop both analytic and learned foothold planning;
- invalidate the old swing reference and zero its error channels;
- preserve the external locomotion command, but expose a zero effective motion
  command while stabilizing;
- keep running the existing motor policy every control step.

The motor policy receives the existing proprioceptive history and recovery
indicator plus two explicit booleans for confirmed left- and right-foot
contact. No additional policy network is introduced.

While stabilizing:

- foothold, swing, touchdown, and commanded-velocity tracking rewards are
  masked;
- body-orientation, angular-velocity, legal-contact, action-regularization,
  termination, and the bounded per-step recovery cost remain active;
- AMP/style reward is masked for stabilization samples because the motion
  dataset does not define contact-recovery behavior;
- no planner PPO event is created.

The policy cannot exit stabilization through a learned discrete action. Exit
is controlled by measured physical state.

## Stability and Exit

Stability requires all of the following for a consecutive dwell window:

- at least one confirmed foot contact;
- body tilt inside a recovery-exit bound;
- body angular velocity inside a recovery-exit bound;
- body velocity no longer divergent;
- the confirmed support foot is not slipping beyond its bound.

Entry and exit use hysteresis: exit bounds are stricter than failure-entry
bounds. Numerical bounds must be calibrated from successful normal HOLD data,
not introduced as arbitrary constants.

After stability is confirmed:

- with left-only support, left becomes support and right becomes next swing;
- with right-only support, right becomes support and left becomes next swing;
- with bilateral support, preserve normal alternation from the last confirmed
  touchdown;
- restore the external locomotion command;
- enter normal HOLD and generate a completely fresh foothold plan from current
  world-frame support position, current body yaw, and current terrain data;
- start SWING only after planning, scoring, and transaction locking complete.

With no confirmed support, stabilization continues until the existing fall or
episode termination condition resets the environment.

## Removed Legacy Behavior

The implementation removes:

- the fixed analytic recovery foothold;
- recovery step length, lookahead, maximum length, and fixed width parameters;
- geometry-only acceptance of a danger-cylinder-unsafe recovery target;
- recovery-step pending/active routing used only by that target;
- the second HOLD transaction used to execute the analytic recovery step.

Existing diagnostics should retain the original anomaly reason and add the
chosen response: accepted touchdown, downward contact search, support
reassignment, planning retry, or autonomous stabilization.

## Validation

Unit tests must cover:

- confirmed early contact truncates swing and does not enter stabilization
  when physical stability is retained;
- late contact preserves horizontal target and support assignment while the
  vertical reference searches downward;
- planning invalidity remains in HOLD and never starts swing;
- support loss uses the remaining confirmed foot when stable;
- no-contact or unstable support enters autonomous stabilization;
- planner caches and learned-planner events remain inactive during
  stabilization;
- stability must hold continuously before exit;
- recovery exit reconstructs support/swing roles from actual contacts and
  creates a fresh plan in the correct world/support frames;
- old analytic recovery parameters and route paths are no longer used.

A short vectorized training diagnostic must additionally report event counts,
stabilization entry/success/duration, post-exit re-entry rate, late-contact
search success, and planning-retry success before any long training run.

## Rationale and References

Contact-triggered controllers commonly truncate swing on early touchdown and
search downward on late touchdown before starting a new planning loop:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC9465384/
- https://www.research-collection.ethz.ch/server/api/core/bitstreams/83e2ce5f-5c49-4d34-8f50-748a0018dc5f/content

Online modification of contact schedules and footholds based on measured
hybrid events improves robustness over fixed schedules:

- https://arxiv.org/abs/2303.04781

Contact-conditioned learned policies support multiple contact modes and their
transitions, while explicit target tracking remains compatible with different
upstream foothold planners:

- https://arxiv.org/abs/2408.00776
- https://arxiv.org/abs/2606.08253
