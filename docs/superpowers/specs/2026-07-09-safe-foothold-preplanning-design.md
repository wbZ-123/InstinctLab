# Safe Foothold Preplanning Design

## Goal

Choose a reachable foothold whose sole boundary does not penetrate terrain edge
cylinders, while deviating as little as possible from the velocity-derived
nominal foothold. Confirm a valid next foothold before allowing the next swing
to begin.

## Planning Time

Planning occurs during double support:

1. Confirm that the current swing foot has touched down.
2. Promote that foot to the new stance foot.
3. Select the opposite foot as the next swing foot.
4. Freeze the new stance-foot frame.
5. Plan and validate the next foothold.
6. Enter swing only after the plan is valid.

At episode startup, the same process runs after both contacts are confirmed and
before the initial left swing.

The existing order—enter swing and then generate the target—must be replaced.
Plan availability must be separate from active-plan validity so that a missing
next plan keeps the robot in double support instead of creating a failure after
liftoff.

## Safety Model

The reachable ellipse constrains only the candidate foothold center. It models
the ankle/sole-center workspace, not the shoe outline.

Landing safety uses the lowest layer of the existing `10 x 5 x 2`
`leg_volume_points` grid. The 26 perimeter points of its `10 x 5` sole layer
are transformed around every candidate center using the planned foot yaw. A
candidate is safe only if all 26 points have zero penetration into the
registered terrain edge cylinders.

The upper point layer remains part of the existing volume-penetration reward
and is not used for landing selection.

## Local Candidate Search

Start from the velocity-derived target after the existing ellipse projection.
If its 26-point footprint is safe, accept it without searching.

If unsafe, generate 32 candidate centers using:

- radii `0.025`, `0.050`, `0.075`, and `0.100` metres;
- forward, backward, left, right, and four diagonal directions;
- axes defined by the desired planar velocity, with a stable stance-frame
  fallback when desired planar speed is near zero.

All candidates are generated and evaluated as batched tensors. Candidates whose
centers lie outside the reachable ellipse are invalid. The remaining candidates
are expanded to 26-point sole boundaries and queried against edge cylinders.
The implementation evaluates all local candidates in parallel, rather than
running sequential GPU/CPU-dependent ring checks.

## Candidate Ranking

Terrain-cylinder safety and ellipse reachability are hard constraints.
Candidates passing both constraints are ranked by:

1. distance from the nominal foothold;
2. deviation of realized velocity from desired velocity;
3. lateral displacement, weighted more heavily than forward/backward
   displacement.

This preserves the commanded travel direction when possible. The selected
foothold is allowed to move forward or backward before moving laterally.
`feasible_velocity_f` is recomputed from the selected target rather than copied
from the nominal target.

## Global Fallback and Blocked Behavior

If the 10 cm local search has no safe result, perform a coarser search over the
entire reachable ellipse. This fallback prioritizes any safe step over velocity
tracking while retaining the same ranking as a tie-breaker.

If the entire reachable ellipse contains no safe foothold:

- keep both feet in support;
- do not enter swing;
- expose that no next plan is available;
- report zero feasible velocity;
- retry planning while remaining stable.

The current pose-based command normally points toward one target for 8–12
seconds and can remain nearly constant while the robot is stationary. Therefore
waiting alone is not a recovery mechanism. In training, a configurable blocked
timeout requests command/target resampling. In deployment, the same blocked
signal is handed to the higher-level navigation system. The planner must not
silently select a foothold that still penetrates a danger cylinder.

## Data and Diagnostics

Planner data will distinguish:

- nominal ellipse-constrained target;
- selected safe target;
- local-search success;
- global-fallback success;
- blocked/no-safe-target;
- selected candidate displacement;
- selected candidate cost.

The foothold monitor developed in the preceding milestone will log safe-target,
fallback, and blocked rates before these signals are used for reward tuning.

## Performance Constraints

- Search runs only when preparing a new swing, not at every physics step.
- The already-safe nominal target takes the fast path.
- Candidate and sole-point checks remain GPU-batched.
- No per-candidate Python loop, `.cpu()`, or `.item()` is allowed in the
  training path.
- Candidate tensors are temporary and are not retained as trajectory history.

## Verification

Pure tensor tests must cover:

- nominal safe target passes through unchanged;
- unsafe nominal target selects the nearest direction-preserving safe target;
- ellipse-external candidates are rejected;
- one penetrated sole-perimeter point rejects the whole candidate;
- selected feasible velocity matches the adjusted target;
- local failure invokes global search;
- complete failure reports blocked and never starts swing;
- zero desired planar velocity uses a deterministic fallback axis;
- batched environments choose independently without NaN or Inf.

Visualization must show nominal target, candidate set, selected target, ellipse,
danger cylinders, and the selected 26-point footprint. The Isaac smoke test must
confirm that a blocked plan remains in double support and a valid plan enters
swing only after target data is latched.
