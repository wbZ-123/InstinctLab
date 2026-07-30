# Event-Gated Learned Foothold PPO Design

## Goal

Train the existing 29-dimensional low-level motor action and the new
2-dimensional learned foothold action in one rollout without pretending that
they have the same control frequency, exploration scale, reward semantics, or
PPO likelihood.

The environment-facing action remains 31-dimensional. The existing foothold
coordinate, terrain-height, geometry, safety-score, HOLD preparation, and
SWING-lock contracts do not change.

## Motivation

The first end-to-end learned-planner implementation proved that the data path
works, but it reused the original PPO action distribution unchanged:

- all 31 action dimensions share the same initial standard deviation;
- the 2 foothold dimensions only affect the environment at planning events;
- `ActorCritic.get_actions_log_prob()` sums all 31 dimensions every control
  step;
- adaptive KL and entropy therefore include unused foothold samples;
- the learned-planning reward is mixed into the same reward stream as
  low-level locomotion.

With a policy standard deviation near `0.9`, the normalized foothold action
corresponds to approximately `0.378 m` longitudinal and `0.225 m` lateral
one-standard-deviation exploration under the current `0.42 m × 0.25 m`
reachability ellipse. This is not equivalent to the physical meaning of a
joint-position action.

The 4096-environment resume run remained finite through iteration 372, then a
PPO minibatch update contaminated the shared actor parameters with non-finite
values. The last completed iteration had a pre-clipping gradient norm of
`44.44` and a final adaptive learning rate of `2.56e-4`. The new design must
remove the action-semantics mismatch and detect the first non-finite
intermediate before an optimizer step can corrupt the model.

## Non-goals

- Do not create a second large standalone planner network.
- Do not change the learned foothold output from final support-frame
  horizontal coordinates to a deployment-time danger-cylinder representation.
- Do not change terrain-height lookup, reachability, or the `0.25 m` hard step
  height limit.
- Do not make the analytic candidate search part of the learned route.
- Do not silently skip non-finite updates and continue with unknown optimizer
  state.
- Do not change PPO behavior for tasks where the learned foothold planner is
  disabled.

## Architecture

The actor keeps one shared observation encoder and one shared actor body. Its
31-dimensional Gaussian action is treated as two logical groups:

```text
dimensions 0..28  : low-level motor action
dimensions 29..30 : high-level foothold action
```

This logical split avoids a second terrain encoder and preserves the current
environment action interface. The PPO implementation, however, must retain
per-dimension log probabilities long enough to form two independent clipped
surrogate objectives.

The learned-planner mode selects a project-local PPO extension. The extension
must be registered only when the learned planner is enabled; original parkour,
locomotion, play, and old-checkpoint runs continue to use the original
`WasabiPPO`.

## Event Contract

For every environment transition, the environment publishes a boolean
`foothold_action_event` that is true exactly when the two foothold action
dimensions from that transition are evaluated to prepare or lock a foothold.

The event is causal:

```text
policy samples action at step t
→ environment consumes its foothold dimensions, or does not consume them
→ transition t stores the resulting event mask
```

The mask is stored in rollout storage beside the action, old mean, old
standard deviation, reward, and done flag. It is not reconstructed later from
phase, contact state, reward value, or a sampled monitor statistic.

Terminal transitions retain their event value. Resetting planner buffers must
not erase the event attached to the transition that caused the reset.

## Reward and Advantage Contract

Learned-planner training adds a second reward group:

1. `execution`: the existing locomotion, AMP, stability, trajectory-tracking,
   and touchdown rewards;
2. `foothold_planning`: the bounded learned foothold planning score evaluated
   only at foothold action events.

The critic therefore predicts two values and rollout storage computes two GAE
advantages.

- The motor surrogate uses only the execution advantage.
- The foothold surrogate uses only the foothold-planning advantage and only
  samples whose stored event mask is true.
- A minibatch with no foothold events contributes exactly zero foothold
  surrogate and zero foothold entropy; it is not treated as an error.
- Event loss is averaged over event samples, not over every control sample.
  Therefore its magnitude does not shrink merely because HOLD planning events
  are sparse.

The foothold-planning reward remains bounded to `[-1, 1]`. Hard geometric
invalidity maps to `-1`; safe nominal identity and continuous penetration
scores retain their existing definitions.

## PPO Likelihood and KL Contract

For each minibatch, reconstruct per-dimension old log probabilities from the
stored action, old mean, and old standard deviation. Compute current
per-dimension log probabilities from the current distribution.

```text
motor log probability
    = sum of dimensions 0..28

foothold log probability
    = sum of dimensions 29..30
```

The two PPO ratios and clipped surrogates are computed independently.

- Motor KL is evaluated on every sample.
- Foothold KL is evaluated only on event samples.
- Motor and foothold KL are logged separately.
- Entropy is also logged and weighted separately.
- The original scalar 31-dimensional likelihood must not be used by the
  event-gated algorithm.

During the first validation implementation, adaptive learning-rate changes are
driven by motor KL only. Foothold KL is bounded by its own clipped objective
and is diagnostic. This prevents sparse foothold samples from repeatedly
changing the learning rate used by the mature low-level controller.

The adaptive learning rate is updated at most once per PPO iteration from the
mean motor KL, not once per minibatch. Original tasks retain their existing
scheduler behavior.

## Exploration-Scale Contract

Motor and foothold dimensions retain separate configured initial standard
deviations.

Motor dimensions keep the original policy value. Foothold exploration is
specified in meters and converted through the same reachability radii used by
the decoder:

```text
normalized_std_x = exploration_std_x_m / reachability_radius_x_m
normalized_std_y = exploration_std_y_m / reachability_radius_y_m
```

This prevents a second independent normalized tuning convention. The
configuration records both physical exploration values and their reachability
source. Tests verify the conversion.

The first runtime calibration must report:

- normalized and meter-valued foothold standard deviations;
- fraction of raw foothold samples outside `[-1, 1]`;
- fraction projected by the reachability ellipse;
- event count per PPO iteration.

No final physical exploration value is declared calibrated until these
measurements are collected. A conservative test value may be used for a short
validation run, but it must remain labeled uncalibrated in configuration and
documentation.

## Numerical-Safety Contract

The event-gated PPO performs finite checks at the following boundaries:

1. observations and critic observations entering a minibatch;
2. stored action, old mean, old standard deviation, returns, and advantages;
3. current action mean and standard deviation;
4. motor and foothold log-probability deltas and ratios;
5. every scalar loss before backward;
6. gradient norm before optimizer step;
7. parameters and optimizer state immediately after optimizer step.

When a check fails, training raises a diagnostic error before the next
minibatch. The diagnostic identifies the first failing boundary, iteration,
epoch, minibatch, current learning rate, motor event count, motor KL, foothold
KL, and extrema of the relevant tensor.

If the gradient norm is non-finite, the optimizer step is not executed.
Training must never silently continue after a non-finite gradient.

## Checkpoint Initialization

The failed 300/310-iteration learned-planner checkpoints are diagnostic
artifacts and are not training initialization.

The first validation run initializes from a stable 30000-iteration
29-dimensional explicit-foothold-tracking checkpoint:

- copy all shape-compatible encoder, actor-body, AMP, and critic parameters;
- copy the 29 motor-output rows and biases;
- initialize the 2 foothold-output rows independently;
- copy the original critic output into the execution critic output;
- initialize the foothold-planning critic output independently;
- initialize weights connected to newly appended nominal-foothold observation
  columns without altering the copied old columns;
- do not load incompatible optimizer state;
- emit a parameter-by-parameter migration report and fail if an unexpected
  tensor is skipped.

All weights remain trainable after initialization. This is joint fine-tuning,
not a frozen planner-only stage.

The implementation must also support a fully fresh run from random
initialization. After the migrated validation is stable, a from-scratch run is
the required independence check.

## Compatibility

When learned foothold planning is disabled:

- action dimension remains 29;
- reward group count remains unchanged;
- original `WasabiPPO` is used;
- checkpoint loading behavior is unchanged;
- no event-mask tensor is allocated;
- no learned-planner loss or diagnostics are computed.

When learned foothold planning is enabled, loading a checkpoint must explicitly
select either:

- strict resume of an event-gated checkpoint; or
- audited migration from a 29-action base checkpoint.

Implicit partial loading is forbidden.

## Verification

Unit tests must cover:

- exact motor/foothold action slicing;
- meter-to-normalized foothold standard-deviation conversion;
- event-mask capture before reset;
- event-mask rollout storage and minibatch indexing;
- motor loss independence from non-event foothold actions;
- zero planner loss for a minibatch with no events;
- planner loss using only planner advantage at event samples;
- separate motor and planner KL/entropy;
- a non-finite gradient preventing optimizer step;
- compatibility path retaining original 29-action behavior;
- audited 29-to-31 action and 1-to-2 critic checkpoint migration.

Runtime acceptance requires:

1. 64 environments for 10 iterations with all finite diagnostics;
2. 4096 environments for 100 iterations without non-finite checks firing;
3. motor KL, foothold KL, gradient norm, event rate, foothold projection rate,
   and both reward streams present in TensorBoard;
4. no material regression in the original learned-disabled 4096-environment
   performance baseline;
5. only after those gates pass, a longer learned-planner run.
