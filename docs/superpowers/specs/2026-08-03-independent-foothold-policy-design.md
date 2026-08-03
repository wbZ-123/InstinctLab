# Independent Foothold Policy Design

## Problem

The first long scratch run showed that the event-gated foothold policy is not
stable when its two action dimensions share the motor MoE body and optimizer.
The early 100-iteration acceptance run did not expose the failure:

- the initial normalized foothold standard deviations were `0.119` and `0.20`;
- by iteration 2,000 they had collapsed to `0.0027` and `0.0065`;
- the motor KL remained near its `0.01` target while foothold KL reached about
  `693`;
- repeated episode-length collapses preceded the permanent collapse around
  iterations 16,450--16,620.

The existing design assumption that the clipped foothold surrogate alone
bounds foothold KL is therefore invalid. A zero foothold entropy coefficient
also provides no mechanism to preserve physically meaningful exploration.

## Architecture

Keep one environment action tensor with the existing ordering:

```text
dimensions 0..28  : motor actions
dimensions 29..30 : normalized foothold XY proposal
```

Internally, produce those groups using two policy modules:

1. The existing depth encoder and MoE motor actor produce only the 29 motor
   actions.
2. A small foothold MLP produces the two foothold actions.
3. The foothold MLP consumes the planner-relevant encoded observation through
   a detached tensor. Foothold loss must not update the depth encoder, motor
   MoE experts, or motor gate.
4. Do not add a second depth CNN. Perception is computed once, keeping the
   additional learning and inference cost small.
5. Concatenate the independently sampled motor and foothold actions only at
   the environment interface.

The two policies still train online in the same rollout. Isolation applies to
their direct gradients and optimizer state, not to their physical interaction:
the foothold selected by the planner still changes the reference trajectory
that the motor policy must execute.

## Optimization Boundaries

Motor and foothold policies have separate:

- optimizers and learning rates;
- Gaussian standard-deviation parameters;
- clipped surrogate losses;
- gradient clipping;
- KL measurements and adaptive learning-rate decisions.

The motor update must not change foothold-policy parameters. The foothold
update must not change motor-policy or shared-encoder parameters.

The foothold optimizer runs only when a minibatch contains causal foothold
events. A minibatch without events performs no foothold optimizer step.

## Foothold Exploration

The initial physical standard deviation remains `0.05 m` in each horizontal
axis because this is the already documented touchdown-reward zero crossing.
It is normalized by the existing reachability radii:

```text
std_x = 0.05 / 0.42
std_y = 0.05 / 0.25
```

The policy must also enforce a separately configured physical minimum standard
deviation. That minimum is a calibration parameter and must be logged in both
meters and normalized units. It must not silently default to the numerical
`1e-15` floor used by the generic motor policy.

An entropy coefficient may shape exploration above the minimum, but the hard
physical floor is the safety net that prevents the observed sub-millimetre
collapse.

## KL Control

Motor KL continues to control only the motor optimizer learning rate.
Foothold KL independently controls only the foothold optimizer learning rate.
Neither KL may change the other optimizer.

The foothold update has an explicit KL ceiling. If the mean event-only
foothold KL exceeds the ceiling during an epoch, remaining foothold minibatch
updates for that PPO iteration are skipped. This is an optimization guard, not
an environment fallback, and it does not suppress non-finite errors.

## Checkpoints and Play

New checkpoints store both policy modules, both standard deviations, and both
optimizer states. Play loads both modules and still emits the same 31-value
environment action tensor.

Legacy 29-action initialization remains supported:

- copy the existing encoder, motor MoE, motor critic, motor standard
  deviations, and discriminator exactly;
- initialize the independent foothold MLP, foothold critic, foothold standard
  deviation, and foothold optimizer explicitly;
- never reinterpret the last two rows of a shared 31-action head as an
  independent planner checkpoint.

The unstable 31-action shared-head checkpoints are diagnostic artifacts and
are not migration sources for the independent planner architecture.

## Diagnostics and Acceptance

Log at least:

- motor and foothold learning rates;
- motor and foothold KL;
- motor and foothold gradient norms;
- foothold event count;
- foothold standard deviation in normalized units and meters;
- number of foothold updates skipped by the KL ceiling.

Before another long run, verify:

1. A foothold-only backward/update leaves every motor and encoder parameter
   bitwise unchanged.
2. A motor-only backward/update leaves every foothold parameter bitwise
   unchanged.
3. The foothold physical standard deviation never falls below its configured
   floor.
4. Excess foothold KL reduces or stops foothold updates without changing the
   motor learning rate.
5. Learned-disabled training remains byte-for-byte on the original 29-action
   algorithm path.
6. A 4096-environment short run has finite losses, nonzero foothold events,
   bounded foothold KL, and no abrupt episode-length collapse before a long run
   is authorized.

