# Learned Foothold Planner SAC Design

## Goal

Replace only the learned foothold planner's PPO update with a small off-policy
SAC learner while preserving the 29-dimensional motor PPO, AMP, MoE backbone,
terrain queries, danger-cylinder checks, swing clearance checks, and gait state
machine.

## Scope and non-goals

- The motor policy remains the existing `WasabiPPO`/`EventGatedWasabiPPO`
  PPO path.
- The planner continues to output two continuous normalized XY actions.
- Existing nominal-target generation, terrain-height lookup, safety scoring,
  route selection, and recovery behavior are unchanged.
- No new candidate-search algorithm or new reward term is introduced.
- SAC is used only for planner actor/critic learning; it does not replace AMP
  discriminator training.

## Architecture

The combined policy keeps the existing motor actor and planner observation
encoder. A planner SAC module owns:

1. a stochastic Gaussian actor producing the same two raw planner actions as
   the current action term;
2. two Q networks taking planner observation and planner action;
3. two slowly updated target Q networks;
4. an automatically tuned entropy temperature;
5. a bounded replay buffer containing only planner-event transitions.

The actor is sampled during every rollout control step because the event gate
is reported by the environment after the action is sent; only transitions whose
event mask is true enter replay. The existing foothold action term continues
to map the raw two-vector into its unit disk, so the SAC replay stores the raw
action and the Q networks use that same raw action. This avoids changing the
planner's XY scaling while replacing its optimizer. The motor actor is
evaluated and updated by PPO exactly as before. Planner and motor parameters
remain disjoint; planner SAC gradients cannot modify motor or AMP parameters.

## Data flow

At each environment step:

```text
current observation
  -> motor PPO actor -> 29 motor actions
  -> planner SAC actor (only on a foothold event) -> 2 XY actions
  -> existing terrain/safety/state-machine logic
  -> reward and next observation
```

When `learned_foothold_action_event` is true, the trainer appends:

```text
(raw_policy_observation, planner_action, foothold_reward,
 raw_next_policy_observation, done)
```

to the replay buffer. Non-event control steps are not planner transitions.
Planner rewards retain the current scalar semantics, including safety,
nominal-point proximity, and command-direction terms.

## SAC update

After each rollout, the motor PPO update runs through its existing five epochs
and four mini-batches. The planner performs a bounded number of SAC updates
only when the replay buffer contains at least one full batch. Each update uses
the standard clipped-double-Q target:

```text
y = r + gamma * (1 - done) *
    [min(Q1_target(next_obs, next_action),
         Q2_target(next_obs, next_action))
     - alpha * log_pi(next_action | next_obs)]
```

The critic minimizes the two mean-squared Bellman errors. The actor maximizes
the minimum current Q value minus the entropy term. The temperature is updated
toward target entropy `-2`, matching the two-dimensional planner action. The
Gaussian log-probability is evaluated in raw-action space; no tanh correction is
applied because the existing unit-disk action mapping remains the execution
boundary.
Target networks use Polyak averaging. SAC updates are skipped, rather than
blocking PPO, until the warm-up event count is reached.

The initial implementation uses conservative, configurable defaults:

- replay capacity: 100,000 planner events;
- batch size: 256;
- warm-up: 1,024 planner events;
- actor, critic, and temperature learning rates: `1e-4`;
- `gamma = 0.99`;
- target-network `tau = 0.005`;
- at most two SAC gradient updates per environment rollout.

The replay stores the flattened policy observation (about 3k values for the
current G1 camera layout) so the dedicated planner depth encoder can remain
trainable during SAC actor updates. The default 100,000-event buffer is
therefore device-resident and can occupy roughly 2.4 GB in float32 at full
capacity; reduce `sac_replay_capacity` for memory-limited experiments. These
values are planner-only and do not alter the motor PPO hyperparameters.

## Checkpoint compatibility

New checkpoints save the SAC actor, both critics, target critics, their
optimizers, temperature state, and replay statistics. Loading an old
event-gated PPO checkpoint keeps the motor network and attempts to reuse the
existing two-dimensional planner actor weights; SAC critics, target critics,
temperature, and replay are initialized fresh. Loading a new SAC checkpoint
requires these SAC keys and fails with a clear error if they are absent.

## Numerical and safety behavior

- All SAC losses and parameters are checked for finite values at the same
  periodic boundary used by the existing learner.
- Gradients are clipped to the existing `max_grad_norm` value.
- Non-finite replay samples or targets cause the SAC update to be skipped with
  a diagnostic counter; they never alter the motor PPO update.
- The planner's action remains subject to the existing geometric and
  dangerous-cylinder checks. SAC does not bypass execution validation.

## Runtime and diagnostics

The environment collection loop is unchanged. Runtime overhead is limited to
the bounded SAC updates and their small networks. New metrics include replay
size, SAC update count, critic losses, actor loss, temperature, planner Q
values, and skipped-update count. Existing PPO, AMP, recovery, and foothold
route metrics remain available.

## Testing strategy

Before integration:

1. Test replay-buffer capacity, event-only insertion, terminal transitions,
   and shape validation.
2. Test SAC target calculation, twin-Q minimum, entropy term, Polyak update,
   and automatic-temperature update on deterministic tensors.
3. Test that a rollout with no planner events leaves SAC parameters unchanged
   while motor PPO can still update.
4. Test old-checkpoint loading and new-checkpoint round-trip behavior.
5. Run the existing foothold test suite, then a 64-environment short run and a
   4096-environment performance smoke test before any long training run.
