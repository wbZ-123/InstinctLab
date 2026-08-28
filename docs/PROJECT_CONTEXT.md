# Project context

This checkout extends Hiking in the Wild with an explicit foothold planner for
the G1 parkour task.  The motor controller remains the original 29-dimensional
AMP/MoE PPO policy.  The final two action dimensions are the learned planner's
raw XY foothold action; the existing environment action term still maps them
into the normalized reachability ellipse and all terrain, danger-cylinder,
clearance, contact, and recovery checks remain authoritative.

## Current planner learner

When `ENABLE_LEARNED_FOOTHOLD_PLANNER=1`, training defaults to the hybrid
`EventGatedWasabiSAC` algorithm.  Motor PPO and the AMP discriminator are
unchanged.  Planner actions are sampled with the existing actor head and are
trained by a small SAC learner only for transitions marked by
`learned_foothold_action_event`; non-event control steps never enter replay.
The planner has a dedicated depth encoder, while the shared motor encoder is
detached for planner updates so SAC cannot change motor features.

The legacy event-gated planner PPO remains available for comparison with
`LEARNED_FOOTHOLD_ALGORITHM=ppo` or
`--learned_foothold_algorithm ppo`.  New SAC checkpoints save the planner
critics, target critics, temperature, optimizer states, and event replay.  A
legacy PPO checkpoint can initialize the motor and planner actor weights, but
its SAC critics and replay start fresh.

## Training

The wrapper sets the repository and local Instinct-RL source on `PYTHONPATH`
and supplies the motion-reference paths.  A normal SAC run is:

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=planner_sac_4096env_30000it \
NUM_ENVS=4096 MAX_ITERATIONS=30000 SAVE_INTERVAL=2000 \
./scripts/foothold_train.sh 2>&1 | tee logs/planner_sac_4096env_30000it.txt
```

To use the old planner PPO comparison mode, add
`LEARNED_FOOTHOLD_ALGORITHM=ppo`.  To resume a SAC checkpoint, use the normal
`--resume --load_run <run> --checkpoint model_<iteration>.pt` options through
the wrapper.  To initialize SAC from an older planner PPO checkpoint, pass
`LEARNED_FOOTHOLD_BASE_CHECKPOINT=<path>`; only the actor/motor weights are
reused and SAC state is intentionally fresh.

## Diagnostics and validation

TensorBoard includes `Train/sac_*` metrics for replay size, event count, update
count, actor/critic losses, entropy temperature, Q values, and skipped
non-finite updates.  Existing motor, AMP, foothold, contact, clearance, and
recovery metrics are still the source of truth for behavior.  Run the focused
or complete foothold tests before a long job:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PWD/third_party/instinct_rl:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold
```

The default 100,000-event replay is device-resident.  With the current roughly
3k-value observation it can use about 2.4 GB of float32 GPU memory at full
capacity; reduce `sac_replay_capacity` in the agent config if memory is tight.
