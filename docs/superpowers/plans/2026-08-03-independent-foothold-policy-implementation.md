# Independent Foothold Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unstable shared 31-output MoE policy with a 29-output motor MoE and an independently optimized two-output foothold MLP while preserving the environment action interface.

**Architecture:** A project-local actor-critic sends the existing encoded policy observation normally to the motor MoE. The foothold MLP receives a detached copy plus a lightweight planner-only encoding of the same deployable depth image. The event-gated PPO owns disjoint motor and foothold optimizers, standard deviations, KL schedules, and gradient clipping; their sampled actions are concatenated only at the environment boundary.

**Tech Stack:** Python 3.11, PyTorch, Instinct-RL PPO/WASABI, IsaacLab configuration, pytest.

## Global Constraints

- Keep learned-disabled parkour on the original 29-action `WasabiPPO` path.
- Keep the learned environment action ordering as 29 motor values followed by two normalized foothold XY values.
- Keep the original motor depth encoder unchanged and add a small planner-only
  depth branch owned by the foothold optimizer.
- Initial foothold exploration remains `0.05 m` per axis.
- Minimum foothold exploration is the existing `0.02 m` touchdown tolerance; maximum is the existing `0.05 m` initial exploration source.
- Motor and foothold desired KL both use the existing PPO target `0.01`, but control separate optimizers.
- Use the existing PPO `2 × desired_kl` convention as the foothold early-stop ceiling.
- Use the existing motor entropy coefficient `0.006` as the initial foothold entropy coefficient instead of introducing an unrelated value.
- Do not accept the unstable shared-head 31-action checkpoints as independent-planner resume or migration sources.

---

### Task 1: Independent actor-critic parameter boundary

**Files:**
- Create: `source/instinctlab/instinctlab/learning/independent_foothold_actor_critic.py`
- Modify: `source/instinctlab/instinctlab/learning/__init__.py`
- Test: `tests/parkour/foothold/test_independent_foothold_actor_critic.py`

**Interfaces:**
- Produces `IndependentFootholdMoEActorCritic` for unit tests without an encoder.
- Produces `IndependentFootholdEncoderMoEActorCritic` for the parkour policy config.
- Produces `motor_parameters()` and `foothold_parameters()` as disjoint exhaustive tuples.
- Produces `motor_std`, `foothold_std`, `clip_motor_std()`, and `clip_foothold_std()`.

- [ ] **Step 1: Write failing construction and inference tests**

Construct the unencoded class with four observations, 31 environment actions,
29 motor actions, two critics, and foothold hidden sizes `[128, 64]`. Assert:

```python
policy.act_inference(obs).shape == (batch_size, 31)
policy.actor(obs).shape == (batch_size, 29)
policy.foothold_actor(obs.detach()).shape == (batch_size, 2)
policy.evaluate(critic_obs).shape == (batch_size, 2)
```

Assert that motor and foothold parameter IDs are disjoint and their union is
the complete actor-critic parameter set.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_independent_foothold_actor_critic.py
```

Expected: import failure because the independent actor-critic module does not
exist.

- [ ] **Step 3: Implement the minimal independent actor-critic**

Keep `self.actor` as the original 29-output `MoeLayer` so legacy motor keys
retain their names. Add a separate MLP:

```python
self.foothold_actor = nn.Sequential(
    nn.Linear(self.mlp_input_dim_a, 128),
    nn.ELU(),
    nn.Linear(128, 64),
    nn.ELU(),
    nn.Linear(64, 2),
)
```

`update_distribution()` and `act_inference()` concatenate motor output with
`foothold_actor(observations.detach())`. `evaluate()` feeds the execution
critic normally and the foothold critic with detached critic features.

Replace the generic 31-value `std` parameter after base construction with
separate `motor_std` and `foothold_std` parameters. Implement explicit clipping
methods that mutate the actual parameters rather than a temporary concatenated
tensor.

- [ ] **Step 4: Add gradient-isolation tests**

Backpropagate a sum of only the last two output dimensions and assert every
motor/encoder gradient is absent or zero. Backpropagate only the first 29
dimensions and assert every foothold gradient is absent or zero. Repeat the
same check for critic value index zero versus one.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Task 1 command. Expected: all tests pass.

---

### Task 2: Disjoint event-gated PPO updates and exploration bounds

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
- Test: `tests/parkour/foothold/test_event_gated_foothold_ppo.py`

**Interfaces:**
- `EventGatedWasabiPPO` receives `foothold_learning_rate`,
  `foothold_desired_kl`, `foothold_min_std_m`, `foothold_max_std_m`, and
  `foothold_kl_stop_multiplier`.
- `optimizer` remains the motor optimizer for upstream compatibility.
- `foothold_optimizer` updates only `actor_critic.foothold_parameters()`.

- [ ] **Step 1: Write failing optimizer-isolation tests**

Build the independent actor-critic and algorithm. Assert optimizer parameter
IDs are disjoint and exhaustive. Execute one synthetic motor-only update and
assert foothold parameters are bitwise unchanged; execute one event foothold
update and assert motor parameters are bitwise unchanged.

- [ ] **Step 2: Write failing standard-deviation-bound tests**

Set `foothold_std` below `0.02 / radius` and above `0.05 / radius`, run the
policy clipping boundary, and assert exact normalized physical limits:

```python
minimum = torch.tensor([0.02 / 0.42, 0.02 / 0.25])
maximum = torch.tensor([0.05 / 0.42, 0.05 / 0.25])
```

Assert motor standard deviations are unaffected.

- [ ] **Step 3: Write failing independent-KL guard tests**

Feed a minibatch whose event-only foothold KL exceeds `2 * 0.01`. Assert the
foothold optimizer does not step, the skip counter increments, and the motor
optimizer remains eligible to step. Assert changing foothold KL never changes
the motor learning rate.

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_event_gated_foothold_ppo.py \
tests/parkour/foothold/test_independent_foothold_actor_critic.py
```

Expected: failures for the absent separate optimizers, bounds, and KL guard.

- [ ] **Step 5: Split policy losses and optimizer steps**

Return separate motor/foothold surrogate, value, and entropy losses. Build one
motor loss and one foothold loss. Zero both optimizers, backpropagate the sum of
only the enabled losses, clip each disjoint parameter group, and step each
optimizer independently. A minibatch with no event never steps the foothold
optimizer.

- [ ] **Step 6: Add foothold KL early stopping and adaptive rate**

Use event-only foothold KL. Once it exceeds `2 * foothold_desired_kl`, skip the
current and remaining foothold policy steps in that PPO iteration. Adjust only
`foothold_learning_rate` and `foothold_optimizer`; retain the existing motor-KL
schedule for `optimizer`.

- [ ] **Step 7: Enforce physical exploration bounds**

Normalize the configured meter bounds through the reachability radii. After
each update clamp only `foothold_std` to those limits. Configure foothold
entropy from the existing `entropy_coef`, not zero.

- [ ] **Step 8: Run the focused tests and verify GREEN**

Run the Task 2 command. Expected: all tests pass.

---

### Task 3: Checkpoint and legacy initialization compatibility

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Modify: `source/instinctlab/instinctlab/learning/foothold_checkpoint.py`
- Modify: `scripts/instinct_rl/cli_args.py`
- Modify: `scripts/instinct_rl/train.py`
- Test: `tests/parkour/foothold/test_foothold_checkpoint.py`
- Test: `tests/parkour/foothold/test_train_save_interval.py`

**Interfaces:**
- New checkpoint key `foothold_optimizer_state_dict`.
- New checkpoint scalar `foothold_learning_rate`.
- Legacy 29-action migration copies `actor.*` directly, maps `std` to
  `motor_std`, and deliberately initializes `foothold_actor.*`,
  `foothold_std`, and `critics.1.*`.

- [ ] **Step 1: Write failing round-trip checkpoint tests**

Save and load the algorithm state. Assert both optimizer states, both learning
rates, both standard deviations, and both actor modules round-trip exactly.

- [ ] **Step 2: Write failing legacy migration tests**

Migrate a real-shape 29-action destination fixture. Assert original motor actor
keys and weights are copied exactly, newly appended observation columns are
zero initialized, and all planner-specific keys retain their explicit fresh
initialization. Assert a shared 31-action checkpoint is rejected with a clear
architecture error.

- [ ] **Step 3: Run checkpoint tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_foothold_checkpoint.py \
tests/parkour/foothold/test_train_save_interval.py
```

- [ ] **Step 4: Implement checkpoint state and migration**

Extend event-gated `state_dict()`/`load_state_dict()` without modifying
upstream Instinct-RL. Update the audited migration key mapping for
`motor_std`, keep motor actor names unchanged, and initialize the independent
planner deliberately. Resume synchronization restores both optimizer-rate
scalars.

- [ ] **Step 5: Run checkpoint tests and verify GREEN**

Run the Task 3 command. Expected: all tests pass.

---

### Task 4: Configuration, play, and diagnostics

**Files:**
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
- Modify: `scripts/instinct_rl/play_learned_config.py`
- Modify: `tests/parkour/foothold/inspect_foothold_tensorboard.py`
- Test: `tests/parkour/foothold/test_play_learned_config.py`
- Test: `tests/parkour/foothold/test_inspect_foothold_tensorboard.py`

**Interfaces:**
- Learned mode selects
  `instinctlab.learning.independent_foothold_actor_critic:IndependentFootholdEncoderMoEActorCritic`.
- Learned-disabled mode retains `EncoderMoEActorCritic` and 29 actions.
- Diagnostics expose motor/planner KL, learning rate, gradient norm, normalized
  and meter-valued foothold standard deviation, event count, and KL skips.

- [ ] **Step 1: Write failing configuration-path tests**

Assert enabling learned mode selects the project-local independent policy and
sets physically sourced exploration/KL values. Assert disabled mode is
unchanged.

- [ ] **Step 2: Write failing diagnostic tests**

Assert the inspector reports all independent optimizer and exploration tags
and distinguishes missing tags from numeric zeros.

- [ ] **Step 3: Implement configuration and diagnostics**

Add only the independent-policy fields needed by the local class. Keep the
saved agent configuration sufficient for play to reconstruct the exact class.
Extend logging/inspection without adding per-step device synchronizations to
collection.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_play_learned_config.py \
tests/parkour/foothold/test_inspect_foothold_tensorboard.py
```

---

### Task 5: Regression and runtime acceptance

**Files:**
- Modify only if a test exposes an in-scope defect.

**Interfaces:**
- Produces evidence authorizing or rejecting a new long training run.

- [ ] **Step 1: Run syntax and focused learning tests**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold/test_independent_foothold_actor_critic.py \
tests/parkour/foothold/test_event_gated_foothold_ppo.py \
tests/parkour/foothold/test_foothold_rollout_storage.py \
tests/parkour/foothold/test_foothold_checkpoint.py
```

- [ ] **Step 2: Run the complete foothold suite**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
tests/parkour/foothold
```

Expected: no regression; learned-disabled tests remain unchanged.

- [ ] **Step 3: Run a learned-disabled smoke test**

Run two iterations with 64 environments and confirm 29 actions, one reward
group, original policy class, and original `WasabiPPO`.

- [ ] **Step 4: Run a learned-enabled 64-environment smoke test**

Run ten iterations and confirm 31 environment actions, two disjoint optimizer
groups, finite values, nonzero foothold events, bounded foothold standard
deviation, and finite motor/foothold KL.

- [ ] **Step 5: Run a learned-enabled 4096-environment acceptance test**

Run 100 iterations only after the 64-environment smoke passes. Accept the
architecture for long training only when:

- no non-finite guard fires;
- foothold KL remains bounded by the guard;
- motor and foothold gradient-isolation diagnostics remain valid;
- physical foothold standard deviation remains within `0.02--0.05 m`;
- no repeated abrupt episode-length collapse appears;
- collection and learning timings are recorded against the previous baseline.
