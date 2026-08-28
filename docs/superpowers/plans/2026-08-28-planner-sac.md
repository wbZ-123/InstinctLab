# Learned Foothold Planner SAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Replace only the learned foothold planner's event-gated PPO optimizer with a bounded SAC learner while preserving the motor PPO/AMP/MoE path and all existing foothold geometry/state-machine behavior.

**Architecture:** Keep the existing combined actor-critic for motor rollout and planner action generation, but expose planner features and raw two-dimensional actions. Add a local planner SAC module with twin Q networks, target Q networks, automatic entropy temperature, and an event-only replay buffer. The hybrid algorithm records planner transitions during `process_env_step`, updates motor PPO and AMP as before, then performs at most two planner SAC updates.

**Tech Stack:** PyTorch, existing local `instinctlab.learning` modules, IsaacLab runner configuration, pytest.

## Global Constraints

- Motor action dimension remains 29; planner action dimension remains 2.
- Planner raw actions retain the existing unit-disk conversion in `LearnedFootholdAction`.
- Existing terrain queries, danger-cylinder checks, swing clearance, routing, recovery, AMP, and MoE logic are unchanged.
- Planner parameters and motor parameters remain disjoint.
- No SAC update may block or mutate the motor PPO update when planner events are absent or insufficient.
- Replay and SAC tensors remain on the configured training device.

---

### Task 1: Add event-only replay storage

**Files:**
- Create: `source/instinctlab/instinctlab/learning/foothold_sac_replay.py`
- Test: `tests/parkour/foothold/test_foothold_sac_replay.py`

**Interfaces:**
- `FootholdReplayBatch(obs, actions, rewards, next_obs, dones)` is a named tuple of tensors.
- `FootholdReplayBuffer(capacity: int, obs_dim: int, action_dim: int, device: torch.device)` exposes `add(...)`, `sample(batch_size)`, `__len__()`, `state_dict()`, and `load_state_dict(...)`.

- [x] Write failing tests for event insertion, circular capacity, terminal flags, sample shapes, invalid dimensions, and empty sampling.
- [x] Run the focused replay tests after implementation; the pre-implementation failure check was intentionally skipped because the module was implemented in the same work session.
- [x] Implement a preallocated device-resident circular buffer. `add` must detach inputs, validate `[batch, obs_dim]`, `[batch, action_dim]`, `[batch]` shapes, and update `size`/`position` without storing non-event transitions.
- [x] Implement uniform sampling with `torch.randint`; reject batches larger than the current size and serialize all tensors plus capacity, size, and position.
- [x] Rerun the focused test and then the existing foothold tests.

### Task 2: Implement the standalone planner SAC update

**Files:**
- Create: `source/instinctlab/instinctlab/learning/foothold_sac.py`
- Test: `tests/parkour/foothold/test_foothold_sac.py`

**Interfaces:**
- `FootholdSACConfig` contains `obs_dim`, `action_dim=2`, `hidden_dims=(128, 128)`, `replay_capacity=100000`, `batch_size=256`, `warmup_events=1024`, `updates_per_rollout=2`, `actor_lr=1e-4`, `critic_lr=1e-4`, `alpha_lr=1e-4`, `gamma=0.99`, `tau=0.005`, `target_entropy=-2.0`, and `max_grad_norm=1.0`.
- `FootholdSAC` exposes `act(features, deterministic=False)`, `observe(...)`, `update()`, `state_dict()`, `load_state_dict(...)`, and diagnostics including replay size, update count, actor/critic losses, alpha, and skipped-update count.

- [x] Write failing deterministic-tensor tests for twin-Q minimum target, terminal target masking, Polyak averaging, automatic alpha update, warm-up skip, and finite-value rejection.
- [x] Run the focused SAC tests after implementation; the pre-implementation failure check was intentionally skipped because the module was implemented in the same work session.
- [x] Implement MLP actor and twin Q networks. The actor must use the existing raw-action convention: sample from a diagonal Gaussian in raw action space and leave unit-disk processing to the environment action term; do not add a tanh correction.
- [x] Implement critic target `r + gamma*(1-done)*(min(target_q1,target_q2)-alpha*log_pi(next_action))`, critic MSE updates, actor Q/entropy update, alpha loss, gradient clipping, and Polyak target updates.
- [x] Implement `observe` through the replay buffer and make `update` return zeroed diagnostics without changing parameters when warm-up or finite checks fail.
- [x] Rerun focused tests and verify all SAC numerical tests pass.

### Task 3: Expose planner features without changing motor behavior

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/independent_foothold_actor_critic.py`
- Test: `tests/parkour/foothold/test_independent_foothold_actor_critic.py`

**Interfaces:**
- Add `planner_features(observations, detach_shared: bool = True) -> Tensor`.
- Add `planner_distribution_from_features(features) -> Normal` and `sample_planner_action(observations, deterministic=False) -> (raw_action, log_prob)`.

- [x] Write failing tests that verify planner feature shape, raw action shape `[batch, 2]`, deterministic repeatability, and detached shared encoder gradients.
- [x] Implement the helpers by reusing the existing encoded observation and dedicated planner depth encoder exactly once per call. Detach the shared motor encoder output for SAC updates; keep planner head/depth-encoder gradients available only where explicitly requested.
- [x] Preserve `act`, `evaluate`, motor parameter grouping, action normalization, and old checkpoint key names.
- [x] Run the focused actor-critic tests and the existing checkpoint/foothold tests.

### Task 4: Integrate planner SAC into the hybrid algorithm

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/foothold_rollout_storage.py`
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Modify: `source/instinctlab/instinctlab/learning/__init__.py`
- Test: `tests/parkour/foothold/test_event_gated_foothold_sac.py`

**Interfaces:**
- Add `next_observations` to `FootholdTransition` and a matching storage tensor.
- Add `EventGatedWasabiSAC`, preserving the runner-facing `act`, `process_env_step`, `compute_returns`, `update`, `state_dict`, and `load_state_dict` interfaces.

- [x] Write failing integration tests for event-only replay insertion, correct foothold reward index, no-event SAC no-op, motor PPO update independence, and planner SAC diagnostics.
- [x] Run focused integration tests after implementation; the pre-implementation failure check was intentionally skipped because the class was implemented in the same work session.
- [x] In `process_env_step`, validate the existing three event masks, copy `next_obs` into the transition, and let the current superclass compute auxiliary rewards exactly once. Store only event rows after reward augmentation, using the planner feature helper with detached shared features.
- [x] In `update`, run the existing motor PPO and AMP phases unchanged, then call bounded SAC updates. Planner SAC must never use PPO advantages, PPO KL stopping, or planner PPO optimizer state.
- [x] Add periodic finite checks and ensure a failed/empty SAC update increments a diagnostic counter without affecting motor/AMP updates.
- [x] Rerun focused integration tests and all existing foothold tests.

### Task 5: Wire configuration and checkpoint compatibility

**Files:**
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Modify: `scripts/foothold_train.sh`
- Test: `tests/parkour/foothold/test_planner_sac_config.py`

**Interfaces:**
- Learned planner enablement selects `EventGatedWasabiSAC` while legacy `EventGatedWasabiPPO` remains importable for old checkpoints/tests.
- `LEARNED_FOOTHOLD_ALGORITHM` accepts `sac` (default when learned planner is enabled) and `ppo` (explicit legacy comparison mode).

- [x] Add config tests for SAC selection, defaults, legacy PPO opt-in, and invalid algorithm values.
- [x] Implement config fields and factory selection without changing task observations, rewards, or planner geometry.
- [x] Extend hybrid checkpoint state with SAC networks, optimizers, alpha state, replay state, and version marker. Loading an old PPO checkpoint retains motor and planner actor weights while initializing missing SAC state; new SAC checkpoints restore SAC state or fail clearly when it is incomplete.
- [x] Update the shell wrapper's printed algorithm and validation messages; preserve existing environment variables and resume flags.
- [x] Run config and checkpoint tests.

### Task 6: Runtime smoke tests and documentation

**Files:**
- Modify: `docs/PROJECT_CONTEXT.md`
- Modify: `docs/foothold_planner_implementation.md`
- Test: `tests/parkour/foothold/test_train_save_interval.py`

- [x] Add SAC metrics and exact training/resume commands to `docs/PROJECT_CONTEXT.md`, including the fact that old PPO planner checkpoints initialize SAC critics afresh.
- [x] Run the complete foothold test suite with the approved hiking Python environment (`467 passed, 1 skipped`).
- [ ] Run a 64-environment short training smoke test and verify replay warm-up, SAC updates, finite diagnostics, and unchanged motor/AMP metrics (blocked here because the execution host exposes no CUDA GPU).
- [ ] Run a 4096-environment performance smoke test and compare collection time to the `before-sac-20260828` baseline (blocked here because the execution host exposes no CUDA GPU).
- [ ] Do not start a 30000-iteration run until the GPU smoke tests pass and the new checkpoint can be loaded by play.
