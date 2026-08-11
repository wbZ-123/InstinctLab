# Event-Gated Learned Foothold PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the 29 motor actions every control step and the 2 learned
foothold actions only at causal planning events, with separate reward
advantages, likelihoods, KL/entropy statistics, exploration scales, and
finite-value protection.

**Architecture:** Keep the current 31-dimensional environment action and shared
actor network. Add a project-local `EventGatedWasabiPPO` that reconstructs
per-dimension likelihoods from stored actions/means/stds, uses execution
advantage for dimensions `0:29`, and uses foothold-planning advantage for
dimensions `29:31` only where a stored event mask is true. The original
`WasabiPPO` remains untouched and is selected whenever learned foothold
planning is disabled.

**Tech Stack:** Python 3.11, PyTorch, IsaacLab manager-based environments,
Instinct-RL PPO/WASABI extension points, pytest, TensorBoard.

## Global Constraints

- Do not edit `/home/zhangweibo/instinct_rl`; all PPO specialization lives in
  the InstinctLab-foothold repository.
- Learned-disabled tasks retain 29 actions, one reward group, original
  `WasabiPPO`, and existing checkpoint behavior.
- Learned-enabled tasks use action slices `motor=0:29` and
  `foothold=29:31`; startup validates the total action dimension is exactly 31.
- A foothold event mask records whether the current transition's two foothold
  action values were consumed. It is never inferred from a nonzero reward.
- Motor loss uses execution advantage index 0.
- Foothold loss uses foothold-planning advantage index 1 and event samples
  only.
- AMP/discriminator auxiliary reward is added only to execution reward group
  0, never foothold-planning group 1.
- Terrain query, coordinate transforms, safety scoring, HOLD preparation, and
  SWING target lock are unchanged.
- The initial physical foothold exploration standard deviation is sourced from
  the configured `0.05 m` touchdown reward zero-crossing and converted by the
  existing reachability radii `0.42 m` and `0.25 m`.
- A non-finite loss or gradient raises before `optimizer.step()`.
- Strict resume and 29-to-31 initialization are separate explicit operations.

---

## File Structure

- Create `source/instinctlab/instinctlab/learning/__init__.py`: register the
  project-local algorithm only on explicit learned-planner setup.
- Create `source/instinctlab/instinctlab/learning/foothold_rollout_storage.py`:
  rollout storage carrying the causal foothold event mask.
- Create `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`:
  pure grouped-likelihood helpers and `EventGatedWasabiPPO`.
- Create `source/instinctlab/instinctlab/learning/foothold_checkpoint.py`:
  audited 29-action/one-value to 31-action/two-value migration.
- Modify `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`:
  monotonic per-environment learned-evaluation generation counter.
- Modify `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`:
  increment the generation exactly when the current action is evaluated.
- Modify `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py`:
  capture generation before/after `env.step()` and publish the causal mask.
- Modify `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`:
  split learned-planning reward into its own group.
- Modify `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`:
  learned-only algorithm parameters and AMP reward routing.
- Modify `scripts/instinct_rl/train.py`: explicit algorithm registration,
  strict resume, and audited base-checkpoint initialization.
- Modify `scripts/instinct_rl/cli_args.py`: base-checkpoint initialization CLI.
- Modify `scripts/foothold_train.sh`: environment variables for strict resume
  versus initialization.
- Create focused tests under `tests/parkour/foothold/`.

---

### Task 1: Record the exact transition that consumes a foothold action

**Files:**
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Modify: `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py`
- Modify: `tests/parkour/foothold/test_foothold_planner_data.py`
- Create: `tests/parkour/foothold/test_foothold_event_mask.py`

**Interfaces:**
- Produces planner data:
  `learned_foothold_event_generation: torch.Tensor[int64]` with shape
  `(num_envs,)`.
- Produces step extra:
  `extras["learned_foothold_action_event"]: torch.Tensor[bool]` with shape
  `(num_envs,)`.

- [ ] **Step 1: Write failing generation and wrapper tests**

Add a data-field assertion:

```python
assert "learned_foothold_event_generation" in FootholdPlannerData.__annotations__
```

Create pure helper tests in `test_foothold_event_mask.py`:

```python
import torch

from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import (
    foothold_event_from_generation,
)


def test_generation_change_marks_only_consumed_transitions():
    before = torch.tensor([4, 7, 9], dtype=torch.int64)
    after = torch.tensor([5, 7, 10], dtype=torch.int64)
    assert foothold_event_from_generation(before, after).tolist() == [
        True,
        False,
        True,
    ]


def test_generation_is_monotonic_across_environment_reset():
    before = torch.tensor([11], dtype=torch.int64)
    after_reset = torch.tensor([12], dtype=torch.int64)
    assert foothold_event_from_generation(before, after_reset).item()
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd /home/zhangweibo/InstinctLab-foothold
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_foothold_planner_data.py \
  tests/parkour/foothold/test_foothold_event_mask.py
```

Expected: import/field failures because the generation contract is absent.

- [ ] **Step 3: Implement the generation contract**

Add the data field:

```python
learned_foothold_event_generation: torch.Tensor | None = None
```

Allocate it as `torch.int64` in planner buffer initialization. Increment only
inside `_prepare_learned_footholds()` after `env_ids.numel() > 0`:

```python
assert self._data.learned_foothold_event_generation is not None
self._data.learned_foothold_event_generation[env_ids] += 1
```

Do not clear this counter in planner reset or
`clear_learned_foothold_buffers()`.

Add the wrapper helper:

```python
def foothold_event_from_generation(
    before: torch.Tensor,
    after: torch.Tensor,
) -> torch.Tensor:
    if before.shape != after.shape:
        raise ValueError("Foothold event generation shapes must match.")
    if before.dtype != torch.int64 or after.dtype != torch.int64:
        raise TypeError("Foothold event generation must use torch.int64.")
    return after != before
```

In wrapper `step()`, capture the counter before `self.env.step(actions)`, read
it again afterward, and attach the cloned boolean result to `extras`. When the
learned action term is absent, do not read planner data and do not add the key.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py \
  source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py \
  source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py \
  tests/parkour/foothold/test_foothold_planner_data.py \
  tests/parkour/foothold/test_foothold_event_mask.py
git commit -m "feat: record causal foothold action events"
```

---

### Task 2: Store event masks in PPO rollout minibatches

**Files:**
- Create: `source/instinctlab/instinctlab/learning/__init__.py`
- Create: `source/instinctlab/instinctlab/learning/foothold_rollout_storage.py`
- Create: `tests/parkour/foothold/test_foothold_rollout_storage.py`

**Interfaces:**
- Produces `FootholdRolloutStorage`.
- Produces `FootholdTransition` with
  `foothold_action_event: torch.Tensor | None`.
- Minibatches expose `foothold_action_event` aligned with their sampled rows.

- [ ] **Step 1: Write failing storage tests**

```python
import torch

from instinctlab.learning.foothold_rollout_storage import (
    FootholdRolloutStorage,
    FootholdTransition,
)


def test_storage_keeps_event_mask_aligned_with_transition():
    storage = FootholdRolloutStorage(
        2, 1, [3], [3], [31], num_rewards=2, device="cpu"
    )
    transition = FootholdTransition()
    transition.observations = torch.zeros(2, 3)
    transition.critic_observations = torch.zeros(2, 3)
    transition.actions = torch.zeros(2, 31)
    transition.rewards = torch.zeros(2, 2)
    transition.dones = torch.zeros(2, dtype=torch.long)
    transition.values = torch.zeros(2, 2)
    transition.actions_log_prob = torch.zeros(2)
    transition.action_mean = torch.zeros(2, 31)
    transition.action_sigma = torch.ones(2, 31)
    transition.foothold_action_event = torch.tensor([True, False])

    storage.add_transitions(transition)

    assert storage.foothold_action_event[0].tolist() == [True, False]
```

Also test that `get_minibatch_from_selection()` returns the same mask for
selected `(T, B)` indices.

- [ ] **Step 2: Run test and verify RED**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_foothold_rollout_storage.py
```

- [ ] **Step 3: Implement storage subclass**

Subclass Instinct-RL `RolloutStorage`, allocate:

```python
self.foothold_action_event = torch.zeros(
    num_transitions_per_env,
    num_envs,
    dtype=torch.bool,
    device=self.device,
)
```

`add_transitions()` validates the event is present and boolean before calling
`super().add_transitions()`, then copies it at the current pre-increment
storage index. Override `get_minibatch_from_selection()` to append the selected
mask to a project-local minibatch named tuple without changing external
Instinct-RL classes.

- [ ] **Step 4: Run test and verify GREEN**

Run the Task 2 pytest command. Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  source/instinctlab/instinctlab/learning/__init__.py \
  source/instinctlab/instinctlab/learning/foothold_rollout_storage.py \
  tests/parkour/foothold/test_foothold_rollout_storage.py
git commit -m "feat: store foothold PPO event masks"
```

---

### Task 3: Split execution and planning reward groups

**Files:**
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
- Modify: `tests/parkour/foothold/test_learned_foothold_planner.py`
- Modify: `tests/parkour/foothold/test_reward_foothold.py`

**Interfaces:**
- Learned-disabled `RewardsCfg` has only `rewards`.
- Learned-enabled configuration adds `foothold_planning`.
- Reward tensor order is `[execution, foothold_planning]`.
- Algorithm config sets
  `auxiliary_reward_per_env_reward_coefs = [1.0, 0.0]`.

- [ ] **Step 1: Write failing configuration tests**

Add tests that instantiate the environment config and assert:

```python
cfg = G1ParkourEnvCfg()
assert cfg.rewards.foothold_planning is None

cfg.enable_learned_foothold_planner()
assert cfg.rewards.rewards.learned_foothold_planning is None
assert cfg.rewards.foothold_planning.learned_foothold_planning.weight == 1.0
```

Test the agent config:

```python
cfg = G1ParkourPPORunnerCfg()
assert cfg.algorithm.auxiliary_reward_per_env_reward_coefs == [1.0]

cfg.enable_event_gated_foothold_ppo()
assert cfg.algorithm.auxiliary_reward_per_env_reward_coefs == [1.0, 0.0]
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_learned_foothold_planner.py \
  tests/parkour/foothold/test_reward_foothold.py
```

- [ ] **Step 3: Implement separate reward group**

Create a configclass containing only:

```python
@configclass
class LearnedFootholdPlanningRewards:
    learned_foothold_planning = RewTerm(
        func=mdp.learned_foothold_planning_event_reward,
        weight=1.0,
        params={
            "sensor_name": "foothold_planner",
            "reachability_radius_x": 0.42,
            "reachability_radius_y": 0.25,
        },
    )
```

`RewardsCfg.foothold_planning` defaults to `None`. The enable method constructs
the group using `FlatProviderConfig.outer_radius_x/y` and removes the current
term from the execution group. Add
`G1ParkourPPORunnerCfg.enable_event_gated_foothold_ppo()` to select two reward
groups and the project-local algorithm parameters.

- [ ] **Step 4: Run tests and verify GREEN**

Run Task 3 tests. Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add \
  source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py \
  source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py \
  tests/parkour/foothold/test_learned_foothold_planner.py \
  tests/parkour/foothold/test_reward_foothold.py
git commit -m "feat: split foothold planning reward group"
```

---

### Task 4: Implement pure grouped PPO math and physical exploration conversion

**Files:**
- Create: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Create: `tests/parkour/foothold/test_event_gated_foothold_ppo.py`

**Interfaces:**
- `normalized_foothold_std(std_m, radii_m) -> torch.Tensor`
- `grouped_log_prob(distribution, actions, motor_dim) -> tuple[Tensor, Tensor]`
- `event_masked_mean(values, event_mask) -> Tensor`
- `grouped_clipped_surrogates(new_motor_log_prob, old_motor_log_prob,
  new_foothold_log_prob, old_foothold_log_prob, execution_advantage,
  foothold_advantage, event_mask, clip_param) -> tuple[Tensor, Tensor]`

- [ ] **Step 1: Write failing pure-math tests**

```python
def test_physical_std_uses_reachability_source():
    result = normalized_foothold_std(
        std_m=(0.05, 0.05),
        radii_m=(0.42, 0.25),
    )
    torch.testing.assert_close(
        result,
        torch.tensor([0.05 / 0.42, 0.05 / 0.25]),
    )


def test_non_event_foothold_changes_do_not_change_motor_ratio():
    # Keep motor log probabilities fixed and alter only dimensions 29:31.
    # Assert motor ratio remains one.


def test_planner_loss_uses_only_event_rows_and_planner_advantage():
    # Two rows: only row 0 is an event. Change row 1 planner likelihood and
    # execution advantage; assert planner surrogate is unchanged.


def test_no_event_minibatch_has_zero_planner_loss():
    assert event_masked_mean(
        torch.tensor([3.0, 7.0]),
        torch.tensor([False, False]),
    ).item() == 0.0
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_event_gated_foothold_ppo.py
```

- [ ] **Step 3: Implement pure helpers**

Use per-dimension `Normal.log_prob()` and explicit slices:

```python
log_prob_each = distribution.log_prob(actions)
motor = log_prob_each[:, :motor_action_dim].sum(dim=-1)
foothold = log_prob_each[:, motor_action_dim:].sum(dim=-1)
```

For planner event averaging:

```python
def event_masked_mean(values, event_mask):
    selected = values[event_mask.bool()]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()
```

Compute independent PPO ratios and apply `clip_param` independently. Motor
surrogate consumes `advantages[:, 0]`; planner surrogate consumes
`advantages[:, 1]`.

- [ ] **Step 4: Run tests and verify GREEN**

Run Task 4 tests. Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add \
  source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py \
  tests/parkour/foothold/test_event_gated_foothold_ppo.py
git commit -m "feat: add event-gated foothold PPO math"
```

---

### Task 5: Integrate EventGatedWasabiPPO with finite-value protection

**Files:**
- Modify: `source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py`
- Modify: `source/instinctlab/instinctlab/learning/__init__.py`
- Modify: `tests/parkour/foothold/test_event_gated_foothold_ppo.py`

**Interfaces:**
- Produces `EventGatedWasabiPPO`.
- Produces `register_event_gated_foothold_algorithm()`.
- Required constructor arguments:
  `motor_action_dim=29`,
  `execution_reward_index=0`,
  `foothold_reward_index=1`,
  `foothold_initial_std_m=(0.05, 0.05)`,
  `foothold_reachability_radii_m=(0.42, 0.25)`,
  `foothold_surrogate_coef=1.0`,
  `foothold_entropy_coef=0.0`.

- [ ] **Step 1: Write failing algorithm behavior tests**

Use a small real `ActorCritic` and fake minibatch to verify:

- `init_storage()` returns `FootholdRolloutStorage`;
- `process_env_step()` rejects a missing event key;
- the last two entries of `actor_critic.std` equal the physical conversion;
- `compute_losses()` returns separate
  `motor_surrogate_loss`, `foothold_surrogate_loss`,
  `motor_kl`, `foothold_kl`, and `foothold_event_count`;
- a NaN loss raises `FloatingPointError`;
- a NaN gradient prevents a counting optimizer's `step()` from being called.

- [ ] **Step 2: Run tests and verify RED**

Run Task 4's pytest file. Expected: failures for the absent class.

- [ ] **Step 3: Implement the project-local algorithm**

Define `EventGatedWasabiPPO(WasabiAlgoMixin, PPO)` in the project-local
module. Its implementation must provide the overrides listed below rather than
modifying either upstream base class.

Override:

- `__init__()` to validate action/reward indices and store separate
  coefficients;
- `init_storage()` to construct `FootholdTransition` and
  `FootholdRolloutStorage`, then construct WASABI discriminator storage exactly
  as `WasabiAlgoMixin.init_storage()` does;
- `process_env_step()` to copy
  `infos["learned_foothold_action_event"]` into the transition before the base
  storage add;
- `compute_losses()` to use Task 4 grouped math and the existing clipped value
  losses for both critics;
- `gradient_step()` to check loss, gradients, and parameters before/after the
  optimizer boundary;
- `update()` to aggregate motor KL and adjust adaptive learning rate once after
  the complete PPO minibatch loop, then run the unchanged WASABI discriminator
  update.

Registration is explicit:

```python
def register_event_gated_foothold_algorithm() -> None:
    import instinct_rl.algorithms as algorithms

    algorithms.EventGatedWasabiPPO = EventGatedWasabiPPO
```

Do not register during package import.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_event_gated_foothold_ppo.py \
  tests/parkour/foothold/test_foothold_rollout_storage.py
```

- [ ] **Step 5: Commit Task 5**

```bash
git add \
  source/instinctlab/instinctlab/learning/__init__.py \
  source/instinctlab/instinctlab/learning/event_gated_foothold_ppo.py \
  tests/parkour/foothold/test_event_gated_foothold_ppo.py
git commit -m "feat: add event-gated WASABI PPO"
```

---

### Task 6: Wire learned-only algorithm selection and diagnostics

**Files:**
- Modify: `scripts/instinct_rl/train.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
- Modify: `scripts/foothold_train.sh`
- Modify: `tests/parkour/foothold/test_train_save_interval.py`

**Interfaces:**
- `--enable_learned_foothold_planner` selects both environment and algorithm.
- Learned-disabled runs retain `class_name="WasabiPPO"`.
- Learned-enabled runs select `class_name="EventGatedWasabiPPO"`.

- [ ] **Step 1: Write failing opt-in tests**

Assert train setup calls, in order:

```python
env_cfg.enable_learned_foothold_planner()
agent_cfg.enable_event_gated_foothold_ppo()
register_event_gated_foothold_algorithm()
```

Assert the wrapper prints:

```text
learned_foothold_planner: 1
learned_foothold_algorithm: EventGatedWasabiPPO
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_train_save_interval.py
```

- [ ] **Step 3: Implement opt-in wiring**

In `train.py`, only inside the learned planner branch, configure the reward
groups, select the algorithm, register it, and print:

```text
[INFO]: Learned foothold PPO: motor_actions=29 foothold_actions=2
[INFO]: Learned foothold exploration: x=0.05m/0.42m=0.119048 y=0.05m/0.25m=0.200000
```

Validate environment action count and reward count after `gym.make()` and
before runner construction. Expected counts are 31 and 2.

- [ ] **Step 4: Run tests and verify GREEN**

Run Task 6 tests. Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```bash
git add \
  scripts/instinct_rl/train.py \
  source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py \
  scripts/foothold_train.sh \
  tests/parkour/foothold/test_train_save_interval.py
git commit -m "feat: wire learned foothold PPO opt in"
```

---

### Task 7: Add audited 29-to-31 checkpoint initialization

**Files:**
- Create: `source/instinctlab/instinctlab/learning/foothold_checkpoint.py`
- Modify: `scripts/instinct_rl/cli_args.py`
- Modify: `scripts/instinct_rl/train.py`
- Modify: `scripts/foothold_train.sh`
- Create: `tests/parkour/foothold/test_foothold_checkpoint.py`

**Interfaces:**
- CLI:
  `--initialize_learned_foothold_from /absolute/path/model_N.pt`.
- Strict `--resume` remains unchanged and cannot be combined with initialization.
- Produces a migration report with exact copied, expanded, initialized, and
  rejected parameter names.

- [ ] **Step 1: Write failing migration tests**

Build small fake state dictionaries with:

- input weight expanding by an explicitly supplied policy-input delta;
- four MoE actor output weights expanding from 29 to 31 rows;
- `std` expanding from 29 to 31;
- legacy `critic.*` parameters mapping to the execution critic
  `critics.0.*`, while `critics.1.*` remains freshly initialized;
- one unexpected mismatch.

Assert:

```python
assert migrated["actor.experts.0.6.weight"][:29].equal(source_weight)
assert migrated["critics.0.experts.0.6.weight"].equal(source_critic)
assert "actor.experts.0.6.weight[29:31]" in report.initialized
assert "critics.1.experts.0.6.weight" in report.initialized
assert report.unexpected == []
```

Assert an unexpected mismatch raises instead of being silently skipped.

- [ ] **Step 2: Run test and verify RED**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_foothold_checkpoint.py
```

- [ ] **Step 3: Implement audited migration**

Implement migration against the destination model's initialized state dict.
Allowed transformations are only:

- expand the first actor/execution-critic input matrices by the exact,
  audited flattened observation delta. For the current configuration this is
  exactly `3`: the current nominal-foot coordinates in a dedicated
  observation term. The legacy foothold history remains unchanged, and the
  last-action history continues to contain only the 29 motor actions;
- append exactly 2 rows to every final actor expert weight/bias and copy the
  first 29;
- map the legacy single `critic.*` module to `critics.0.*` and retain the
  initialized `critics.1.*` planning critic;
- append exactly 2 entries to `std`, copy the first 29, and initialize the last
  two with the physical conversion.

All equal-shape tensors copy exactly. Any other mismatch raises. Load
discriminator weights strictly. Do not load optimizer moments, but initialize
the fresh actor-critic optimizer with the source checkpoint's saved scalar
learning rate, and start runner iteration at zero.

- [ ] **Step 4: Wire mutually exclusive CLI modes**

Reject:

```text
--resume + --initialize_learned_foothold_from
```

The wrapper exposes:

```bash
LEARNED_FOOTHOLD_BASE_CHECKPOINT=/absolute/path/model_N.pt
```

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_foothold_checkpoint.py \
  tests/parkour/foothold/test_train_save_interval.py
```

- [ ] **Step 6: Commit Task 7**

```bash
git add \
  source/instinctlab/instinctlab/learning/foothold_checkpoint.py \
  scripts/instinct_rl/cli_args.py \
  scripts/instinct_rl/train.py \
  scripts/foothold_train.sh \
  tests/parkour/foothold/test_foothold_checkpoint.py \
  tests/parkour/foothold/test_train_save_interval.py
git commit -m "feat: migrate base policy to learned foothold PPO"
```

---

### Task 8: Full regression and staged runtime acceptance

**Files:**
- Modify: `docs/foothold_planner_implementation.md`
- Modify: `docs/foothold_parameter_audit.md`
- Modify: `tests/parkour/foothold/inspect_foothold_tensorboard.py`
- Modify: `tests/parkour/foothold/test_inspect_foothold_tensorboard.py`

**Interfaces:**
- Inspector reports motor/planner KL, event count, planner projection rate,
  both reward groups, gradient norm, and learning rate.

- [x] **Step 1: Write failing inspector tests**

Add scalar fixtures for:

```text
Train/motor_kl
Train/foothold_kl
Train/foothold_event_count
Train/foothold_raw_out_of_range_fraction
Train/foothold_ellipse_projection_fraction
Train/grad_norm
Loss/learning_rate
```

Assert missing finite-safety or event metrics produce a BAD inspection result.

- [x] **Step 2: Run inspector tests and verify RED**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold/test_inspect_foothold_tensorboard.py
```

- [x] **Step 3: Implement inspector and documentation updates**

Document the `0.05 m` exploration source as the touchdown reward zero-crossing
and mark it runtime-uncalibrated until the acceptance runs report projection
statistics.

- [x] **Step 4: Run the complete foothold unit suite**

```bash
cd /home/zhangweibo/InstinctLab-foothold
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
/home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q \
  tests/parkour/foothold
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [x] **Step 5: Run learned-disabled compatibility smoke**

Use 64 environments and 2 iterations without
`ENABLE_LEARNED_FOOTHOLD_PLANNER`. Confirm 29 actions, one reward, and original
`WasabiPPO`.

- [x] **Step 6: Run learned-enabled 64-environment smoke**

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=event_gated_foothold_64env_10it \
NUM_ENVS=64 \
MAX_ITERATIONS=10 \
SAVE_INTERVAL=10 \
./scripts/foothold_train.sh
```

Acceptance: 31 actions, two rewards, finite diagnostics, nonzero event count,
and `model_10.pt`.

- [x] **Step 7: Run 4096-environment numerical acceptance**

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=event_gated_foothold_4096env_100it \
NUM_ENVS=4096 \
MAX_ITERATIONS=100 \
SAVE_INTERVAL=100 \
./scripts/foothold_train.sh
```

Acceptance:

- reaches `model_100.pt`;
- no finite guard fires;
- motor and planner KL remain finite;
- planner event count is nonzero;
- both reward streams are finite;
- TensorBoard inspector passes.

- [x] **Step 8: Compare learned-disabled performance**

Run the original 4096-environment 100-iteration baseline under the same GPU
conditions. Event-gated code must add no learned-planner tensors or material
collection-time regression when disabled.

- [ ] **Step 9: Commit Task 8**

```bash
git add \
  docs/foothold_planner_implementation.md \
  docs/foothold_parameter_audit.md \
  tests/parkour/foothold/inspect_foothold_tensorboard.py \
  tests/parkour/foothold/test_inspect_foothold_tensorboard.py
git commit -m "docs: verify event-gated foothold training"
```

#### Runtime acceptance record (2026-07-31)

- Full foothold suite: `284 passed, 1 skipped`.
- Learned-disabled smoke:
  - 29 actions (`joint_pos` only);
  - one reward group;
  - original `WasabiPPO`;
  - completed two iterations and wrote `model_2.pt`.
- Learned-enabled 64-environment smoke:
  - 31 actions ordered as 29 motor + 2 foothold;
  - two reward groups;
  - finite motor/planner KL and nonzero foothold events.
- The first 4096-environment diagnostic exposed four non-finite planner
  returns.  Root cause was the edge-cylinder kernel's undefined penetration
  direction when a sole point lies exactly on a cylinder centerline.  The
  safety scorer now conservatively converts that undefined penetration to its
  existing full-penalty depth instead of allowing NaN to enter GAE.
- Final 4096-environment acceptance reached `model_100.pt` without a finite
  guard, traceback, or empty event update:
  - motor KL mean `0.00748`;
  - foothold KL mean `0.00977`;
  - foothold event count range `664.75` to `4826.5` per reported update;
  - execution and planner reward streams remained finite;
  - stable mean collection time `5.546 s`;
  - stable mean learning time `1.814 s`;
  - stable mean total iteration time `7.360 s`.
- Learned-disabled runtime contains no learned action, observation expansion,
  second reward, second critic, or event-gated storage.  The learned-enabled
  collection time remains close to the previously measured clean 4096-env
  baseline; the expected remaining overhead is concentrated in the second
  critic and grouped PPO learning phase.
