# G1 Flat Foothold Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately registered G1 flat-ground foothold-tracking task that trains a 29-DoF four-expert MoE policy from a fixed 44-D foothold command, with resumable curriculum state, play-time visualization, and optional W&B monitoring.

**Architecture:** Pure PyTorch modules own foothold geometry, gait transitions, flat target generation, swing references, and curriculum state; these can be tested without launching Isaac Sim. A single Isaac Lab `CommandTerm` adapts those modules to robot/contact state and exposes one authoritative runtime state to observations, rewards, metrics, and markers. New environment and agent configurations inherit the existing G1 parkour task but leave the existing task IDs unchanged.

**Tech Stack:** Python 3.10+, PyTorch 2.7, Isaac Lab manager-based environments, Instinct-RL `WasabiPPO`, Gymnasium, pytest 9, TensorBoard, optional Weights & Biases.

## Global Constraints

- Work only in branch `feat/foothold-01-flat-tracking` at `/home/zhangweibo/InstinctLab-foothold`.
- Preserve `Instinct-Parkour-G1-v0` and `Instinct-Parkour-G1-Play-v0` behavior.
- Register new IDs `Instinct-Parkour-Foothold-G1-v0` and `Instinct-Parkour-Foothold-G1-Play-v0`.
- Keep the actor action dimension at 29 and use four MoE experts from the first training run.
- Keep the actor foothold interface exactly 44 dimensions; raw depth and binary contacts are excluded from actor observations.
- Use virtual sole centers and sole polygons derived from ankle links; never use the ankle-link origin as the touchdown target.
- Use a gravity-aligned stance-sole frame frozen at swing start.
- Start AMP discriminator training immediately, but schedule its reward coefficient through `0.00`, `0.02`, `0.05`, and `0.10`.
- Do not launch a 4096-environment run until unit, one-environment, 64-environment, and 256-environment gates pass.
- Do not hard-code user-specific dataset paths; use the existing motion configuration or an environment variable.
- Each implementation task follows red-green-refactor and ends with a focused Git commit.

---

## File Map

**Create**

- `source/instinctlab/instinctlab/tasks/parkour/foothold/__init__.py`: public pure-domain API.
- `source/instinctlab/instinctlab/tasks/parkour/foothold/types.py`: command layout, gait-state enum, immutable result records.
- `source/instinctlab/instinctlab/tasks/parkour/foothold/geometry.py`: sole transforms and frozen-frame coordinate conversion.
- `source/instinctlab/instinctlab/tasks/parkour/foothold/flat_provider.py`: curriculum-bounded flat foothold target generation.
- `source/instinctlab/instinctlab/tasks/parkour/foothold/trajectory.py`: two-segment quintic swing reference.
- `source/instinctlab/instinctlab/tasks/parkour/foothold/state_machine.py`: vectorized phase/contact transition logic.
- `source/instinctlab/instinctlab/tasks/parkour/foothold/curriculum.py`: serializable per-environment curriculum state.
- `source/instinctlab/instinctlab/tasks/parkour/foothold/metrics.py`: touchdown percentile reduction.
- `source/instinctlab/instinctlab/tasks/parkour/mdp/commands/foothold_command.py`: Isaac Lab adapter and authoritative runtime buffers.
- `source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_foothold_flat_cfg.py`: train/play environments.
- `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_foothold_cfg.py`: MoE, PPO, AMP, and rollout settings.
- `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/foothold_runner.py`: environment-state checkpoint adapter and optional W&B lifecycle.
- `tests/parkour/foothold/`: CPU unit tests for every pure module and runner protocol.

**Modify**

- `source/instinctlab/instinctlab/tasks/parkour/mdp/commands/commands_cfg.py`: add `FootholdCommandCfg`.
- `source/instinctlab/instinctlab/tasks/parkour/mdp/commands/__init__.py`: export command and config.
- `source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py`: phase-gated swing, touchdown, and support rewards.
- `source/instinctlab/instinctlab/tasks/parkour/mdp/curriculums.py`: feed episode outcomes into curriculum.
- `source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py`: register two new task IDs.
- `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/__init__.py`: export the local runner.
- `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py`: expose environment training-state protocol.
- `scripts/instinct_rl/cli_args.py`: add W&B CLI options.
- `scripts/instinct_rl/train.py`: construct the local runner and initialize optional W&B.
- `source/instinctlab/instinctlab/tasks/parkour/scripts/play.py`: toggle and update foothold debug markers.
- `source/instinctlab/instinctlab/tasks/parkour/README.md`: commands, validation ladder, acceptance thresholds.

---

### Task 1: Freeze the 44-D command and sole-frame mathematics

**Files:**
- Create: `source/instinctlab/instinctlab/tasks/parkour/foothold/__init__.py`
- Create: `source/instinctlab/instinctlab/tasks/parkour/foothold/types.py`
- Create: `source/instinctlab/instinctlab/tasks/parkour/foothold/geometry.py`
- Test: `tests/parkour/foothold/test_types.py`
- Test: `tests/parkour/foothold/test_geometry.py`

**Interfaces:**
- Produces: `FOOTHOLD_COMMAND_DIM: int = 44`, `CommandSlice`, `GaitState`, `SoleGeometry`, `make_frozen_stance_frame(origin_w, yaw_w)`, `world_to_frozen(points_w, frame)`, and `frozen_to_world(points_f, frame)`.

- [ ] **Step 1: Write failing layout and round-trip tests**

```python
import torch

from instinctlab.tasks.parkour.foothold.geometry import (
    SoleGeometry,
    frozen_to_world,
    make_frozen_stance_frame,
    world_to_frozen,
)
from instinctlab.tasks.parkour.foothold.types import CommandSlice, FOOTHOLD_COMMAND_DIM


def test_command_layout_is_contiguous_and_44d():
    slices = [member.value for member in CommandSlice]
    assert slices[0].start == 0
    assert all(left.stop == right.start for left, right in zip(slices, slices[1:]))
    assert slices[-1].stop == FOOTHOLD_COMMAND_DIM == 44


def test_frozen_stance_frame_round_trip():
    origin = torch.tensor([[1.0, 2.0, 0.3]])
    heading = torch.tensor([torch.pi / 2])
    frame = make_frozen_stance_frame(origin, heading)
    point_w = torch.tensor([[1.2, 2.1, 0.35]])
    torch.testing.assert_close(frozen_to_world(world_to_frozen(point_w, frame), frame), point_w)


def test_sole_center_is_offset_from_ankle():
    geometry = SoleGeometry(center_offset_b=torch.tensor([0.02, 0.0, -0.058]), half_length=0.12, half_width=0.055)
    center = geometry.center_world(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(center, torch.tensor([[0.02, 0.0, -0.058]]))
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `conda run -n hiking python -m pytest tests/parkour/foothold/test_types.py tests/parkour/foothold/test_geometry.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'instinctlab.tasks.parkour.foothold'`.

- [ ] **Step 3: Implement the exact layout and geometry API**

```python
class CommandSlice(Enum):
    SWING_ONE_HOT = slice(0, 2)
    PHASE_SIN_COS = slice(2, 4)
    NORMALIZED_TIME = slice(4, 5)
    CURRENT_POSITION = slice(5, 8)
    CURRENT_YAW_SIN_COS = slice(8, 10)
    CURRENT_NORMAL = slice(10, 13)
    NEXT_POSITION = slice(13, 16)
    NEXT_YAW_SIN_COS = slice(16, 18)
    NEXT_NORMAL = slice(18, 21)
    FEASIBLE_VELOCITY = slice(21, 24)
    APEX_HEIGHT = slice(24, 25)
    CORRIDOR_HEIGHTS = slice(25, 33)
    CORRIDOR_CONFIDENCES = slice(33, 41)
    PLANNER_VALID = slice(41, 42)
    NEXT_VALID = slice(42, 43)
    TERRAIN_CONFIDENCE = slice(43, 44)


FOOTHOLD_COMMAND_DIM = 44


class GaitState(IntEnum):
    HOLD = 0
    LEFT_SWING = 1
    RIGHT_SWING = 2
    TOUCHDOWN_CONFIRM = 3
    EARLY_CONTACT = 4
    OVERDUE = 5
    STANCE_LOST = 6
    PLAN_INVALID = 7
    RECOVERY = 8
```

Implement `FrozenFrame(origin_w: Tensor, cos_yaw: Tensor, sin_yaw: Tensor)` with explicit yaw rotation formulas, and `SoleGeometry.center_world()` plus four polygon corners using quaternion rotation from `isaaclab.utils.math` only behind a function-local import. This keeps the layout tests importable on CPU without Isaac Sim.

- [ ] **Step 4: Run the focused tests**

Run: `conda run -n hiking python -m pytest tests/parkour/foothold/test_types.py tests/parkour/foothold/test_geometry.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/foothold tests/parkour/foothold/test_types.py tests/parkour/foothold/test_geometry.py
git commit -m "feat(foothold): define command and sole frames"
```

### Task 2: Generate curriculum-bounded flat footholds

**Files:**
- Create: `source/instinctlab/instinctlab/tasks/parkour/foothold/flat_provider.py`
- Test: `tests/parkour/foothold/test_flat_provider.py`

**Interfaces:**
- Consumes: `FrozenFrame`.
- Produces: `FlatProviderConfig`, `FlatTargetBatch`, and `sample_flat_targets(stance_xy, swing_side, desired_velocity, level, generator, cfg)`.

Freeze the later replacement boundary now:

```python
@dataclass(frozen=True)
class TerrainCorridor:
    heights: torch.Tensor
    confidences: torch.Tensor
    support_margin: torch.Tensor
    edge_risk: torch.Tensor
    unknown_fraction: torch.Tensor


@dataclass(frozen=True)
class FootholdPlanRequest:
    stance_pose_w: torch.Tensor
    swing_side: torch.Tensor
    desired_velocity_f: torch.Tensor
    curriculum_level: torch.Tensor


@dataclass(frozen=True)
class FootholdPlan:
    current_pose_f: torch.Tensor
    next_pose_f: torch.Tensor
    current_normal_f: torch.Tensor
    next_normal_f: torch.Tensor
    feasible_velocity_f: torch.Tensor
    apex_height: torch.Tensor
    planner_valid: torch.Tensor
    next_valid: torch.Tensor
    terrain_confidence: torch.Tensor


class TerrainProvider(Protocol):
    def query_corridor(self, start_w: torch.Tensor, goal_w: torch.Tensor) -> TerrainCorridor:
        pass


class FootholdPlanner(Protocol):
    def plan(self, request: FootholdPlanRequest, terrain: TerrainCorridor) -> FootholdPlan:
        pass
```

`TerrainCorridor` contains eight heights, eight confidences, support margin, edge risk, and unknown fraction. `FootholdPlan` contains current/next pose, normals, feasible velocity, apex height, and three validity/confidence scalars. The flat provider returns zero heights, unit confidence/support, and zero edge/unknown values; a later oracle map or depth provider can replace it without changing the 44-D actor interface.

- [ ] **Step 1: Write tests for ellipse limits, side separation, and deterministic sampling**

```python
def test_level_zero_targets_stay_inside_inner_ellipse():
    generator = torch.Generator().manual_seed(7)
    result = sample_flat_targets(
        stance_xy=torch.zeros(4096, 2),
        swing_side=torch.arange(4096).remainder(2),
        desired_velocity=torch.tensor([[0.5, 0.0]]).repeat(4096, 1),
        level=torch.zeros(4096, dtype=torch.long),
        generator=generator,
        cfg=FlatProviderConfig(),
    )
    normalized = (result.position_f[:, 0] / 0.22).square() + (result.position_f[:, 1].abs() / 0.12).square()
    assert normalized.max() <= 1.0 + 1e-5
    assert torch.all(result.position_f[result.swing_side == 0, 1] >= 0.06)
    assert torch.all(result.position_f[result.swing_side == 1, 1] <= -0.06)


def test_same_seed_repeats_targets():
    kwargs = dict(
        stance_xy=torch.zeros(8, 2),
        swing_side=torch.arange(8).remainder(2),
        desired_velocity=torch.zeros(8, 2),
        level=torch.ones(8, dtype=torch.long),
        cfg=FlatProviderConfig(),
    )
    first = sample_flat_targets(generator=torch.Generator().manual_seed(3), **kwargs)
    second = sample_flat_targets(generator=torch.Generator().manual_seed(3), **kwargs)
    torch.testing.assert_close(first.position_f, second.position_f)
```

- [ ] **Step 2: Verify failure, implement, and verify success**

Run before implementation: `conda run -n hiking python -m pytest tests/parkour/foothold/test_flat_provider.py -q`

Expected before: import failure. Implement vectorized uniform disk sampling, scale by level radii `[(0.22, 0.12), (0.30, 0.17), (0.38, 0.22)]`, add velocity feed-forward `0.16 * desired_velocity`, clamp to the fixed outer ellipse `(0.42, 0.25)`, enforce `0.06 m` lateral separation, and return flat normals `[0, 0, 1]`, zero corridor heights, unit confidences, and valid flags.

Run after: `conda run -n hiking python -m pytest tests/parkour/foothold/test_flat_provider.py -q`

Expected after: `2 passed`.

- [ ] **Step 3: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/foothold/flat_provider.py tests/parkour/foothold/test_flat_provider.py
git commit -m "feat(foothold): add flat target provider"
```

### Task 3: Add the gait state machine and swing reference

**Files:**
- Create: `source/instinctlab/instinctlab/tasks/parkour/foothold/state_machine.py`
- Create: `source/instinctlab/instinctlab/tasks/parkour/foothold/trajectory.py`
- Test: `tests/parkour/foothold/test_state_machine.py`
- Test: `tests/parkour/foothold/test_trajectory.py`

**Interfaces:**
- Produces: `GaitMachineConfig(reset_hold_s=0.4, swing_s=0.32, contact_confirm_s=0.04, overdue_s=0.12)`, `advance_gait(state, contact, planner_valid, dt)`, and `quintic_swing_reference(start, goal, phase, apex_height)`.

- [ ] **Step 1: Write transition and boundary-condition tests**

```python
def test_reset_holds_for_point_four_seconds():
    state = initial_gait_state(1, device="cpu")
    for _ in range(19):
        state = advance_gait(state, contact=torch.tensor([[True, True]]), planner_valid=torch.tensor([True]), dt=0.02)
        assert state.mode.item() == GaitState.HOLD
    state = advance_gait(state, contact=torch.tensor([[True, True]]), planner_valid=torch.tensor([True]), dt=0.02)
    assert state.mode.item() == GaitState.LEFT_SWING


def test_invalid_plan_enters_recovery_through_plan_invalid():
    state = initial_gait_state(1, device="cpu")
    state.hold_time[:] = 0.4
    state = advance_gait(state, contact=torch.tensor([[True, True]]), planner_valid=torch.tensor([False]), dt=0.02)
    assert state.mode.item() == GaitState.PLAN_INVALID


def test_quintic_reference_has_zero_endpoint_velocity():
    start = torch.tensor([[0.0, 0.1, 0.0]])
    goal = torch.tensor([[0.3, 0.1, 0.0]])
    at_start = quintic_swing_reference(start, goal, torch.tensor([0.0]), apex_height=torch.tensor([0.10]))
    at_end = quintic_swing_reference(start, goal, torch.tensor([1.0]), apex_height=torch.tensor([0.10]))
    torch.testing.assert_close(at_start.position, start)
    torch.testing.assert_close(at_end.position, goal)
    torch.testing.assert_close(at_start.velocity, torch.zeros_like(start))
    torch.testing.assert_close(at_end.velocity, torch.zeros_like(goal), atol=1e-6, rtol=0.0)
```

- [ ] **Step 2: Implement and run the tests**

Use quintic blend `s(u)=10u^3-15u^4+6u^5`; split vertical motion at `u=0.5` so both halves meet at `max(start_z, goal_z) + apex_height`. In `advance_gait`, debounce contact for two 20 ms samples, latch touchdown once, classify contact before phase `0.65` as `EARLY_CONTACT`, classify no contact after nominal swing plus `0.12 s` as `OVERDUE`, and send invalid/stance-lost states to `RECOVERY`.

Run: `conda run -n hiking python -m pytest tests/parkour/foothold/test_state_machine.py tests/parkour/foothold/test_trajectory.py -q`

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/foothold/state_machine.py source/instinctlab/instinctlab/tasks/parkour/foothold/trajectory.py tests/parkour/foothold
git commit -m "feat(foothold): add gait state and swing reference"
```

### Task 4: Integrate one authoritative Isaac Lab command term

**Files:**
- Create: `source/instinctlab/instinctlab/tasks/parkour/mdp/commands/foothold_command.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/mdp/commands/commands_cfg.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/mdp/commands/__init__.py`
- Test: `tests/parkour/foothold/test_command_packing.py`

**Interfaces:**
- Produces: `FootholdCommand(CommandTerm)`, `FootholdCommandCfg`, `command: Tensor[N,44]`, `swing_reference_w`, `touchdown_latched`, `target_error`, `state_dict()`, and `load_state_dict()`.

- [ ] **Step 1: Test command packing without simulation**

```python
def test_pack_command_writes_every_named_slice():
    fields = {}
    for index, command_slice in enumerate(CommandSlice):
        width = command_slice.value.stop - command_slice.value.start
        fields[command_slice] = torch.full((2, width), float(index))
    packed = pack_foothold_command(fields)
    assert packed.shape == (2, 44)
    for command_slice, expected in fields.items():
        torch.testing.assert_close(packed[:, command_slice.value], expected)
```

- [ ] **Step 2: Implement `pack_foothold_command` and the command adapter**

`FootholdCommand.__init__` resolves left/right ankle body IDs once, creates `SoleGeometry(center_offset_b=[0.02, 0.0, -0.058], half_length=0.12, half_width=0.055)`, allocates all buffers on `env.device`, and rejects a packed tensor whose last dimension is not 44. `_resample_command(env_ids)` freezes the current stance frame and obtains current/next targets from `sample_flat_targets`. `_update_command()` advances gait, evaluates the swing reference, computes errors and metrics, and packs the 44-D tensor. `_set_debug_vis_impl()` and `_debug_vis_callback()` own current target, next target, sole polygon, reference path, support/edge/unknown overlays, and state-color markers.

Add this exact config surface:

```python
@configclass
class FootholdCommandCfg(CommandTermCfg):
    class_type: type = FootholdCommand
    asset_name: str = "robot"
    left_ankle_body: str = "left_ankle_roll_link"
    right_ankle_body: str = "right_ankle_roll_link"
    sole_center_offset: tuple[float, float, float] = (0.02, 0.0, -0.058)
    sole_half_length: float = 0.12
    sole_half_width: float = 0.055
    reset_hold_s: float = 0.4
    nominal_swing_s: float = 0.32
    apex_height: float = 0.10
    debug_vis: bool = False
```

- [ ] **Step 3: Run CPU tests and a one-environment construction check**

Run: `conda run -n hiking python -m pytest tests/parkour/foothold/test_command_packing.py -q`

Expected: pass.

Run: `conda run -n hiking python scripts/list_envs.py | rg 'Instinct-Parkour'`

Expected at this task: existing parkour IDs still list; new IDs are added in Task 8.

- [ ] **Step 4: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/mdp/commands tests/parkour/foothold/test_command_packing.py
git commit -m "feat(foothold): integrate foothold command term"
```

### Task 5: Add phase-gated observations and rewards

**Files:**
- Modify: `source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py`
- Create: `source/instinctlab/instinctlab/tasks/parkour/mdp/foothold_observations.py`
- Test: `tests/parkour/foothold/test_rewards.py`

**Interfaces:**
- Produces: `foothold_command(env)`, `foothold_critic_state(env)`, `swing_position_reward`, `swing_velocity_reward`, `touchdown_pose_reward`, `support_polygon_reward`, and `slip_penalty`.

- [ ] **Step 1: Test phase masks and one-shot touchdown**

```python
def test_swing_reward_is_zero_outside_swing():
    error = torch.tensor([0.00, 0.05])
    mask = torch.tensor([False, True])
    result = gaussian_phase_reward(error, mask, sigma=0.04)
    torch.testing.assert_close(result, torch.tensor([0.0, torch.exp(torch.tensor(-1.5625))]))


def test_touchdown_reward_only_uses_new_latch():
    pose_error = torch.tensor([0.0, 0.0])
    new_touchdown = torch.tensor([True, False])
    torch.testing.assert_close(touchdown_latched_reward(pose_error, new_touchdown, sigma=0.05), torch.tensor([1.0, 0.0]))
```

- [ ] **Step 2: Implement terms and run tests**

Use `exp(-(error/sigma)^2)` for positive tracking terms. Apply swing terms only in `LEFT_SWING`/`RIGHT_SWING`, touchdown terms only on the rising edge of `touchdown_latched`, and support/slip terms only for the stance foot. Read every target, phase, and mask from `env.command_manager.get_term("foothold")`; do not recompute planner state in reward functions.

Run: `conda run -n hiking python -m pytest tests/parkour/foothold/test_rewards.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py source/instinctlab/instinctlab/tasks/parkour/mdp/foothold_observations.py tests/parkour/foothold/test_rewards.py
git commit -m "feat(foothold): add phase gated rewards"
```

### Task 6: Make curriculum and AMP scheduling resumable

**Files:**
- Create: `source/instinctlab/instinctlab/tasks/parkour/foothold/curriculum.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/mdp/curriculums.py`
- Test: `tests/parkour/foothold/test_curriculum.py`

**Interfaces:**
- Produces: `FootholdCurriculumState.state_dict()`, `load_state_dict()`, `update(outcomes)`, and `amp_reward_coef(iteration)`.

- [ ] **Step 1: Test promotion, serialization, and AMP breakpoints**

```python
def test_curriculum_round_trip_and_promotion():
    state = FootholdCurriculumState.create(4, device="cpu")
    state.update(success=torch.ones(4, dtype=torch.bool), touchdown_error=torch.zeros(4))
    restored = FootholdCurriculumState.create(4, device="cpu")
    restored.load_state_dict(state.state_dict())
    torch.testing.assert_close(restored.level, state.level)
    assert restored.global_frontier == state.global_frontier


def test_amp_schedule():
    assert amp_reward_coef(0) == 0.0
    assert amp_reward_coef(2_000) == 0.02
    assert amp_reward_coef(6_000) == 0.05
    assert amp_reward_coef(12_000) == 0.10
```

- [ ] **Step 2: Implement exact update rules**

Maintain per-environment `level`, `success_ema`, `error_ema`, and a global frontier. Promote after EMA success `>=0.90` and EMA touchdown error `<=0.05 m`; demote below success `0.60`; clamp levels to `[0,2]`; expose quotas `[1.0,0.0,0.0]`, `[0.6,0.4,0.0]`, and `[0.3,0.4,0.3]`. Implement AMP knots `[(0,0.00),(2000,0.02),(6000,0.05),(12000,0.10)]` with right-continuous step values.

Run: `conda run -n hiking python -m pytest tests/parkour/foothold/test_curriculum.py -q`

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/foothold/curriculum.py source/instinctlab/instinctlab/tasks/parkour/mdp/curriculums.py tests/parkour/foothold/test_curriculum.py
git commit -m "feat(foothold): add resumable curriculum schedule"
```

### Task 7: Extend checkpointing and optional W&B monitoring

**Files:**
- Create: `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/foothold_runner.py`
- Modify: `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/__init__.py`
- Modify: `source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py`
- Modify: `scripts/instinct_rl/cli_args.py`
- Modify: `scripts/instinct_rl/train.py`
- Test: `tests/parkour/foothold/test_runner_state.py`

**Interfaces:**
- Produces: `InstinctLabOnPolicyRunner`, `InstinctRlVecEnvWrapper.training_state_dict()`, and `InstinctRlVecEnvWrapper.load_training_state_dict(state)`.

- [ ] **Step 1: Test environment-state checkpoint protocol with fakes**

```python
def test_runner_adds_and_restores_environment_state(tmp_path):
    runner = make_fake_runner(tmp_path)
    runner.save(tmp_path / "model.pt")
    runner.env.training_value = -1
    runner.load(tmp_path / "model.pt")
    assert runner.env.training_value == 17
```

- [ ] **Step 2: Implement the runner adapter**

Add wrapper methods that delegate to `self.unwrapped.training_state_dict()` and `self.unwrapped.load_training_state_dict(state)` when present, otherwise return an empty dictionary and accept an empty dictionary. Subclass `OnPolicyRunner`. In `save`, call the parent, reopen the checkpoint with `weights_only=True`, add `environment_state=self.env.training_state_dict()`, and atomically replace the file through a sibling `.tmp` path. In `load`, read `environment_state`, call the parent, and then restore it; when loading an older checkpoint without that key, print one warning and retain level zero. Override `rollout_step` to assign `self.alg.discriminator_reward_coef = amp_reward_coef(self.current_learning_iteration)` before delegating to the parent.

Override `log` and, under `torch.inference_mode()`, evaluate the actor gate logits with the current flattened policy observation:

```python
actor_layer = self.alg.actor_critic.actor
actor_moe = actor_layer[0] if isinstance(actor_layer, torch.nn.Sequential) else actor_layer
gate_load = torch.softmax(actor_moe.gate(locs["obs"]), dim=-1).mean(dim=0)
gate_entropy = -(gate_load * gate_load.clamp_min(1e-8).log()).sum()
for expert_index, load in enumerate(gate_load):
    self.writer_mp_add_scalar(f"MoE/actor_expert_{expert_index}_load", load.item(), self.current_learning_iteration)
self.writer_mp_add_scalar("MoE/actor_gate_entropy", gate_entropy.item(), self.current_learning_iteration)
```

- [ ] **Step 3: Add explicit monitoring CLI**

Add:

```python
arg_group.add_argument("--wandb", action="store_true", help="Mirror TensorBoard metrics to W&B.")
arg_group.add_argument("--wandb_project", default="instinctlab-foothold")
arg_group.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="offline")
```

When `--wandb` is set on rank zero, call `wandb.init(project=args_cli.wandb_project, name=os.path.basename(log_dir), dir=log_dir, mode=args_cli.wandb_mode, sync_tensorboard=True, config={"task": args_cli.task})`; always call `wandb.finish()` in a `finally` block. Without the flag, do not import `wandb`.

- [ ] **Step 4: Run tests and CLI help**

Run: `conda run -n hiking python -m pytest tests/parkour/foothold/test_runner_state.py -q`

Expected: pass.

Run: `conda run -n hiking python scripts/instinct_rl/train.py --help | rg 'wandb_(project|mode)'`

Expected: both options print before simulator launch exits.

- [ ] **Step 5: Commit**

```bash
git add source/instinctlab/instinctlab/utils/wrappers/instinct_rl scripts/instinct_rl tests/parkour/foothold/test_runner_state.py
git commit -m "feat(foothold): resume curriculum and monitor training"
```

### Task 8: Register isolated train/play configurations

**Files:**
- Create: `source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_foothold_flat_cfg.py`
- Create: `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_foothold_cfg.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py`
- Test: `tests/parkour/foothold/test_config_contract.py`

**Interfaces:**
- Produces: `G1FootholdFlatEnvCfg`, `G1FootholdFlatEnvCfg_PLAY`, and `G1FootholdPPORunnerCfg`.

- [ ] **Step 1: Write registry/config contract tests**

```python
def test_foothold_policy_contract():
    cfg = G1FootholdPPORunnerCfg()
    assert cfg.policy.num_moe_experts == 4
    assert cfg.policy.class_name == "MoEActorCritic"
    assert cfg.num_steps_per_env in (32, 48)
    assert cfg.algorithm.discriminator_reward_coef == 0.0


def test_existing_and_new_ids_are_registered():
    ids = set(gym.registry)
    assert "Instinct-Parkour-G1-v0" in ids
    assert "Instinct-Parkour-Foothold-G1-v0" in ids
    assert "Instinct-Parkour-Foothold-G1-Play-v0" in ids
```

- [ ] **Step 2: Implement the new environment configuration**

Inherit `G1ParkourEnvCfg`; replace rough terrain with a plane; disable the depth-camera observation; add command name `foothold`; define actor observations as base angular velocity, projected gravity, joint position/velocity, previous action histories, and the current 44-D command only. Define critic observations as actor terms plus base linear velocity, raw/debounced contacts, sole pose/velocity, target errors, support/edge values, and oracle agreement. Configure the phase-gated reward groups from Task 5. The play class uses one environment and enables command debug visualization.

- [ ] **Step 3: Implement the agent config**

Use:

```python
@configclass
class FootholdMoEPolicyCfg(InstinctRlMoEActorCriticCfg):
    init_noise_std = 1.0
    num_moe_experts = 4
    moe_gate_hidden_dims = [128]
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"


@configclass
class G1FootholdPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 30_000
    save_interval = 500
    experiment_name = "g1_foothold_flat"
    policy = FootholdMoEPolicyCfg()
    algorithm = FootholdAmpAlgoCfg()
```

Start `FootholdAmpAlgoCfg.discriminator_reward_coef = 0.0`; the runner applies the schedule. Register train/play IDs with the new environment and same agent entry point.

- [ ] **Step 4: Run configuration tests and list environments**

Run: `conda run -n hiking python -m pytest tests/parkour/foothold/test_config_contract.py -q`

Expected: pass.

Run: `conda run -n hiking python scripts/list_envs.py | rg 'Instinct-Parkour-(Foothold-)?G1'`

Expected: old IDs and both new foothold IDs appear.

- [ ] **Step 5: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/config/g1 tests/parkour/foothold/test_config_contract.py
git commit -m "feat(foothold): register flat tracking task"
```

### Task 9: Add play visualization, metrics, and the validation ladder

**Files:**
- Modify: `source/instinctlab/instinctlab/tasks/parkour/scripts/play.py`
- Create: `source/instinctlab/instinctlab/tasks/parkour/foothold/metrics.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/README.md`
- Test: `tests/parkour/foothold/test_metric_reduction.py`

**Interfaces:**
- Produces: median/P95 metric reduction and documented smoke commands.

- [ ] **Step 1: Test exact percentile reduction**

```python
import pytest
import torch

from instinctlab.tasks.parkour.foothold.metrics import touchdown_statistics


def test_touchdown_percentiles():
    values = torch.arange(1, 101, dtype=torch.float32)
    result = touchdown_statistics(values)
    assert result["median"] == pytest.approx(50.5)
    assert result["p95"] == pytest.approx(95.05)
```

- [ ] **Step 2: Implement metrics and visualization controls**

Implement:

```python
def touchdown_statistics(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {"median": float("nan"), "p95": float("nan")}
    quantiles = torch.quantile(values, torch.tensor([0.5, 0.95], device=values.device))
    return {"median": quantiles[0].item(), "p95": quantiles[1].item()}
```

Compute XY/Z/yaw/timing median and P95 only on new touchdown latches; log early contact, overdue, plan-invalid, recovery, fall, slip, impact, support margin, curriculum distribution, AMP coefficient, PPO/Wasabi losses, MoE expert load/entropy, and FPS/GPU metrics. Add `--foothold_debug_vis` to play; stable colors are current target green, next target cyan, sole polygons white, reference trajectory yellow, early contact orange, overdue magenta, invalid/recovery red. Show gait state, phase, target error, and validity as viewer text.

- [ ] **Step 3: Document and run the validation ladder**

Document these commands in order:

```bash
conda run -n hiking python -m pytest tests/parkour/foothold -q
conda run -n hiking python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task Instinct-Parkour-Foothold-G1-Play-v0 --num_envs 1 --foothold_debug_vis
conda run -n hiking python scripts/instinct_rl/train.py --task Instinct-Parkour-Foothold-G1-v0 --num_envs 64 --max_iterations 10 --headless --wandb --wandb_mode offline
conda run -n hiking python scripts/instinct_rl/train.py --task Instinct-Parkour-Foothold-G1-v0 --num_envs 256 --max_iterations 500 --headless --wandb --wandb_mode offline
```

The 256-environment gate requires no NaN/Inf, successful checkpoint resume, nonzero reward gradients, stable memory use, and decreasing touchdown error. Run short paired 256-environment comparisons with AMP held at `0.00` and `0.10` before accepting the scheduled maximum. The later 4096-environment acceptance gate is XY median `<=0.025 m`, XY P95 `<=0.050 m`, Z P95 `<=0.020 m`, yaw P95 `<=8 deg`, timing P95 `<=0.080 s`, early contact `<1%`, 20-step success `>=90%`, and falls `<2%`.

- [ ] **Step 4: Run unit and 64-environment gates**

Run: first and third commands above.

Expected: all unit tests pass; 10 iterations finish, create a checkpoint, and contain no NaN/Inf. Resume that checkpoint for two more iterations and confirm curriculum level/frontier and iteration continue rather than reset.

- [ ] **Step 5: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/foothold/metrics.py source/instinctlab/instinctlab/tasks/parkour/scripts/play.py source/instinctlab/instinctlab/tasks/parkour/README.md tests/parkour/foothold/test_metric_reduction.py
git commit -m "feat(foothold): visualize and validate tracking"
```

---

## Final Review Gate

- [ ] Run `conda run -n hiking python -m pytest tests/parkour/foothold -q`; expect all tests to pass.
- [ ] Run `pre-commit run --all-files`; expect every hook to pass.
- [ ] Run `git diff main...HEAD --check`; expect no output.
- [ ] Confirm the old G1 parkour IDs still instantiate with unchanged configuration classes.
- [ ] Confirm the actor observation contains 44 current-command values, no command history, no raw depth, and no contact bits.
- [ ] Confirm a checkpoint round-trip restores iteration, policy, normalizers, curriculum level/EMA/frontier, gait buffers, and AMP schedule position.
- [ ] Do not start the 4096-environment run until the 256-environment gate is reviewed and accepted.
