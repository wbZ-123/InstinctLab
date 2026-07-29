# Learned Explicit Foothold Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a learned explicit foothold planner path where the policy supplies
a normalized 2D foothold action, the planner decodes it through the existing
reachability ellipse into final support-frame `x_f, y_f`, and the existing
joint action policy tracks the resulting analytic swing trajectory.

**Architecture:** Use IsaacLab's action manager to add a separate normalized 2D
high-level foothold action term beside the existing joint-position action term.
During confirmed double-support HOLD, the foothold planner decodes the current
action with `FlatProviderConfig.outer_radius_x/y`, converts it through the
strict local/world height-query contract, and caches the latest valid prepared
target. At the new-SWING transition it locks that target and ignores later
high-level outputs until touchdown.

**Tech Stack:** Python, PyTorch tensors, IsaacLab manager-based env/action manager, existing `FootholdPlanner` sensor, existing `instinctlab_foothold` pure planning utilities, pytest.

## Global Constraints

- The action term stores only normalized `u_x, u_y ∈ [-1, 1]`.
- The planner maps normalized output into final explicit `x_f, y_f` using the
  existing `FlatProviderConfig.outer_radius_x/y` reachability ellipse.
- Do not define an independent learned-action range in meters.
- Terrain height query always uses world-frame `x_w, y_w`; never local `x_f, y_f` directly.
- `z_f = z_w - support_foot_z_w`.
- Learned foothold action is consumed only while both feet are confirmed in
  HOLD, with a one-shot new-SWING fallback when no prepared target exists.
- The accepted target is locked and learned foothold output is ignored during active SWING.
- Network never predicts terrain height.
- Danger-cylinder information is training reward/diagnostic only, not policy observation.
- Existing explicit planner remains available as nominal-prior, debug, and fallback path.
- Learned planner path must be disabled by default for old checkpoints and existing play commands.
- No per-step dense candidate enumeration in the learned planner path.
- The current actor depth-image observation is the learned planner's terrain
  input; do not add danger-cylinder or mesh internals to actor observations.
- Replace duplicate local/world final-target composition in `FootholdPlanner`
  with the shared helpers from `instinctlab_foothold.frame_transform`.

---

## File Structure

- Create `source/instinctlab/instinctlab_foothold/frame_transform.py`: pure local/world foothold transform helpers with height-query contract.
- Modify `source/instinctlab/instinctlab_foothold/__init__.py`: export new transform helpers.
- Create `tests/parkour/foothold/test_frame_transform.py`: coordinate-system regression tests.
- Create `source/instinctlab/instinctlab/envs/mdp/actions/foothold_actions.py`: action term storing clipped normalized foothold output.
- Modify `source/instinctlab/instinctlab/envs/mdp/actions/action_cfg.py`: config for learned foothold action term.
- Modify `source/instinctlab/instinctlab/envs/mdp/actions/__init__.py`: export new action term/config.
- Create `tests/parkour/foothold/test_learned_foothold_action.py`: action scaling/bounds tests.
- Modify `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py`: learned-planner toggles and bounds.
- Modify `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`: learned-planner diagnostics.
- Modify `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`: consume learned foothold action at planning events and lock target.
- Modify `tests/parkour/foothold/test_foothold_planner_data.py`: data field tests.
- Create `tests/parkour/foothold/test_learned_foothold_planner.py`: event timing and coordinate tests.
- Modify `source/instinctlab/instinctlab/envs/mdp/observations/foothold.py`: expose nominal prior and learned planner state, not danger-cylinder info.
- Modify `tests/parkour/foothold/test_observation_foothold.py`: observation shape/content tests.
- Modify `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`: planning reward terms for learned foothold output.
- Modify `tests/parkour/foothold/test_reward_foothold.py`: reward tests.
- Modify `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`: opt-in learned planner action/reward config.
- Modify `scripts/instinct_rl/train.py`: CLI flag applies the learned planner config method.
- Modify `scripts/foothold_train.sh`: env-var switch for learned planner training.
- Modify `scripts/instinct_rl/play.py`, `scripts/instinct_rl/play_debug.py`: debug learned local/world target fields.
- Modify `tests/parkour/foothold/test_play_debug.py`: debug payload tests.

---

### Task 1: Add frame transform utilities with hard coordinate tests

**Files:**
- Create: `source/instinctlab/instinctlab_foothold/frame_transform.py`
- Modify: `source/instinctlab/instinctlab_foothold/__init__.py`
- Test: `tests/parkour/foothold/test_frame_transform.py`

**Interfaces:**
- Produces: `planner_frame_to_world_xy(origin_w: torch.Tensor, target_xy_f: torch.Tensor, yaw_w: torch.Tensor) -> torch.Tensor`
- Produces: `apply_world_height_to_planner_target(origin_w: torch.Tensor, target_xy_f: torch.Tensor, yaw_w: torch.Tensor, terrain_height_query_w: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
- Later tasks consume these helpers inside `FootholdPlanner`.

- [x] **Step 1: Write the failing coordinate tests**

Create `tests/parkour/foothold/test_frame_transform.py`:

```python
import math

import torch

from instinctlab_foothold import (
    apply_world_height_to_planner_target,
    planner_frame_to_world_xy,
)


def test_planner_frame_to_world_xy_uses_origin_and_yaw():
    origin_w = torch.tensor([[1.0, 2.0, 0.3]])
    target_xy_f = torch.tensor([[0.2, 0.0]])
    yaw_w = torch.tensor([math.pi / 2.0])

    xy_w = planner_frame_to_world_xy(origin_w, target_xy_f, yaw_w)

    torch.testing.assert_close(xy_w, torch.tensor([[1.0, 2.2]]), atol=1.0e-6, rtol=0.0)


def test_apply_world_height_queries_world_xy_and_returns_local_z():
    queried = []

    def terrain_query(points_xy_w: torch.Tensor):
        queried.append(points_xy_w.clone())
        return torch.tensor([0.75]), torch.tensor([True])

    origin_w = torch.tensor([[1.0, 2.0, 0.25]])
    target_xy_f = torch.tensor([[0.2, 0.0]])
    yaw_w = torch.tensor([math.pi / 2.0])

    target_f, target_w, valid = apply_world_height_to_planner_target(
        origin_w=origin_w,
        target_xy_f=target_xy_f,
        yaw_w=yaw_w,
        terrain_height_query_w=terrain_query,
    )

    torch.testing.assert_close(queried[0], torch.tensor([[1.0, 2.2]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(target_w, torch.tensor([[1.0, 2.2, 0.75]]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(target_f, torch.tensor([[0.2, 0.0, 0.5]]), atol=1.0e-6, rtol=0.0)
    assert valid.tolist() == [True]
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/zhangweibo/InstinctLab-foothold
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_frame_transform.py
```

Expected: FAIL with import error for `planner_frame_to_world_xy`.

- [x] **Step 3: Implement the minimal transform helpers**

Create `source/instinctlab/instinctlab_foothold/frame_transform.py`:

```python
from __future__ import annotations

from collections.abc import Callable

import torch


TerrainHeightQuery = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def planner_frame_to_world_xy(
    origin_w: torch.Tensor,
    target_xy_f: torch.Tensor,
    yaw_w: torch.Tensor,
) -> torch.Tensor:
    cos_yaw = torch.cos(yaw_w)
    sin_yaw = torch.sin(yaw_w)
    x_w = origin_w[:, 0] + cos_yaw * target_xy_f[:, 0] - sin_yaw * target_xy_f[:, 1]
    y_w = origin_w[:, 1] + sin_yaw * target_xy_f[:, 0] + cos_yaw * target_xy_f[:, 1]
    return torch.stack([x_w, y_w], dim=-1)


def apply_world_height_to_planner_target(
    *,
    origin_w: torch.Tensor,
    target_xy_f: torch.Tensor,
    yaw_w: torch.Tensor,
    terrain_height_query_w: TerrainHeightQuery,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_xy_w = planner_frame_to_world_xy(origin_w, target_xy_f, yaw_w)
    terrain_z_w, valid = terrain_height_query_w(target_xy_w)
    terrain_z_w = terrain_z_w.to(device=target_xy_f.device, dtype=target_xy_f.dtype)
    valid = valid.to(device=target_xy_f.device, dtype=torch.bool) & torch.isfinite(terrain_z_w)
    target_w = torch.cat([target_xy_w, terrain_z_w[:, None]], dim=-1)
    target_f = torch.cat([target_xy_f, (terrain_z_w - origin_w[:, 2])[:, None]], dim=-1)
    return target_f, target_w, valid
```

Update `source/instinctlab/instinctlab_foothold/__init__.py`:

```python
from .frame_transform import apply_world_height_to_planner_target, planner_frame_to_world_xy
```

and add both names to `__all__`.

- [x] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add source/instinctlab/instinctlab_foothold/frame_transform.py source/instinctlab/instinctlab_foothold/__init__.py tests/parkour/foothold/test_frame_transform.py
git commit -m "feat: add foothold frame height transform"
```

---

### Task 2: Store a normalized learned foothold action

**Files:**
- Create: `source/instinctlab/instinctlab/envs/mdp/actions/foothold_actions.py`
- Modify: `source/instinctlab/instinctlab/envs/mdp/actions/action_cfg.py`
- Modify: `source/instinctlab/instinctlab/envs/mdp/actions/__init__.py`
- Test: `tests/parkour/foothold/test_learned_foothold_action.py`

**Interfaces:**
- Produces config: `LearnedFootholdActionCfg`
- Produces action term: `LearnedFootholdAction`
- Stores on env:
  - `env.learned_foothold_action_raw: torch.Tensor` with shape `(num_envs, 2)`
  - `env.learned_foothold_action_normalized: torch.Tensor` with shape `(num_envs, 2)`
- Later planner task consumes `env.learned_foothold_action_normalized` and
  performs the only meter-valued mapping.

- [ ] **Step 1: Write failing tests**

Change `tests/parkour/foothold/test_learned_foothold_action.py` so the action
term only clamps policy output. It must not contain or apply meter-valued
foothold bounds.

```python
import torch

from instinctlab.envs.mdp.actions.foothold_actions import normalize_foothold_action


def test_normalize_foothold_action_clamps_without_meter_scaling():
    raw = torch.tensor([[-2.0, 0.25], [0.5, 2.0]])
    normalized = normalize_foothold_action(raw)
    torch.testing.assert_close(
        normalized,
        torch.tensor([[-1.0, 0.25], [0.5, 1.0]]),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_learned_foothold_action.py
```

Expected: FAIL because `normalize_foothold_action` does not exist.

- [ ] **Step 3: Implement normalized action storage**

Change `foothold_actions.py` to:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .action_cfg import LearnedFootholdActionCfg


def normalize_foothold_action(raw_action: torch.Tensor) -> torch.Tensor:
    return torch.clamp(raw_action, -1.0, 1.0)


class LearnedFootholdAction(ActionTerm):
    cfg: "LearnedFootholdActionCfg"

    def __init__(self, cfg: "LearnedFootholdActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 2, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        env.learned_foothold_action_raw = self._raw_actions
        env.learned_foothold_action_normalized = self._processed_actions

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = normalize_foothold_action(actions)

    def apply_actions(self) -> None:
        return None
```

Add `LearnedFootholdActionCfg` to `action_cfg.py`:

```python
from isaaclab.managers.action_manager import ActionTermCfg

from . import foothold_actions


@configclass
class LearnedFootholdActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = foothold_actions.LearnedFootholdAction
    asset_name: str = "robot"
```

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add source/instinctlab/instinctlab/envs/mdp/actions tests/parkour/foothold/test_learned_foothold_action.py
git commit -m "fix: normalize learned foothold action"
```

---

### Task 3: Decode, prepare, and lock learned footholds

**Files:**
- Create: `source/instinctlab/instinctlab_foothold/learned_target.py`
- Modify: `source/instinctlab/instinctlab_foothold/__init__.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_cfg.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py`
- Modify: `source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py`
- Test: `tests/parkour/foothold/test_foothold_planner_data.py`
- Test: `tests/parkour/foothold/test_learned_foothold_planner.py`

**Interfaces:**
- Consumes: `env.learned_foothold_action_normalized` from Task 2.
- Consumes: `apply_world_height_to_planner_target` from Task 1.
- Produces data fields:
  - `learned_foothold_enabled`
  - `learned_foothold_action_normalized`
  - `learned_foothold_decoded_f`
  - `learned_foothold_prepared_f`
  - `learned_foothold_prepared_w`
  - `learned_foothold_prepared_valid`
  - `learned_foothold_locked`
  - `learned_foothold_target_f`
  - `learned_foothold_target_w`
  - `learned_foothold_used`
  - `learned_foothold_height_valid`
  - `learned_foothold_safety_valid`

- [ ] **Step 1: Write failing planner data test**

Extend `tests/parkour/foothold/test_foothold_planner_data.py` to assert new fields exist and initialize to `None` by dataclass default.

```python
from instinctlab.sensors.foothold_planner.foothold_planner_data import FootholdPlannerData


def test_foothold_planner_data_has_learned_foothold_fields():
    data = FootholdPlannerData()
    assert data.learned_foothold_action_normalized is None
    assert data.learned_foothold_decoded_f is None
    assert data.learned_foothold_prepared_f is None
    assert data.learned_foothold_prepared_w is None
    assert data.learned_foothold_prepared_valid is None
    assert data.learned_foothold_locked is None
    assert data.learned_foothold_target_f is None
    assert data.learned_foothold_target_w is None
    assert data.learned_foothold_used is None
    assert data.learned_foothold_height_valid is None
    assert data.learned_foothold_safety_valid is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_foothold_planner_data.py
```

Expected: FAIL because fields do not exist.

- [ ] **Step 3: Add cfg/data fields**

Add to `FootholdPlannerCfg`:

```python
enable_learned_foothold: bool = False
learned_foothold_step_height_limit_m: float = 0.25
```

Do not add learned-planner meter-valued x/y bounds. Decode with
`self._flat_provider_cfg.outer_radius_x/y`.

Add dataclass fields listed above to `FootholdPlannerData`.

Initialize runtime tensors in planner buffer initialization with shapes `(num_envs, 2)`, `(num_envs, 3)`, or `(num_envs,)`.

- [ ] **Step 4: Write event-locking tests**

Create `tests/parkour/foothold/test_learned_foothold_planner.py` with pure
helper-level tests. Put the pure helpers in
`instinctlab_foothold/learned_target.py` so tests do not require Isaac Sim:

```python
import torch

from instinctlab_foothold import (
    decode_normalized_foothold,
    learned_foothold_event_masks,
)


def test_decode_normalized_foothold_uses_shared_reachability_ellipse():
    normalized = torch.tensor([[1.0, 1.0]])
    target = decode_normalized_foothold(
        normalized,
        radius_x=0.42,
        radius_y=0.25,
    )
    usage = (
        (target[:, 0] / 0.42).square()
        + (target[:, 1] / 0.25).square()
    )
    torch.testing.assert_close(usage, torch.ones_like(usage))


def test_event_masks_prepare_in_confirmed_hold_and_lock_only_on_new_swing():
    prepare, lock = learned_foothold_event_masks(
        hold=torch.tensor([True, True, False, False]),
        both_contacts_confirmed=torch.tensor([True, False, True, True]),
        new_swing=torch.tensor([False, False, True, False]),
        enable=True,
    )
    assert prepare.tolist() == [True, False, False, False]
    assert lock.tolist() == [False, False, True, False]
```

- [ ] **Step 5: Implement HOLD preparation and SWING locking**

During confirmed-contact HOLD, if `cfg.enable_learned_foothold` and
`env.learned_foothold_action_normalized` exists:

```python
learned_xy_f = decode_normalized_foothold(
    self._env.learned_foothold_action_normalized[prepare_env_ids],
    radius_x=self._flat_provider_cfg.outer_radius_x,
    radius_y=self._flat_provider_cfg.outer_radius_y,
)
prepared_f, prepared_w, terrain_valid = apply_world_height_to_planner_target(
    origin_w=prepare_stance_pos_w,
    target_xy_f=learned_xy_f,
    yaw_w=base_yaw_w[prepare],
    terrain_height_query_w=self._query_target_terrain_height_at_xy_w,
)
```

Then apply hard gates:

```python
height_valid = terrain_valid & (
    torch.abs(prepared_f[:, 2])
    <= self.cfg.learned_foothold_step_height_limit_m
)
```

Store only valid prepared targets. At `new_swing`, use the cached prepared
target. If none exists, evaluate the current output once through the same
pipeline. Lock the accepted target and trajectory. Do not read learned actions
again during active SWING. Clear prepared/locked state after touchdown, reset,
recovery transition, or plan invalidation.

Replace `_compose_world_from_frame` and `_apply_terrain_height_to_target` uses
for final target construction with `apply_world_height_to_planner_target`.
Keep unrelated vector-rotation utilities only where still needed.

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_foothold_planner_data.py tests/parkour/foothold/test_learned_foothold_planner.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add source/instinctlab/instinctlab_foothold/learned_target.py source/instinctlab/instinctlab_foothold/__init__.py source/instinctlab/instinctlab/sensors/foothold_planner tests/parkour/foothold/test_foothold_planner_data.py tests/parkour/foothold/test_learned_foothold_planner.py
git commit -m "feat: prepare and lock learned foothold targets"
```

---

### Task 4: Add learned planner observations without simulator-only inputs

**Files:**
- Modify: `source/instinctlab/instinctlab/envs/mdp/observations/foothold.py`
- Test: `tests/parkour/foothold/test_observation_foothold.py`

**Interfaces:**
- Produces observation terms containing nominal prior and learned planner state.
- Must not include danger-cylinder center/radius or candidate safety masks.

- [ ] **Step 1: Write failing tests**

Add a test that the foothold observation includes `raw_unclipped_foothold_f`, `target_foothold_f`, gait mode/phase, and not safe-target candidate counts when `include_debug_diagnostics=False`.

```python
def test_foothold_observation_exposes_nominal_prior_without_danger_diagnostics(fake_env):
    obs = foothold_planner_observation(fake_env, sensor_name="foothold_planner", include_debug_diagnostics=False)
    assert obs.shape[-1] >= 1
    assert not torch.isnan(obs).any()
```

Use the existing fake-env patterns in `test_observation_foothold.py`; do not introduce Isaac Sim dependency.

- [ ] **Step 2: Run test to verify it fails if the new parameter is missing**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_observation_foothold.py
```

- [ ] **Step 3: Implement observation update**

Add optional parameter:

```python
include_debug_diagnostics: bool = False
```

Keep policy-safe observation values to:

```text
gait mode / phase
swing side
nominal prior raw_unclipped_foothold_f
current locked target_foothold_f
target error to swing foot
learned_foothold_used flag
learned_foothold_height_valid flag
```

The actor already receives `depth_image` in `ObservationsCfg.PolicyCfg`.
Do not add a second terrain patch and do not expose danger-cylinder data.

Do not include candidate counts, danger-cylinder internals, or penetration statistics in policy observation.

- [ ] **Step 4: Run tests**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add source/instinctlab/instinctlab/envs/mdp/observations/foothold.py tests/parkour/foothold/test_observation_foothold.py
git commit -m "feat: expose learned foothold planner observations"
```

---

### Task 5: Add planning rewards for learned explicit footholds

**Files:**
- Modify: `source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py`
- Test: `tests/parkour/foothold/test_reward_foothold.py`

**Interfaces:**
- Produces reward terms:
  - `learned_foothold_nominal_deviation_l2`
  - `learned_foothold_height_valid_indicator`
  - `learned_foothold_safety_valid_indicator`
  - `learned_foothold_planning_event_indicator`

- [ ] **Step 1: Write failing reward tests**

Add tests with fake planner data:

```python
def test_learned_foothold_nominal_deviation_penalizes_large_departure(fake_env):
    reward = learned_foothold_nominal_deviation_l2(fake_env, sensor_name="foothold_planner")
    assert reward.shape == (fake_env.num_envs,)
    assert torch.all(reward >= 0.0)
```

And:

```python
def test_learned_foothold_height_valid_indicator_reports_hard_gate(fake_env):
    value = learned_foothold_height_valid_indicator(fake_env, sensor_name="foothold_planner")
    assert value.tolist() == [1.0, 0.0]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_reward_foothold.py
```

- [ ] **Step 3: Implement reward terms**

Use existing `_foothold_planner_data`. Do not use danger-cylinder internals as observation; reward may use planner diagnostics.

```python
def learned_foothold_nominal_deviation_l2(env, sensor_name: str = "foothold_planner"):
    data = _foothold_planner_data(env, sensor_name)
    return torch.sum((data.learned_foothold_target_f[:, :2] - data.raw_unclipped_foothold_f[:, :2]).square(), dim=-1)
```

Indicator rewards should return float tensors.

- [ ] **Step 4: Run tests**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add source/instinctlab/instinctlab/envs/mdp/rewards/foothold.py tests/parkour/foothold/test_reward_foothold.py
git commit -m "feat: add learned foothold planning rewards"
```

---

### Task 6: Add opt-in parkour config for learned explicit planner

**Files:**
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Modify: `scripts/instinct_rl/train.py`
- Modify: `scripts/foothold_train.sh`
- Test: `tests/parkour/foothold/test_learned_foothold_config.py`

**Interfaces:**
- Produces config class or toggle that adds `learned_foothold` action term only when explicitly enabled.
- Produces train CLI flag `--enable_learned_explicit_foothold_planner`.
- Produces wrapper switch `ENABLE_LEARNED_FOOTHOLD_PLANNER=1` for `scripts/foothold_train.sh`.
- Existing default task remains old-checkpoint compatible.

- [ ] **Step 1: Write failing config tests**

Create `tests/parkour/foothold/test_learned_foothold_config.py`:

```python
from instinctlab.tasks.parkour.config.parkour_env_cfg import G1ParkourEnvCfg


def test_default_parkour_config_does_not_enable_learned_foothold_action():
    cfg = G1ParkourEnvCfg()
    assert not hasattr(cfg.actions, "learned_foothold") or cfg.actions.learned_foothold is None


def test_learned_foothold_config_can_be_enabled_explicitly():
    cfg = G1ParkourEnvCfg()
    cfg.enable_learned_explicit_foothold_planner()
    assert cfg.actions.learned_foothold is not None
    assert cfg.scene.foothold_planner.enable_learned_foothold is True
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_learned_foothold_config.py
```

- [ ] **Step 3: Implement opt-in config method**

In `parkour_env_cfg.py`, add:

```python
def enable_learned_explicit_foothold_planner(self):
    self.actions.learned_foothold = mdp.LearnedFootholdActionCfg(
        asset_name="robot",
    )
    self.scene.foothold_planner.enable_learned_foothold = True
```

If `ActionsCfg` needs the field declared, set:

```python
learned_foothold = None
```

- [ ] **Step 4: Add train CLI flag**

In `scripts/instinct_rl/train.py`, add an argparse flag near the other train-specific arguments:

```python
parser.add_argument(
    "--enable_learned_explicit_foothold_planner",
    action="store_true",
    default=False,
    help="Enable the learned explicit foothold planner action/reward path.",
)
```

Inside `main(...)`, after `env_cfg.sim.device = ...` and before distributed setup, call the opt-in method:

```python
    if args_cli.enable_learned_explicit_foothold_planner:
        if not hasattr(env_cfg, "enable_learned_explicit_foothold_planner"):
            raise AttributeError(
                f"Task config {type(env_cfg).__name__} does not support learned explicit foothold planner."
            )
        env_cfg.enable_learned_explicit_foothold_planner()
```

- [ ] **Step 5: Add train-wrapper switch**

In `scripts/foothold_train.sh`, add an optional command-array extension after the `cmd=(...)` block:

```bash
if [[ "${ENABLE_LEARNED_FOOTHOLD_PLANNER:-0}" == "1" ]]; then
    cmd+=("--enable_learned_explicit_foothold_planner")
fi
```

Also print the switch near the other run metadata:

```bash
echo "[foothold_train] enable_learned_foothold_planner: ${ENABLE_LEARNED_FOOTHOLD_PLANNER:-0}"
```

The final implementation must make this dry-run command show `--enable_learned_explicit_foothold_planner` explicitly:

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 DRY_RUN=1 ./scripts/foothold_train.sh
```

- [ ] **Step 6: Run tests**

Run the same pytest command plus action tests. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py scripts/instinct_rl/train.py scripts/foothold_train.sh tests/parkour/foothold/test_learned_foothold_config.py
git commit -m "feat: add opt-in learned foothold planner config"
```

---

### Task 7: Add play/debug visibility for learned local/world targets

**Files:**
- Modify: `scripts/instinct_rl/play_debug.py`
- Modify: `scripts/instinct_rl/play.py`
- Test: `tests/parkour/foothold/test_play_debug.py`

**Interfaces:**
- Debug payload includes:
  - learned action local `x_f, y_f`
  - learned target local `x_f, y_f, z_f`
  - learned target world `x_w, y_w, z_w`
  - height valid flag
  - safety valid flag

- [ ] **Step 1: Write failing debug payload test**

Extend `test_play_debug.py` to build fake data with learned fields and assert local/world fields appear in formatted debug output.

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold/test_play_debug.py
```

- [ ] **Step 3: Implement debug fields**

Add fields to the existing foothold debug payload builder. Use clear names:

```text
learned_action_f
learned_target_f
learned_target_w
learned_height_valid
learned_safety_valid
```

- [ ] **Step 4: Run tests**

Run same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/instinct_rl/play.py scripts/instinct_rl/play_debug.py tests/parkour/foothold/test_play_debug.py
git commit -m "feat: show learned foothold planner debug fields"
```

---

### Task 8: Smoke test and benchmark before long training

**Files:**
- No required source files.
- Optional docs update: `docs/foothold_parameter_audit.md` if current project uses it for run notes.

**Interfaces:**
- Confirms old default config still works.
- Confirms opt-in learned config starts and changes action dimension.
- Confirms collection time impact is measured before 30000 iteration training.

- [ ] **Step 1: Run unit regression suite**

```bash
cd /home/zhangweibo/InstinctLab-foothold
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" /home/zhangweibo/miniconda3/envs/hiking/bin/python -m pytest -q tests/parkour/foothold
```

Expected: all foothold tests pass.

- [ ] **Step 2: Run default 4096-env 100-iteration baseline**

```bash
RUN_NAME=perf_default_after_learned_foothold_4096_100it \
MAX_ITERATIONS=100 \
NUM_ENVS=4096 \
./scripts/foothold_train.sh 2>&1 | tee logs/perf_default_after_learned_foothold_4096_100it.txt
```

Expected: collection time remains close to current default because learned planner is disabled.

- [ ] **Step 3: Run learned planner opt-in 4096-env 100-iteration benchmark**

Use the explicit wrapper switch implemented in Task 6:

```bash
ENABLE_LEARNED_FOOTHOLD_PLANNER=1 \
RUN_NAME=perf_learned_foothold_4096_100it \
MAX_ITERATIONS=100 \
NUM_ENVS=4096 \
./scripts/foothold_train.sh 2>&1 | tee logs/perf_learned_foothold_4096_100it.txt
```

Expected: run starts; if collection time increases sharply, stop before long training and profile action/planner event code.

- [ ] **Step 4: Play smoke test with learned planner debug enabled**

Use a small/short checkpoint run or no-resume random policy only to verify debug plumbing and marker behavior. Do not judge walking quality from random policy.

- [ ] **Step 5: Commit benchmark notes if docs are updated**

```bash
git add docs/foothold_parameter_audit.md
git commit -m "docs: record learned foothold planner smoke benchmarks"
```

---

## Self-Review

- Spec coverage: The plan covers coordinate-system conversion, event timing, action expansion, observations, rewards, opt-in config, debug, and benchmark gates.
- Placeholder scan: No intentionally unresolved commands remain. Task 6 defines `ENABLE_LEARNED_FOOTHOLD_PLANNER=1`; Task 8 uses the same switch.
- Type consistency: The plan consistently uses final explicit foothold `x_f, y_f`, world height query, and `z_f = z_w - support_foot_z_w`.
- Scope: This is a large feature but each task has an independently testable deliverable. The plan intentionally leaves network architecture internals to the existing PPO actor output path by using action-manager expansion first.
