# Foothold Planner Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-overhead raw foothold-planner diagnostics to the existing Monitor Manager and TensorBoard logging path without changing rewards, observations, or policy behavior.

**Architecture:** A focused `FootholdPlannerMonitorTerm` reads the existing planner sensor once per environment step and maintains only per-environment scalar accumulators. It reports sanitized step summaries and completed-episode summaries through the existing `MonitorManager`, and the parkour `MonitorCfg` registers the term.

**Tech Stack:** Python 3.11, PyTorch, IsaacLab manager configuration, InstinctLab `MonitorTerm`/`MonitorTermCfg`, pytest.

## Global Constraints

- Do not change planner algorithms, reward weights, observations, or the RL runner.
- Keep all persistent monitor tensors on the simulation device and outside autograd.
- Never retain per-step trajectory history or synchronize GPU values to CPU inside `update()`.
- Empty denominators and non-finite planner inputs must produce finite zero-valued logs.
- The monitor must support partial environment resets.
- Use the existing `GaitState` enum instead of duplicating gait-state integers.

---

## File Structure

- Create `source/instinctlab/instinctlab/monitors/foothold.py`: foothold-specific monitor term and its accumulator lifecycle.
- Modify `source/instinctlab/instinctlab/monitors/__init__.py`: export the new monitor term.
- Modify `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`: register one monitor term.
- Create `tests/parkour/foothold/test_foothold_monitor.py`: isolated unit tests for events, statistics, non-finite inputs, and partial reset.
- Modify `tests/parkour/foothold/smoke_foothold_planner.py`: assert and print monitor registration/log keys.

### Task 1: Implement and unit-test raw episode accumulation

**Files:**
- Create: `tests/parkour/foothold/test_foothold_monitor.py`
- Create: `source/instinctlab/instinctlab/monitors/foothold.py`

**Interfaces:**
- Consumes: `env.scene.sensors[sensor_name].data: FootholdPlannerData`
- Produces: `FootholdPlannerMonitorTerm(cfg: MonitorTermCfg, env: ManagerBasedRLEnv)`, implementing `update(dt)`, `reset_idx(env_ids)`, and `get_log(is_episode=False)`.

- [ ] **Step 1: Write a module loader and failing construction test**

Create `tests/parkour/foothold/test_foothold_monitor.py` with lightweight stubs for the Isaac-dependent base class, then load the new module directly:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def _load_monitor_module():
    source_root = (
        Path(__file__).resolve().parents[3] / "source" / "instinctlab"
    )
    instinctlab_package = ModuleType("instinctlab")
    instinctlab_package.__path__ = [str(source_root / "instinctlab")]
    monitors_package = ModuleType("instinctlab.monitors")
    monitors_package.__path__ = [
        str(source_root / "instinctlab" / "monitors")
    ]
    sys.modules["instinctlab"] = instinctlab_package
    sys.modules["instinctlab.monitors"] = monitors_package

    manager_module = ModuleType("instinctlab.monitors.monitor_manager")

    class MonitorTerm:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env
            self.device = env.device

    manager_module.MonitorTerm = MonitorTerm
    sys.modules["instinctlab.monitors.monitor_manager"] = manager_module

    path = source_root / "instinctlab" / "monitors" / "foothold.py"
    spec = importlib.util.spec_from_file_location(
        "instinctlab.monitors.foothold_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_env(data):
    planner = SimpleNamespace(data=data)
    return SimpleNamespace(
        num_envs=data.gait_mode.shape[0],
        device=torch.device("cpu"),
        scene=SimpleNamespace(sensors={"foothold_planner": planner}),
    )


def _make_cfg():
    return SimpleNamespace(params={"sensor_name": "foothold_planner"})


def _make_data(num_envs=2):
    return SimpleNamespace(
        gait_mode=torch.zeros(num_envs, dtype=torch.long),
        touchdown_accepted=torch.zeros(num_envs, dtype=torch.bool),
        swing_clearance_safe=torch.ones(num_envs, dtype=torch.bool),
        swing_clearance_penetration=torch.zeros(num_envs),
        default_swing_apex_height=torch.full((num_envs,), 0.08),
        swing_apex_height=torch.full((num_envs,), 0.08),
        planner_valid=torch.ones(num_envs, dtype=torch.bool),
    )


def test_monitor_constructs_compact_per_environment_buffers():
    module = _load_monitor_module()
    data = _make_data(num_envs=3)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    assert monitor._step_count.shape == (3,)
    assert monitor._step_count.device.type == "cpu"
    assert monitor._step_count.dtype == torch.float32
```

- [ ] **Step 2: Run the construction test and verify RED**

Run:

```bash
cd ~/InstinctLab-foothold
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
  pytest -q tests/parkour/foothold/test_foothold_monitor.py::test_monitor_constructs_compact_per_environment_buffers
```

Expected: FAIL because `source/instinctlab/instinctlab/monitors/foothold.py` does not exist.

- [ ] **Step 3: Add the monitor skeleton and compact buffers**

Create `source/instinctlab/instinctlab/monitors/foothold.py`:

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from instinctlab_foothold import GaitState

from .monitor_manager import MonitorTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .monitor_cfg import MonitorTermCfg


class FootholdPlannerMonitorTerm(MonitorTerm):
    """Accumulate finite raw diagnostics from a foothold planner sensor."""

    _SUM_BUFFER_NAMES = (
        "_step_count",
        "_swing_step_count",
        "_touchdown_accepted_count",
        "_touchdown_confirm_count",
        "_early_contact_count",
        "_overdue_count",
        "_stance_lost_count",
        "_clearance_sample_count",
        "_clearance_safe_count",
        "_penetration_sum",
        "_apex_delta_sum",
        "_invalid_plan_count",
        "_nonfinite_count",
    )
    _MAX_BUFFER_NAMES = ("_penetration_max", "_apex_delta_max")

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_name = cfg.params.get("sensor_name", "foothold_planner")
        try:
            self._planner = env.scene.sensors[sensor_name]
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                f"Foothold planner monitor cannot find sensor '{sensor_name}'."
            ) from exc

        shape = (env.num_envs,)
        for name in self._SUM_BUFFER_NAMES + self._MAX_BUFFER_NAMES:
            setattr(
                self,
                name,
                torch.zeros(shape, dtype=torch.float32, device=self.device),
            )
        self._previous_touchdown_accepted = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._previous_touchdown_confirm = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._last_episode_log: dict[str, torch.Tensor] = {}
```

- [ ] **Step 4: Run the construction test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Add failing tests for state/event semantics and clearance statistics**

Append:

```python
def test_update_counts_states_events_and_clearance_without_double_counting():
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    env = _make_env(data)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), env)

    data.gait_mode[:] = torch.tensor([1, 4])
    data.touchdown_accepted[:] = torch.tensor([True, False])
    data.swing_clearance_safe[:] = torch.tensor([False, True])
    data.swing_clearance_penetration[:] = torch.tensor([0.03, 0.50])
    data.swing_apex_height[:] = torch.tensor([0.14, 0.20])
    data.planner_valid[:] = torch.tensor([True, False])
    monitor.update(dt=0.02)
    monitor.update(dt=0.02)

    torch.testing.assert_close(monitor._step_count, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(
        monitor._swing_step_count, torch.tensor([2.0, 0.0])
    )
    torch.testing.assert_close(
        monitor._touchdown_accepted_count, torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(
        monitor._early_contact_count, torch.tensor([0.0, 2.0])
    )
    torch.testing.assert_close(
        monitor._clearance_sample_count, torch.tensor([2.0, 0.0])
    )
    torch.testing.assert_close(
        monitor._penetration_sum, torch.tensor([0.06, 0.0])
    )
    torch.testing.assert_close(
        monitor._penetration_max, torch.tensor([0.03, 0.0])
    )
    torch.testing.assert_close(
        monitor._apex_delta_sum, torch.tensor([0.12, 0.0])
    )
    torch.testing.assert_close(
        monitor._invalid_plan_count, torch.tensor([0.0, 2.0])
    )


def test_touchdown_confirm_counts_only_mode_entry():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.gait_mode[:] = 3
    monitor.update(dt=0.02)
    monitor.update(dt=0.02)
    data.gait_mode[:] = 0
    monitor.update(dt=0.02)
    data.gait_mode[:] = 3
    monitor.update(dt=0.02)

    torch.testing.assert_close(
        monitor._touchdown_confirm_count, torch.tensor([2.0])
    )
```

- [ ] **Step 6: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
  pytest -q tests/parkour/foothold/test_foothold_monitor.py
```

Expected: FAIL because `update()` is not implemented.

- [ ] **Step 7: Implement finite, no-grad per-step accumulation**

Add these methods to `FootholdPlannerMonitorTerm`:

```python
    @staticmethod
    def _require(data, field: str) -> torch.Tensor:
        value = getattr(data, field, None)
        if value is None:
            raise RuntimeError(
                f"Foothold planner monitor requires data field '{field}'."
            )
        return value

    @torch.no_grad()
    def update(self, dt: float):
        del dt
        data = self._planner.data
        gait_mode = self._require(data, "gait_mode")
        touchdown_accepted = self._require(
            data, "touchdown_accepted"
        ).bool()
        clearance_safe = self._require(data, "swing_clearance_safe").bool()
        penetration_raw = self._require(
            data, "swing_clearance_penetration"
        )
        default_apex = self._require(data, "default_swing_apex_height")
        adjusted_apex = self._require(data, "swing_apex_height")
        planner_valid = self._require(data, "planner_valid").bool()

        swing = (gait_mode == GaitState.LEFT_SWING) | (
            gait_mode == GaitState.RIGHT_SWING
        )
        touchdown_confirm = gait_mode == GaitState.TOUCHDOWN_CONFIRM
        accepted_edge = touchdown_accepted & ~self._previous_touchdown_accepted
        confirm_edge = touchdown_confirm & ~self._previous_touchdown_confirm

        penetration_finite = torch.isfinite(penetration_raw)
        apex_delta_raw = adjusted_apex - default_apex
        apex_finite = torch.isfinite(apex_delta_raw)
        sample_finite = penetration_finite & apex_finite
        clearance_sample = swing & sample_finite
        penetration = torch.nan_to_num(
            penetration_raw, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        apex_delta = torch.nan_to_num(
            apex_delta_raw, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)

        self._step_count += 1.0
        self._swing_step_count += swing.float()
        self._touchdown_accepted_count += accepted_edge.float()
        self._touchdown_confirm_count += confirm_edge.float()
        self._early_contact_count += (
            gait_mode == GaitState.EARLY_CONTACT
        ).float()
        self._overdue_count += (gait_mode == GaitState.OVERDUE).float()
        self._stance_lost_count += (
            gait_mode == GaitState.STANCE_LOST
        ).float()
        self._clearance_sample_count += clearance_sample.float()
        self._clearance_safe_count += (
            clearance_sample & clearance_safe
        ).float()
        self._penetration_sum += penetration * clearance_sample.float()
        self._penetration_max = torch.maximum(
            self._penetration_max,
            penetration * clearance_sample.float(),
        )
        self._apex_delta_sum += apex_delta * clearance_sample.float()
        self._apex_delta_max = torch.maximum(
            self._apex_delta_max,
            apex_delta * clearance_sample.float(),
        )
        self._invalid_plan_count += (~planner_valid).float()
        self._nonfinite_count += (~sample_finite).float()

        self._previous_touchdown_accepted.copy_(touchdown_accepted)
        self._previous_touchdown_confirm.copy_(touchdown_confirm)
```

- [ ] **Step 8: Run the monitor tests and verify GREEN**

Run the command from Step 6.

Expected: `3 passed`.

- [ ] **Step 9: Commit event accumulation**

```bash
git add \
  source/instinctlab/instinctlab/monitors/foothold.py \
  tests/parkour/foothold/test_foothold_monitor.py
git diff --cached --check
git commit -m "feat(foothold): accumulate planner monitor metrics"
```

### Task 2: Produce finite episode logs and support partial resets

**Files:**
- Modify: `tests/parkour/foothold/test_foothold_monitor.py`
- Modify: `source/instinctlab/instinctlab/monitors/foothold.py`

**Interfaces:**
- Consumes: Task 1 accumulator buffers.
- Produces: finite dictionaries from `get_log()` with stable TensorBoard key names.

- [ ] **Step 1: Write failing tests for episode summaries and partial reset**

Append:

```python
def test_partial_reset_reports_completed_env_and_preserves_other_env():
    module = _load_monitor_module()
    data = _make_data(num_envs=2)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.gait_mode[:] = torch.tensor([1, 1])
    data.swing_clearance_safe[:] = torch.tensor([True, False])
    data.swing_clearance_penetration[:] = torch.tensor([0.0, 0.04])
    data.swing_apex_height[:] = torch.tensor([0.10, 0.16])
    monitor.update(dt=0.02)
    monitor.reset_idx(torch.tensor([0]))
    episode = monitor.get_log(is_episode=True)

    assert episode["swing_fraction"].item() == 1.0
    assert episode["clearance_safe_fraction"].item() == 1.0
    assert episode["penetration_mean"].item() == 0.0
    assert monitor._step_count[0].item() == 0.0
    assert monitor._step_count[1].item() == 1.0
    torch.testing.assert_close(
        monitor._penetration_sum[1], torch.tensor(0.04)
    )


def test_nonfinite_and_empty_samples_log_finite_zero():
    module = _load_monitor_module()
    data = _make_data(num_envs=1)
    monitor = module.FootholdPlannerMonitorTerm(_make_cfg(), _make_env(data))

    data.swing_clearance_penetration[:] = float("nan")
    data.swing_apex_height[:] = float("inf")
    monitor.update(dt=0.02)
    monitor.reset_idx(torch.tensor([0]))
    episode = monitor.get_log(is_episode=True)

    assert episode["nonfinite_fraction"].item() == 1.0
    assert episode["penetration_mean"].item() == 0.0
    assert episode["apex_delta_mean"].item() == 0.0
    assert all(torch.isfinite(value) for value in episode.values())
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
  pytest -q tests/parkour/foothold/test_foothold_monitor.py
```

Expected: FAIL because `reset_idx()` and `get_log()` do not produce summaries.

- [ ] **Step 3: Implement safe summaries, snapshot-before-clear, and step logs**

Add:

```python
    @staticmethod
    def _safe_ratio(
        numerator: torch.Tensor, denominator: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(
            denominator > 0.0,
            numerator / denominator.clamp_min(1.0),
            torch.zeros_like(numerator),
        )

    def _summarize(self, env_ids) -> dict[str, torch.Tensor]:
        step_count = self._step_count[env_ids]
        clearance_count = self._clearance_sample_count[env_ids]
        values = {
            "swing_fraction": self._safe_ratio(
                self._swing_step_count[env_ids], step_count
            ).mean(),
            "touchdown_accepted_step_rate": self._safe_ratio(
                self._touchdown_accepted_count[env_ids], step_count
            ).mean(),
            "touchdown_confirm_step_rate": self._safe_ratio(
                self._touchdown_confirm_count[env_ids], step_count
            ).mean(),
            "early_contact_fraction": self._safe_ratio(
                self._early_contact_count[env_ids], step_count
            ).mean(),
            "overdue_fraction": self._safe_ratio(
                self._overdue_count[env_ids], step_count
            ).mean(),
            "stance_lost_fraction": self._safe_ratio(
                self._stance_lost_count[env_ids], step_count
            ).mean(),
            "clearance_safe_fraction": self._safe_ratio(
                self._clearance_safe_count[env_ids], clearance_count
            ).mean(),
            "penetration_mean": self._safe_ratio(
                self._penetration_sum[env_ids], clearance_count
            ).mean(),
            "penetration_max": self._penetration_max[env_ids].max(),
            "apex_delta_mean": self._safe_ratio(
                self._apex_delta_sum[env_ids], clearance_count
            ).mean(),
            "apex_delta_max": self._apex_delta_max[env_ids].max(),
            "plan_invalid_fraction": self._safe_ratio(
                self._invalid_plan_count[env_ids], step_count
            ).mean(),
            "nonfinite_fraction": self._safe_ratio(
                self._nonfinite_count[env_ids], step_count
            ).mean(),
        }
        return {
            key: torch.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
            for key, value in values.items()
        }

    @torch.no_grad()
    def reset_idx(self, env_ids: Sequence[int] | slice):
        self._last_episode_log = self._summarize(env_ids)
        for name in self._SUM_BUFFER_NAMES + self._MAX_BUFFER_NAMES:
            getattr(self, name)[env_ids] = 0.0
        self._previous_touchdown_accepted[env_ids] = False
        self._previous_touchdown_confirm[env_ids] = False

    def get_log(
        self, is_episode: bool = False
    ) -> dict[str, torch.Tensor]:
        if is_episode:
            return self._last_episode_log
        return self._summarize(slice(None))
```

Before calling `.max()` in `_summarize`, normalize tensor-form `env_ids` and
return scalar zeros if it selects no environments. Use this exact guard:

```python
        selected_step_count = self._step_count[env_ids]
        if selected_step_count.numel() == 0:
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            return {
                key: zero
                for key in (
                    "swing_fraction",
                    "touchdown_accepted_step_rate",
                    "touchdown_confirm_step_rate",
                    "early_contact_fraction",
                    "overdue_fraction",
                    "stance_lost_fraction",
                    "clearance_safe_fraction",
                    "penetration_mean",
                    "penetration_max",
                    "apex_delta_mean",
                    "apex_delta_max",
                    "plan_invalid_fraction",
                    "nonfinite_fraction",
                )
            }
```

- [ ] **Step 4: Run all monitor tests and verify GREEN**

Run the command from Step 2.

Expected: `5 passed`.

- [ ] **Step 5: Run the existing foothold unit suite**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
  pytest -q tests/parkour/foothold
```

Expected: all tests pass with no regression.

- [ ] **Step 6: Commit episode logging**

```bash
git add \
  source/instinctlab/instinctlab/monitors/foothold.py \
  tests/parkour/foothold/test_foothold_monitor.py
git diff --cached --check
git commit -m "feat(foothold): report finite planner episode metrics"
```

### Task 3: Register the monitor in the parkour environment

**Files:**
- Modify: `source/instinctlab/instinctlab/monitors/__init__.py`
- Modify: `source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- Modify: `tests/parkour/foothold/smoke_foothold_planner.py`

**Interfaces:**
- Consumes: `FootholdPlannerMonitorTerm` from Tasks 1–2.
- Produces: active Monitor Manager term named `foothold_planner` and TensorBoard tags prefixed with `Episode_Monitor/foothold_planner_`.

- [ ] **Step 1: Export the new monitor class**

Modify `source/instinctlab/instinctlab/monitors/__init__.py`:

```python
from .foothold import *
from .monitor_cfg import *
from .monitor_manager import *
from .monitors import *
```

- [ ] **Step 2: Import monitor configuration types in the parkour config**

Add alongside the other InstinctLab imports in
`source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`:

```python
from instinctlab.monitors import FootholdPlannerMonitorTerm, MonitorTermCfg
```

- [ ] **Step 3: Register the term in `MonitorCfg`**

Replace the empty class:

```python
@configclass
class MonitorCfg:
    foothold_planner = MonitorTermCfg(
        func=FootholdPlannerMonitorTerm,
        params={"sensor_name": "foothold_planner"},
    )
```

- [ ] **Step 4: Add smoke assertions immediately after environment reset**

In `tests/parkour/foothold/smoke_foothold_planner.py`, after obtaining
`base_env = env.unwrapped`, add:

```python
monitor = base_env.monitor_manager.active_terms["foothold_planner"]
print(
    "[SMOKE] monitor terms:",
    sorted(base_env.monitor_manager.active_terms.keys()),
    flush=True,
)
monitor_log = monitor.get_log(is_episode=False)
print("[SMOKE] foothold monitor:", monitor_log, flush=True)
assert "clearance_safe_fraction" in monitor_log
assert "plan_invalid_fraction" in monitor_log
assert all(torch.isfinite(value) for value in monitor_log.values())
```

- [ ] **Step 5: Run static/unit verification**

Run:

```bash
PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
  pytest -q tests/parkour/foothold
python -m compileall -q \
  source/instinctlab/instinctlab/monitors/foothold.py \
  source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py \
  tests/parkour/foothold/smoke_foothold_planner.py
git diff --check
```

Expected: all pytest tests pass; compilation and diff checks produce no output.

- [ ] **Step 6: Run the real Isaac smoke test**

Run:

```bash
../IsaacLab/isaaclab.sh -p \
  tests/parkour/foothold/smoke_foothold_planner.py \
  --headless \
  --task Instinct-Parkour-Target-Amp-G1-Play-v0 \
  --num_envs 1
```

Expected:

```text
[INFO] Monitor Manager: <MonitorManager> contains 1 active groups.
[SMOKE] monitor terms: ['foothold_planner']
[SMOKE] env closed
[SMOKE] closing Isaac app
```

No traceback, NaN, or Inf may appear in the monitor output.

- [ ] **Step 7: Commit environment integration**

```bash
git add \
  source/instinctlab/instinctlab/monitors/__init__.py \
  source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py \
  tests/parkour/foothold/smoke_foothold_planner.py
git diff --cached --check
git commit -m "feat(foothold): register planner diagnostics monitor"
```

### Task 4: Verify training and TensorBoard output

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes: registered monitor and existing `scripts/foothold_train.sh`.
- Produces: evidence that logging works without changing training behavior.

- [ ] **Step 1: Run a one-iteration training integration check**

Run:

```bash
cd ~/InstinctLab-foothold
NUM_ENVS=64 MAX_ITERATIONS=1 \
RUN_NAME=foothold_monitor_1it \
bash scripts/foothold_train.sh
```

Expected: training reaches iteration 1 and exits without NaN, traceback, or
changes to the active reward-term table.

- [ ] **Step 2: Confirm event files contain monitor tags**

Run:

```bash
tensorboard --inspect \
  --logdir logs/instinct_rl/g1_parkour \
  | rg "Episode_Monitor/foothold_planner_|Step_Monitor/foothold_planner_"
```

Expected: output includes at least:

```text
Episode_Monitor/foothold_planner_clearance_safe_fraction
Episode_Monitor/foothold_planner_plan_invalid_fraction
Episode_Monitor/foothold_planner_swing_fraction
```

- [ ] **Step 3: Record final repository evidence**

Run:

```bash
git status --short --branch
git log -4 --oneline
```

Expected: clean working tree with the three implementation commits above the
design/plan commits.
