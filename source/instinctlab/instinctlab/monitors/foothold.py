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