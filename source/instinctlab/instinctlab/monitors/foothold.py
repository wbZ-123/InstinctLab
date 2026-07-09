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

        step_count = selected_step_count
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