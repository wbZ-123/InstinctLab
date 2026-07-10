from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from instinctlab_foothold import GaitState

from .monitor_manager import MonitorTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .monitor_cfg import MonitorTermCfg


class FootholdPlannerMonitorTerm(MonitorTerm):
    """Accumulate finite raw diagnostics from a foothold planner sensor.

    Most gait/clearance fractions use simulation steps as their denominator.
    Safe-target search metrics are different: target search only happens when
    a new swing target is planned, so valid/fallback/score metrics are divided
    by ``safe_target_search_count``.  ``safe_target_search_rate`` is the bridge
    back to step time: search events divided by total steps.
    """

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
        "_safe_target_search_count",
        "_safe_target_final_valid_count",
        "_safe_target_fallback_count",
        "_safe_target_score_sum",
        "_safe_target_nominal_inside_ellipse_count",
        "_safe_target_nominal_obstacle_safe_count",
        "_safe_target_nominal_valid_count",
        "_safe_target_candidate_count_sum",
        "_safe_target_candidate_inside_ellipse_count_sum",
        "_safe_target_candidate_obstacle_safe_count_sum",
        "_safe_target_candidate_valid_count_sum",
    )
    _MAX_BUFFER_NAMES = (
        "_penetration_max",
        "_apex_delta_max",
        "_safe_target_score_max",
    )

    def __init__(self, cfg: MonitorTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_name = cfg.params.get("sensor_name", "foothold_planner")
        debug_event_path = cfg.params.get("debug_event_path")
        self._debug_event_path = (
            Path(debug_event_path) if debug_event_path is not None else None
        )
        self._debug_event_max_count = int(
            cfg.params.get("debug_event_max_count", 0)
        )
        self._debug_event_count = 0
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
        safe_target_search_performed = self._require(
            data, "safe_target_search_performed"
        ).bool()
        safe_target_final_valid = self._require(
            data, "safe_target_final_valid"
        ).bool()
        safe_target_used_fallback = self._require(
            data, "safe_target_used_fallback"
        ).bool()
        safe_target_score_raw = self._require(data, "safe_target_score")
        safe_target_nominal_inside_ellipse = self._require(
            data, "safe_target_nominal_inside_ellipse"
        ).bool()
        safe_target_nominal_obstacle_safe = self._require(
            data, "safe_target_nominal_obstacle_safe"
        ).bool()
        safe_target_nominal_valid = self._require(
            data, "safe_target_nominal_valid"
        ).bool()
        safe_target_candidate_count_raw = self._require(
            data, "safe_target_candidate_count"
        )
        safe_target_candidate_inside_ellipse_count_raw = self._require(
            data, "safe_target_candidate_inside_ellipse_count"
        )
        safe_target_candidate_obstacle_safe_count_raw = self._require(
            data, "safe_target_candidate_obstacle_safe_count"
        )
        safe_target_candidate_valid_count_raw = self._require(
            data, "safe_target_candidate_valid_count"
        )

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
        safe_target_score = torch.nan_to_num(
            safe_target_score_raw, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        safe_target_candidate_count = torch.nan_to_num(
            safe_target_candidate_count_raw, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        safe_target_candidate_inside_ellipse_count = torch.nan_to_num(
            safe_target_candidate_inside_ellipse_count_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        safe_target_candidate_obstacle_safe_count = torch.nan_to_num(
            safe_target_candidate_obstacle_safe_count_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        safe_target_candidate_valid_count = torch.nan_to_num(
            safe_target_candidate_valid_count_raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
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
        # Safe-target search is an event, not a per-step state.  Only count the
        # valid/fallback/score fields when a new search actually happened.
        search_sample = safe_target_search_performed.float()
        self._safe_target_search_count += search_sample
        self._safe_target_final_valid_count += (
            safe_target_search_performed & safe_target_final_valid
        ).float()
        self._safe_target_fallback_count += (
            safe_target_search_performed & safe_target_used_fallback
        ).float()
        self._safe_target_score_sum += safe_target_score * search_sample
        self._safe_target_score_max = torch.maximum(
            self._safe_target_score_max,
            safe_target_score * search_sample,
        )
        self._safe_target_nominal_inside_ellipse_count += (
            safe_target_search_performed & safe_target_nominal_inside_ellipse
        ).float()
        self._safe_target_nominal_obstacle_safe_count += (
            safe_target_search_performed & safe_target_nominal_obstacle_safe
        ).float()
        self._safe_target_nominal_valid_count += (
            safe_target_search_performed & safe_target_nominal_valid
        ).float()
        self._safe_target_candidate_count_sum += (
            safe_target_candidate_count * search_sample
        )
        self._safe_target_candidate_inside_ellipse_count_sum += (
            safe_target_candidate_inside_ellipse_count * search_sample
        )
        self._safe_target_candidate_obstacle_safe_count_sum += (
            safe_target_candidate_obstacle_safe_count * search_sample
        )
        self._safe_target_candidate_valid_count_sum += (
            safe_target_candidate_valid_count * search_sample
        )

        self._previous_touchdown_accepted.copy_(touchdown_accepted)
        self._previous_touchdown_confirm.copy_(touchdown_confirm)
        self._maybe_dump_debug_events(
            safe_target_search_performed=safe_target_search_performed,
            safe_target_final_valid=safe_target_final_valid,
            planner_valid=planner_valid,
        )

    @staticmethod
    def _round_float(value: float) -> float:
        return round(float(value), 6)

    @classmethod
    def _tensor_row_to_list(
        cls,
        value: torch.Tensor | None,
        env_id: int,
    ) -> list[float] | None:
        if value is None:
            return None
        row = value[env_id].detach().cpu().reshape(-1).tolist()
        return [cls._round_float(item) for item in row]

    @classmethod
    def _tensor_scalar(
        cls,
        value: torch.Tensor | None,
        env_id: int,
    ) -> float | bool | int | None:
        if value is None:
            return None
        item = value[env_id].detach().cpu().item()
        if isinstance(item, bool):
            return item
        if isinstance(item, int):
            return item
        return cls._round_float(float(item))

    @staticmethod
    def _classify_invalid_safe_target(
        *,
        nominal_valid: bool,
        candidate_count: float,
        candidate_inside_ellipse_count: float,
        candidate_obstacle_safe_count: float,
        candidate_valid_count: float,
    ) -> str:
        if nominal_valid:
            return "inconsistent_final_invalid"
        if candidate_count <= 0.0:
            return "nominal_invalid_no_candidates"
        if candidate_inside_ellipse_count <= 0.0:
            return "candidate_outside_ellipse"
        if candidate_obstacle_safe_count <= 0.0:
            return "candidate_obstacle_blocked"
        if candidate_valid_count <= 0.0:
            return "candidate_constraints_intersection_empty"
        return "unknown_invalid"

    def _maybe_dump_debug_events(
        self,
        *,
        safe_target_search_performed: torch.Tensor,
        safe_target_final_valid: torch.Tensor,
        planner_valid: torch.Tensor,
    ) -> None:
        if self._debug_event_path is None or self._debug_event_max_count <= 0:
            return
        if self._debug_event_count >= self._debug_event_max_count:
            return

        data = self._planner.data
        invalid_event = safe_target_search_performed & ~safe_target_final_valid
        if not torch.any(invalid_event).item():
            return

        env_ids = torch.nonzero(invalid_event, as_tuple=False).flatten()
        self._debug_event_path.parent.mkdir(parents=True, exist_ok=True)
        with self._debug_event_path.open("a", encoding="utf-8") as file:
            for env_id_tensor in env_ids:
                if self._debug_event_count >= self._debug_event_max_count:
                    break
                env_id = int(env_id_tensor.item())
                nominal_valid = bool(
                    data.safe_target_nominal_valid[env_id].detach().cpu().item()
                )
                candidate_count = float(
                    data.safe_target_candidate_count[env_id].detach().cpu().item()
                )
                candidate_inside_count = float(
                    data.safe_target_candidate_inside_ellipse_count[env_id]
                    .detach()
                    .cpu()
                    .item()
                )
                candidate_obstacle_safe_count = float(
                    data.safe_target_candidate_obstacle_safe_count[env_id]
                    .detach()
                    .cpu()
                    .item()
                )
                candidate_valid_count = float(
                    data.safe_target_candidate_valid_count[env_id]
                    .detach()
                    .cpu()
                    .item()
                )
                event = {
                    "event": "safe_target_invalid",
                    "reason": self._classify_invalid_safe_target(
                        nominal_valid=nominal_valid,
                        candidate_count=candidate_count,
                        candidate_inside_ellipse_count=candidate_inside_count,
                        candidate_obstacle_safe_count=candidate_obstacle_safe_count,
                        candidate_valid_count=candidate_valid_count,
                    ),
                    "env_id": env_id,
                    "step_count": self._tensor_scalar(self._step_count, env_id),
                    "planner_valid": bool(
                        planner_valid[env_id].detach().cpu().item()
                    ),
                    "safe_target_final_valid": False,
                    "safe_target_used_fallback": self._tensor_scalar(
                        data.safe_target_used_fallback, env_id
                    ),
                    "safe_target_score": self._tensor_scalar(
                        data.safe_target_score, env_id
                    ),
                    "safe_target_nominal_inside_ellipse": self._tensor_scalar(
                        data.safe_target_nominal_inside_ellipse, env_id
                    ),
                    "safe_target_nominal_obstacle_safe": self._tensor_scalar(
                        data.safe_target_nominal_obstacle_safe, env_id
                    ),
                    "safe_target_nominal_valid": nominal_valid,
                    "safe_target_candidate_count": self._round_float(
                        candidate_count
                    ),
                    "safe_target_candidate_inside_ellipse_count": self._round_float(
                        candidate_inside_count
                    ),
                    "safe_target_candidate_obstacle_safe_count": self._round_float(
                        candidate_obstacle_safe_count
                    ),
                    "safe_target_candidate_valid_count": self._round_float(
                        candidate_valid_count
                    ),
                    "raw_unclipped_foothold_f": self._tensor_row_to_list(
                        getattr(data, "raw_unclipped_foothold_f", None), env_id
                    ),
                    "target_foothold_f": self._tensor_row_to_list(
                        getattr(data, "target_foothold_f", None), env_id
                    ),
                    "desired_velocity_f": self._tensor_row_to_list(
                        getattr(data, "desired_velocity_f", None), env_id
                    ),
                    "feasible_velocity_f": self._tensor_row_to_list(
                        getattr(data, "feasible_velocity_f", None), env_id
                    ),
                    "swing_clearance_safe": self._tensor_scalar(
                        getattr(data, "swing_clearance_safe", None), env_id
                    ),
                    "swing_clearance_penetration": self._tensor_scalar(
                        getattr(data, "swing_clearance_penetration", None), env_id
                    ),
                    "default_swing_apex_height": self._tensor_scalar(
                        getattr(data, "default_swing_apex_height", None), env_id
                    ),
                    "swing_apex_height": self._tensor_scalar(
                        getattr(data, "swing_apex_height", None), env_id
                    ),
                }
                file.write(json.dumps(event, sort_keys=True) + "\n")
                self._debug_event_count += 1

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
                    "safe_target_search_rate",
                    "safe_target_final_valid_fraction",
                    "safe_target_fallback_fraction",
                    "safe_target_score_mean",
                    "safe_target_score_max",
                    "safe_target_nominal_inside_ellipse_fraction",
                    "safe_target_nominal_obstacle_safe_fraction",
                    "safe_target_nominal_valid_fraction",
                    "safe_target_candidate_count_mean",
                    "safe_target_candidate_inside_ellipse_count_mean",
                    "safe_target_candidate_obstacle_safe_count_mean",
                    "safe_target_candidate_valid_count_mean",
                )
            }

        step_count = selected_step_count
        clearance_count = self._clearance_sample_count[env_ids]
        safe_target_search_count = self._safe_target_search_count[env_ids]
        total_safe_target_search_count = safe_target_search_count.sum()
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
            # Fraction of all simulation steps where a new safe-target search
            # was performed.
            "safe_target_search_rate": self._safe_ratio(
                safe_target_search_count, step_count
            ).mean(),
            # Fraction of search events that ended with an executable target.
            "safe_target_final_valid_fraction": self._safe_ratio(
                self._safe_target_final_valid_count[env_ids].sum(),
                total_safe_target_search_count,
            ),
            # Fraction of search events that replaced the nominal target.
            "safe_target_fallback_fraction": self._safe_ratio(
                self._safe_target_fallback_count[env_ids].sum(),
                total_safe_target_search_count,
            ),
            # Mean XY correction distance over search events.
            "safe_target_score_mean": self._safe_ratio(
                self._safe_target_score_sum[env_ids].sum(),
                total_safe_target_search_count,
            ),
            "safe_target_score_max": self._safe_target_score_max[
                env_ids
            ].max(),
            "safe_target_nominal_inside_ellipse_fraction": self._safe_ratio(
                self._safe_target_nominal_inside_ellipse_count[env_ids].sum(),
                total_safe_target_search_count,
            ),
            "safe_target_nominal_obstacle_safe_fraction": self._safe_ratio(
                self._safe_target_nominal_obstacle_safe_count[env_ids].sum(),
                total_safe_target_search_count,
            ),
            "safe_target_nominal_valid_fraction": self._safe_ratio(
                self._safe_target_nominal_valid_count[env_ids].sum(),
                total_safe_target_search_count,
            ),
            "safe_target_candidate_count_mean": self._safe_ratio(
                self._safe_target_candidate_count_sum[env_ids].sum(),
                total_safe_target_search_count,
            ),
            "safe_target_candidate_inside_ellipse_count_mean": self._safe_ratio(
                self._safe_target_candidate_inside_ellipse_count_sum[env_ids].sum(),
                total_safe_target_search_count,
            ),
            "safe_target_candidate_obstacle_safe_count_mean": self._safe_ratio(
                self._safe_target_candidate_obstacle_safe_count_sum[env_ids].sum(),
                total_safe_target_search_count,
            ),
            "safe_target_candidate_valid_count_mean": self._safe_ratio(
                self._safe_target_candidate_valid_count_sum[env_ids].sum(),
                total_safe_target_search_count,
            ),
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
