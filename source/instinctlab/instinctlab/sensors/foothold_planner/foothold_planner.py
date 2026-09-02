from __future__ import annotations

import re
from dataclasses import replace
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.sensors import SensorBase
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.utils.math import convert_quat
from isaaclab.utils.warp import raycast_mesh
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema

from instinctlab_foothold import (
    FlatProviderConfig,
    GaitMachineConfig,
    GaitMachineState,
    GaitState,
    SoleGeometry,
    advance_gait,
    apply_world_height_to_planner_target,
    apply_late_touchdown_descent,
    clear_learned_foothold_buffers,
    ContactEvent,
    EventResponse,
    StabilityBounds,
    load_stability_bounds,
    response_for_event,
    support_roles_from_contacts,
    evaluate_safe_foothold_target,
    finalize_learned_foothold_route_outcome,
    gait_phase,
    initial_gait_state,
    adjust_apex_for_edge_clearance,
    classify_learned_foothold_route,
    learned_foothold_event_masks,
    learned_foothold_swing_ready,
    learned_foothold_transaction_ready,
    lock_prepared_learned_foothold,
    make_recovery_foothold_target,
    nominal_foothold_prepare_mask,
    prepare_learned_foothold_target,
    quintic_swing_reference,
    reachable_ellipse_usage,
    reframe_cached_world_foothold,
    route_nominal_and_learned_footholds,
    select_preflight_target_w,
    sample_flat_targets,
    store_learned_foothold_preparation,
)

from .foothold_planner_data import FootholdPlannerData

from instinctlab_foothold.target_search import (
    PenetrationObstacle as TargetSearchObstacle,
    make_sole_perimeter_points_xy,
    search_safe_foothold_target,
)

if TYPE_CHECKING:
    from instinctlab_foothold.clearance import (
        PenetrationObstacle as ClearanceObstacle,
    )
    from .foothold_planner_cfg import FootholdPlannerCfg


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return world yaw from a quaternion in wxyz convention."""
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def _select_sole_roles(
    *,
    left_sole_w: torch.Tensor,
    right_sole_w: torch.Tensor,
    swing_side: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stance and swing soles for the supplied gait-side state."""
    if left_sole_w.shape != right_sole_w.shape:
        raise ValueError("left and right sole tensors must share one shape.")
    if left_sole_w.shape[:-1] != swing_side.shape:
        raise ValueError("swing_side must match the sole batch shape.")

    swing_is_left = swing_side == 0
    stance_w = torch.where(
        swing_is_left.unsqueeze(-1),
        right_sole_w,
        left_sole_w,
    )
    swing_w = torch.where(
        swing_is_left.unsqueeze(-1),
        left_sole_w,
        right_sole_w,
    )
    return stance_w, swing_w


def _rotate_vector_yaw(
    vector_f: torch.Tensor,
    yaw_w: torch.Tensor,
) -> torch.Tensor:
    cos_yaw = torch.cos(yaw_w)
    sin_yaw = torch.sin(yaw_w)
    vector_w = vector_f.clone()
    vector_w[..., 0] = cos_yaw * vector_f[..., 0] - sin_yaw * vector_f[..., 1]
    vector_w[..., 1] = sin_yaw * vector_f[..., 0] + cos_yaw * vector_f[..., 1]
    return vector_w


def _compose_world_from_frame(
    origin_w: torch.Tensor,
    vector_f: torch.Tensor,
    yaw_w: torch.Tensor,
) -> torch.Tensor:
    return origin_w + _rotate_vector_yaw(vector_f, yaw_w)


def _apply_terrain_height_to_target(
    *,
    target_foothold_w: torch.Tensor,
    target_foothold_f: torch.Tensor,
    stance_pos_w: torch.Tensor,
    terrain_height_w: torch.Tensor,
    terrain_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return foothold targets with valid terrain heights applied to z.

    Horizontal foothold selection remains unchanged.  Only valid, finite
    terrain heights replace the target world z.  The local-frame target z is
    kept consistent with the corrected world target so observations/debug data
    do not disagree with the actual reference.
    """
    finite_height = torch.isfinite(terrain_height_w)
    valid = terrain_valid & finite_height

    corrected_w = target_foothold_w.clone()
    corrected_w[:, 2] = torch.where(
        valid,
        terrain_height_w,
        corrected_w[:, 2],
    )

    corrected_f = target_foothold_f.clone()
    corrected_f[:, 2] = corrected_w[:, 2] - stance_pos_w[:, 2]
    return corrected_w, corrected_f, valid


def _make_required_body_paths_glob(
    robot_prim_path: str,
    body_names: Sequence[str],
) -> str:
    """Build a PhysX rigid-body glob for only the planner-required bodies."""
    unique_body_names = sorted(set(body_names))
    if not unique_body_names:
        raise ValueError("body_names must not be empty.")

    body_names_regex = (
        r"("
        + "|".join(re.escape(name) for name in unique_body_names)
        + r")"
    )
    body_paths_regex = f"{robot_prim_path}/{body_names_regex}"
    return body_paths_regex.replace(".*", "*")


def _adaptive_step_hold_s(
    desired_velocity_f: torch.Tensor,
    *,
    base_hold_s: float,
    min_hold_s: float,
    velocity_scale_s_per_mps: float,
) -> torch.Tensor:
    """Return velocity-adaptive double-support hold time per environment."""
    if min_hold_s < 0.0:
        raise ValueError("min_hold_s must be non-negative.")
    if base_hold_s < min_hold_s:
        raise ValueError("base_hold_s must be greater than or equal to min_hold_s.")
    if velocity_scale_s_per_mps < 0.0:
        raise ValueError("velocity_scale_s_per_mps must be non-negative.")

    planar_speed = torch.linalg.norm(desired_velocity_f[:, :2], dim=-1)
    hold_s = base_hold_s - velocity_scale_s_per_mps * planar_speed
    return torch.clamp(
        hold_s,
        min=min_hold_s,
        max=base_hold_s,
    )


def _flat_target_level_from_curriculum_scale(
    scale: torch.Tensor,
    *,
    num_levels: int,
) -> torch.Tensor:
    """Map a [0, 1] curriculum scale to a flat-provider target level."""
    if num_levels <= 0:
        raise ValueError("num_levels must be positive.")

    scale = torch.nan_to_num(scale, nan=0.0, posinf=1.0, neginf=0.0)
    scale = scale.clamp(0.0, 1.0)
    if num_levels == 1:
        return torch.zeros_like(scale, dtype=torch.long)

    return torch.floor(scale * num_levels).clamp(
        min=0,
        max=num_levels - 1,
    ).to(dtype=torch.long)


flat_target_level_from_curriculum_scale = (
    _flat_target_level_from_curriculum_scale
)


def _derive_flat_provider_config(cfg: object) -> FlatProviderConfig:
    """Derive flat-target sampler timing from planner gait timing.

    ``flat_target_lookahead_phase`` is a temporary calibration parameter: it
    means "the expected touchdown phase inside the nominal swing interval".
    """
    if cfg.swing_duration_s <= 0.0:
        raise ValueError("swing_duration_s must be positive.")
    if not 0.0 < cfg.flat_target_lookahead_phase <= 1.0:
        raise ValueError("flat_target_lookahead_phase must be in (0, 1].")

    return replace(
        FlatProviderConfig(),
        velocity_lookahead_s=cfg.swing_duration_s * cfg.flat_target_lookahead_phase,
    )


def _clear_safe_target_event_buffers(
    data: FootholdPlannerData,
    env_ids: Sequence[int] | torch.Tensor | slice,
) -> None:
    """Clear per-step safe-target event buffers for selected environments.

    Safe-target diagnostics are event fields, not persistent state.  They must
    be reset before each planner update so monitor terms do not repeatedly
    count a previous swing-planning event.
    """

    if data.safe_target_search_performed is not None:
        data.safe_target_search_performed[env_ids] = False
    if data.safe_target_final_valid is not None:
        data.safe_target_final_valid[env_ids] = True
    if data.safe_target_used_fallback is not None:
        data.safe_target_used_fallback[env_ids] = False
    if data.safe_target_score is not None:
        data.safe_target_score[env_ids] = 0.0
    if data.safe_target_nominal_inside_ellipse is not None:
        data.safe_target_nominal_inside_ellipse[env_ids] = True
    if data.safe_target_nominal_obstacle_safe is not None:
        data.safe_target_nominal_obstacle_safe[env_ids] = True
    if data.safe_target_nominal_valid is not None:
        data.safe_target_nominal_valid[env_ids] = True
    if data.safe_target_candidate_count is not None:
        data.safe_target_candidate_count[env_ids] = 0.0
    if data.safe_target_candidate_inside_ellipse_count is not None:
        data.safe_target_candidate_inside_ellipse_count[env_ids] = 0.0
    if data.safe_target_candidate_obstacle_safe_count is not None:
        data.safe_target_candidate_obstacle_safe_count[env_ids] = 0.0
    if data.safe_target_candidate_valid_count is not None:
        data.safe_target_candidate_valid_count[env_ids] = 0.0
    if data.safe_target_final_max_penetration_depth is not None:
        data.safe_target_final_max_penetration_depth[env_ids] = 0.0


def _clear_foothold_plan_buffers(
    data: FootholdPlannerData,
    env_ids: Sequence[int] | torch.Tensor | slice,
) -> None:
    """Clear persistent foothold plan buffers for selected environments.

    These fields describe the currently active swing plan.  After an
    environment reset there is no active plan yet, so leaving the previous
    target around makes debug markers, observations, and reward terms see a
    stale foothold while the gait state is already back in HOLD.
    """

    if getattr(data, "target_foothold_w", None) is not None:
        data.target_foothold_w[env_ids] = 0.0
    if getattr(data, "target_foothold_f", None) is not None:
        data.target_foothold_f[env_ids] = 0.0
    if getattr(data, "swing_start_pos_w", None) is not None:
        data.swing_start_pos_w[env_ids] = 0.0
    if getattr(data, "raw_unclipped_foothold_f", None) is not None:
        data.raw_unclipped_foothold_f[env_ids] = 0.0
    if getattr(data, "nominal_foothold_prepared", None) is not None:
        data.nominal_foothold_prepared[env_ids] = False
    if getattr(data, "nominal_feasible_velocity_f", None) is not None:
        data.nominal_feasible_velocity_f[env_ids] = 0.0
    if getattr(data, "nominal_curriculum_residual_f", None) is not None:
        data.nominal_curriculum_residual_f[env_ids] = 0.0
    if getattr(data, "nominal_curriculum_radius_f", None) is not None:
        data.nominal_curriculum_radius_f[env_ids] = 0.0
    if getattr(data, "nominal_curriculum_usage", None) is not None:
        data.nominal_curriculum_usage[env_ids] = 0.0
    if getattr(data, "nominal_frame_origin_w", None) is not None:
        data.nominal_frame_origin_w[env_ids] = 0.0
    if getattr(data, "nominal_frame_yaw_w", None) is not None:
        data.nominal_frame_yaw_w[env_ids] = 0.0
    if getattr(data, "nominal_foothold_w", None) is not None:
        data.nominal_foothold_w[env_ids] = 0.0
    if getattr(data, "nominal_geometric_valid", None) is not None:
        data.nominal_geometric_valid[env_ids] = False
    if getattr(data, "nominal_safety_valid", None) is not None:
        data.nominal_safety_valid[env_ids] = False
    if getattr(data, "nominal_safety_score", None) is not None:
        data.nominal_safety_score[env_ids] = 0.0
    if getattr(data, "learned_foothold_lock_geometric_valid", None) is not None:
        data.learned_foothold_lock_geometric_valid[env_ids] = True
    if getattr(data, "target_terrain_valid", None) is not None:
        data.target_terrain_valid[env_ids] = True
    if getattr(data, "feasible_velocity_f", None) is not None:
        data.feasible_velocity_f[env_ids] = 0.0
    if getattr(data, "target_delta_f", None) is not None:
        data.target_delta_f[env_ids] = 0.0
    if getattr(data, "curriculum_residual_f", None) is not None:
        data.curriculum_residual_f[env_ids] = 0.0
    if getattr(data, "curriculum_radius_f", None) is not None:
        data.curriculum_radius_f[env_ids] = 0.0
    if getattr(data, "curriculum_usage", None) is not None:
        data.curriculum_usage[env_ids] = 0.0
    if getattr(data, "target_ellipse_max_x", None) is not None:
        data.target_ellipse_max_x[env_ids] = 0.0
    if getattr(data, "target_ellipse_usage", None) is not None:
        data.target_ellipse_usage[env_ids] = 0.0
    if getattr(data, "default_swing_apex_height", None) is not None:
        data.default_swing_apex_height[env_ids] = 0.0
    if getattr(data, "swing_apex_height", None) is not None:
        data.swing_apex_height[env_ids] = 0.0
    if getattr(data, "swing_clearance_safe", None) is not None:
        data.swing_clearance_safe[env_ids] = True
    if getattr(data, "swing_clearance_penetration", None) is not None:
        data.swing_clearance_penetration[env_ids] = 0.0
    if getattr(data, "swing_clearance_deepest_phase", None) is not None:
        data.swing_clearance_deepest_phase[env_ids] = 0.0
    if getattr(data, "swing_clearance_start_penetration", None) is not None:
        data.swing_clearance_start_penetration[env_ids] = 0.0
    if getattr(data, "swing_clearance_goal_penetration", None) is not None:
        data.swing_clearance_goal_penetration[env_ids] = 0.0
    if getattr(data, "swing_clearance_start_escape_safe", None) is not None:
        data.swing_clearance_start_escape_safe[env_ids] = False
    if getattr(data, "swing_preflight_safe", None) is not None:
        data.swing_preflight_safe[env_ids] = True
    if getattr(data, "swing_preflight_ready", None) is not None:
        data.swing_preflight_ready[env_ids] = False


def _apply_startup_hold_gate(
    *,
    data: FootholdPlannerData,
    gait_state: GaitMachineState,
    selected_env_ids: torch.Tensor,
    startup_hold_mask: torch.Tensor,
    reset_hold_s: float,
) -> None:
    """Keep selected environments in HOLD during episode-start stabilisation."""

    if not torch.any(startup_hold_mask).item():
        return

    startup_env_ids = selected_env_ids[startup_hold_mask]
    gait_state.mode[startup_hold_mask] = GaitState.HOLD
    gait_state.elapsed_s[startup_hold_mask] = 0.0
    gait_state.hold_required_s[startup_hold_mask] = reset_hold_s
    gait_state.swing_has_lifted[startup_hold_mask] = False
    gait_state.recovery_step_pending[startup_hold_mask] = False
    gait_state.recovery_step_active[startup_hold_mask] = False

    if data.gait_mode is not None:
        data.gait_mode[startup_env_ids] = GaitState.HOLD
    if data.phase is not None:
        data.phase[startup_env_ids] = 0.0
    if data.touchdown_accepted is not None:
        data.touchdown_accepted[startup_env_ids] = False
    if data.swing_has_lifted is not None:
        data.swing_has_lifted[startup_env_ids] = False
    if data.recovery_step_active is not None:
        data.recovery_step_active[startup_env_ids] = False

    _clear_foothold_plan_buffers(data, startup_env_ids)
    clear_learned_foothold_buffers(data, startup_env_ids)


class FootholdPlanner(SensorBase):
    """Foothold planner sensor.

    This sensor computes foothold targets and swing references, then exposes
    them through ``sensor.data`` for rewards, observations, and debug
    visualization.
    """

    cfg: FootholdPlannerCfg

    def __init__(self, cfg: FootholdPlannerCfg):
        super().__init__(cfg)
        if cfg.startup_hold_s < 0.0:
            raise ValueError("startup_hold_s must be non-negative.")
        if cfg.target_terrain_raycast_start_height_m <= 0.0:
            raise ValueError(
                "target_terrain_raycast_start_height_m must be positive."
            )
        if cfg.target_terrain_raycast_max_dist_m <= 0.0:
            raise ValueError(
                "target_terrain_raycast_max_dist_m must be positive."
            )
        self._data = FootholdPlannerData()
        self._flat_target_curriculum_scale: torch.Tensor | None = None
        self._virtual_obstacles: dict[str, object] = {}
        self._sole_geometry = SoleGeometry(
            center_offset_b=torch.tensor(cfg.sole_center_offset_b),
            half_length=cfg.sole_half_length,
            half_width=cfg.sole_half_width,
        )
        self._flat_provider_cfg = _derive_flat_provider_config(cfg)
        self._stability_bounds: StabilityBounds | None = None
        if cfg.enable_contact_adaptive_recovery:
            if not cfg.recovery_stability_calibration_path:
                raise ValueError(
                    "contact-adaptive recovery requires "
                    "recovery_stability_calibration_path"
                )
            self._stability_bounds = load_stability_bounds(
                cfg.recovery_stability_calibration_path
            )
        self._gait_cfg = GaitMachineConfig(
            reset_hold_s=cfg.reset_hold_s,
            swing_s=cfg.swing_duration_s,
            contact_confirm_s=cfg.contact_confirm_s,
            stance_lost_confirm_s=cfg.stance_lost_confirm_s,
            hold_contact_lost_confirm_s=cfg.hold_contact_lost_confirm_s,
            early_contact_phase=cfg.early_contact_phase,
            overdue_s=cfg.overdue_s,
            recovery_hold_s=cfg.recovery_hold_s,
            step_hold_s=cfg.step_hold_s,
        )

    @property
    def data(self) -> FootholdPlannerData:
        self._update_outdated_buffers()
        return self._data
    
    def register_virtual_obstacles(
        self,
        virtual_obstacles: dict[str, object],
    ) -> None:
        """Register terrain virtual obstacles for swing clearance checks.

        This mirrors VolumePoints.register_virtual_obstacles. The planner only
        stores obstacle handles here; swing-clearance adjustment can consume them
        later during reference generation.
        """
        self._virtual_obstacles.update(virtual_obstacles)

    def set_desired_velocity(
        self,
        desired_velocity_f: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set the desired velocity used by the flat foothold sampler.

        Args:
            desired_velocity_f: Desired ``[vx, vy, wz]`` commands in the
                planner frame. Shape can be ``(3,)`` for all environments or
                ``(num_selected_envs, 3)`` for a subset.
            env_ids: Environment ids to update. Defaults to all environments.
        """
        if self._data.desired_velocity_f is None:
            raise RuntimeError(
                "FootholdPlanner desired velocity buffer is not initialized."
            )

        if env_ids is None:
            resolved_env_ids = slice(None)
            resolved_env_id_tensor = torch.arange(
                self._num_envs,
                device=self._device,
                dtype=torch.long,
            )
            num_selected_envs = self._num_envs
        elif isinstance(env_ids, slice):
            resolved_env_ids = env_ids
            resolved_env_id_tensor = torch.arange(
                self._num_envs,
                device=self._device,
                dtype=torch.long,
            )[env_ids]
            num_selected_envs = resolved_env_id_tensor.shape[0]
        else:
            resolved_env_ids = env_ids
            resolved_env_id_tensor = torch.as_tensor(
                env_ids,
                device=self._device,
                dtype=torch.long,
            )
            num_selected_envs = resolved_env_id_tensor.shape[0]

        desired_velocity_f = desired_velocity_f.to(
            device=self._device,
            dtype=self._data.desired_velocity_f.dtype,
        )
        if desired_velocity_f.shape == (3,):
            desired_velocity_f = desired_velocity_f.unsqueeze(0).expand(
                num_selected_envs,
                -1,
            )
        if desired_velocity_f.shape != (num_selected_envs, 3):
            raise ValueError(
                "desired_velocity_f must have shape (3,) or "
                f"({num_selected_envs}, 3), got "
                f"{tuple(desired_velocity_f.shape)}."
            )

        # A prepared HOLD plan is a transaction for the next swing.  Keep the
        # live command buffer current, but do not invalidate that transaction
        # when heading feedback changes the command on every control tick.  A
        # new snapshot is consumed when the next HOLD plan is prepared.
        self._data.desired_velocity_f[resolved_env_ids] = desired_velocity_f

    def set_flat_target_curriculum_scale(
        self,
        scale: float | torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set the curriculum scale used to choose flat-target sampling level."""
        if self._flat_target_curriculum_scale is None:
            return

        if env_ids is None:
            resolved_env_ids = slice(None)
            num_selected_envs = self._num_envs
        elif isinstance(env_ids, slice):
            resolved_env_ids = env_ids
            num_selected_envs = torch.arange(
                self._num_envs,
                device=self._device,
            )[env_ids].shape[0]
        else:
            resolved_env_ids = env_ids
            num_selected_envs = len(env_ids)

        scale_tensor = torch.as_tensor(
            scale,
            device=self._device,
            dtype=self._flat_target_curriculum_scale.dtype,
        )
        if scale_tensor.ndim == 0:
            scale_tensor = scale_tensor.expand(num_selected_envs)
        if scale_tensor.shape != (num_selected_envs,):
            raise ValueError(
                "scale must be scalar or match the number of selected environments."
            )

        self._flat_target_curriculum_scale[resolved_env_ids] = (
            torch.nan_to_num(scale_tensor, nan=0.0, posinf=1.0, neginf=0.0)
            .clamp(0.0, 1.0)
        )

    @property
    def flat_target_curriculum_scale(self) -> torch.Tensor | None:
        """Read-only view of the per-environment flat-target curriculum scale."""

        return self._flat_target_curriculum_scale

    def _query_target_terrain_height_at_xy_w(
        self,
        target_xy_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Query terrain mesh height at final target XY with one downward ray.

        The planner uses this only when a new target is created, not every
        control step.  Missing terrain mesh data returns ``None`` so flat-plane
        unit tests and non-standard scenes keep their existing z fallback.
        """
        if not self.cfg.enable_target_terrain_height:
            return None

        mesh = RayCaster.meshes.get(self.cfg.target_terrain_mesh_prim_path)
        if mesh is None:
            return None

        ray_starts = torch.zeros(
            target_xy_w.shape[0],
            3,
            device=target_xy_w.device,
            dtype=target_xy_w.dtype,
        )
        ray_starts[:, :2] = target_xy_w
        ray_starts[:, 2] = self.cfg.target_terrain_raycast_start_height_m
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[:, 2] = -1.0

        ray_hits_w, ray_distance, _, _ = raycast_mesh(
            ray_starts=ray_starts.unsqueeze(1),
            ray_directions=ray_directions.unsqueeze(1),
            mesh=mesh,
            max_dist=self.cfg.target_terrain_raycast_max_dist_m,
            return_distance=True,
        )
        assert ray_distance is not None
        ray_hits_w = ray_hits_w.squeeze(1)
        ray_distance = ray_distance.squeeze(1)
        terrain_height_w = (
            ray_hits_w[:, 2] + self.cfg.target_terrain_height_offset_m
        )
        terrain_valid = torch.isfinite(ray_distance) & torch.isfinite(
            terrain_height_w
        )
        return terrain_height_w, terrain_valid

    def _prepare_learned_footholds(
        self,
        *,
        env_ids: torch.Tensor,
        stance_pos_w: torch.Tensor,
        base_yaw_w: torch.Tensor,
    ) -> None:
        """Evaluate current high-level proposals during confirmed HOLD."""

        if env_ids.numel() == 0:
            return

        assert self._data.learned_foothold_action_normalized is not None
        assert self._data.raw_unclipped_foothold_f is not None
        assert self._data.learned_foothold_event_generation is not None
        self._data.learned_foothold_event_generation[env_ids] += 1

        def terrain_query_w(
            target_xy_w: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = self._query_target_terrain_height_at_xy_w(target_xy_w)
            if result is not None:
                return result
            return (
                torch.full(
                    (target_xy_w.shape[0],),
                    float("nan"),
                    device=target_xy_w.device,
                    dtype=target_xy_w.dtype,
                ),
                torch.zeros(
                    target_xy_w.shape[0],
                    device=target_xy_w.device,
                    dtype=torch.bool,
                ),
            )

        preparation = prepare_learned_foothold_target(
            normalized_action=(
                self._data.learned_foothold_action_normalized[env_ids]
            ),
            nominal_xy_f=self._data.raw_unclipped_foothold_f[env_ids, :2],
            origin_w=stance_pos_w,
            yaw_w=base_yaw_w,
            radius_x=self._flat_provider_cfg.outer_radius_x,
            radius_y=self._flat_provider_cfg.outer_radius_y,
            max_adjustment_x=self.cfg.learned_foothold_max_adjustment_x_m,
            max_adjustment_y=self.cfg.learned_foothold_max_adjustment_y_m,
            max_step_height_m=self.cfg.max_foothold_step_height_m,
            terrain_height_query_w=terrain_query_w,
        )

        obstacle = self._virtual_obstacles.get("edges")
        if obstacle is None:
            safety_valid = torch.ones_like(preparation.geometric_valid)
            safety_score = torch.ones(
                env_ids.shape[0],
                device=self._device,
                dtype=preparation.target_f.dtype,
            )
            penetrating_point_count = torch.zeros_like(safety_score)
            penetrating_point_ratio = torch.zeros_like(safety_score)
            total_penetration_depth = torch.zeros_like(safety_score)
            safety_margin_score = torch.ones_like(safety_score)
            minimum_signed_clearance = torch.full_like(
                safety_score,
                self.cfg.safe_target_clearance_reference_m,
            )
        else:
            evaluation = evaluate_safe_foothold_target(
                target_f=preparation.target_f,
                support_foot_f=torch.zeros_like(preparation.target_f),
                target_origin_w=stance_pos_w,
                target_yaw_w=base_yaw_w,
                obstacle=cast("TargetSearchObstacle", obstacle),
                ellipse_half_length=(
                    self._flat_provider_cfg.outer_radius_x
                ),
                ellipse_half_width=(
                    self._flat_provider_cfg.outer_radius_y
                ),
                foot_points_xy=self._safe_target_foot_points_xy,
                safety_margin=self.cfg.safe_target_search_margin_m,
                max_penetrating_points=(
                    self.cfg.safe_target_max_penetrating_points
                ),
                clearance_reference_m=(
                    self.cfg.safe_target_clearance_reference_m
                ),
            )
            safety_valid = evaluation.obstacle_safe
            safety_score = evaluation.safety_score
            penetrating_point_count = (
                evaluation.penetrating_point_count
            )
            penetrating_point_ratio = (
                evaluation.penetrating_point_ratio
            )
            total_penetration_depth = (
                evaluation.total_penetration_depth
            )
            safety_margin_score = evaluation.safety_margin_score
            minimum_signed_clearance = evaluation.minimum_signed_clearance

        store_learned_foothold_preparation(
            data=self._data,
            env_ids=env_ids,
            preparation=preparation,
            safety_valid=safety_valid,
            safety_score=safety_score,
            penetrating_point_count=penetrating_point_count,
            penetrating_point_ratio=penetrating_point_ratio,
            total_penetration_depth=total_penetration_depth,
            safety_margin_score=safety_margin_score,
            minimum_signed_clearance=minimum_signed_clearance,
        )

    def _prepare_nominal_footholds(
        self,
        *,
        env_ids: torch.Tensor,
        swing_side: torch.Tensor,
        recovery_step: torch.Tensor,
        stance_pos_w: torch.Tensor,
        base_yaw_w: torch.Tensor,
    ) -> None:
        """Generate once in HOLD and cache the exact nominal plan seen by policy."""

        if env_ids.numel() == 0:
            return
        assert self._data.raw_unclipped_foothold_f is not None
        assert self._data.nominal_foothold_prepared is not None
        assert self._data.nominal_feasible_velocity_f is not None
        assert self._data.nominal_curriculum_residual_f is not None
        assert self._data.nominal_curriculum_radius_f is not None
        assert self._data.nominal_curriculum_usage is not None
        assert self._data.nominal_frame_origin_w is not None
        assert self._data.nominal_frame_yaw_w is not None
        assert self._data.nominal_foothold_w is not None
        assert self._data.nominal_geometric_valid is not None
        assert self._data.nominal_safety_valid is not None
        assert self._data.nominal_safety_score is not None

        # This frame remains authoritative until the next nominal is prepared.
        # It keeps the nominal observation, learned action decoding, safety
        # evaluation, and cached world target in one coordinate system.
        self._data.nominal_frame_origin_w[env_ids] = stance_pos_w
        self._data.nominal_frame_yaw_w[env_ids] = base_yaw_w

        desired_velocity = self._data.desired_velocity_f[env_ids]
        level = _flat_target_level_from_curriculum_scale(
            self._flat_target_curriculum_scale[env_ids],
            num_levels=len(self._flat_provider_cfg.curriculum_radius_x),
        )
        stance_xy_f = torch.zeros(
            env_ids.shape[0],
            2,
            device=self._device,
            dtype=desired_velocity.dtype,
        )
        flat_result = sample_flat_targets(
            stance_xy=stance_xy_f,
            swing_side=swing_side,
            desired_velocity=desired_velocity,
            level=level,
            generator=self._generator,
            cfg=self._flat_provider_cfg,
            # The learned planner already has PPO exploration.  Its analytic
            # prior must remain deterministic for the full HOLD transaction;
            # otherwise the policy sees a moving "correct" foothold.
            enable_curriculum_residual=not self.cfg.enable_learned_foothold,
        )

        target_f = flat_result.position_f.clone()
        feasible_velocity_f = flat_result.feasible_velocity_f.clone()
        curriculum_residual_f = flat_result.curriculum_residual_f.clone()
        curriculum_radius_f = flat_result.curriculum_radius_f.clone()
        curriculum_usage = flat_result.curriculum_usage.clone()

        if torch.any(recovery_step).item():
            recovery_target_f = make_recovery_foothold_target(
                swing_side=swing_side[recovery_step],
                desired_velocity_f=desired_velocity[recovery_step],
                step_length_m=self.cfg.recovery_step_length_m,
                velocity_lookahead_s=(
                    self.cfg.recovery_step_velocity_lookahead_s
                ),
                max_step_length_m=self.cfg.recovery_step_max_length_m,
                step_width_m=self.cfg.recovery_step_width_m,
                dtype=target_f.dtype,
                device=self._device,
            )
            target_f[recovery_step] = recovery_target_f
            feasible_velocity_f[recovery_step] = (
                self._feasible_velocity_from_target(
                    target_foothold_f=recovery_target_f,
                    swing_side=swing_side[recovery_step],
                    yaw_velocity_f=torch.zeros(
                        recovery_target_f.shape[0],
                        device=self._device,
                        dtype=target_f.dtype,
                    ),
                )
            )
            curriculum_residual_f[recovery_step] = 0.0
            curriculum_radius_f[recovery_step] = 0.0
            curriculum_usage[recovery_step] = 0.0

        def terrain_query_w(
            target_xy_w: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = self._query_target_terrain_height_at_xy_w(target_xy_w)
            if result is not None:
                return result
            return (
                torch.full(
                    (target_xy_w.shape[0],),
                    float("nan"),
                    device=target_xy_w.device,
                    dtype=target_xy_w.dtype,
                ),
                torch.zeros(
                    target_xy_w.shape[0],
                    device=target_xy_w.device,
                    dtype=torch.bool,
                ),
            )

        target_f, target_w, height_valid = (
            apply_world_height_to_planner_target(
                origin_w=stance_pos_w,
                target_xy_f=target_f[:, :2],
                yaw_w=base_yaw_w,
                terrain_height_query_w=terrain_query_w,
            )
        )
        geometric_valid = height_valid & (
            torch.abs(target_f[:, 2])
            < self.cfg.max_foothold_step_height_m
        )
        obstacle = self._virtual_obstacles.get("edges")
        if obstacle is None:
            safety_valid = torch.ones_like(geometric_valid)
            safety_score = torch.ones(
                env_ids.shape[0],
                device=self._device,
                dtype=target_f.dtype,
            )
        else:
            evaluation = evaluate_safe_foothold_target(
                target_f=target_f,
                support_foot_f=torch.zeros_like(target_f),
                target_origin_w=stance_pos_w,
                target_yaw_w=base_yaw_w,
                obstacle=cast("TargetSearchObstacle", obstacle),
                ellipse_half_length=(
                    self._flat_provider_cfg.outer_radius_x
                ),
                ellipse_half_width=(
                    self._flat_provider_cfg.outer_radius_y
                ),
                foot_points_xy=self._safe_target_foot_points_xy,
                safety_margin=self.cfg.safe_target_search_margin_m,
                max_penetrating_points=(
                    self.cfg.safe_target_max_penetrating_points
                ),
            )
            safety_valid = evaluation.obstacle_safe
            safety_score = evaluation.safety_score

        self._data.raw_unclipped_foothold_f[env_ids] = target_f
        self._data.nominal_feasible_velocity_f[env_ids] = (
            feasible_velocity_f
        )
        self._data.nominal_curriculum_residual_f[env_ids] = (
            curriculum_residual_f
        )
        self._data.nominal_curriculum_radius_f[env_ids] = (
            curriculum_radius_f
        )
        self._data.nominal_curriculum_usage[env_ids] = curriculum_usage
        self._data.nominal_foothold_w[env_ids] = target_w
        self._data.nominal_geometric_valid[env_ids] = geometric_valid
        self._data.nominal_safety_valid[env_ids] = safety_valid
        self._data.nominal_safety_score[env_ids] = safety_score
        self._data.nominal_foothold_prepared[env_ids] = True

    def _feasible_velocity_from_target(
        self,
        target_foothold_f: torch.Tensor,
        swing_side: torch.Tensor,
        yaw_velocity_f: torch.Tensor,
    ) -> torch.Tensor:
        """Infer the realized planar velocity from a final relative target."""
        side_sign = torch.where(
            swing_side == 0,
            torch.ones_like(swing_side),
            -torch.ones_like(swing_side),
        ).to(dtype=target_foothold_f.dtype)
        lookahead_s = self._flat_provider_cfg.velocity_lookahead_s
        realized_velocity_x = target_foothold_f[:, 0] / lookahead_s
        realized_velocity_y = (
            target_foothold_f[:, 1]
            - side_sign * self._flat_provider_cfg.nominal_step_width
        ) / lookahead_s
        return torch.stack(
            (
                realized_velocity_x,
                realized_velocity_y,
                yaw_velocity_f,
            ),
            dim=-1,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)

        if not hasattr(self, "_gait_state"):
            return

        if env_ids is None:
            reset_env_ids = slice(None)
            reset_env_id_tensor = torch.arange(
                self._num_envs,
                device=self._device,
                dtype=torch.long,
            )
        elif isinstance(env_ids, slice):
            reset_env_ids = env_ids
            reset_env_id_tensor = torch.arange(
                self._num_envs,
                device=self._device,
                dtype=torch.long,
            )[env_ids]
        else:
            reset_env_ids = env_ids
            reset_env_id_tensor = torch.as_tensor(
                env_ids,
                device=self._device,
                dtype=torch.long,
            )

        reset_state = initial_gait_state(
            num_envs=reset_env_id_tensor.shape[0],
            device=self._device,
            env_ids=reset_env_id_tensor,
        )
        self._write_gait_state(reset_env_ids, reset_state)

        if self._data.gait_mode is not None:
            self._data.gait_mode[reset_env_ids] = reset_state.mode
        if self._data.swing_side is not None:
            self._data.swing_side[reset_env_ids] = reset_state.swing_side
        if self._data.phase is not None:
            self._data.phase[reset_env_ids] = 0.0
        if self._data.default_swing_reference_pos_w is not None:
            self._data.default_swing_reference_pos_w[reset_env_ids] = 0.0
        if self._data.swing_reference_pos_w is not None:
            self._data.swing_reference_pos_w[reset_env_ids] = 0.0
        if self._data.swing_reference_vel_w is not None:
            self._data.swing_reference_vel_w[reset_env_ids] = 0.0
        if self._data.actual_swing_foot_vel_w is not None:
            self._data.actual_swing_foot_vel_w[reset_env_ids] = 0.0
        if self._data.default_swing_apex_height is not None:
            self._data.default_swing_apex_height[reset_env_ids] = 0.0
        if self._data.swing_apex_height is not None:
            self._data.swing_apex_height[reset_env_ids] = 0.0
        if self._data.swing_clearance_safe is not None:
            self._data.swing_clearance_safe[reset_env_ids] = True
        if self._data.swing_clearance_penetration is not None:
            self._data.swing_clearance_penetration[reset_env_ids] = 0.0
        if self._data.swing_clearance_deepest_phase is not None:
            self._data.swing_clearance_deepest_phase[reset_env_ids] = 0.0
        if self._data.swing_clearance_start_penetration is not None:
            self._data.swing_clearance_start_penetration[reset_env_ids] = 0.0
        if self._data.swing_clearance_goal_penetration is not None:
            self._data.swing_clearance_goal_penetration[reset_env_ids] = 0.0
        if self._data.swing_clearance_start_escape_safe is not None:
            self._data.swing_clearance_start_escape_safe[reset_env_ids] = False
        if self._data.touchdown_accepted is not None:
            self._data.touchdown_accepted[reset_env_ids] = False
        if self._data.touchdown_xy_error is not None:
            self._data.touchdown_xy_error[reset_env_ids] = 0.0
        if self._data.touchdown_z_error is not None:
            self._data.touchdown_z_error[reset_env_ids] = 0.0
        if self._data.touchdown_xy_ok is not None:
            self._data.touchdown_xy_ok[reset_env_ids] = False
        if self._data.touchdown_z_ok is not None:
            self._data.touchdown_z_ok[reset_env_ids] = False
        if self._data.touchdown_swing_contact is not None:
            self._data.touchdown_swing_contact[reset_env_ids] = False
        if self._data.touchdown_within_tolerance is not None:
            self._data.touchdown_within_tolerance[reset_env_ids] = False
        if self._data.swing_has_lifted is not None:
            self._data.swing_has_lifted[reset_env_ids] = False
        if self._data.recovery_step_active is not None:
            self._data.recovery_step_active[reset_env_ids] = False
        if self._data.confirmed_foot_contact is not None:
            self._data.confirmed_foot_contact[reset_env_ids] = False
        for name in (
            "body_tilt_rad",
            "body_angular_speed_rad_s",
            "body_horizontal_speed_m_s",
            "support_slip_m_s",
        ):
            value = getattr(self._data, name, None)
            if value is not None:
                value[reset_env_ids] = 0.0
        if self._data.stabilization_active is not None:
            self._data.stabilization_active[reset_env_ids] = False
        if self._data.stabilization_ready is not None:
            self._data.stabilization_ready[reset_env_ids] = False
        if self._data.event_response is not None:
            self._data.event_response[reset_env_ids] = EventResponse.NONE
        if self._data.planning_failure is not None:
            self._data.planning_failure[reset_env_ids] = False
        if self._data.planner_valid is not None:
            self._data.planner_valid[reset_env_ids] = True
        if self._data.safe_target_search_performed is not None:
            self._data.safe_target_search_performed[reset_env_ids] = False
        if self._data.safe_target_final_valid is not None:
            self._data.safe_target_final_valid[reset_env_ids] = True
        if self._data.safe_target_used_fallback is not None:
            self._data.safe_target_used_fallback[reset_env_ids] = False
        if self._data.safe_target_score is not None:
            self._data.safe_target_score[reset_env_ids] = 0.0
        if self._data.safe_target_nominal_inside_ellipse is not None:
            self._data.safe_target_nominal_inside_ellipse[reset_env_ids] = True
        if self._data.safe_target_nominal_obstacle_safe is not None:
            self._data.safe_target_nominal_obstacle_safe[reset_env_ids] = True
        if self._data.safe_target_nominal_valid is not None:
            self._data.safe_target_nominal_valid[reset_env_ids] = True
        if self._data.safe_target_candidate_count is not None:
            self._data.safe_target_candidate_count[reset_env_ids] = 0.0
        if self._data.safe_target_candidate_inside_ellipse_count is not None:
            self._data.safe_target_candidate_inside_ellipse_count[reset_env_ids] = 0.0
        if self._data.safe_target_candidate_obstacle_safe_count is not None:
            self._data.safe_target_candidate_obstacle_safe_count[reset_env_ids] = 0.0
        if self._data.safe_target_candidate_valid_count is not None:
            self._data.safe_target_candidate_valid_count[reset_env_ids] = 0.0
        if self._data.safe_target_final_max_penetration_depth is not None:
            self._data.safe_target_final_max_penetration_depth[
                reset_env_ids
            ] = 0.0
        if self._data.raw_unclipped_foothold_f is not None:
            self._data.raw_unclipped_foothold_f[reset_env_ids] = 0.0
        if self._data.flat_target_level is not None:
            self._data.flat_target_level[reset_env_ids] = 0
        if self._data.velocity_lookahead_s is not None:
            self._data.velocity_lookahead_s[reset_env_ids] = (
                self._flat_provider_cfg.velocity_lookahead_s
            )
        _clear_foothold_plan_buffers(self._data, reset_env_ids)
        clear_learned_foothold_buffers(self._data, reset_env_ids)
        if hasattr(self, "_previous_sole_valid"):
            self._previous_sole_valid[reset_env_ids] = False
            self._previous_left_sole_pos_w[reset_env_ids] = 0.0
            self._previous_right_sole_pos_w[reset_env_ids] = 0.0

    def _initialize_impl(self):
        super()._initialize_impl()

        self._generator = torch.Generator(device=self._device)

        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        robot_prim_path = self.cfg.prim_path
        template_robot_prim_path = (
            f"{self._parent_prims[0].GetPath().pathString}/"
            f"{robot_prim_path.rsplit('/', 1)[-1]}"
        )
        body_names = []
        for prim in sim_utils.get_all_matching_child_prims(
            template_robot_prim_path,
            predicate=lambda p: p.HasAPI(PhysxSchema.PhysxRigidBodyAPI),
            depth=1,
        ):
            body_names.append(prim.GetName())

        if not body_names:
            raise RuntimeError(
                "FootholdPlanner could not find any rigid bodies under "
                f"'{template_robot_prim_path}'."
            )

        left_ids, left_names = string_utils.resolve_matching_names(
            self.cfg.left_ankle_body_name,
            body_names,
            preserve_order=True,
        )
        right_ids, right_names = string_utils.resolve_matching_names(
            self.cfg.right_ankle_body_name,
            body_names,
            preserve_order=True,
        )
        base_ids, base_names = string_utils.resolve_matching_names(
            self.cfg.base_body_name,
            body_names,
            preserve_order=True,
        )

        if len(left_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one left ankle body, "
                f"but found {left_names} for pattern "
                f"{self.cfg.left_ankle_body_name!r}."
            )
        if len(right_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one right ankle body, "
                f"but found {right_names} for pattern "
                f"{self.cfg.right_ankle_body_name!r}."
            )
        if len(base_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one base body, "
                f"but found {base_names} for pattern "
                f"{self.cfg.base_body_name!r}."
            )

        required_body_names = (
            body_names[left_ids[0]],
            body_names[right_ids[0]],
            body_names[base_ids[0]],
        )
        body_paths_glob = _make_required_body_paths_glob(
            robot_prim_path,
            required_body_names,
        )

        self._robot_body_physx_view = (
            self._physics_sim_view.create_rigid_body_view(body_paths_glob)
        )
        self._num_robot_bodies = (
            self._robot_body_physx_view.count // self._num_envs
        )
        self._robot_body_names = [
            path.split("/")[-1]
            for path in self._robot_body_physx_view.prim_paths[
                : self._num_robot_bodies
            ]
        ]

        left_ids, left_names = string_utils.resolve_matching_names(
            self.cfg.left_ankle_body_name,
            self._robot_body_names,
            preserve_order=True,
        )
        right_ids, right_names = string_utils.resolve_matching_names(
            self.cfg.right_ankle_body_name,
            self._robot_body_names,
            preserve_order=True,
        )
        base_ids, base_names = string_utils.resolve_matching_names(
            self.cfg.base_body_name,
            self._robot_body_names,
            preserve_order=True,
        )

        if len(left_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one left ankle body in "
                "the reduced rigid-body PhysX view, "
                f"but found {left_names} for pattern "
                f"{self.cfg.left_ankle_body_name!r}."
            )
        if len(right_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one right ankle body in "
                "the reduced rigid-body PhysX view, "
                f"but found {right_names} for pattern "
                f"{self.cfg.right_ankle_body_name!r}."
            )
        if len(base_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one base body in "
                "the reduced rigid-body PhysX view, "
                f"but found {base_names} for pattern "
                f"{self.cfg.base_body_name!r}."
            )

        self._base_body_id = base_ids[0]
        self._left_ankle_body_id = left_ids[0]
        self._right_ankle_body_id = right_ids[0]

        contact_body_names = []
        for prim in sim_utils.get_all_matching_child_prims(
            template_robot_prim_path,
            predicate=lambda p: p.HasAPI(PhysxSchema.PhysxContactReportAPI),
            depth=1,
        ):
            contact_body_names.append(prim.GetName())

        if not contact_body_names:
            raise RuntimeError(
                "FootholdPlanner could not find any contact-reporting "
                f"bodies under '{template_robot_prim_path}'."
            )

        left_contact_ids, left_contact_names = (
            string_utils.resolve_matching_names(
                self.cfg.left_contact_body_name,
                contact_body_names,
                preserve_order=True,
            )
        )
        right_contact_ids, right_contact_names = (
            string_utils.resolve_matching_names(
                self.cfg.right_contact_body_name,
                contact_body_names,
                preserve_order=True,
            )
        )

        if len(left_contact_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one left contact body, "
                f"but found {left_contact_names} for pattern "
                f"{self.cfg.left_contact_body_name!r}."
            )
        if len(right_contact_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one right contact body, "
                f"but found {right_contact_names} for pattern "
                f"{self.cfg.right_contact_body_name!r}."
            )

        required_contact_body_names = (
            contact_body_names[left_contact_ids[0]],
            contact_body_names[right_contact_ids[0]],
        )
        contact_body_paths_glob = _make_required_body_paths_glob(
            robot_prim_path,
            required_contact_body_names,
        )

        self._contact_physx_view = (
            self._physics_sim_view.create_rigid_contact_view(
                contact_body_paths_glob,
                max_contact_data_count=len(required_contact_body_names)
                * self._num_envs,
            )
        )
        self._num_contact_bodies = (
            self._contact_physx_view.sensor_count // self._num_envs
        )
        self._contact_body_names = sorted(set(required_contact_body_names))

        left_contact_ids, left_contact_names = (
            string_utils.resolve_matching_names(
                self.cfg.left_contact_body_name,
                self._contact_body_names,
                preserve_order=True,
            )
        )
        right_contact_ids, right_contact_names = (
            string_utils.resolve_matching_names(
                self.cfg.right_contact_body_name,
                self._contact_body_names,
                preserve_order=True,
            )
        )

        if len(left_contact_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one left contact body in "
                "the reduced contact PhysX view, "
                f"but found {left_contact_names} for pattern "
                f"{self.cfg.left_contact_body_name!r}."
            )
        if len(right_contact_ids) != 1:
            raise RuntimeError(
                "FootholdPlanner expected exactly one right contact body in "
                "the reduced contact PhysX view, "
                f"but found {right_contact_names} for pattern "
                f"{self.cfg.right_contact_body_name!r}."
            )

        self._left_contact_body_id = left_contact_ids[0]
        self._right_contact_body_id = right_contact_ids[0]

        self._data.gait_mode = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.long,
        )
        self._data.swing_side = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.long,
        )
        self._data.phase = torch.zeros(
            self._num_envs,
            device=self._device,
        )

        self._data.target_foothold_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.target_foothold_f = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.learned_foothold_enabled = torch.full(
            (self._num_envs,),
            self.cfg.enable_learned_foothold,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_action_normalized = torch.zeros(
            self._num_envs,
            2,
            device=self._device,
        )
        self._data.learned_foothold_decoded_f = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.learned_foothold_prepared_f = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.learned_foothold_prepared_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.learned_foothold_prepared_valid = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_lock_geometric_valid = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.target_terrain_valid = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_locked = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_target_f = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.learned_foothold_target_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.learned_foothold_used = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_height_valid = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_geometric_valid = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_safety_valid = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_evaluated = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_transaction_evaluated = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_event_generation = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.int64,
        )
        self._data.learned_foothold_route_event = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_route_use_nominal = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_route_use_learned = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_route_initial_executable = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.learned_foothold_route_outcome = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.long,
        )
        self._data.learned_foothold_safety_score = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.learned_foothold_safety_margin_score = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.learned_foothold_minimum_signed_clearance = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.learned_foothold_penetrating_point_count = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.learned_foothold_penetrating_point_ratio = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.learned_foothold_total_penetration_depth = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._safe_target_foot_points_xy = make_sole_perimeter_points_xy(
            foot_length=self.cfg.safe_target_foot_length_m,
            foot_width=self.cfg.safe_target_foot_width_m,
            num_x=self.cfg.safe_target_foot_grid_num_x,
            num_y=self.cfg.safe_target_foot_grid_num_y,
            device=self._device,
            dtype=torch.float,
        )
        self._data.desired_velocity_f = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.feasible_velocity_f = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.safe_target_search_performed = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.safe_target_final_valid = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.safe_target_used_fallback = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.safe_target_score = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.safe_target_nominal_inside_ellipse = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.safe_target_nominal_obstacle_safe = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.safe_target_nominal_valid = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.safe_target_candidate_count = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.safe_target_candidate_inside_ellipse_count = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.safe_target_candidate_obstacle_safe_count = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.safe_target_candidate_valid_count = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.safe_target_final_max_penetration_depth = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.raw_unclipped_foothold_f = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.nominal_foothold_prepared = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.nominal_feasible_velocity_f = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.nominal_curriculum_residual_f = torch.zeros(
            self._num_envs,
            2,
            device=self._device,
        )
        self._data.nominal_curriculum_radius_f = torch.zeros(
            self._num_envs,
            2,
            device=self._device,
        )
        self._data.nominal_curriculum_usage = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.nominal_frame_origin_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.nominal_frame_yaw_w = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.nominal_foothold_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.nominal_geometric_valid = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.nominal_safety_valid = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.nominal_safety_score = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.flat_target_level = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.long,
        )
        self._data.velocity_lookahead_s = torch.full(
            (self._num_envs,),
            self._flat_provider_cfg.velocity_lookahead_s,
            device=self._device,
        )
        self._data.target_delta_f = torch.zeros(
            self._num_envs,
            2,
            device=self._device,
        )
        self._data.curriculum_residual_f = torch.zeros(
            self._num_envs,
            2,
            device=self._device,
        )
        self._data.curriculum_radius_f = torch.zeros(
            self._num_envs,
            2,
            device=self._device,
        )
        self._data.curriculum_usage = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.target_ellipse_max_x = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.target_ellipse_usage = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.default_swing_reference_pos_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.swing_reference_pos_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.swing_reference_vel_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.swing_duration_s = torch.full(
            (self._num_envs,),
            self.cfg.swing_duration_s,
            device=self._device,
        )
        self._data.default_swing_apex_height = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.swing_apex_height = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.swing_clearance_safe = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.swing_clearance_penetration = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.swing_clearance_deepest_phase = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.swing_clearance_start_penetration = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.swing_clearance_goal_penetration = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.swing_clearance_start_escape_safe = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.swing_preflight_safe = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.swing_preflight_ready = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.actual_stance_foot_pos_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.actual_swing_foot_pos_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.actual_swing_foot_vel_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.swing_start_pos_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._data.foot_contact = torch.zeros(
            self._num_envs,
            2,
            device=self._device,
            dtype=torch.bool,
        )
        self._previous_left_sole_pos_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._previous_right_sole_pos_w = torch.zeros(
            self._num_envs,
            3,
            device=self._device,
        )
        self._previous_sole_valid = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.confirmed_foot_contact = torch.zeros(
            self._num_envs,
            2,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.body_tilt_rad = torch.zeros(self._num_envs, device=self._device)
        self._data.body_angular_speed_rad_s = torch.zeros(
            self._num_envs, device=self._device
        )
        self._data.body_horizontal_speed_m_s = torch.zeros(
            self._num_envs, device=self._device
        )
        self._data.support_slip_m_s = torch.zeros(
            self._num_envs, device=self._device
        )
        self._data.stabilization_active = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.bool
        )
        self._data.stabilization_ready = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.bool
        )
        self._data.event_response = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self._data.planning_failure = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.bool
        )

        self._data.touchdown_accepted = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.touchdown_xy_error = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.touchdown_z_error = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._data.touchdown_xy_ok = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.touchdown_z_ok = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.touchdown_swing_contact = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.touchdown_within_tolerance = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.swing_has_lifted = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.recovery_step_active = torch.zeros(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._data.planner_valid = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
        )
        self._flat_target_curriculum_scale = torch.zeros(
            self._num_envs,
            device=self._device,
        )
        self._gait_state = initial_gait_state(
            num_envs=self._num_envs,
            device=self._device,
        )

    def _select_gait_state(
        self,
        env_ids,
    ) -> GaitMachineState:
        return GaitMachineState(
            mode=self._gait_state.mode[env_ids],
            swing_side=self._gait_state.swing_side[env_ids],
            elapsed_s=self._gait_state.elapsed_s[env_ids],
            hold_elapsed_s=self._gait_state.hold_elapsed_s[env_ids],
            hold_required_s=self._gait_state.hold_required_s[env_ids],
            contact_elapsed_s=self._gait_state.contact_elapsed_s[env_ids],
            no_contact_elapsed_s=self._gait_state.no_contact_elapsed_s[
                env_ids
            ],
            swing_has_lifted=self._gait_state.swing_has_lifted[env_ids],
            recovery_step_pending=self._gait_state.recovery_step_pending[
                env_ids
            ],
            recovery_step_active=self._gait_state.recovery_step_active[
                env_ids
            ],
            stabilization_elapsed_s=(
                None
                if self._gait_state.stabilization_elapsed_s is None
                else self._gait_state.stabilization_elapsed_s[env_ids]
            ),
            late_search_elapsed_s=(
                None
                if self._gait_state.late_search_elapsed_s is None
                else self._gait_state.late_search_elapsed_s[env_ids]
            ),
            planning_failure=(
                None
                if self._gait_state.planning_failure is None
                else self._gait_state.planning_failure[env_ids]
            ),
        )

    def _write_gait_state(
        self,
        env_ids,
        state: GaitMachineState,
    ) -> None:
        self._gait_state.mode[env_ids] = state.mode
        self._gait_state.swing_side[env_ids] = state.swing_side
        self._gait_state.elapsed_s[env_ids] = state.elapsed_s
        self._gait_state.hold_elapsed_s[env_ids] = state.hold_elapsed_s
        self._gait_state.hold_required_s[env_ids] = state.hold_required_s
        self._gait_state.contact_elapsed_s[env_ids] = (
            state.contact_elapsed_s
        )
        self._gait_state.no_contact_elapsed_s[env_ids] = (
            state.no_contact_elapsed_s
        )
        self._gait_state.swing_has_lifted[env_ids] = state.swing_has_lifted
        self._gait_state.recovery_step_pending[env_ids] = (
            state.recovery_step_pending
        )
        self._gait_state.recovery_step_active[env_ids] = (
            state.recovery_step_active
        )
        if (
            self._gait_state.stabilization_elapsed_s is not None
            and state.stabilization_elapsed_s is not None
        ):
            self._gait_state.stabilization_elapsed_s[env_ids] = (
                state.stabilization_elapsed_s
            )
        if (
            self._gait_state.late_search_elapsed_s is not None
            and state.late_search_elapsed_s is not None
        ):
            self._gait_state.late_search_elapsed_s[env_ids] = (
                state.late_search_elapsed_s
            )
        if (
            self._gait_state.planning_failure is not None
            and state.planning_failure is not None
        ):
            self._gait_state.planning_failure[env_ids] = state.planning_failure

    def _startup_hold_mask(self, selected_env_ids: torch.Tensor) -> torch.Tensor:
        if self.cfg.startup_hold_s <= 0.0:
            return torch.zeros(
                selected_env_ids.shape[0],
                device=self._device,
                dtype=torch.bool,
            )
        return self._timestamp[selected_env_ids] < self.cfg.startup_hold_s - 1.0e-6

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = slice(None)

        if isinstance(env_ids, slice):
            selected_env_ids = torch.arange(
                self._num_envs,
                device=self._device,
            )[env_ids]
        else:
            selected_env_ids = torch.as_tensor(
                env_ids,
                device=self._device,
                dtype=torch.long,
            )
        startup_hold_mask = self._startup_hold_mask(selected_env_ids)

        assert self._data.planner_valid is not None
        assert self._data.touchdown_accepted is not None
        assert self._data.actual_stance_foot_pos_w is not None
        assert self._data.actual_swing_foot_pos_w is not None
        assert self._data.swing_start_pos_w is not None
        assert self._data.swing_side is not None
        assert self._data.target_foothold_f is not None
        assert self._data.desired_velocity_f is not None
        assert self._data.feasible_velocity_f is not None
        assert self._data.target_foothold_w is not None
        assert self._data.gait_mode is not None
        assert self._data.phase is not None
        assert self._data.swing_reference_pos_w is not None
        assert self._data.swing_reference_vel_w is not None
        assert self._data.swing_duration_s is not None
        assert self._data.actual_swing_foot_vel_w is not None
        assert self._data.foot_contact is not None
        assert self._data.touchdown_xy_error is not None
        assert self._data.touchdown_z_error is not None
        assert self._data.touchdown_xy_ok is not None
        assert self._data.touchdown_z_ok is not None
        assert self._data.touchdown_swing_contact is not None
        assert self._data.touchdown_within_tolerance is not None
        assert self._data.swing_has_lifted is not None
        assert self._data.recovery_step_active is not None

        self._data.planner_valid[env_ids] = True
        if self._data.learned_foothold_evaluated is not None:
            self._data.learned_foothold_evaluated[env_ids] = False
        if self._data.learned_foothold_route_event is not None:
            self._data.learned_foothold_route_event[env_ids] = False
        if self._data.learned_foothold_route_use_nominal is not None:
            self._data.learned_foothold_route_use_nominal[env_ids] = False
        if self._data.learned_foothold_route_use_learned is not None:
            self._data.learned_foothold_route_use_learned[env_ids] = False
        if (
            self._data.learned_foothold_route_initial_executable
            is not None
        ):
            self._data.learned_foothold_route_initial_executable[
                env_ids
            ] = False
        if self._data.learned_foothold_route_outcome is not None:
            self._data.learned_foothold_route_outcome[env_ids] = 0
        _clear_safe_target_event_buffers(self._data, env_ids)

        robot_body_poses_w = self._robot_body_physx_view.get_transforms().view(
            self._num_envs,
            self._num_robot_bodies,
            7,
        )[env_ids]
        left_ankle_pos_w = robot_body_poses_w[
            :,
            self._left_ankle_body_id,
            :3,
        ]
        right_ankle_pos_w = robot_body_poses_w[
            :,
            self._right_ankle_body_id,
            :3,
        ]

        left_ankle_quat_w = convert_quat(
            robot_body_poses_w[:, self._left_ankle_body_id, 3:],
            to="wxyz",
        )
        right_ankle_quat_w = convert_quat(
            robot_body_poses_w[:, self._right_ankle_body_id, 3:],
            to="wxyz",
        )
        base_quat_w = convert_quat(
            robot_body_poses_w[:, self._base_body_id, 3:],
            to="wxyz",
        )
        base_yaw_w = _yaw_from_quat_wxyz(base_quat_w)

        left_sole_pos_w = self._sole_geometry.center_world(
            left_ankle_pos_w,
            left_ankle_quat_w,
        )
        right_sole_pos_w = self._sole_geometry.center_world(
            right_ankle_pos_w,
            right_ankle_quat_w,
        )

        stance_sole_pos_w, swing_sole_pos_w = _select_sole_roles(
            left_sole_w=left_sole_pos_w,
            right_sole_w=right_sole_pos_w,
            swing_side=self._data.swing_side[env_ids],
        )
        self._data.actual_stance_foot_pos_w[env_ids] = stance_sole_pos_w
        self._data.actual_swing_foot_pos_w[env_ids] = swing_sole_pos_w

        foot_target_error = torch.linalg.norm(
            (
                self._data.actual_swing_foot_pos_w[env_ids]
                - self._data.target_foothold_w[env_ids]
            )[:, :2],
            dim=-1,
        )
        foot_height_error = torch.abs(
            self._data.actual_swing_foot_pos_w[env_ids, 2]
            - self._data.target_foothold_w[env_ids, 2]
        )

        net_contact_forces_w = (
            self._contact_physx_view.get_net_contact_forces(
                dt=self._sim_physics_dt,
            ).view(
                self._num_envs,
                self._num_contact_bodies,
                3,
            )[env_ids]
        )
        foot_contact_forces_w = net_contact_forces_w[
            :,
            [self._left_contact_body_id, self._right_contact_body_id],
            :,
        ]
        contact = (
            torch.linalg.norm(foot_contact_forces_w, dim=-1)
            > self.cfg.contact_force_threshold_n
        )
        self._data.foot_contact[env_ids] = contact

        # Stability diagnostics use the same reduced PhysX body view as the
        # planner.  Velocities are world-frame [linear, angular] values, and
        # the tilt is derived from the pelvis z-axis rather than a guessed
        # Euler-angle convention.
        robot_body_vels_w = self._robot_body_physx_view.get_velocities().view(
            self._num_envs,
            self._num_robot_bodies,
            6,
        )[env_ids]
        base_vel_w = robot_body_vels_w[:, self._base_body_id]
        left_ankle_velocity_w = robot_body_vels_w[:, self._left_ankle_body_id]
        right_ankle_velocity_w = robot_body_vels_w[:, self._right_ankle_body_id]
        left_sole_velocity_w = self._sole_geometry.center_velocity_world(
            left_ankle_velocity_w[:, :3],
            left_ankle_velocity_w[:, 3:],
            left_ankle_quat_w,
        )
        right_sole_velocity_w = self._sole_geometry.center_velocity_world(
            right_ankle_velocity_w[:, :3],
            right_ankle_velocity_w[:, 3:],
            right_ankle_quat_w,
        )
        base_quat = base_quat_w
        pelvis_z_world_z = 1.0 - 2.0 * (
            base_quat[:, 1].square() + base_quat[:, 2].square()
        )
        body_tilt_rad = torch.acos(
            pelvis_z_world_z.clamp(min=-1.0, max=1.0)
        )
        self._data.body_tilt_rad[env_ids] = body_tilt_rad
        self._data.body_angular_speed_rad_s[env_ids] = torch.linalg.vector_norm(
            base_vel_w[:, 3:],
            dim=-1,
        )
        self._data.body_horizontal_speed_m_s[env_ids] = torch.linalg.vector_norm(
            base_vel_w[:, :2],
            dim=-1,
        )
        previous_valid = self._previous_sole_valid[env_ids]
        left_slip = torch.linalg.vector_norm(
            left_sole_pos_w[:, :2]
            - self._previous_left_sole_pos_w[env_ids, :2],
            dim=-1,
        ) / self.cfg.control_dt_s
        right_slip = torch.linalg.vector_norm(
            right_sole_pos_w[:, :2]
            - self._previous_right_sole_pos_w[env_ids, :2],
            dim=-1,
        ) / self.cfg.control_dt_s
        support_side_for_slip = 1 - self._gait_state.swing_side[env_ids]
        support_slip = torch.where(
            support_side_for_slip == 0,
            left_slip,
            right_slip,
        )
        self._data.support_slip_m_s[env_ids] = torch.where(
            previous_valid,
            support_slip,
            torch.zeros_like(support_slip),
        )
        self._previous_left_sole_pos_w[env_ids] = left_sole_pos_w
        self._previous_right_sole_pos_w[env_ids] = right_sole_pos_w
        self._previous_sole_valid[env_ids] = True

        selected_env_count = contact.shape[0]
        selected_rows = torch.arange(
            selected_env_count,
            device=self._device,
        )
        current_swing_side = self._data.swing_side[env_ids]
        swing_foot_contact = contact[
            selected_rows,
            current_swing_side,
        ]
        touchdown_xy_ok = (
            foot_target_error <= self.cfg.touchdown_xy_tolerance_m
        )
        touchdown_z_ok = (
            foot_height_error <= self.cfg.touchdown_z_tolerance_m
        )
        self._data.touchdown_xy_error[env_ids] = foot_target_error
        self._data.touchdown_z_error[env_ids] = foot_height_error
        self._data.touchdown_xy_ok[env_ids] = touchdown_xy_ok
        self._data.touchdown_z_ok[env_ids] = touchdown_z_ok
        self._data.touchdown_swing_contact[env_ids] = swing_foot_contact
        self._data.touchdown_within_tolerance[env_ids] = (
            touchdown_xy_ok & touchdown_z_ok
        )
        current_swing_has_lifted = self._gait_state.swing_has_lifted[
            env_ids
        ]
        self._data.swing_has_lifted[env_ids] = current_swing_has_lifted
        self._data.touchdown_accepted[env_ids] = (
            swing_foot_contact
            & current_swing_has_lifted
            & touchdown_xy_ok
            & touchdown_z_ok
        )
        previous_gait_state = self._select_gait_state(env_ids)
        step_hold_s = torch.zeros_like(
            previous_gait_state.hold_required_s,
        )
        next_contact_elapsed_s = torch.where(
            contact,
            previous_gait_state.contact_elapsed_s + self.cfg.control_dt_s,
            torch.zeros_like(previous_gait_state.contact_elapsed_s),
        )
        next_no_contact_elapsed_s = torch.where(
            contact,
            torch.zeros_like(previous_gait_state.no_contact_elapsed_s),
            previous_gait_state.no_contact_elapsed_s
            + self.cfg.control_dt_s,
        )
        confirmed_contact = (
            next_contact_elapsed_s
            >= self.cfg.contact_confirm_s - 1.0e-6
        )
        self._data.confirmed_foot_contact[env_ids] = confirmed_contact
        both_contacts_confirmed = torch.all(
            confirmed_contact,
            dim=-1,
        )
        stance_side = 1 - previous_gait_state.swing_side
        new_support_confirmed = confirmed_contact[
            selected_rows,
            stance_side,
        ]
        confirmed_contact_lost = (
            next_no_contact_elapsed_s
            >= self.cfg.hold_contact_lost_confirm_s - 1.0e-6
        )
        any_contact_lost = torch.any(
            confirmed_contact_lost,
            dim=-1,
        )
        new_support_lost = confirmed_contact_lost[
            selected_rows,
            stance_side,
        ]
        initial_stabilization_hold = (
            (previous_gait_state.mode == GaitState.HOLD)
            & ~previous_gait_state.recovery_step_pending
            & (
                (previous_gait_state.hold_required_s < 0.0)
                | (
                    previous_gait_state.hold_required_s
                    >= self.cfg.reset_hold_s - 1.0e-6
                )
            )
        )
        recovery_mode = previous_gait_state.mode == GaitState.RECOVERY
        recovery_hold = (
            recovery_mode
            | previous_gait_state.recovery_step_pending
        )
        strict_double_support = (
            startup_hold_mask
            | initial_stabilization_hold
        )
        hold_contact_ready = torch.where(
            strict_double_support,
            both_contacts_confirmed,
            torch.where(
                recovery_hold,
                torch.any(confirmed_contact, dim=-1),
                new_support_confirmed,
            ),
        )
        hold_contact_lost = torch.where(
            strict_double_support,
            any_contact_lost,
            # During a recovery step, the non-support foot is expected to be
            # airborne.  Only the selected support foot may trigger contact
            # loss; using any_contact_lost here would immediately re-enter
            # recovery whenever the swing foot stayed lifted.
            new_support_lost,
        )

        # A support loss invalidates the pending recovery transaction.  Clear
        # it before reading the cache so the next confirmed support relation
        # gets a fresh analytic target in the new support frame.  Active
        # swings keep their locked target until the state machine reports the
        # failure; only HOLD/recovery planning is cleared here.
        assert self._data.planning_failure is not None
        discard_pending_plan = self._data.planning_failure[
            selected_env_ids
        ].clone() | (
            (
                (previous_gait_state.mode == GaitState.HOLD)
                | (previous_gait_state.mode == GaitState.RECOVERY)
            )
            & hold_contact_lost
        ) | (
            (previous_gait_state.mode == GaitState.RECOVERY)
            & ~previous_gait_state.recovery_step_pending
        )
        if torch.any(discard_pending_plan).item():
            discard_env_ids = selected_env_ids[discard_pending_plan]
            _clear_foothold_plan_buffers(self._data, discard_env_ids)
            clear_learned_foothold_buffers(self._data, discard_env_ids)

        nominal_ready_before_update = torch.ones_like(
            both_contacts_confirmed,
        )
        if self.cfg.enable_learned_foothold:
            assert self._data.nominal_foothold_prepared is not None
            nominal_ready_before_update = (
                self._data.nominal_foothold_prepared[env_ids].clone()
            )
            prepare_nominal = nominal_foothold_prepare_mask(
                hold=previous_gait_state.mode == GaitState.HOLD,
                hold_contact_ready=hold_contact_ready,
                nominal_ready=nominal_ready_before_update,
                startup_hold=startup_hold_mask,
            )
            # A single-support recovery HOLD is a real one-shot planning
            # transaction.  Its nominal recovery target is prepared with the
            # same frozen support frame as a normal HOLD; the learned route
            # may then replace it after the next control-cycle evaluation.
            if self.cfg.enable_contact_adaptive_recovery:
                prepare_nominal &= ~recovery_mode
            if torch.any(prepare_nominal).item():
                recovery_step = previous_gait_state.recovery_step_pending[
                    prepare_nominal
                ].clone()
                self._prepare_nominal_footholds(
                    env_ids=selected_env_ids[prepare_nominal],
                    swing_side=previous_gait_state.swing_side[
                        prepare_nominal
                    ],
                    recovery_step=recovery_step,
                    stance_pos_w=self._data.actual_stance_foot_pos_w[
                        selected_env_ids[prepare_nominal]
                    ],
                    base_yaw_w=base_yaw_w[prepare_nominal],
                )
        prepare_learned, _ = learned_foothold_event_masks(
            hold=previous_gait_state.mode == GaitState.HOLD,
            hold_contact_ready=hold_contact_ready,
            nominal_ready=nominal_ready_before_update,
            new_swing=torch.zeros_like(both_contacts_confirmed),
            enable=self.cfg.enable_learned_foothold,
        )
        prepare_learned &= ~startup_hold_mask
        # Legacy recovery steps remain analytic-only.  Contact-adaptive
        # recovery is different: after a confirmed single support it enters a
        # HOLD transaction and evaluates one learned proposal for the other
        # foot.
        if not self.cfg.enable_contact_adaptive_recovery:
            prepare_learned &= ~previous_gait_state.recovery_step_pending
        if self.cfg.enable_contact_adaptive_recovery:
            prepare_learned &= ~recovery_mode
        # Evaluate at most one learned action per HOLD transaction. An unsafe
        # action keeps its single PPO penalty, but it never becomes an
        # executable foothold. Recovery uses the same 3-D safety gate as
        # normal walking; its separate responsibility is contact stabilization.
        if self.cfg.enable_learned_foothold:
            assert (
                self._data.learned_foothold_transaction_evaluated is not None
            )
            assert self._data.learned_foothold_prepared_valid is not None
            assert self._data.learned_foothold_prepared_w is not None
            assert self._data.learned_foothold_geometric_valid is not None
            assert self._data.learned_foothold_safety_valid is not None
            assert self._data.swing_preflight_ready is not None
            assert self._data.swing_preflight_safe is not None
            nominal_route_ready = (
                self._data.nominal_foothold_prepared[selected_env_ids]
                & self._data.nominal_geometric_valid[selected_env_ids]
                & self._data.nominal_safety_valid[selected_env_ids]
            )
            already_usable = learned_foothold_transaction_ready(
                nominal_route_ready=nominal_route_ready,
                transaction_evaluated=(
                    self._data.learned_foothold_transaction_evaluated[
                        selected_env_ids
                    ]
                ),
                learned_prepared_valid=self._data.learned_foothold_prepared_valid[
                    selected_env_ids
                ],
                learned_geometric_valid=self._data.learned_foothold_geometric_valid[
                    selected_env_ids
                ],
                learned_safety_valid=self._data.learned_foothold_safety_valid[
                    selected_env_ids
                ],
            )
            if self.cfg.enable_contact_adaptive_recovery:
                # A recovery transaction is one-shot even when its learned
                # proposal is geometrically invalid.  Re-sampling it every
                # control step would sever the action-to-outcome relation and
                # flood PPO with duplicate events.
                already_usable = torch.where(
                    previous_gait_state.recovery_step_pending,
                    self._data.learned_foothold_transaction_evaluated[
                        selected_env_ids
                    ],
                    already_usable,
                )
            prepare_learned &= ~already_usable
        if torch.any(prepare_learned).item():
            prepare_env_ids = selected_env_ids[prepare_learned]
            assert self._data.nominal_frame_origin_w is not None
            assert self._data.nominal_frame_yaw_w is not None
            assert self._data.swing_preflight_ready is not None
            # A new high-level proposal replaces every result derived from
            # the previous proposal. Its clearance must be checked afresh.
            self._data.swing_preflight_ready[prepare_env_ids] = False
            self._prepare_learned_footholds(
                env_ids=prepare_env_ids,
                stance_pos_w=self._data.nominal_frame_origin_w[
                    prepare_env_ids
                ],
                base_yaw_w=self._data.nominal_frame_yaw_w[
                    prepare_env_ids
                ],
            )

        swing_ready = torch.ones_like(
            both_contacts_confirmed,
            dtype=torch.bool,
        )
        plan_wait_expired = torch.zeros_like(swing_ready)
        if self.cfg.enable_learned_foothold:
            assert self._data.nominal_foothold_prepared is not None
            assert self._data.nominal_geometric_valid is not None
            assert self._data.nominal_safety_valid is not None
            assert (
                self._data.learned_foothold_transaction_evaluated is not None
            )
            assert self._data.learned_foothold_prepared_valid is not None
            assert self._data.learned_foothold_geometric_valid is not None
            nominal_route_ready = (
                self._data.nominal_foothold_prepared[selected_env_ids]
                & self._data.nominal_geometric_valid[selected_env_ids]
                & self._data.nominal_safety_valid[selected_env_ids]
            )
            swing_ready = learned_foothold_swing_ready(
                nominal_route_ready=nominal_route_ready,
                transaction_evaluated=(
                    self._data.learned_foothold_transaction_evaluated[
                        selected_env_ids
                    ]
                ),
                learned_prepared_valid=(
                    self._data.learned_foothold_prepared_valid[
                        selected_env_ids
                    ]
                ),
                learned_geometric_valid=(
                    self._data.learned_foothold_geometric_valid[
                        selected_env_ids
                    ]
                ),
                learned_safety_valid=(
                    self._data.learned_foothold_safety_valid[
                        selected_env_ids
                    ]
                ),
                recovery_step=previous_gait_state.recovery_step_pending,
            )
            if self.cfg.enable_contact_adaptive_recovery:
                swing_ready &= ~recovery_mode
            plan_wait_expired = (
                self._data.nominal_foothold_prepared[selected_env_ids]
                & self._data.learned_foothold_transaction_evaluated[
                    selected_env_ids
                ]
                & ~swing_ready
            )

        # Preflight the complete swing transaction while still in HOLD.  The
        # same frozen nominal support frame and world target are used again at
        # the transition below, so clearance cannot retroactively invalidate an
        # already-started swing.
        if self.cfg.enable_learned_foothold:
            assert self._data.swing_preflight_safe is not None
            assert self._data.swing_preflight_ready is not None
            assert self._data.nominal_foothold_prepared is not None
            assert self._data.nominal_geometric_valid is not None
            assert self._data.nominal_safety_valid is not None
            assert self._data.nominal_foothold_w is not None
            assert self._data.nominal_frame_origin_w is not None
            assert self._data.nominal_frame_yaw_w is not None
            assert (
                self._data.learned_foothold_transaction_evaluated is not None
            )
            assert self._data.learned_foothold_prepared_valid is not None
            assert self._data.learned_foothold_prepared_w is not None
            assert self._data.learned_foothold_geometric_valid is not None
            assert self._data.learned_foothold_safety_valid is not None
            assert self._data.learned_foothold_target_w is not None
            preflight_window = (
                (previous_gait_state.mode == GaitState.HOLD)
                & hold_contact_ready
            )
            preflight_route = route_nominal_and_learned_footholds(
                nominal_geometric_valid=(
                    self._data.nominal_geometric_valid[selected_env_ids]
                ),
                nominal_safety_valid=(
                    self._data.nominal_safety_valid[selected_env_ids]
                ),
                learned_prepared=(
                    self._data.learned_foothold_prepared_valid[selected_env_ids]
                ),
                learned_geometric_valid=(
                    self._data.learned_foothold_geometric_valid[selected_env_ids]
                ),
                learned_safety_valid=(
                    self._data.learned_foothold_safety_valid[selected_env_ids]
                ),
                recovery_step=previous_gait_state.recovery_step_pending,
            )
            # Preserve a failed normal-walk preflight through the reward step
            # that owns the learned event. Recovery treats obstacle clearance
            # as a soft learning signal, so it is not converted into a failed
            # route below.
            retry_failed_preflight = (
                preflight_window
                & self._data.learned_foothold_transaction_evaluated[
                    selected_env_ids
                ]
                & self._data.swing_preflight_ready[selected_env_ids]
                & ~self._data.swing_preflight_safe[selected_env_ids]
                & self._data.nominal_geometric_valid[selected_env_ids]
                & self._data.nominal_safety_valid[selected_env_ids]
            )
            if torch.any(retry_failed_preflight).item():
                retry_preflight_env_ids = selected_env_ids[
                    retry_failed_preflight
                ]
                self._data.swing_preflight_ready[
                    retry_preflight_env_ids
                ] = False
            preflight_candidate = (
                preflight_window
                & preflight_route.executable
                & ~self._data.swing_preflight_ready[selected_env_ids]
            )
            if torch.any(preflight_candidate).item():
                preflight_env_ids = selected_env_ids[preflight_candidate]
                preflight_target_w = select_preflight_target_w(
                    route_use_learned=(
                        preflight_route.use_learned[preflight_candidate]
                    ),
                    learned_prepared_w=(
                        self._data.learned_foothold_prepared_w[
                            preflight_env_ids
                        ]
                    ),
                    nominal_target_w=(
                        self._data.nominal_foothold_w[preflight_env_ids]
                    ),
                )
                preflight_start_w = self._data.actual_swing_foot_pos_w[
                    preflight_env_ids
                ]
                # Lock the exact trajectory start while HOLD is still stable.
                # The later SWING transaction consumes this snapshot instead
                # of rereading a foot position from a moved support frame.
                self._data.swing_start_pos_w[preflight_env_ids] = (
                    preflight_start_w
                )
                preflight_yaw_w = self._data.nominal_frame_yaw_w[
                    preflight_env_ids
                ]
                preflight_safe = torch.ones(
                    preflight_env_ids.shape[0],
                    device=self._device,
                    dtype=torch.bool,
                )
                preflight_default_apex = torch.full(
                    (preflight_env_ids.shape[0],),
                    self.cfg.swing_apex_height_m,
                    device=self._device,
                    dtype=preflight_target_w.dtype,
                )
                preflight_apex = preflight_default_apex.clone()
                preflight_penetration = torch.zeros_like(preflight_apex)
                preflight_deepest_phase = torch.zeros_like(preflight_apex)
                preflight_start_penetration = torch.zeros_like(preflight_apex)
                preflight_goal_penetration = torch.zeros_like(preflight_apex)
                preflight_start_escape_safe = torch.zeros_like(
                    preflight_safe,
                )
                edge_obstacle = self._virtual_obstacles.get("edges")
                if self.cfg.enable_edge_clearance and edge_obstacle is not None:
                    adjustment = adjust_apex_for_edge_clearance(
                        obstacle=cast("ClearanceObstacle", edge_obstacle),
                        start=preflight_start_w,
                        goal=preflight_target_w,
                        default_apex_height=preflight_default_apex,
                        max_apex_height=self.cfg.clearance_max_apex_height_m,
                        apex_step=self.cfg.clearance_apex_step_m,
                        sample_spacing=self.cfg.clearance_sample_spacing_m,
                        swing_duration_s=self.cfg.swing_duration_s,
                        foot_points_xy=self._safe_target_foot_points_xy,
                        foot_yaw_w=preflight_yaw_w,
                        allow_start_penetration_escape=True,
                        goal_max_penetrating_points=(
                            self.cfg.safe_target_max_penetrating_points
                        ),
                    )
                    obstacle_preflight_safe = adjustment.is_safe
                    preflight_safe = obstacle_preflight_safe
                    preflight_apex = adjustment.apex_height
                    preflight_penetration = (
                        adjustment.penetration.max_penetration_depth
                    )
                    preflight_deepest_phase = (
                        adjustment.penetration.deepest_phase
                    )
                    preflight_start_penetration = (
                        adjustment.penetration.start_penetration_depth
                    )
                    preflight_goal_penetration = (
                        adjustment.penetration.goal_penetration_depth
                    )
                    preflight_start_escape_safe = (
                        adjustment.penetration.start_escape_safe
                    )
                self._data.swing_preflight_safe[preflight_env_ids] = preflight_safe
                self._data.swing_preflight_ready[preflight_env_ids] = True
                assert self._data.default_swing_apex_height is not None
                assert self._data.swing_apex_height is not None
                assert self._data.swing_clearance_penetration is not None
                assert self._data.swing_clearance_deepest_phase is not None
                assert self._data.swing_clearance_start_penetration is not None
                assert self._data.swing_clearance_goal_penetration is not None
                assert self._data.swing_clearance_start_escape_safe is not None
                self._data.default_swing_apex_height[preflight_env_ids] = (
                    preflight_default_apex
                )
                self._data.swing_apex_height[preflight_env_ids] = preflight_apex
                self._data.swing_clearance_safe[preflight_env_ids] = preflight_safe
                self._data.swing_clearance_penetration[preflight_env_ids] = (
                    preflight_penetration
                )
                self._data.swing_clearance_deepest_phase[preflight_env_ids] = (
                    preflight_deepest_phase
                )
                self._data.swing_clearance_start_penetration[
                    preflight_env_ids
                ] = preflight_start_penetration
                self._data.swing_clearance_goal_penetration[preflight_env_ids] = (
                    preflight_goal_penetration
                )
                self._data.swing_clearance_start_escape_safe[
                    preflight_env_ids
                ] = preflight_start_escape_safe
                failed_preflight = preflight_env_ids[~preflight_safe]
                if failed_preflight.numel() > 0:
                    # Keep the learned action's single PPO outcome, but remove
                    # a failed normal-walk action from execution routing. The
                    # persistent HOLD latch prevents resampling; the next
                    # cycle preflights the safe nominal fallback instead.
                    # If no safe fallback exists, the state machine handles
                    # the failed recovery transaction explicitly below.
                    self._data.learned_foothold_prepared_valid[
                        failed_preflight
                    ] = False
                failed_preflight_mask = (
                    preflight_candidate
                    & ~self._data.swing_preflight_safe[selected_env_ids]
                )
                learned_fallback_available = (
                    failed_preflight_mask
                    & preflight_route.use_learned
                    & nominal_route_ready
                )
                failed_without_fallback = (
                    failed_preflight_mask & ~learned_fallback_available
                )
                plan_wait_expired = (
                    plan_wait_expired | failed_without_fallback
                )
                swing_ready = swing_ready & self._data.swing_preflight_safe[
                    selected_env_ids
                ]

        if self.cfg.enable_learned_foothold:
            # A failed HOLD preflight is a planning retry, not a physical
            # recovery event.  The latter is reserved for contact/liftoff
            # failures during an already locked swing.
            planning_failure = plan_wait_expired & hold_contact_ready
        else:
            planning_failure = torch.zeros_like(swing_ready)

        event_response = None
        stabilization_ready = None
        late_search_exhausted = None
        support_available = torch.any(confirmed_contact, dim=-1)
        recovery_contact_stable = torch.all(confirmed_contact, dim=-1)
        if self.cfg.enable_contact_adaptive_recovery:
            assert self._stability_bounds is not None
            assert self._data.body_tilt_rad is not None
            assert self._data.body_angular_speed_rad_s is not None
            assert self._data.body_horizontal_speed_m_s is not None
            assert self._data.support_slip_m_s is not None
            # Each foot has already passed the existing contact-confirmation
            # interval before appearing in ``confirmed_contact``.  A single
            # confirmed support is enough to open the one-shot recovery HOLD;
            # zero confirmed supports remain in RECOVERY.  Double support is
            # still reported for diagnostics and normal gait hand-off.
            stabilization_ready = support_available
            event = torch.full_like(
                previous_gait_state.mode,
                ContactEvent.NONE,
            )
            swing_contact_confirmed = confirmed_contact[
                selected_rows,
                previous_gait_state.swing_side,
            ]
            event[
                (previous_gait_state.mode == GaitState.EARLY_CONTACT)
                & swing_contact_confirmed
            ] = ContactEvent.EARLY_CONTACT
            event[previous_gait_state.mode == GaitState.OVERDUE] = (
                ContactEvent.LATE_CONTACT
            )
            event[
                (previous_gait_state.mode == GaitState.STANCE_LOST)
                | (previous_gait_state.mode == GaitState.HOLD_CONTACT_LOST)
            ] = ContactEvent.SUPPORT_LOST
            event[
                (previous_gait_state.mode == GaitState.PLAN_INVALID)
                | (
                    (previous_gait_state.mode == GaitState.HOLD)
                    & ~self._data.planner_valid[env_ids]
                )
            ] = ContactEvent.PLAN_INVALID
            late_elapsed = (
                previous_gait_state.late_search_elapsed_s
                if previous_gait_state.late_search_elapsed_s is not None
                else torch.zeros_like(previous_gait_state.elapsed_s)
            )
            # The only permitted late-contact descent is the existing
            # touchdown Z tolerance.  At ``late_speed`` this takes exactly
            # ``overdue_s``; do not silently turn the 6 cm touchdown tolerance
            # into a 25 cm downward-search range.
            late_search_exhausted = late_elapsed >= self.cfg.overdue_s
            event_response = response_for_event(
                event,
                support_stable=support_available,
                late_search_available=~late_search_exhausted,
                late_touchdown_confirmed=(
                    (previous_gait_state.mode == GaitState.OVERDUE)
                    & swing_contact_confirmed
                    & self._data.touchdown_accepted[env_ids]
                ),
            )
            event_response[previous_gait_state.mode == GaitState.RECOVERY] = (
                EventResponse.STABILIZE
            )
            planning_failure &= support_available
            self._data.event_response[env_ids] = event_response
            self._data.stabilization_ready[env_ids] = stabilization_ready
            self._data.planning_failure[env_ids] = planning_failure
        elif self._data.planning_failure is not None:
            self._data.planning_failure[env_ids] = planning_failure

        gait_state = advance_gait(
            state=previous_gait_state,
            contact=contact,
            touchdown_accepted=self._data.touchdown_accepted[env_ids],
            planner_valid=self._data.planner_valid[env_ids],
            dt=self.cfg.control_dt_s,
            cfg=self._gait_cfg,
            step_hold_s=step_hold_s,
            swing_ready=swing_ready,
            hold_contact_ready=hold_contact_ready,
            hold_contact_lost=hold_contact_lost,
            plan_wait_expired=plan_wait_expired,
            event_response=event_response,
            stabilization_ready=stabilization_ready,
            # Keep the timer predicate identical to the readiness predicate:
            # two confirmed contacts on every consecutive control step.
            stability_current=recovery_contact_stable,
            late_search_exhausted=late_search_exhausted,
            planning_failure=planning_failure,
        )
        # ``advance_gait`` swaps ``swing_side`` on touchdown/recovery entry.
        # Refresh the physical foot roles immediately so every downstream
        # HOLD cache is anchored to the *new* support foot, not to the support
        # foot from the just-finished step.
        stance_sole_pos_w, swing_sole_pos_w = _select_sole_roles(
            left_sole_w=left_sole_pos_w,
            right_sole_w=right_sole_pos_w,
            swing_side=gait_state.swing_side,
        )
        self._data.actual_stance_foot_pos_w[env_ids] = stance_sole_pos_w
        self._data.actual_swing_foot_pos_w[env_ids] = swing_sole_pos_w
        swing_is_left = gait_state.swing_side == 0
        self._data.actual_swing_foot_vel_w[env_ids] = torch.where(
            swing_is_left[:, None],
            left_sole_velocity_w,
            right_sole_velocity_w,
        )
        if self._data.stabilization_active is not None:
            self._data.stabilization_active[env_ids] = (
                gait_state.mode == GaitState.RECOVERY
            )
        entered_hold = (
            (gait_state.mode == GaitState.HOLD)
            & (previous_gait_state.mode != GaitState.HOLD)
        )
        entered_hold &= ~startup_hold_mask
        if self.cfg.enable_learned_foothold and torch.any(
            entered_hold
        ).item():
            entered_hold_env_ids = selected_env_ids[entered_hold]
            clear_learned_foothold_buffers(
                self._data,
                entered_hold_env_ids,
            )
            assert self._data.nominal_foothold_prepared is not None
            self._data.nominal_foothold_prepared[
                entered_hold_env_ids
            ] = False
            assert self._data.swing_preflight_ready is not None
            self._data.swing_preflight_ready[entered_hold_env_ids] = False
        _apply_startup_hold_gate(
            data=self._data,
            gait_state=gait_state,
            selected_env_ids=selected_env_ids,
            startup_hold_mask=startup_hold_mask,
            reset_hold_s=self.cfg.reset_hold_s,
        )
        self._write_gait_state(env_ids, gait_state)
        self._data.swing_has_lifted[env_ids] = gait_state.swing_has_lifted
        self._data.recovery_step_active[env_ids] = (
            gait_state.recovery_step_active
        )

        self._data.gait_mode[env_ids] = gait_state.mode
        self._data.swing_side[env_ids] = gait_state.swing_side
        self._data.phase[env_ids] = gait_phase(
            gait_state,
            self._gait_cfg,
        )

        stance_pos_w = self._data.actual_stance_foot_pos_w[env_ids]
        num_selected_envs = stance_pos_w.shape[0]
        active_swing = (
            (gait_state.mode == GaitState.LEFT_SWING)
            | (gait_state.mode == GaitState.RIGHT_SWING)
        )
        new_swing = active_swing & (gait_state.elapsed_s <= 1.0e-6)
        # Snapshot the recovery role before any route-locking logic consumes
        # it.  Keep the full selected-environment mask for the lock route and
        # slice it to the new-swing batch below; mixing those two shapes would
        # misclassify recovery targets in partially active batches.
        recovery_step_by_env = gait_state.recovery_step_active
        learned_use = torch.zeros_like(new_swing)
        if self.cfg.enable_learned_foothold:
            _, lock_learned = learned_foothold_event_masks(
                hold=gait_state.mode == GaitState.HOLD,
                hold_contact_ready=hold_contact_ready,
                nominal_ready=torch.ones_like(new_swing),
                new_swing=new_swing,
                enable=True,
            )
            if torch.any(lock_learned).item():
                lock_env_ids = selected_env_ids[lock_learned]
                lock_safety_valid = self._data.learned_foothold_safety_valid[
                    lock_env_ids
                ]
                learned_use[lock_learned] = (
                    lock_prepared_learned_foothold(
                        data=self._data,
                        env_ids=lock_env_ids,
                        safety_valid=lock_safety_valid,
                    )
                )

        if torch.any(new_swing).item():
            new_swing_env_ids = selected_env_ids[new_swing]
            recovery_step = recovery_step_by_env[new_swing]
            new_swing_stance_pos_w = stance_pos_w[new_swing].clone()
            new_swing_side = gait_state.swing_side[new_swing]
            new_swing_count = new_swing_stance_pos_w.shape[0]
            new_swing_base_yaw_w = base_yaw_w[new_swing].clone()
            use_preflight = torch.zeros(
                new_swing_count,
                device=self._device,
                dtype=torch.bool,
            )
            if self.cfg.enable_learned_foothold:
                assert self._data.nominal_foothold_prepared is not None
                assert self._data.nominal_frame_origin_w is not None
                assert self._data.nominal_frame_yaw_w is not None
                frozen_frame = self._data.nominal_foothold_prepared[
                    new_swing_env_ids
                ]
                new_swing_stance_pos_w = torch.where(
                    frozen_frame[:, None],
                    self._data.nominal_frame_origin_w[new_swing_env_ids],
                    new_swing_stance_pos_w,
                )
                new_swing_base_yaw_w = torch.where(
                    frozen_frame,
                    self._data.nominal_frame_yaw_w[new_swing_env_ids],
                    new_swing_base_yaw_w,
                )
                assert self._data.swing_preflight_ready is not None
                use_preflight = self._data.swing_preflight_ready[
                    new_swing_env_ids
                ]

            desired_velocity = self._data.desired_velocity_f[
                new_swing_env_ids
            ]
            level = _flat_target_level_from_curriculum_scale(
                self._flat_target_curriculum_scale[new_swing_env_ids],
                num_levels=len(self._flat_provider_cfg.curriculum_radius_x),
            )
            stance_xy_f = torch.zeros(
                new_swing_count,
                2,
                device=self._device,
                dtype=new_swing_stance_pos_w.dtype,
            )

            use_preflight_start = torch.zeros(
                new_swing_count,
                device=self._device,
                dtype=torch.bool,
            )
            if self.cfg.enable_learned_foothold:
                assert self._data.swing_preflight_ready is not None
                use_preflight_start = self._data.swing_preflight_ready[
                    new_swing_env_ids
                ]
            live_swing_start_w = self._data.actual_swing_foot_pos_w[env_ids][
                new_swing
            ]
            self._data.swing_start_pos_w[new_swing_env_ids] = torch.where(
                use_preflight_start[:, None],
                self._data.swing_start_pos_w[new_swing_env_ids],
                live_swing_start_w,
            )
            if self.cfg.enable_learned_foothold:
                assert self._data.nominal_foothold_prepared is not None
                missing_nominal = ~self._data.nominal_foothold_prepared[
                    new_swing_env_ids
                ]
                if torch.any(missing_nominal).item():
                    # Startup/reset fallback. Normal HOLD operation publishes
                    # this cache one policy cycle before it is consumed.
                    self._prepare_nominal_footholds(
                        env_ids=new_swing_env_ids[missing_nominal],
                        swing_side=new_swing_side[missing_nominal],
                        recovery_step=recovery_step[missing_nominal],
                        stance_pos_w=new_swing_stance_pos_w[
                            missing_nominal
                        ],
                        base_yaw_w=new_swing_base_yaw_w[
                            missing_nominal
                        ],
                    )
                assert self._data.raw_unclipped_foothold_f is not None
                assert self._data.nominal_feasible_velocity_f is not None
                assert self._data.nominal_curriculum_residual_f is not None
                assert self._data.nominal_curriculum_radius_f is not None
                assert self._data.nominal_curriculum_usage is not None
                target_foothold_f = self._data.raw_unclipped_foothold_f[
                    new_swing_env_ids
                ].clone()
                feasible_velocity_f = (
                    self._data.nominal_feasible_velocity_f[
                        new_swing_env_ids
                    ].clone()
                )
                curriculum_residual_f = (
                    self._data.nominal_curriculum_residual_f[
                        new_swing_env_ids
                    ].clone()
                )
                curriculum_radius_f = (
                    self._data.nominal_curriculum_radius_f[
                        new_swing_env_ids
                    ].clone()
                )
                curriculum_usage = self._data.nominal_curriculum_usage[
                    new_swing_env_ids
                ].clone()
            else:
                flat_result = sample_flat_targets(
                    stance_xy=stance_xy_f,
                    swing_side=new_swing_side,
                    desired_velocity=desired_velocity,
                    level=level,
                    generator=self._generator,
                    cfg=self._flat_provider_cfg,
                )
                target_foothold_f = flat_result.position_f
                feasible_velocity_f = flat_result.feasible_velocity_f
                curriculum_residual_f = (
                    flat_result.curriculum_residual_f.clone()
                )
                curriculum_radius_f = (
                    flat_result.curriculum_radius_f.clone()
                )
                curriculum_usage = flat_result.curriculum_usage.clone()
                if torch.any(recovery_step).item():
                    recovery_target_f = make_recovery_foothold_target(
                        swing_side=new_swing_side[recovery_step],
                        desired_velocity_f=desired_velocity[recovery_step],
                        step_length_m=self.cfg.recovery_step_length_m,
                        velocity_lookahead_s=(
                            self.cfg.recovery_step_velocity_lookahead_s
                        ),
                        max_step_length_m=(
                            self.cfg.recovery_step_max_length_m
                        ),
                        step_width_m=self.cfg.recovery_step_width_m,
                        dtype=target_foothold_f.dtype,
                        device=self._device,
                    )
                    target_foothold_f[recovery_step] = recovery_target_f
                    feasible_velocity_f[recovery_step] = (
                        self._feasible_velocity_from_target(
                            target_foothold_f=recovery_target_f,
                            swing_side=new_swing_side[recovery_step],
                            yaw_velocity_f=torch.zeros(
                                recovery_target_f.shape[0],
                                device=self._device,
                                dtype=target_foothold_f.dtype,
                            ),
                        )
                    )
                    curriculum_residual_f[recovery_step] = 0.0
                    curriculum_radius_f[recovery_step] = 0.0
                    curriculum_usage[recovery_step] = 0.0

                if self._data.raw_unclipped_foothold_f is not None:
                    self._data.raw_unclipped_foothold_f[
                        new_swing_env_ids
                    ] = target_foothold_f

            learned_use_new_swing = learned_use[new_swing]
            cached_target_w = None
            if self.cfg.enable_learned_foothold:
                assert self._data.nominal_geometric_valid is not None
                assert self._data.nominal_safety_valid is not None
                assert self._data.nominal_foothold_w is not None
                assert self._data.learned_foothold_prepared_valid is not None
                assert self._data.learned_foothold_geometric_valid is not None
                assert self._data.learned_foothold_target_w is not None
                assert self._data.learned_foothold_route_event is not None
                assert self._data.learned_foothold_route_use_nominal is not None
                assert self._data.learned_foothold_route_use_learned is not None
                assert (
                    self._data.learned_foothold_route_initial_executable
                    is not None
                )
                assert self._data.learned_foothold_route_outcome is not None
                assert (
                    self._data.learned_foothold_transaction_evaluated
                    is not None
                )
                assert self._data.swing_preflight_ready is not None
                assert self._data.swing_preflight_safe is not None
                route = route_nominal_and_learned_footholds(
                    nominal_geometric_valid=(
                        self._data.nominal_geometric_valid[
                            new_swing_env_ids
                        ]
                    ),
                    nominal_safety_valid=(
                        self._data.nominal_safety_valid[
                            new_swing_env_ids
                        ]
                    ),
                    learned_prepared=(
                        self._data.learned_foothold_prepared_valid[
                            new_swing_env_ids
                        ]
                    ),
                    learned_geometric_valid=(
                        self._data.learned_foothold_geometric_valid[
                            new_swing_env_ids
                        ]
                    ),
                    learned_safety_valid=(
                        self._data.learned_foothold_safety_valid[
                            new_swing_env_ids
                        ]
                    ),
                    recovery_step=recovery_step,
                )
                learned_use_new_swing = route.use_learned
                cached_target_w = self._data.nominal_foothold_w[
                    new_swing_env_ids
                ].clone()
                self._data.learned_foothold_route_event[
                    new_swing_env_ids
                ] = True
                self._data.learned_foothold_route_use_nominal[
                    new_swing_env_ids
                ] = route.use_nominal
                self._data.learned_foothold_route_use_learned[
                    new_swing_env_ids
                ] = route.use_learned
                self._data.learned_foothold_route_initial_executable[
                    new_swing_env_ids
                ] = route.executable
                self._data.learned_foothold_route_outcome[
                    new_swing_env_ids
                ] = classify_learned_foothold_route(
                    recovery_step=recovery_step.bool(),
                    transaction_evaluated=(
                        self._data.learned_foothold_transaction_evaluated[
                            new_swing_env_ids
                        ]
                    ),
                    learned_geometric_valid=(
                        self._data.learned_foothold_geometric_valid[
                            new_swing_env_ids
                        ]
                    ),
                    learned_safety_valid=(
                        self._data.learned_foothold_safety_valid[
                            new_swing_env_ids
                        ]
                    ),
                    preflight_ready=self._data.swing_preflight_ready[
                        new_swing_env_ids
                    ],
                    preflight_safe=self._data.swing_preflight_safe[
                        new_swing_env_ids
                    ],
                    route_use_learned=route.use_learned,
                )
                self._data.planner_valid[new_swing_env_ids] = (
                    route.executable
                )
                self._data.learned_foothold_used[
                    new_swing_env_ids
                ] = route.use_learned
                if torch.any(~route.executable).item():
                    local_new_swing_ids = torch.nonzero(
                        new_swing,
                        as_tuple=False,
                    ).flatten()
                    invalid_local_ids = local_new_swing_ids[
                        ~route.executable
                    ]
                    invalid_env_ids = new_swing_env_ids[
                        ~route.executable
                    ]
                    gait_state.mode[invalid_local_ids] = (
                        GaitState.PLAN_INVALID
                    )
                    self._gait_state.mode[invalid_env_ids] = (
                        GaitState.PLAN_INVALID
                    )
                    self._data.gait_mode[invalid_env_ids] = (
                        GaitState.PLAN_INVALID
                    )
            if torch.any(learned_use_new_swing).item():
                learned_env_ids = new_swing_env_ids[
                    learned_use_new_swing
                ]
                target_foothold_f[learned_use_new_swing] = (
                    self._data.learned_foothold_target_f[
                        learned_env_ids
                    ]
                )
                feasible_velocity_f[learned_use_new_swing] = (
                    self._feasible_velocity_from_target(
                        target_foothold_f=target_foothold_f[
                            learned_use_new_swing
                        ],
                        swing_side=new_swing_side[
                            learned_use_new_swing
                        ],
                        yaw_velocity_f=feasible_velocity_f[
                            learned_use_new_swing,
                            2,
                        ],
                    )
                )
                assert cached_target_w is not None
                cached_target_w[learned_use_new_swing] = (
                    self._data.learned_foothold_target_w[learned_env_ids]
                )

            if self.cfg.enable_learned_foothold:
                assert cached_target_w is not None
                target_foothold_f, lock_geometric_valid = (
                    reframe_cached_world_foothold(
                        target_w=cached_target_w,
                        current_origin_w=new_swing_stance_pos_w,
                        current_yaw_w=new_swing_base_yaw_w,
                        radius_x=self._flat_provider_cfg.outer_radius_x,
                        radius_y=self._flat_provider_cfg.outer_radius_y,
                        max_step_height_m=(
                            self.cfg.max_foothold_step_height_m
                        ),
                    )
                )
                lock_valid = (
                    self._data.planner_valid[new_swing_env_ids]
                    & lock_geometric_valid
                )
                if self._data.learned_foothold_lock_geometric_valid is not None:
                    self._data.learned_foothold_lock_geometric_valid[
                        new_swing_env_ids
                    ] = lock_geometric_valid
                self._data.planner_valid[new_swing_env_ids] = lock_valid
                feasible_velocity_f = self._feasible_velocity_from_target(
                    target_foothold_f=target_foothold_f,
                    swing_side=new_swing_side,
                    yaw_velocity_f=feasible_velocity_f[:, 2],
                )
                if torch.any(~lock_valid).item():
                    local_new_swing_ids = torch.nonzero(
                        new_swing,
                        as_tuple=False,
                    ).flatten()
                    invalid_local_ids = local_new_swing_ids[~lock_valid]
                    invalid_env_ids = new_swing_env_ids[~lock_valid]
                    gait_state.mode[invalid_local_ids] = GaitState.PLAN_INVALID
                    self._gait_state.mode[invalid_env_ids] = (
                        GaitState.PLAN_INVALID
                    )
                    self._data.gait_mode[invalid_env_ids] = (
                        GaitState.PLAN_INVALID
                    )

            target_search_obstacle = self._virtual_obstacles.get("edges")
            if (
                self.cfg.enable_safe_target_search
                and not self.cfg.enable_learned_foothold
                and target_search_obstacle is not None
            ):
                support_foot_f = torch.zeros_like(target_foothold_f)

                foot_points_xy = self._safe_target_foot_points_xy.to(
                    dtype=target_foothold_f.dtype,
                )

                terrain_height_query_w = None
                if (
                    self.cfg.enable_target_terrain_height
                    and RayCaster.meshes.get(
                        self.cfg.target_terrain_mesh_prim_path
                    )
                    is not None
                ):
                    terrain_height_query_w = (
                        self._query_target_terrain_height_at_xy_w
                    )

                safe_result = search_safe_foothold_target(
                    nominal_target_f=target_foothold_f,
                    raw_target_f=target_foothold_f,
                    support_foot_f=support_foot_f,
                    target_origin_w=new_swing_stance_pos_w,
                    target_yaw_w=new_swing_base_yaw_w,
                    desired_velocity_f=desired_velocity,
                    obstacle=cast(
                        "TargetSearchObstacle",
                        target_search_obstacle,
                    ),
                    ellipse_half_length=self._flat_provider_cfg.outer_radius_x,
                    ellipse_half_width=self._flat_provider_cfg.outer_radius_y,
                    foot_points_xy=foot_points_xy,
                    candidate_radii=torch.tensor(
                        self.cfg.safe_target_search_radii_m,
                        device=self._device,
                        dtype=target_foothold_f.dtype,
                    ),
                    candidate_directions=torch.tensor(
                        self.cfg.safe_target_search_directions,
                        device=self._device,
                        dtype=target_foothold_f.dtype,
                    ),
                    safety_margin=self.cfg.safe_target_search_margin_m,
                    terrain_height_query_w=terrain_height_query_w,
                    max_step_height_m=self.cfg.max_foothold_step_height_m,
                )

                target_foothold_f = safe_result.target_f
                feasible_velocity_f = self._feasible_velocity_from_target(
                    target_foothold_f=target_foothold_f,
                    swing_side=new_swing_side,
                    yaw_velocity_f=feasible_velocity_f[:, 2],
                )
                self._data.planner_valid[new_swing_env_ids] = (
                    safe_result.valid
                )
                if torch.any(~safe_result.valid).item():
                    local_new_swing_ids = torch.nonzero(
                        new_swing,
                        as_tuple=False,
                    ).flatten()
                    invalid_local_ids = local_new_swing_ids[
                        ~safe_result.valid
                    ]
                    invalid_env_ids = new_swing_env_ids[
                        ~safe_result.valid
                    ]
                    gait_state.mode[invalid_local_ids] = (
                        GaitState.PLAN_INVALID
                    )
                    self._gait_state.mode[invalid_env_ids] = (
                        GaitState.PLAN_INVALID
                    )
                    self._data.gait_mode[invalid_env_ids] = (
                        GaitState.PLAN_INVALID
                    )

                if self._data.safe_target_search_performed is not None:
                    self._data.safe_target_search_performed[
                        new_swing_env_ids
                    ] = True
                if self._data.safe_target_final_valid is not None:
                    self._data.safe_target_final_valid[new_swing_env_ids] = (
                        safe_result.valid
                    )
                if self._data.safe_target_used_fallback is not None:
                    self._data.safe_target_used_fallback[new_swing_env_ids] = (
                        safe_result.used_fallback
                    )
                if self._data.safe_target_score is not None:
                    self._data.safe_target_score[new_swing_env_ids] = (
                        safe_result.selected_score
                    )
                if self._data.safe_target_nominal_inside_ellipse is not None:
                    self._data.safe_target_nominal_inside_ellipse[
                        new_swing_env_ids
                    ] = safe_result.nominal_inside_ellipse
                if self._data.safe_target_nominal_obstacle_safe is not None:
                    self._data.safe_target_nominal_obstacle_safe[
                        new_swing_env_ids
                    ] = safe_result.nominal_obstacle_safe
                if self._data.safe_target_nominal_valid is not None:
                    self._data.safe_target_nominal_valid[
                        new_swing_env_ids
                    ] = safe_result.nominal_valid
                if self._data.safe_target_candidate_count is not None:
                    self._data.safe_target_candidate_count[
                        new_swing_env_ids
                    ] = safe_result.candidate_count
                if self._data.safe_target_candidate_inside_ellipse_count is not None:
                    self._data.safe_target_candidate_inside_ellipse_count[
                        new_swing_env_ids
                    ] = safe_result.candidate_inside_ellipse_count
                if self._data.safe_target_candidate_obstacle_safe_count is not None:
                    self._data.safe_target_candidate_obstacle_safe_count[
                        new_swing_env_ids
                    ] = safe_result.candidate_obstacle_safe_count
                if self._data.safe_target_candidate_valid_count is not None:
                    self._data.safe_target_candidate_valid_count[
                        new_swing_env_ids
                    ] = safe_result.candidate_valid_count
                if self._data.safe_target_final_max_penetration_depth is not None:
                    self._data.safe_target_final_max_penetration_depth[
                        new_swing_env_ids
                    ] = safe_result.final_max_penetration_depth
            else:
                if self._data.safe_target_search_performed is not None:
                    self._data.safe_target_search_performed[
                        new_swing_env_ids
                    ] = False
                if self._data.safe_target_final_valid is not None:
                    self._data.safe_target_final_valid[
                        new_swing_env_ids
                    ] = self._data.planner_valid[new_swing_env_ids]
                if self._data.safe_target_used_fallback is not None:
                    self._data.safe_target_used_fallback[new_swing_env_ids] = False
                if self._data.safe_target_score is not None:
                    self._data.safe_target_score[new_swing_env_ids] = 0.0
                if self._data.safe_target_final_max_penetration_depth is not None:
                    self._data.safe_target_final_max_penetration_depth[
                        new_swing_env_ids
                    ] = 0.0

            if cached_target_w is not None:
                target_foothold_w = cached_target_w
            else:
                target_foothold_w = _compose_world_from_frame(
                    new_swing_stance_pos_w,
                    target_foothold_f,
                    new_swing_base_yaw_w,
                )

            terrain_query = self._query_target_terrain_height_at_xy_w(
                target_foothold_w[:, :2]
            )
            if terrain_query is not None:
                terrain_height_w, queried_terrain_valid = terrain_query
                apply_terrain_mask = torch.ones_like(
                    queried_terrain_valid,
                    dtype=torch.bool,
                )
                if self.cfg.enable_learned_foothold:
                    # A preflight transaction has already queried this exact
                    # world XY, checked the complete trajectory, and frozen
                    # its world target.  Never replace its target z after
                    # the HOLD -> SWING transition.
                    apply_terrain_mask = ~use_preflight

                terrain_valid = torch.ones_like(
                    queried_terrain_valid,
                    dtype=torch.bool,
                )
                if torch.any(apply_terrain_mask).item():
                    corrected_w, corrected_f, corrected_valid = (
                        _apply_terrain_height_to_target(
                            target_foothold_w=target_foothold_w[
                                apply_terrain_mask
                            ],
                            target_foothold_f=target_foothold_f[
                                apply_terrain_mask
                            ],
                            stance_pos_w=new_swing_stance_pos_w[
                                apply_terrain_mask
                            ],
                            terrain_height_w=terrain_height_w[
                                apply_terrain_mask
                            ],
                            terrain_valid=queried_terrain_valid[
                                apply_terrain_mask
                            ],
                        )
                    )
                    target_foothold_w[apply_terrain_mask] = corrected_w
                    target_foothold_f[apply_terrain_mask] = corrected_f
                    terrain_valid[apply_terrain_mask] = corrected_valid
                if self._data.target_terrain_valid is not None:
                    self._data.target_terrain_valid[new_swing_env_ids] = (
                        terrain_valid
                    )
                invalid_terrain = ~terrain_valid
                invalid_terrain_env_ids = new_swing_env_ids[invalid_terrain]
                if invalid_terrain_env_ids.numel() > 0:
                    local_new_swing_ids = torch.nonzero(
                        new_swing,
                        as_tuple=False,
                    ).flatten()
                    invalid_local_ids = local_new_swing_ids[invalid_terrain]
                    gait_state.mode[invalid_local_ids] = (
                        GaitState.PLAN_INVALID
                    )
                    self._gait_state.mode[invalid_terrain_env_ids] = (
                        GaitState.PLAN_INVALID
                    )
                    self._data.gait_mode[invalid_terrain_env_ids] = (
                        GaitState.PLAN_INVALID
                    )
                    self._data.planner_valid[invalid_terrain_env_ids] = False

            self._data.target_foothold_f[new_swing_env_ids] = target_foothold_f
            self._data.feasible_velocity_f[new_swing_env_ids] = feasible_velocity_f
            if self._data.flat_target_level is not None:
                self._data.flat_target_level[new_swing_env_ids] = level
            if self._data.velocity_lookahead_s is not None:
                self._data.velocity_lookahead_s[new_swing_env_ids] = (
                    self._flat_provider_cfg.velocity_lookahead_s
                )
            target_delta_f = target_foothold_f[:, :2]
            target_ellipse_max_x = (
                self._flat_provider_cfg.outer_radius_x
                * torch.sqrt(
                    torch.clamp(
                        1.0
                        - (
                            target_delta_f[:, 1]
                            / self._flat_provider_cfg.outer_radius_y
                        ).square(),
                        min=0.0,
                    )
                )
            )
            target_ellipse_usage = reachable_ellipse_usage(
                target_delta_f,
                radius_x=self._flat_provider_cfg.outer_radius_x,
                radius_y=self._flat_provider_cfg.outer_radius_y,
            )
            if self._data.target_delta_f is not None:
                self._data.target_delta_f[new_swing_env_ids] = target_delta_f
            if self._data.curriculum_residual_f is not None:
                self._data.curriculum_residual_f[new_swing_env_ids] = (
                    curriculum_residual_f
                )
            if self._data.curriculum_radius_f is not None:
                self._data.curriculum_radius_f[new_swing_env_ids] = (
                    curriculum_radius_f
                )
            if self._data.curriculum_usage is not None:
                self._data.curriculum_usage[new_swing_env_ids] = (
                    curriculum_usage
                )
            if self._data.target_ellipse_max_x is not None:
                self._data.target_ellipse_max_x[new_swing_env_ids] = (
                    target_ellipse_max_x
                )
            if self._data.target_ellipse_usage is not None:
                self._data.target_ellipse_usage[new_swing_env_ids] = (
                    target_ellipse_usage
                )
            self._data.target_foothold_w[new_swing_env_ids] = target_foothold_w

            assert self._data.default_swing_apex_height is not None
            assert self._data.swing_apex_height is not None
            assert self._data.swing_clearance_safe is not None
            assert self._data.swing_clearance_penetration is not None
            assert self._data.swing_clearance_deepest_phase is not None
            assert self._data.swing_clearance_start_penetration is not None
            assert self._data.swing_clearance_goal_penetration is not None
            assert self._data.swing_clearance_start_escape_safe is not None

            new_default_apex_height = torch.full(
                (new_swing_count,),
                self.cfg.swing_apex_height_m,
                device=self._device,
                dtype=target_foothold_f.dtype,
            )
            new_apex_height = new_default_apex_height
            new_clearance_safe = torch.ones(
                new_swing_count,
                device=self._device,
                dtype=torch.bool,
            )
            new_clearance_penetration = torch.zeros(
                new_swing_count,
                device=self._device,
                dtype=target_foothold_f.dtype,
            )
            new_clearance_deepest_phase = torch.zeros(
                new_swing_count,
                device=self._device,
                dtype=target_foothold_f.dtype,
            )
            new_clearance_start_penetration = torch.zeros_like(
                new_clearance_penetration
            )
            new_clearance_goal_penetration = torch.zeros_like(
                new_clearance_penetration
            )
            new_clearance_start_escape_safe = torch.zeros(
                new_swing_count,
                device=self._device,
                dtype=torch.bool,
            )

            if self.cfg.enable_learned_foothold:
                if torch.any(use_preflight).item():
                    new_default_apex_height[use_preflight] = (
                        self._data.default_swing_apex_height[
                            new_swing_env_ids[use_preflight]
                        ]
                    )
                    new_apex_height[use_preflight] = (
                        self._data.swing_apex_height[
                            new_swing_env_ids[use_preflight]
                        ]
                    )
                    new_clearance_safe[use_preflight] = (
                        self._data.swing_clearance_safe[
                            new_swing_env_ids[use_preflight]
                        ]
                    )
                    new_clearance_penetration[use_preflight] = (
                        self._data.swing_clearance_penetration[
                            new_swing_env_ids[use_preflight]
                        ]
                    )
                    new_clearance_deepest_phase[use_preflight] = (
                        self._data.swing_clearance_deepest_phase[
                            new_swing_env_ids[use_preflight]
                        ]
                    )
                    new_clearance_start_penetration[use_preflight] = (
                        self._data.swing_clearance_start_penetration[
                            new_swing_env_ids[use_preflight]
                        ]
                    )
                    new_clearance_goal_penetration[use_preflight] = (
                        self._data.swing_clearance_goal_penetration[
                            new_swing_env_ids[use_preflight]
                        ]
                    )
                    new_clearance_start_escape_safe[use_preflight] = (
                        self._data.swing_clearance_start_escape_safe[
                            new_swing_env_ids[use_preflight]
                        ]
                    )

            if self.cfg.enable_edge_clearance:
                edge_obstacle = self._virtual_obstacles.get("edges")
                missing_preflight = ~use_preflight
                if edge_obstacle is not None and torch.any(missing_preflight).item():
                    missing_env_ids = new_swing_env_ids[missing_preflight]
                    edge_obstacle = cast("ClearanceObstacle", edge_obstacle)
                    apex_adjustment = adjust_apex_for_edge_clearance(
                        obstacle=edge_obstacle,
                        start=self._data.swing_start_pos_w[missing_env_ids],
                        goal=self._data.target_foothold_w[missing_env_ids],
                        default_apex_height=new_default_apex_height[missing_preflight],
                        max_apex_height=self.cfg.clearance_max_apex_height_m,
                        apex_step=self.cfg.clearance_apex_step_m,
                        sample_spacing=self.cfg.clearance_sample_spacing_m,
                        swing_duration_s=self.cfg.swing_duration_s,
                        foot_points_xy=self._safe_target_foot_points_xy,
                        foot_yaw_w=new_swing_base_yaw_w[missing_preflight],
                        allow_start_penetration_escape=True,
                        goal_max_penetrating_points=(
                            self.cfg.safe_target_max_penetrating_points
                        ),
                    )
                    new_apex_height[missing_preflight] = apex_adjustment.apex_height
                    new_clearance_safe[missing_preflight] = apex_adjustment.is_safe
                    new_clearance_penetration[missing_preflight] = (
                        apex_adjustment.penetration.max_penetration_depth
                    )
                    new_clearance_deepest_phase[missing_preflight] = (
                        apex_adjustment.penetration.deepest_phase
                    )
                    new_clearance_start_penetration[missing_preflight] = (
                        apex_adjustment.penetration.start_penetration_depth
                    )
                    new_clearance_goal_penetration[missing_preflight] = (
                        apex_adjustment.penetration.goal_penetration_depth
                    )
                    new_clearance_start_escape_safe[missing_preflight] = (
                        apex_adjustment.penetration.start_escape_safe
                    )

            self._data.default_swing_apex_height[new_swing_env_ids] = (
                new_default_apex_height
            )
            self._data.swing_apex_height[new_swing_env_ids] = new_apex_height
            self._data.swing_clearance_safe[new_swing_env_ids] = (
                new_clearance_safe
            )
            self._data.swing_clearance_penetration[new_swing_env_ids] = (
                new_clearance_penetration
            )
            self._data.swing_clearance_deepest_phase[new_swing_env_ids] = (
                new_clearance_deepest_phase
            )
            self._data.swing_clearance_start_penetration[
                new_swing_env_ids
            ] = new_clearance_start_penetration
            self._data.swing_clearance_goal_penetration[new_swing_env_ids] = (
                new_clearance_goal_penetration
            )
            self._data.swing_clearance_start_escape_safe[
                new_swing_env_ids
            ] = new_clearance_start_escape_safe

            # Route classification is finalized only after the frozen-frame
            # lock, terrain query, and full swing clearance result are known.
            # This prevents a route that was initially accepted from being
            # reported as success when a later post-check invalidated it.
            if self.cfg.enable_learned_foothold:
                assert self._data.learned_foothold_route_outcome is not None
                assert (
                    self._data.learned_foothold_route_initial_executable
                    is not None
                )
                final_route_valid = (
                    self._data.planner_valid[new_swing_env_ids]
                    & new_clearance_safe
                )
                self._data.learned_foothold_route_outcome[
                    new_swing_env_ids
                ] = finalize_learned_foothold_route_outcome(
                    initial_outcome=self._data.learned_foothold_route_outcome[
                        new_swing_env_ids
                    ],
                    route_initial_executable=(
                        self._data.learned_foothold_route_initial_executable[
                            new_swing_env_ids
                        ]
                    ),
                    final_planner_valid=final_route_valid,
                )

        default_apex_height = torch.full(
            (num_selected_envs,),
            self.cfg.swing_apex_height_m,
            device=self._device,
        )

        assert self._data.default_swing_apex_height is not None
        assert self._data.swing_apex_height is not None
        assert self._data.swing_clearance_safe is not None
        assert self._data.swing_clearance_penetration is not None
        assert self._data.swing_clearance_deepest_phase is not None
        assert self._data.swing_clearance_start_penetration is not None
        assert self._data.swing_clearance_goal_penetration is not None
        assert self._data.swing_clearance_start_escape_safe is not None

        default_swing_reference = quintic_swing_reference(
            start=self._data.swing_start_pos_w[env_ids],
            goal=self._data.target_foothold_w[env_ids],
            phase=self._data.phase[env_ids],
            apex_height=default_apex_height,
            swing_duration_s=self.cfg.swing_duration_s,
        )

        cached_default_apex_height = self._data.default_swing_apex_height[env_ids]
        cached_apex_height = self._data.swing_apex_height[env_ids]
        cached_clearance_safe = self._data.swing_clearance_safe[env_ids]
        cached_clearance_penetration = self._data.swing_clearance_penetration[
            env_ids
        ]
        cached_clearance_deepest_phase = (
            self._data.swing_clearance_deepest_phase[env_ids]
        )
        cached_clearance_start_penetration = (
            self._data.swing_clearance_start_penetration[env_ids]
        )
        cached_clearance_goal_penetration = (
            self._data.swing_clearance_goal_penetration[env_ids]
        )
        cached_clearance_start_escape_safe = (
            self._data.swing_clearance_start_escape_safe[env_ids]
        )
        default_apex_height = torch.where(
            active_swing,
            cached_default_apex_height,
            default_apex_height,
        )
        apex_height = torch.where(
            active_swing,
            cached_apex_height,
            default_apex_height,
        )
        clearance_safe = torch.ones(
            num_selected_envs,
            device=self._device,
            dtype=torch.bool,
        )
        clearance_safe = torch.where(
            active_swing,
            cached_clearance_safe,
            clearance_safe,
        )
        clearance_penetration = torch.zeros(
            num_selected_envs,
            device=self._device,
        )
        clearance_deepest_phase = torch.zeros(
            num_selected_envs,
            device=self._device,
        )
        clearance_start_penetration = torch.zeros(
            num_selected_envs,
            device=self._device,
        )
        clearance_goal_penetration = torch.zeros(
            num_selected_envs,
            device=self._device,
        )
        clearance_start_escape_safe = torch.zeros(
            num_selected_envs,
            device=self._device,
            dtype=torch.bool,
        )
        clearance_penetration = torch.where(
            active_swing,
            cached_clearance_penetration,
            clearance_penetration,
        )
        clearance_deepest_phase = torch.where(
            active_swing,
            cached_clearance_deepest_phase,
            clearance_deepest_phase,
        )
        clearance_start_penetration = torch.where(
            active_swing,
            cached_clearance_start_penetration,
            clearance_start_penetration,
        )
        clearance_goal_penetration = torch.where(
            active_swing,
            cached_clearance_goal_penetration,
            clearance_goal_penetration,
        )
        clearance_start_escape_safe = torch.where(
            active_swing,
            cached_clearance_start_escape_safe,
            clearance_start_escape_safe,
        )

        swing_reference = quintic_swing_reference(
            start=self._data.swing_start_pos_w[env_ids],
            goal=self._data.target_foothold_w[env_ids],
            phase=self._data.phase[env_ids],
            apex_height=apex_height,
            swing_duration_s=self.cfg.swing_duration_s,
        )
        late_search_active = gait_state.mode == GaitState.OVERDUE
        if torch.any(late_search_active).item():
            late_elapsed = (
                gait_state.late_search_elapsed_s
                if gait_state.late_search_elapsed_s is not None
                else torch.zeros_like(gait_state.elapsed_s)
            )
            descended_reference = apply_late_touchdown_descent(
                reference=swing_reference,
                late_search_elapsed_s=late_elapsed,
                max_descent_m=self.cfg.touchdown_z_tolerance_m,
                search_duration_s=self.cfg.overdue_s,
            )
            swing_reference = type(swing_reference)(
                position=torch.where(
                    late_search_active[:, None],
                    descended_reference.position,
                    swing_reference.position,
                ),
                velocity=torch.where(
                    late_search_active[:, None],
                    descended_reference.velocity,
                    swing_reference.velocity,
                ),
                acceleration=torch.where(
                    late_search_active[:, None],
                    descended_reference.acceleration,
                    swing_reference.acceleration,
                ),
            )

        assert self._data.default_swing_reference_pos_w is not None
        assert self._data.swing_reference_pos_w is not None
        assert self._data.swing_reference_vel_w is not None
        self._data.default_swing_reference_pos_w[env_ids] = default_swing_reference.position
        self._data.swing_reference_pos_w[env_ids] = swing_reference.position
        self._data.swing_reference_vel_w[env_ids] = swing_reference.velocity
        self._data.default_swing_apex_height[env_ids] = default_apex_height
        self._data.swing_apex_height[env_ids] = apex_height
        self._data.swing_clearance_safe[env_ids] = clearance_safe
        self._data.swing_clearance_penetration[env_ids] = clearance_penetration
        self._data.swing_clearance_deepest_phase[env_ids] = (
            clearance_deepest_phase
        )
        self._data.swing_clearance_start_penetration[env_ids] = (
            clearance_start_penetration
        )
        self._data.swing_clearance_goal_penetration[env_ids] = (
            clearance_goal_penetration
        )
        self._data.swing_clearance_start_escape_safe[env_ids] = (
            clearance_start_escape_safe
        )
