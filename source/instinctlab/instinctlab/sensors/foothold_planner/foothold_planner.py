from __future__ import annotations

import re
from dataclasses import replace
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.sensors import SensorBase
from isaaclab.utils.math import convert_quat
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema

from instinctlab_foothold import (
    FlatProviderConfig,
    GaitMachineConfig,
    GaitMachineState,
    GaitState,
    SoleGeometry,
    advance_gait,
    gait_phase,
    initial_gait_state,
    adjust_apex_for_edge_clearance,
    make_recovery_foothold_target,
    quintic_swing_reference,
    sample_flat_targets,
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
    gait_state.hold_elapsed_s[startup_hold_mask] = 0.0
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
        self._data = FootholdPlannerData()
        self._flat_target_curriculum_scale: torch.Tensor | None = None
        self._virtual_obstacles: dict[str, object] = {}
        self._sole_geometry = SoleGeometry(
            center_offset_b=torch.tensor(cfg.sole_center_offset_b),
            half_length=cfg.sole_half_length,
            half_width=cfg.sole_half_width,
        )
        self._flat_provider_cfg = _derive_flat_provider_config(cfg)
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
        if self._data.default_swing_apex_height is not None:
            self._data.default_swing_apex_height[reset_env_ids] = 0.0
        if self._data.swing_apex_height is not None:
            self._data.swing_apex_height[reset_env_ids] = 0.0
        if self._data.swing_clearance_safe is not None:
            self._data.swing_clearance_safe[reset_env_ids] = True
        if self._data.swing_clearance_penetration is not None:
            self._data.swing_clearance_penetration[reset_env_ids] = 0.0
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
        if self._data.raw_unclipped_foothold_f is not None:
            self._data.raw_unclipped_foothold_f[reset_env_ids] = 0.0
        if self._data.flat_target_level is not None:
            self._data.flat_target_level[reset_env_ids] = 0
        if self._data.velocity_lookahead_s is not None:
            self._data.velocity_lookahead_s[reset_env_ids] = (
                self._flat_provider_cfg.velocity_lookahead_s
            )
        _clear_foothold_plan_buffers(self._data, reset_env_ids)

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
        self._data.raw_unclipped_foothold_f = torch.zeros(
            self._num_envs,
            3,
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

        swing_side = self._data.swing_side[env_ids]
        swing_is_left = swing_side == 0

        self._data.actual_stance_foot_pos_w[env_ids] = torch.where(
            swing_is_left.unsqueeze(-1),
            right_sole_pos_w,
            left_sole_pos_w,
        )

        self._data.actual_swing_foot_pos_w[env_ids] = torch.where(
            swing_is_left.unsqueeze(-1),
            left_sole_pos_w,
            right_sole_pos_w,
        )

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
        )
        step_hold_s = _adaptive_step_hold_s(
            self._data.desired_velocity_f[env_ids],
            base_hold_s=self.cfg.step_hold_s,
            min_hold_s=self.cfg.step_hold_min_s,
            velocity_scale_s_per_mps=self.cfg.step_hold_velocity_scale_s_per_mps,
        )

        gait_state = advance_gait(
            state=self._select_gait_state(env_ids),
            contact=contact,
            touchdown_accepted=self._data.touchdown_accepted[env_ids],
            planner_valid=self._data.planner_valid[env_ids],
            dt=self.cfg.control_dt_s,
            cfg=self._gait_cfg,
            step_hold_s=step_hold_s,
        )
        _apply_startup_hold_gate(
            data=self._data,
            gait_state=gait_state,
            selected_env_ids=selected_env_ids,
            startup_hold_mask=self._startup_hold_mask(selected_env_ids),
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

        if torch.any(new_swing).item():
            new_swing_env_ids = selected_env_ids[new_swing]
            new_swing_stance_pos_w = stance_pos_w[new_swing]
            new_swing_side = gait_state.swing_side[new_swing]
            new_swing_count = new_swing_stance_pos_w.shape[0]

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

            flat_result = sample_flat_targets(
                stance_xy=stance_xy_f,
                swing_side=new_swing_side,
                desired_velocity=desired_velocity,
                level=level,
                generator=self._generator,
                cfg=self._flat_provider_cfg,
            )

            self._data.swing_start_pos_w[new_swing_env_ids] = (
                self._data.actual_swing_foot_pos_w[env_ids][new_swing]
            )

            target_foothold_f = flat_result.position_f
            feasible_velocity_f = flat_result.feasible_velocity_f
            curriculum_residual_f = flat_result.curriculum_residual_f.clone()
            curriculum_radius_f = flat_result.curriculum_radius_f.clone()
            curriculum_usage = flat_result.curriculum_usage.clone()
            recovery_step = gait_state.recovery_step_active[new_swing]
            if torch.any(recovery_step).item():
                recovery_target_f = make_recovery_foothold_target(
                    swing_side=new_swing_side[recovery_step],
                    desired_velocity_f=desired_velocity[recovery_step],
                    step_length_m=self.cfg.recovery_step_length_m,
                    velocity_lookahead_s=(
                        self.cfg.recovery_step_velocity_lookahead_s
                    ),
                    max_step_length_m=self.cfg.recovery_step_max_length_m,
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
                self._data.raw_unclipped_foothold_f[new_swing_env_ids] = (
                    target_foothold_f
                )

            target_search_obstacle = self._virtual_obstacles.get("edges")
            if (
                self.cfg.enable_safe_target_search
                and target_search_obstacle is not None
            ):
                support_foot_f = torch.zeros_like(target_foothold_f)

                foot_points_xy = make_sole_perimeter_points_xy(
                    foot_length=self.cfg.safe_target_foot_length_m,
                    foot_width=self.cfg.safe_target_foot_width_m,
                    num_x=self.cfg.safe_target_foot_grid_num_x,
                    num_y=self.cfg.safe_target_foot_grid_num_y,
                    device=self._device,
                    dtype=target_foothold_f.dtype,
                )

                safe_result = search_safe_foothold_target(
                    nominal_target_f=target_foothold_f,
                    raw_target_f=target_foothold_f,
                    support_foot_f=support_foot_f,
                    target_origin_w=new_swing_stance_pos_w,
                    target_yaw_w=base_yaw_w[new_swing],
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
            else:
                if self._data.safe_target_search_performed is not None:
                    self._data.safe_target_search_performed[
                        new_swing_env_ids
                    ] = False
                if self._data.safe_target_final_valid is not None:
                    self._data.safe_target_final_valid[new_swing_env_ids] = True
                if self._data.safe_target_used_fallback is not None:
                    self._data.safe_target_used_fallback[new_swing_env_ids] = False
                if self._data.safe_target_score is not None:
                    self._data.safe_target_score[new_swing_env_ids] = 0.0

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
            target_ellipse_usage = torch.where(
                target_ellipse_max_x > 1.0e-6,
                torch.abs(target_delta_f[:, 0]) / target_ellipse_max_x,
                torch.zeros_like(target_ellipse_max_x),
            ).clamp(min=0.0)
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
            self._data.target_foothold_w[new_swing_env_ids] = (
                _compose_world_from_frame(
                    new_swing_stance_pos_w,
                    target_foothold_f,
                    base_yaw_w[new_swing],
                )
            )
            self._data.target_foothold_w[new_swing_env_ids, 2] = (
                new_swing_stance_pos_w[:, 2]
            )

            assert self._data.default_swing_apex_height is not None
            assert self._data.swing_apex_height is not None
            assert self._data.swing_clearance_safe is not None
            assert self._data.swing_clearance_penetration is not None

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

            if self.cfg.enable_edge_clearance:
                edge_obstacle = self._virtual_obstacles.get("edges")
                if edge_obstacle is not None:
                    edge_obstacle = cast("ClearanceObstacle", edge_obstacle)
                    apex_adjustment = adjust_apex_for_edge_clearance(
                        obstacle=edge_obstacle,
                        start=self._data.swing_start_pos_w[new_swing_env_ids],
                        goal=self._data.target_foothold_w[new_swing_env_ids],
                        default_apex_height=new_default_apex_height,
                        max_apex_height=self.cfg.clearance_max_apex_height_m,
                        apex_step=self.cfg.clearance_apex_step_m,
                        sample_spacing=self.cfg.clearance_sample_spacing_m,
                        swing_duration_s=self.cfg.swing_duration_s,
                    )
                    new_apex_height = apex_adjustment.apex_height
                    new_clearance_safe = apex_adjustment.is_safe
                    new_clearance_penetration = (
                        apex_adjustment.penetration.max_penetration_depth
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

        default_apex_height = torch.full(
            (num_selected_envs,),
            self.cfg.swing_apex_height_m,
            device=self._device,
        )

        assert self._data.default_swing_apex_height is not None
        assert self._data.swing_apex_height is not None
        assert self._data.swing_clearance_safe is not None
        assert self._data.swing_clearance_penetration is not None

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
        clearance_penetration = torch.where(
            active_swing,
            cached_clearance_penetration,
            clearance_penetration,
        )

        clearance_invalid = active_swing & ~clearance_safe
        if torch.any(clearance_invalid).item():
            invalid_env_ids = selected_env_ids[clearance_invalid]
            self._data.planner_valid[invalid_env_ids] = False
            gait_state.mode[clearance_invalid] = GaitState.PLAN_INVALID
            self._gait_state.mode[invalid_env_ids] = GaitState.PLAN_INVALID
            self._data.gait_mode[invalid_env_ids] = GaitState.PLAN_INVALID

        swing_reference = quintic_swing_reference(
            start=self._data.swing_start_pos_w[env_ids],
            goal=self._data.target_foothold_w[env_ids],
            phase=self._data.phase[env_ids],
            apex_height=apex_height,
            swing_duration_s=self.cfg.swing_duration_s,
        )

        assert self._data.default_swing_reference_pos_w is not None
        assert self._data.swing_reference_pos_w is not None
        self._data.default_swing_reference_pos_w[env_ids] = default_swing_reference.position
        self._data.swing_reference_pos_w[env_ids] = swing_reference.position
        self._data.default_swing_apex_height[env_ids] = default_apex_height
        self._data.swing_apex_height[env_ids] = apex_height
        self._data.swing_clearance_safe[env_ids] = clearance_safe
        self._data.swing_clearance_penetration[env_ids] = clearance_penetration
