from __future__ import annotations

import re
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
    quintic_swing_reference,
    sample_flat_targets,
)

from .foothold_planner_data import FootholdPlannerData

if TYPE_CHECKING:
    from instinctlab_foothold.clearance import PenetrationObstacle
    from .foothold_planner_cfg import FootholdPlannerCfg


class FootholdPlanner(SensorBase):
    """Foothold planner sensor.

    This sensor computes foothold targets and swing references, then exposes
    them through ``sensor.data`` for rewards, observations, and debug
    visualization.
    """

    cfg: FootholdPlannerCfg

    def __init__(self, cfg: FootholdPlannerCfg):
        super().__init__(cfg)
        self._data = FootholdPlannerData()
        self._virtual_obstacles: dict[str, object] = {}
        self._sole_geometry = SoleGeometry(
            center_offset_b=torch.tensor(cfg.sole_center_offset_b),
            half_length=cfg.sole_half_length,
            half_width=cfg.sole_half_width,
        )
        self._flat_provider_cfg = FlatProviderConfig()
        self._gait_cfg = GaitMachineConfig(
            reset_hold_s=cfg.reset_hold_s,
            swing_s=cfg.swing_duration_s,
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

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)

        if not hasattr(self, "_gait_state"):
            return

        if env_ids is None:
            reset_env_ids = slice(None)
            num_reset_envs = self._num_envs
        elif isinstance(env_ids, slice):
            reset_env_ids = env_ids
            num_reset_envs = torch.arange(
                self._num_envs,
                device=self._device,
            )[env_ids].shape[0]
        else:
            reset_env_ids = env_ids
            num_reset_envs = len(env_ids)

        reset_state = initial_gait_state(
            num_envs=num_reset_envs,
            device=self._device,
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
        if self._data.planner_valid is not None:
            self._data.planner_valid[reset_env_ids] = True

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

        body_names_regex = r"(" + "|".join(re.escape(name) for name in body_names) + r")"
        body_paths_regex = f"{robot_prim_path}/{body_names_regex}"
        body_paths_glob = body_paths_regex.replace(".*", "*")

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

        contact_body_names_regex = (
            r"("
            + "|".join(re.escape(name) for name in contact_body_names)
            + r")"
        )
        contact_body_paths_regex = (
            f"{robot_prim_path}/{contact_body_names_regex}"
        )
        contact_body_paths_glob = contact_body_paths_regex.replace(".*", "*")

        self._contact_physx_view = (
            self._physics_sim_view.create_rigid_contact_view(
                contact_body_paths_glob,
                max_contact_data_count=len(contact_body_names)
                * self._num_envs,
            )
        )
        self._num_contact_bodies = (
            self._contact_physx_view.sensor_count // self._num_envs
        )
        self._contact_body_names = contact_body_names

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
        self._data.planner_valid = torch.ones(
            self._num_envs,
            device=self._device,
            dtype=torch.bool,
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
            contact_elapsed_s=self._gait_state.contact_elapsed_s[env_ids],
            no_contact_elapsed_s=self._gait_state.no_contact_elapsed_s[
                env_ids
            ],
            swing_has_lifted=self._gait_state.swing_has_lifted[env_ids],
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
        self._gait_state.contact_elapsed_s[env_ids] = (
            state.contact_elapsed_s
        )
        self._gait_state.no_contact_elapsed_s[env_ids] = (
            state.no_contact_elapsed_s
        )
        self._gait_state.swing_has_lifted[env_ids] = state.swing_has_lifted

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

        self._data.planner_valid[env_ids] = True

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
        self._data.touchdown_accepted[env_ids] = (
            swing_foot_contact
            & (foot_target_error <= self.cfg.touchdown_xy_tolerance_m)
            & (foot_height_error <= self.cfg.touchdown_z_tolerance_m)
        )

        gait_state = advance_gait(
            state=self._select_gait_state(env_ids),
            contact=contact,
            touchdown_accepted=self._data.touchdown_accepted[env_ids],
            planner_valid=self._data.planner_valid[env_ids],
            dt=self.cfg.control_dt_s,
            cfg=self._gait_cfg,
        )
        self._write_gait_state(env_ids, gait_state)

        self._data.gait_mode[env_ids] = gait_state.mode
        self._data.swing_side[env_ids] = gait_state.swing_side
        self._data.phase[env_ids] = gait_phase(
            gait_state,
            self._gait_cfg,
        )

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
            new_swing_side = swing_side[new_swing]
            new_swing_count = new_swing_stance_pos_w.shape[0]

            desired_velocity = self._data.desired_velocity_f[
                new_swing_env_ids
            ]
            level = torch.zeros(
                new_swing_count,
                device=self._device,
                dtype=torch.long,
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
            self._data.target_foothold_f[new_swing_env_ids] = (
                flat_result.position_f
            )
            self._data.feasible_velocity_f[new_swing_env_ids] = (
                flat_result.feasible_velocity_f
            )
            self._data.target_foothold_w[new_swing_env_ids] = (
                new_swing_stance_pos_w + flat_result.position_f
            )
            self._data.target_foothold_w[new_swing_env_ids, 2] = (
                new_swing_stance_pos_w[:, 2]
            )

        default_apex_height = torch.full(
            (num_selected_envs,),
            self.cfg.swing_apex_height_m,
            device=self._device,
        )

        default_swing_reference = quintic_swing_reference(
            start=self._data.swing_start_pos_w[env_ids],
            goal=self._data.target_foothold_w[env_ids],
            phase=self._data.phase[env_ids],
            apex_height=default_apex_height,
            swing_duration_s=self.cfg.swing_duration_s,
        )

        apex_height = default_apex_height
        clearance_safe = torch.ones(
            num_selected_envs,
            device=self._device,
            dtype=torch.bool,
        )
        clearance_penetration = torch.zeros(
            num_selected_envs,
            device=self._device,
        )

        if self.cfg.enable_edge_clearance:
            edge_obstacle = self._virtual_obstacles.get("edges")
            if edge_obstacle is not None:
                edge_obstacle = cast("PenetrationObstacle", edge_obstacle)
                apex_adjustment = adjust_apex_for_edge_clearance(
                    obstacle=edge_obstacle,
                    start=self._data.swing_start_pos_w[env_ids],
                    goal=self._data.target_foothold_w[env_ids],
                    default_apex_height=default_apex_height,
                    max_apex_height=self.cfg.clearance_max_apex_height_m,
                    apex_step=self.cfg.clearance_apex_step_m,
                    sample_spacing=self.cfg.clearance_sample_spacing_m,
                    swing_duration_s=self.cfg.swing_duration_s,
                )
                apex_height = apex_adjustment.apex_height
                clearance_safe = apex_adjustment.is_safe
                clearance_penetration = apex_adjustment.penetration.max_penetration_depth

        swing_reference = quintic_swing_reference(
            start=self._data.swing_start_pos_w[env_ids],
            goal=self._data.target_foothold_w[env_ids],
            phase=self._data.phase[env_ids],
            apex_height=apex_height,
            swing_duration_s=self.cfg.swing_duration_s,
        )

        assert self._data.default_swing_reference_pos_w is not None
        assert self._data.swing_reference_pos_w is not None
        assert self._data.default_swing_apex_height is not None
        assert self._data.swing_apex_height is not None
        assert self._data.swing_clearance_safe is not None
        assert self._data.swing_clearance_penetration is not None

        self._data.default_swing_reference_pos_w[env_ids] = default_swing_reference.position
        self._data.swing_reference_pos_w[env_ids] = swing_reference.position
        self._data.default_swing_apex_height[env_ids] = default_apex_height
        self._data.swing_apex_height[env_ids] = apex_height
        self._data.swing_clearance_safe[env_ids] = clearance_safe
        self._data.swing_clearance_penetration[env_ids] = clearance_penetration
