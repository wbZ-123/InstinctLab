"""Visualize the flat foothold planner with idealized 3D foot spheres.

Run from the repository root with:

    PYTHONPATH="$PWD/source/instinctlab:$PYTHONPATH" \
    ../IsaacLab/isaaclab.sh -p tests/parkour/foothold/visualize_flat_foothold_planner.py

This script does not create a robot. It uses colored spheres to visualize the
idealized foothold planner geometry:

- red: left foot
- blue: right foot
- green: current target foothold
- yellow: current swing reference
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "source", "instinctlab"))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(
    description="Visualize the flat foothold planner with idealized spheres."
)
parser.add_argument("--vx", type=float, default=0.25)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--dt", type=float, default=0.02)
parser.add_argument("--swing-duration", type=float, default=0.80)
parser.add_argument("--reset-x", type=float, default=1.5)
parser.add_argument("--terrain", choices=("flat", "step"), default="step")
parser.add_argument("--step-x", type=float, default=0.6)
parser.add_argument("--step-height", type=float, default=0.18)
parser.add_argument("--edge-demo", action="store_true")
parser.add_argument("--edge-radius", type=float, default=0.05)
parser.add_argument("--edge-z-min", type=float, default=0.0)
parser.add_argument("--edge-z-max", type=float, default=0.22)
parser.add_argument("--apex-step", type=float, default=0.03)
parser.add_argument("--max-apex", type=float, default=0.30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from instinctlab_foothold import (  # noqa: E402
    FlatProviderConfig,
    StepTerrainQuery,
    adjust_apex_for_edge_clearance,
    lift_flat_targets_to_terrain,
    quintic_swing_reference,
    sample_flat_targets,
)

class FakeVerticalCylinderObstacle:
    def __init__(
        self,
        center_xy,
        z_min: float,
        z_max: float,
        radius: float,
    ):
        self.center_xy = torch.as_tensor(center_xy, dtype=torch.float32)
        self.z_min = z_min
        self.z_max = z_max
        self.radius = radius

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        center_xy = self.center_xy.to(device=points.device, dtype=points.dtype)
        delta_xy = points[:, :2] - center_xy
        radial_distance = torch.linalg.norm(delta_xy, dim=-1)

        inside_xy = radial_distance < self.radius
        inside_z = (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        inside = inside_xy & inside_z

        penetration_depth = self.radius - radial_distance
        direction_xy = torch.zeros_like(delta_xy)

        nonzero = radial_distance > 1.0e-8
        direction_xy[nonzero] = delta_xy[nonzero] / radial_distance[nonzero].unsqueeze(-1)
        direction_xy[~nonzero, 0] = 1.0

        offset = torch.zeros_like(points)
        offset[inside, :2] = direction_xy[inside] * penetration_depth[inside].unsqueeze(-1)
        return offset

def make_visualizer() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FootholdPlanner",
        markers={
            "left_foot": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
            "right_foot": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.2, 1.0)
                ),
            ),
            "feasible_target": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                ),
            ),
            "trajectory": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 1.0)
                ),
            ),
            "default_trajectory": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.85, 0.85)
                ),
            ),
            "raw_unclipped_foothold": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.25, 0.0)
                ),
            ),
            "ellipse": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0)
                ),
            ),
            "raw_to_feasible_line": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0)
                ),
            ),
            "swing_goal_line": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.8, 0.0)
                ),
            ),
            "edge_demo": sim_utils.CylinderCfg(
                radius=0.05,
                height=0.22,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0),
                    opacity=0.35,
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def main() -> None:
    device = torch.device(args.device)

    sim_cfg = SimulationCfg(dt=args.dt, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -2.2, 1.4], [0.4, 0.0, 0.0])

    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    visualizer = make_visualizer()

    cfg = FlatProviderConfig(
        outer_radius_x=0.45,
        outer_radius_y=0.35,
        min_lateral_separation=0.06,
        nominal_step_width=0.18,
        velocity_lookahead_s=0.20,
        curriculum_radius_x=(0.0, 0.0, 0.0),
        curriculum_radius_y=(0.0, 0.0, 0.0),
        curriculum_yaw_limit_rad=(0.0, 0.0, 0.0),
    )
    
    terrain_query = StepTerrainQuery(
        step_x_m=args.step_x,
        lower_height_m=0.0,
        upper_height_m=args.step_height if args.terrain == "step" else 0.0,
        edge_half_width_m=0.05,
    )

    # left = 0, right = 1
    def reset_feet():
        left = torch.tensor([[0.0, 0.09, 0.0]], device=device)
        right = torch.tensor([[0.0, -0.09, 0.0]], device=device)
        side = torch.tensor([1], device=device, dtype=torch.long)
        ph = torch.tensor([0.0], device=device)
        return left, right, side, ph
    
    def raw_unclipped_foothold_from_command(
        stance_pos,
        swing_side_tensor,
    ):
        side_sign = torch.where(
            swing_side_tensor == 0,
            torch.tensor(1.0, device=device),
            torch.tensor(-1.0, device=device),
        ).to(dtype=stance_pos.dtype)

        raw = stance_pos.clone()
        raw[:, 0] = (
            raw[:, 0]
            + desired_velocity[:, 0] * cfg.velocity_lookahead_s
        )
        raw[:, 1] = (
            raw[:, 1]
            + side_sign * cfg.nominal_step_width
            + desired_velocity[:, 1] * cfg.velocity_lookahead_s
        )
        raw[:, 2] = 0.0
        return raw
    
    def ellipse_boundary_points(stance_pos):
        theta = torch.linspace(0.0, 2.0 * torch.pi, 64, device=device)
        points = torch.zeros((64, 3), device=device)
        points[:, 0] = stance_pos[0, 0] + cfg.outer_radius_x * torch.cos(theta)
        points[:, 1] = stance_pos[0, 1] + cfg.outer_radius_y * torch.sin(theta)
        points[:, 2] = 0.01
        return points
    
    def line_points(start, end, num_points=20, z_offset=0.02):
        alpha = torch.linspace(
            0.0,
            0.95,
            num_points,
            device=device,
        ).unsqueeze(-1)

        points = (1.0 - alpha) * start + alpha * end
        points[:, 2] = points[:, 2] + z_offset
        return points

    left_foot, right_foot, swing_side, phase = reset_feet()

    desired_velocity = torch.tensor(
        [[args.vx, args.vy, args.wz]],
        device=device,
    )
    level = torch.zeros(1, device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(1)

    default_apex_height = torch.tensor([0.08], device=device)
    current_apex_height = default_apex_height.clone()
    edge_demo_pos = torch.zeros((0, 3), device=device)

    target = right_foot.clone()
    swing_start = right_foot.clone()
    swing_ref_pos = right_foot.clone()
    raw_target = right_foot.clone()
    ellipse_points = torch.zeros((64, 3), device=device)

    sim.reset()
    print("[VIS] setup complete", flush=True)
    print("[VIS] red=left blue=right green=target yellow=swing_ref", flush=True)

    while simulation_app.is_running():
        stance_xy = (
            left_foot[:, :2]
            if swing_side.item() == 1
            else right_foot[:, :2]
        )

        # Resample target at the start of each swing.
        if phase.item() == 0.0:
            stance_pos = (
                left_foot
                if swing_side.item() == 1
                else right_foot
            )

            raw_target = raw_unclipped_foothold_from_command(
                stance_pos,
                swing_side,
            )
            ellipse_points = ellipse_boundary_points(stance_pos)

            flat_result = sample_flat_targets(
                stance_xy=stance_xy,
                swing_side=swing_side,
                desired_velocity=desired_velocity,
                level=level,
                generator=generator,
                cfg=cfg,
            )
            terrain_result = lift_flat_targets_to_terrain(
                flat_result,
                terrain_query,
            )
            target = terrain_result.position_f.to(device)
            current_apex_height = default_apex_height.clone()
            edge_demo_pos = torch.zeros((0, 3), device=device)

            swing_start = (
                right_foot.clone()
                if swing_side.item() == 1
                else left_foot.clone()
            )

            if args.edge_demo:
                default_mid_ref = quintic_swing_reference(
                    start=swing_start,
                    goal=target,
                    phase=torch.tensor([0.5], device=device),
                    apex_height=default_apex_height,
                    swing_duration_s=args.swing_duration,
                )
                edge_center = default_mid_ref.position.clone()
                edge_center[:, 2] = edge_center[:, 2] - 0.02
                edge_center_xy = edge_center[:, :2]

                edge_obstacle = FakeVerticalCylinderObstacle(
                    center_xy=edge_center_xy[0],
                    z_min=edge_center[0, 2].item() - 0.06,
                    z_max=edge_center[0, 2].item() + 0.06,
                    radius=args.edge_radius,
                )

                adjustment = adjust_apex_for_edge_clearance(
                    obstacle=edge_obstacle,
                    start=swing_start,
                    goal=target,
                    default_apex_height=default_apex_height,
                    swing_duration_s=args.swing_duration,
                    apex_step=args.apex_step,
                    max_apex_height=torch.tensor([args.max_apex], device=device),
                    sample_spacing=0.03,
                )
                current_apex_height = adjustment.apex_height

                print(
                    "[VIS] edge_adjustment default_apex=",
                    default_apex_height.detach().cpu().tolist(),
                    "adjusted_apex=",
                    current_apex_height.detach().cpu().tolist(),
                    "safe=",
                    adjustment.is_safe.detach().cpu().tolist(),
                    "max_penetration=",
                    adjustment.penetration.max_penetration_depth.detach().cpu().tolist(),
                    flush=True,
                )

                edge_demo_pos = edge_center

            print(
                "[VIS] raw_unclipped_foothold=",
                raw_target.detach().cpu().tolist(),
                "terrain_target=",
                target.detach().cpu().tolist(),
                "terrain_height=",
                terrain_result.terrain.heights[:, 0].detach().cpu().tolist(),
                "desired_velocity=",
                desired_velocity.detach().cpu().tolist(),
                flush=True,
            )

        ref = quintic_swing_reference(
            start=swing_start,
            goal=target,
            phase=phase,
            apex_height=current_apex_height,
            swing_duration_s=args.swing_duration,
        )
        swing_ref_pos = ref.position
        trajectory_phase = torch.linspace(
            0.0,
            1.0,
            25,
            device=device,
        )
        default_trajectory_ref = quintic_swing_reference(
            start=swing_start.repeat(25, 1),
            goal=target.repeat(25, 1),
            phase=trajectory_phase,
            apex_height=default_apex_height.repeat(25),
            swing_duration_s=args.swing_duration,
        )
        default_trajectory_points = default_trajectory_ref.position
        trajectory_ref = quintic_swing_reference(
            start=swing_start.repeat(25, 1),
            goal=target.repeat(25, 1),
            phase=trajectory_phase,
            apex_height=current_apex_height.repeat(25),
            swing_duration_s=args.swing_duration,
        )
        trajectory_points = trajectory_ref.position

        raw_to_feasible_line = line_points(
            raw_target,
            target,
            num_points=20,
            z_offset=0.025,
        )

        swing_goal_line = line_points(
            swing_start,
            target,
            num_points=20,
            z_offset=0.04,
        )

        # Ideal execution: the swing foot follows the reference exactly.
        if swing_side.item() == 1:
            right_foot = swing_ref_pos.clone()
        else:
            left_foot = swing_ref_pos.clone()

        marker_positions = torch.cat(
            (
                left_foot,
                right_foot,
                target,
                trajectory_points,
                default_trajectory_points,
                raw_target,
                ellipse_points,
                raw_to_feasible_line,
                swing_goal_line,
                edge_demo_pos,
            ),
            dim=0,
        )
        marker_indices = torch.cat(
            (
                torch.tensor([0, 1, 2], device=device, dtype=torch.long),

                # adjusted trajectory 紫色
                torch.full(
                    (trajectory_points.shape[0],),
                    3,
                    device=device,
                    dtype=torch.long,
                ),

                # default trajectory 灰色
                torch.full(
                    (default_trajectory_points.shape[0],),
                    4,
                    device=device,
                    dtype=torch.long,
                ),

                # raw target 橙色
                torch.tensor([5], device=device, dtype=torch.long),

                # ellipse 青色
                torch.full(
                    (ellipse_points.shape[0],),
                    6,
                    device=device,
                    dtype=torch.long,
                ),

                # raw-to-feasible 白线
                torch.full(
                    (raw_to_feasible_line.shape[0],),
                    7,
                    device=device,
                    dtype=torch.long,
                ),

                # swing-start-to-goal 黄线
                torch.full(
                    (swing_goal_line.shape[0],),
                    8,
                    device=device,
                    dtype=torch.long,
                ),

                # edge demo 蓝色柱
                torch.full(
                    (edge_demo_pos.shape[0],),
                    9,
                    device=device,
                    dtype=torch.long,
                ),
            ),
            dim=0,
        )

        visualizer.visualize(
            translations=marker_positions,
            marker_indices=marker_indices,
        )

        center = 0.5 * (left_foot[0] + right_foot[0])
        eye = [
            center[0].item() + 1.2,
            center[1].item() - 1.8,
            1.2,
        ]
        target_view = [
            center[0].item(),
            center[1].item(),
            0.0,
        ]
        sim.set_camera_view(eye, target_view)

        sim.step()

        phase = phase + args.dt / args.swing_duration

        if phase.item() >= 1.0:
            # Snap to target, swap support/swing foot, and start a new step.
            if swing_side.item() == 1:
                right_foot = target.clone()
                swing_side.fill_(0)
            else:
                left_foot = target.clone()
                swing_side.fill_(1)

            phase.zero_()

            center_x = 0.5 * (left_foot[0, 0] + right_foot[0, 0])
            if center_x.item() > args.reset_x:
                print("[VIS] reset to origin", flush=True)
                left_foot, right_foot, swing_side, phase = reset_feet()
                target = right_foot.clone()
                swing_start = right_foot.clone()
                swing_ref_pos = right_foot.clone()
                raw_target = right_foot.clone()
                ellipse_points = torch.zeros((64, 3), device=device)


if __name__ == "__main__":
    main()
    simulation_app.close()