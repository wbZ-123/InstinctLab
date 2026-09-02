from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .foothold_planner import FootholdPlanner


@configclass
class FootholdPlannerCfg(SensorBaseCfg):
    """Configuration for the foothold planner sensor."""

    class_type: type = FootholdPlanner

    robot_name: str = "robot"

    left_ankle_body_name: str = "left_ankle_roll_link"
    right_ankle_body_name: str = "right_ankle_roll_link"
    base_body_name: str = "pelvis"

    left_contact_body_name: str = "left_ankle_roll_link"
    right_contact_body_name: str = "right_ankle_roll_link"

    contact_sensor_name: str = "contact_forces"

    # Active G1 shoe URDF foot-contact envelope relative to ankle_roll_link:
    # x=[-0.054, 0.132], y=[-0.036, 0.036], bottom z=-0.058.
    sole_center_offset_b: tuple[float, float, float] = (0.039, 0.0, -0.058)

    sole_half_length: float = 0.093
    sole_half_width: float = 0.036

    swing_duration_s: float = 0.32
    # Episode-start stabilisation window. During this time the foothold planner
    # stays in HOLD and exposes no active swing/target plan, matching the play
    # diagnosis where a short zero-action warmup prevented immediate
    # bad-orientation resets. Set to 0.0 to preserve legacy behaviour.
    startup_hold_s: float = 0.0
    reset_hold_s: float = 0.40
    contact_confirm_s: float = 0.04
    stance_lost_confirm_s: float = 0.10
    hold_contact_lost_confirm_s: float = 0.10
    early_contact_phase: float = 0.65
    overdue_s: float = 0.12
    # Legacy non-adaptive callers use this fallback timer. In the learned
    # contact-adaptive path, one confirmed support opens the recovery HOLD;
    # zero confirmed supports remain in RECOVERY and no second dwell timer is
    # imposed.
    recovery_hold_s: float = 0.04
    step_hold_s: float = 0.04
    step_hold_min_s: float = 0.0
    step_hold_velocity_scale_s_per_mps: float = 0.02
    control_dt_s: float = 0.02
    recovery_step_length_m: float = 0.04
    recovery_step_velocity_lookahead_s: float = 0.10
    recovery_step_max_length_m: float = 0.12
    recovery_step_width_m: float = 0.18

    contact_force_threshold_n: float = 1.0

    swing_apex_height_m: float = 0.08
    # Temporary calibration parameter: expected touchdown phase inside the
    # nominal swing interval. The flat target velocity lookahead is derived as
    # ``flat_target_lookahead_phase * swing_duration_s``.
    flat_target_lookahead_phase: float = 0.8

    # Project the final planned XY foothold onto the runtime terrain mesh.
    # This keeps the flat planner responsible for horizontal foothold selection
    # while using the terrain surface as the source of truth for target z.
    enable_target_terrain_height: bool = True
    target_terrain_mesh_prim_path: str = "/World/ground"
    # Matches the parkour foot height scanners' 20 m downward ray origin.
    target_terrain_raycast_start_height_m: float = 20.0
    target_terrain_raycast_max_dist_m: float = 40.0
    target_terrain_height_offset_m: float = 0.0
    # Ordinary parkour stair height range tops out at 0.23 m; require the
    # support-to-target height difference to stay strictly below 0.27 m.
    max_foothold_step_height_m: float = 0.27

    # Opt-in while the learned explicit foothold path is integrated.  When
    # disabled, the existing analytical planner remains the only target source.
    # Learned XY actions reuse FlatProviderConfig.outer_radius_x/y and the
    # existing max_foothold_step_height_m instead of defining duplicate limits.
    enable_learned_foothold: bool = False
    # Learned actions are residuals around the frozen analytic nominal point.
    # Bounds are intentionally separate from the physical reachability ellipse:
    # they limit how far the learned planner may alter the analytic gait.
    learned_foothold_max_adjustment_x_m: float = 0.12
    learned_foothold_max_adjustment_y_m: float = 0.10

    # Contact-adaptive recovery is opt-in until a calibration file is
    # available. Its motion/slip fields remain diagnostics; one confirmed
    # support foot opens a single-support recovery HOLD, while zero support
    # keeps RECOVERY active. No extra motion thresholds or second dwell are
    # imposed here.
    enable_contact_adaptive_recovery: bool = False
    recovery_stability_calibration_path: str = ""

    enable_edge_clearance: bool = True
    clearance_max_apex_height_m: float = 0.14
    clearance_apex_step_m: float = 0.03
    clearance_sample_spacing_m: float = 0.03

    enable_safe_target_search: bool = True
    safe_target_search_radii_m: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10)
    safe_target_search_directions: tuple[tuple[float, float], ...] = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    )
    safe_target_search_margin_m: float = 0.0
    # Distance at which a clear sole-perimeter target receives full safety
    # margin reward. This matches the 4 cm virtual edge-cylinder radius.
    safe_target_clearance_reference_m: float = 0.04
    # Endpoint execution may tolerate at most two penetrating sole-perimeter
    # samples. Their safety score remains negative; three or more are invalid.
    safe_target_max_penetrating_points: int = 2
    safe_target_foot_length_m: float = 0.186
    safe_target_foot_width_m: float = 0.072
    safe_target_foot_grid_num_x: int = 10
    safe_target_foot_grid_num_y: int = 5

    touchdown_xy_tolerance_m: float = 0.08
    touchdown_z_tolerance_m: float = 0.06

    
