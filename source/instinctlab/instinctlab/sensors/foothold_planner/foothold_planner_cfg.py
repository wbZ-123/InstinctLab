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

    left_contact_body_name: str = "left_ankle_roll_link"
    right_contact_body_name: str = "right_ankle_roll_link"

    contact_sensor_name: str = "contact_forces"

    sole_center_offset_b: tuple[float, float, float] = (0.0, 0.0, -0.05)

    sole_half_length: float = 0.12
    sole_half_width: float = 0.045

    swing_duration_s: float = 0.32
    reset_hold_s: float = 0.40
    control_dt_s: float = 0.02

    contact_force_threshold_n: float = 1.0

    swing_apex_height_m: float = 0.08

    enable_edge_clearance: bool = True
    clearance_max_apex_height_m: float = 0.30
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
    safe_target_foot_length_m: float = 0.20
    safe_target_foot_width_m: float = 0.10
    safe_target_foot_grid_num_x: int = 10
    safe_target_foot_grid_num_y: int = 5

    touchdown_xy_tolerance_m: float = 0.08
    touchdown_z_tolerance_m: float = 0.06

    
