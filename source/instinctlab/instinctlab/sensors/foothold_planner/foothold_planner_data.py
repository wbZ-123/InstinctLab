from dataclasses import dataclass

import torch


@dataclass
class FootholdPlannerData:
    """Runtime buffers exported by :class:`FootholdPlanner`.

    Coordinate suffixes:
        ``*_w`` is in world frame.
        ``*_f`` is in the planner/support-foot frame used by foothold logic.

    Safe-target fields are event-oriented.  The planner only searches for a
    safe target when a new swing target is planned; consumers should not treat
    the latest cached value as a per-step search result unless
    ``safe_target_search_performed`` is true for that step.
    """

    gait_mode: torch.Tensor | None = None
    swing_side: torch.Tensor | None = None
    phase: torch.Tensor | None = None

    target_foothold_w: torch.Tensor | None = None
    target_foothold_f: torch.Tensor | None = None
    # Learned planner action and event-latched target diagnostics. The action
    # is normalized; all ``*_f`` targets use the support-foot planner frame.
    learned_foothold_enabled: torch.Tensor | None = None
    learned_foothold_action_normalized: torch.Tensor | None = None
    learned_foothold_decoded_f: torch.Tensor | None = None
    learned_foothold_prepared_f: torch.Tensor | None = None
    learned_foothold_prepared_w: torch.Tensor | None = None
    learned_foothold_prepared_valid: torch.Tensor | None = None
    # Event diagnostics only; these flags do not participate in planning.
    learned_foothold_lock_geometric_valid: torch.Tensor | None = None
    target_terrain_valid: torch.Tensor | None = None
    learned_foothold_locked: torch.Tensor | None = None
    learned_foothold_target_f: torch.Tensor | None = None
    learned_foothold_target_w: torch.Tensor | None = None
    learned_foothold_used: torch.Tensor | None = None
    learned_foothold_height_valid: torch.Tensor | None = None
    learned_foothold_geometric_valid: torch.Tensor | None = None
    learned_foothold_safety_valid: torch.Tensor | None = None
    # One-step reward/PPO event pulse. Cleared at the start of every update.
    learned_foothold_evaluated: torch.Tensor | None = None
    # Persistent latch for the current HOLD planning transaction. Cleared only
    # when that transaction is explicitly discarded or a new one begins.
    learned_foothold_transaction_evaluated: torch.Tensor | None = None
    # Monotonic counter incremented exactly when the current high-level action
    # is evaluated. Unlike per-step flags, this is never cleared by reset.
    learned_foothold_event_generation: torch.Tensor | None = None
    # True only on the update where a prepared proposal is routed at the
    # transition into a new swing. This is distinct from HOLD evaluations,
    # which may happen repeatedly before the route is committed.
    learned_foothold_route_event: torch.Tensor | None = None
    learned_foothold_route_use_nominal: torch.Tensor | None = None
    learned_foothold_route_use_learned: torch.Tensor | None = None
    learned_foothold_route_initial_executable: torch.Tensor | None = None
    # Mutually exclusive reason code for the last committed new-SWING route.
    # It is diagnostic only; route_event is the validity pulse for this field.
    learned_foothold_route_outcome: torch.Tensor | None = None
    learned_foothold_safety_score: torch.Tensor | None = None
    learned_foothold_safety_margin_score: torch.Tensor | None = None
    learned_foothold_minimum_signed_clearance: torch.Tensor | None = None
    learned_foothold_penetrating_point_count: torch.Tensor | None = None
    learned_foothold_penetrating_point_ratio: torch.Tensor | None = None
    learned_foothold_total_penetration_depth: torch.Tensor | None = None
    # Exact analytic nominal plan published during HOLD. These auxiliary
    # values keep the later SWING plan identical to the prior seen by policy.
    nominal_foothold_prepared: torch.Tensor | None = None
    nominal_feasible_velocity_f: torch.Tensor | None = None
    nominal_curriculum_residual_f: torch.Tensor | None = None
    nominal_curriculum_radius_f: torch.Tensor | None = None
    nominal_curriculum_usage: torch.Tensor | None = None
    # Authoritative support frame captured when the HOLD nominal is prepared.
    # Learned proposals for that HOLD event must be decoded in this same frame.
    nominal_frame_origin_w: torch.Tensor | None = None
    nominal_frame_yaw_w: torch.Tensor | None = None
    nominal_foothold_w: torch.Tensor | None = None
    nominal_geometric_valid: torch.Tensor | None = None
    nominal_safety_valid: torch.Tensor | None = None
    nominal_safety_score: torch.Tensor | None = None
    desired_velocity_f: torch.Tensor | None = None
    feasible_velocity_f: torch.Tensor | None = None
    default_swing_reference_pos_w: torch.Tensor | None = None
    swing_reference_pos_w: torch.Tensor | None = None
    swing_reference_vel_w: torch.Tensor | None = None
    swing_duration_s: torch.Tensor | None = None
    default_swing_apex_height: torch.Tensor | None = None
    swing_apex_height: torch.Tensor | None = None
    swing_clearance_safe: torch.Tensor | None = None
    swing_clearance_penetration: torch.Tensor | None = None
    swing_clearance_deepest_phase: torch.Tensor | None = None
    swing_clearance_start_penetration: torch.Tensor | None = None
    swing_clearance_goal_penetration: torch.Tensor | None = None
    swing_clearance_start_escape_safe: torch.Tensor | None = None
    # HOLD-time preflight result.  This is the gate used before entering a
    # new SWING; it is not recomputed from a moving support frame afterwards.
    swing_preflight_safe: torch.Tensor | None = None
    swing_preflight_ready: torch.Tensor | None = None
    actual_stance_foot_pos_w: torch.Tensor | None = None
    actual_swing_foot_pos_w: torch.Tensor | None = None
    actual_swing_foot_vel_w: torch.Tensor | None = None
    swing_start_pos_w: torch.Tensor | None = None
    foot_contact: torch.Tensor | None = None
    confirmed_foot_contact: torch.Tensor | None = None
    body_tilt_rad: torch.Tensor | None = None
    body_angular_speed_rad_s: torch.Tensor | None = None
    body_horizontal_speed_m_s: torch.Tensor | None = None
    support_slip_m_s: torch.Tensor | None = None
    stabilization_active: torch.Tensor | None = None
    stabilization_ready: torch.Tensor | None = None
    event_response: torch.Tensor | None = None
    planning_failure: torch.Tensor | None = None

    touchdown_accepted: torch.Tensor | None = None
    touchdown_xy_error: torch.Tensor | None = None
    touchdown_z_error: torch.Tensor | None = None
    touchdown_xy_ok: torch.Tensor | None = None
    touchdown_z_ok: torch.Tensor | None = None
    touchdown_swing_contact: torch.Tensor | None = None
    touchdown_within_tolerance: torch.Tensor | None = None
    swing_has_lifted: torch.Tensor | None = None
    recovery_step_active: torch.Tensor | None = None
    # True when the currently exposed planner output can be executed by the
    # gait state machine. Safe-target search failure must propagate here.
    planner_valid: torch.Tensor | None = None

    # True only on update steps where a new swing target search was executed.
    safe_target_search_performed: torch.Tensor | None = None
    # Result of the current search event. True means the nominal target was
    # safe or a safe fallback candidate was found.
    safe_target_final_valid: torch.Tensor | None = None
    # True for search events where the nominal target was replaced by a nearby
    # safe candidate. False when nominal target is used or no search occurred.
    safe_target_used_fallback: torch.Tensor | None = None
    # XY distance between nominal target and selected fallback target. Zero
    # when nominal target is used, no search occurred, or no valid fallback
    # was found.
    safe_target_score: torch.Tensor | None = None
    # Search-event diagnostics for locating why a target search succeeds or
    # fails.  These values are meaningful only when
    # ``safe_target_search_performed`` is true.
    safe_target_nominal_inside_ellipse: torch.Tensor | None = None
    safe_target_nominal_obstacle_safe: torch.Tensor | None = None
    safe_target_nominal_valid: torch.Tensor | None = None
    safe_target_candidate_count: torch.Tensor | None = None
    safe_target_candidate_inside_ellipse_count: torch.Tensor | None = None
    safe_target_candidate_obstacle_safe_count: torch.Tensor | None = None
    safe_target_candidate_valid_count: torch.Tensor | None = None
    safe_target_final_max_penetration_depth: torch.Tensor | None = None
    # Historical debug target before safe-target search. Despite the name,
    # this currently records the flat-provider nominal target after flat
    # reachability constraints, not the raw velocity-only point.
    raw_unclipped_foothold_f: torch.Tensor | None = None

    # Flat-target curriculum and geometry diagnostics. These fields expose how
    # much of the configured support-foot ellipse is used by the current target
    # without changing the planner target itself.
    flat_target_level: torch.Tensor | None = None
    velocity_lookahead_s: torch.Tensor | None = None
    target_delta_f: torch.Tensor | None = None
    curriculum_residual_f: torch.Tensor | None = None
    curriculum_radius_f: torch.Tensor | None = None
    curriculum_usage: torch.Tensor | None = None
    target_ellipse_max_x: torch.Tensor | None = None
    target_ellipse_usage: torch.Tensor | None = None
