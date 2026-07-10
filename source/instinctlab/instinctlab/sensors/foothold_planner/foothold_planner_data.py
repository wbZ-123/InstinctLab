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
    desired_velocity_f: torch.Tensor | None = None
    feasible_velocity_f: torch.Tensor | None = None
    default_swing_reference_pos_w: torch.Tensor | None = None
    swing_reference_pos_w: torch.Tensor | None = None
    default_swing_apex_height: torch.Tensor | None = None
    swing_apex_height: torch.Tensor | None = None
    swing_clearance_safe: torch.Tensor | None = None
    swing_clearance_penetration: torch.Tensor | None = None
    actual_stance_foot_pos_w: torch.Tensor | None = None
    actual_swing_foot_pos_w: torch.Tensor | None = None
    swing_start_pos_w: torch.Tensor | None = None
    foot_contact: torch.Tensor | None = None

    touchdown_accepted: torch.Tensor | None = None
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
    # Historical debug target before safe-target search. Despite the name,
    # this currently records the flat-provider nominal target after flat
    # reachability constraints, not the raw velocity-only point.
    raw_unclipped_foothold_f: torch.Tensor | None = None
