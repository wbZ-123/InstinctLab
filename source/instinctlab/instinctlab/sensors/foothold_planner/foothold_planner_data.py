from dataclasses import dataclass

import torch


@dataclass
class FootholdPlannerData:
    gait_mode: torch.Tensor | None = None
    swing_side: torch.Tensor | None = None
    phase: torch.Tensor | None = None

    target_foothold_w: torch.Tensor | None = None
    target_foothold_f: torch.Tensor | None = None
    desired_velocity_f: torch.Tensor | None = None
    feasible_velocity_f: torch.Tensor | None = None
    swing_reference_pos_w: torch.Tensor | None = None
    actual_stance_foot_pos_w: torch.Tensor | None = None
    actual_swing_foot_pos_w: torch.Tensor | None = None
    swing_start_pos_w: torch.Tensor | None = None
    foot_contact: torch.Tensor | None = None

    touchdown_accepted: torch.Tensor | None = None
    planner_valid: torch.Tensor | None = None
