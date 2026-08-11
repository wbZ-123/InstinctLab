from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .action_cfg import LearnedFootholdActionCfg


def normalize_foothold_action(raw_action: torch.Tensor) -> torch.Tensor:
    """Clamp the policy output without assigning meter-valued semantics."""

    return torch.clamp(raw_action, -1.0, 1.0)


class LearnedFootholdAction(ActionTerm):
    """Stores normalized foothold output for event-based planner consumption."""

    cfg: "LearnedFootholdActionCfg"

    def __init__(self, cfg: "LearnedFootholdActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 2, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        planner_data = env.scene[cfg.sensor_name].data
        planner_data.learned_foothold_action_normalized = (
            self._processed_actions
        )
        env.learned_foothold_action_raw = self._raw_actions
        env.learned_foothold_action_normalized = self._processed_actions

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = normalize_foothold_action(actions)

    def apply_actions(self) -> None:
        return None
