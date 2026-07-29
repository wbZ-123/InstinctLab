from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .action_cfg import LearnedFootholdActionCfg


def scale_foothold_action(
    raw_action: torch.Tensor,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> torch.Tensor:
    """Scale raw unit policy output to support-frame foothold XY coordinates."""

    clipped = torch.clamp(raw_action, -1.0, 1.0)
    x_min, x_max = x_range
    y_min, y_max = y_range
    x = x_min + 0.5 * (clipped[:, 0] + 1.0) * (x_max - x_min)
    y = y_min + 0.5 * (clipped[:, 1] + 1.0) * (y_max - y_min)
    return torch.stack([x, y], dim=-1)


class LearnedFootholdAction(ActionTerm):
    """Stores a learned explicit foothold target for event-based planner consumption."""

    cfg: "LearnedFootholdActionCfg"

    def __init__(self, cfg: "LearnedFootholdActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 2, device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        env.learned_foothold_action_raw = self._raw_actions
        env.learned_foothold_action_f = self._processed_actions

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
        self._processed_actions[:] = scale_foothold_action(
            actions,
            x_range=self.cfg.x_range,
            y_range=self.cfg.y_range,
        )

    def apply_actions(self) -> None:
        return None
