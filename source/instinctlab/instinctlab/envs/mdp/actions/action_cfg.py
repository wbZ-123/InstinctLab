from dataclasses import MISSING

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import ActionTerm, SceneEntityCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

from . import foothold_actions
from . import joint_actions


@configclass
class ActionOverridenJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the action overridden delayed joint position action term.

    See :class:`ActionOverridenointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.ActionOverridenJointPositionAction

    asset_cfg: SceneEntityCfg = MISSING
    """Whether to override the action with the delayed action. Defaults to False."""

    override_value: float = 0.0
    """Delay in frames before the action is overridden. Defaults to 0."""


@configclass
class LearnedFootholdActionCfg(ActionTermCfg):
    """Configuration for learned explicit foothold planner action output."""

    class_type: type[ActionTerm] = foothold_actions.LearnedFootholdAction

    asset_name: str = "robot"
    """Scene asset used only to satisfy IsaacLab action term initialization."""
