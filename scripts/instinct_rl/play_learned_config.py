"""Restore learned-foothold training mode when playing a checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _algorithm_class_name(agent_cfg: Mapping[str, Any]) -> str | None:
    algorithm_cfg = agent_cfg.get("algorithm")
    if not isinstance(algorithm_cfg, Mapping):
        return None
    class_name = algorithm_cfg.get("class_name")
    return class_name if isinstance(class_name, str) else None


def configure_learned_foothold_play(
    env_cfg: Any,
    saved_agent_cfg: Mapping[str, Any],
    *,
    register_algorithm: Callable[[], None],
) -> bool:
    """Match the play environment and algorithm to the saved checkpoint."""

    if _algorithm_class_name(saved_agent_cfg) != "EventGatedWasabiPPO":
        return False
    enable_planner = getattr(
        env_cfg, "enable_learned_foothold_planner", None
    )
    if not callable(enable_planner):
        raise RuntimeError(
            "Checkpoint uses EventGatedWasabiPPO, but the selected task "
            "does not support the learned foothold planner."
        )
    enable_planner()
    register_algorithm()
    return True
