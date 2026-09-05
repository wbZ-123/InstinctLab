from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from instinctlab_foothold import quintic_swing_reference


MARKER_TARGET = 0
MARKER_REFERENCE = 1
MARKER_ACTUAL = 2
MARKER_START = 3
MARKER_TRAJECTORY = 4
MARKER_NOMINAL = 5


@dataclass(frozen=True)
class FootholdMarkerBatch:
    translations: torch.Tensor
    marker_indices: torch.Tensor


def _safe_getattr(obj: Any, name: str) -> Any | None:
    return getattr(obj, name, None) if obj is not None else None


def _select_env_tensor(value: Any | None, env_id: int) -> torch.Tensor | None:
    if value is None or not hasattr(value, "detach"):
        return None
    tensor = value.detach()
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.shape[0] <= env_id:
        return None
    return tensor[env_id]


def _select_env_int(value: Any | None, env_id: int) -> int | None:
    tensor = _select_env_tensor(value, env_id)
    if tensor is None:
        return None
    return int(tensor.reshape(-1)[0].item())


def _is_active_swing_mode(gait_mode: int | None) -> bool:
    return gait_mode in {
        1,  # LEFT_SWING
        2,  # RIGHT_SWING
        5,  # OVERDUE
        8,  # RECOVERY
    }


def _visible_planner_target_w(
    data: Any,
    *,
    env_id: int,
    gait_mode: int | None,
) -> torch.Tensor | None:
    """Return the latest planner proposal, even when it was not executable."""

    transaction_evaluated = _select_env_int(
        _safe_getattr(data, "learned_foothold_transaction_evaluated"),
        env_id,
    )
    proposal_w = _select_env_tensor(
        _safe_getattr(data, "learned_foothold_prepared_w"),
        env_id,
    )
    if (
        transaction_evaluated
        and proposal_w is not None
        and proposal_w.numel() == 3
        and bool(torch.isfinite(proposal_w).all().item())
    ):
        return proposal_w

    if _is_active_swing_mode(gait_mode):
        return _select_env_tensor(
            _safe_getattr(data, "target_foothold_w"), env_id
        )
    return None


def _visible_nominal_foothold_w(
    data: Any,
    *,
    env_id: int,
) -> torch.Tensor | None:
    """Return the prepared analytic nominal target for comparison."""

    nominal_w = _select_env_tensor(
        _safe_getattr(data, "nominal_foothold_w"),
        env_id,
    )
    if (
        nominal_w is None
        or nominal_w.numel() != 3
        or not bool(torch.isfinite(nominal_w).all().item())
    ):
        return None

    prepared = _safe_getattr(data, "nominal_foothold_prepared")
    if prepared is not None:
        prepared_value = _select_env_int(prepared, env_id)
        if prepared_value is not None and not bool(prepared_value):
            return None
    return nominal_w


def _build_trajectory_points(
    *,
    start_w: torch.Tensor,
    target_w: torch.Tensor,
    apex_height: torch.Tensor,
    samples: int,
    swing_duration_s: float,
) -> torch.Tensor:
    samples = max(int(samples), 2)
    phases = torch.linspace(
        0.0,
        1.0,
        samples,
        device=start_w.device,
        dtype=start_w.dtype,
    )
    reference = quintic_swing_reference(
        start=start_w.reshape(1, 3).expand(samples, 3),
        goal=target_w.reshape(1, 3).expand(samples, 3),
        phase=phases,
        apex_height=apex_height.reshape(1).expand(samples),
        swing_duration_s=swing_duration_s,
    )
    return reference.position


def build_foothold_marker_batch(
    data: Any,
    *,
    env_id: int = 0,
    trajectory_samples: int = 12,
    swing_duration_s: float = 0.8,
) -> FootholdMarkerBatch | None:
    gait_mode = _select_env_int(_safe_getattr(data, "gait_mode"), env_id)
    nominal_w = _visible_nominal_foothold_w(data, env_id=env_id)
    visible_target_w = _visible_planner_target_w(
        data,
        env_id=env_id,
        gait_mode=gait_mode,
    )
    if not _is_active_swing_mode(gait_mode):
        visible_points: list[torch.Tensor] = []
        visible_indices: list[int] = []
        if nominal_w is not None:
            visible_points.append(nominal_w.reshape(1, 3))
            visible_indices.append(MARKER_NOMINAL)
        if visible_target_w is not None:
            visible_points.append(visible_target_w.reshape(1, 3))
            visible_indices.append(MARKER_TARGET)
        if not visible_points:
            return None
        return FootholdMarkerBatch(
            translations=torch.cat(visible_points, dim=0),
            marker_indices=torch.tensor(
                visible_indices,
                device=visible_points[0].device,
                dtype=torch.long,
            ),
        )

    target_w = _select_env_tensor(_safe_getattr(data, "target_foothold_w"), env_id)
    reference_w = _select_env_tensor(_safe_getattr(data, "swing_reference_pos_w"), env_id)
    actual_w = _select_env_tensor(_safe_getattr(data, "actual_swing_foot_pos_w"), env_id)
    start_w = _select_env_tensor(_safe_getattr(data, "swing_start_pos_w"), env_id)
    apex_height = _select_env_tensor(_safe_getattr(data, "swing_apex_height"), env_id)

    required = (
        visible_target_w,
        target_w,
        reference_w,
        actual_w,
        start_w,
        apex_height,
    )
    if any(value is None for value in required):
        return None

    assert visible_target_w is not None
    assert target_w is not None
    assert reference_w is not None
    assert actual_w is not None
    assert start_w is not None
    assert apex_height is not None

    trajectory_w = _build_trajectory_points(
        start_w=start_w,
        target_w=target_w,
        apex_height=apex_height,
        samples=trajectory_samples,
        swing_duration_s=swing_duration_s,
    )

    marker_points = [
        visible_target_w.reshape(1, 3),
        reference_w.reshape(1, 3),
        actual_w.reshape(1, 3),
        start_w.reshape(1, 3),
    ]
    marker_indices_list = [
        MARKER_TARGET,
        MARKER_REFERENCE,
        MARKER_ACTUAL,
        MARKER_START,
    ]
    if nominal_w is not None:
        marker_points.insert(0, nominal_w.reshape(1, 3))
        marker_indices_list.insert(0, MARKER_NOMINAL)
    marker_points.append(trajectory_w)
    marker_indices_list.extend([MARKER_TRAJECTORY] * trajectory_w.shape[0])
    translations = torch.cat(marker_points, dim=0)
    marker_indices = torch.tensor(
        marker_indices_list,
        device=translations.device,
        dtype=torch.long,
    )
    return FootholdMarkerBatch(
        translations=translations,
        marker_indices=marker_indices,
    )


def make_foothold_visualizer():
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FootholdPlannerPlay",
        markers={
            "target_foothold": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.85, 0.0),
                ),
            ),
            "swing_reference": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.9, 1.0),
                ),
            ),
            "actual_swing_foot": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.25),
                ),
            ),
            "swing_start": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0),
                ),
            ),
            "reference_trajectory": sim_utils.SphereCfg(
                radius=0.014,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 1.0),
                ),
            ),
            "nominal_foothold": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def update_foothold_visualizer(
    visualizer: Any,
    data: Any,
    *,
    env_id: int = 0,
    trajectory_samples: int = 12,
    swing_duration_s: float = 0.8,
) -> bool:
    batch = build_foothold_marker_batch(
        data,
        env_id=env_id,
        trajectory_samples=trajectory_samples,
        swing_duration_s=swing_duration_s,
    )
    if batch is None:
        return False
    visualizer.visualize(
        translations=batch.translations,
        marker_indices=batch.marker_indices,
    )
    return True
