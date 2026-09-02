"""Read-only counterfactual diagnostics for the learned foothold policy."""

from __future__ import annotations

from math import prod
from typing import Mapping, Sequence

import torch

from instinctlab.learning.foothold_sac import radial_squash


def _component_slice(
    obs_segments: Mapping[str, Sequence[int]],
    component_name: str,
) -> slice:
    if component_name not in obs_segments:
        raise KeyError(f"observation component not found: {component_name}")
    start = 0
    for name, shape in obs_segments.items():
        width = int(prod(shape))
        if name == component_name:
            return slice(start, start + width)
        start += width
    raise KeyError(f"observation component not found: {component_name}")


def replace_nominal_lateral(
    observations: torch.Tensor,
    obs_segments: Mapping[str, Sequence[int]],
    nominal_y_m: float,
) -> torch.Tensor:
    """Clone observations and replace only nominal foothold Y."""

    component_shape = tuple(obs_segments.get("nominal_foothold", ()))
    if not component_shape:
        raise KeyError("observation component not found: nominal_foothold")
    if int(prod(component_shape)) != 3:
        raise ValueError("nominal_foothold must contain three coordinates")
    component = _component_slice(obs_segments, "nominal_foothold")
    if observations.shape[-1] < component.stop:
        raise ValueError("observation width does not match obs_segments")
    replaced = observations.clone()
    replaced[..., component.start + 1] = float(nominal_y_m)
    return replaced


def _nominal_lateral(
    observations: torch.Tensor,
    obs_segments: Mapping[str, Sequence[int]],
) -> float:
    component = _component_slice(obs_segments, "nominal_foothold")
    return float(observations[0, component.start + 1].detach().cpu().item())


@torch.no_grad()
def diagnose_foothold_policy(
    actor_critic,
    sac,
    observation: torch.Tensor,
    *,
    radius_x_m: float,
    radius_y_m: float,
    nominal_y_values_m: tuple[float, ...] = (-0.18, 0.18),
    q_grid_size: int = 51,
) -> dict:
    """Evaluate actor and critic sensitivity to counterfactual nominal Y."""

    if radius_x_m <= 0.0 or radius_y_m <= 0.0:
        raise ValueError("reachability radii must be positive")
    if q_grid_size < 3:
        raise ValueError("q_grid_size must be at least three")
    if observation.ndim == 1:
        observation = observation.unsqueeze(0)
    if observation.ndim != 2 or observation.shape[0] != 1:
        raise ValueError("diagnostic observation must contain exactly one row")

    obs_segments = actor_critic.obs_segments
    original_features = actor_critic.planner_features(
        observation,
        detach_shared=True,
    )
    result = {
        "original_nominal_y_m": _nominal_lateral(
            observation,
            obs_segments,
        ),
        "counterfactuals": [],
    }
    radii = observation.new_tensor((radius_x_m, radius_y_m))

    for nominal_y_m in nominal_y_values_m:
        counterfactual = replace_nominal_lateral(
            observation,
            obs_segments,
            nominal_y_m,
        )
        features = actor_critic.planner_features(
            counterfactual,
            detach_shared=True,
        )
        distribution = actor_critic.planner_distribution_from_features(
            features
        )
        raw_mean = distribution.mean
        normalized_mean = radial_squash(raw_mean)
        decoded_xy_m = normalized_mean * radii

        fixed_x = normalized_mean[0, 0]
        lateral_limit = torch.sqrt(
            torch.clamp(1.0 - fixed_x.square(), min=0.0)
        )
        lateral_grid = torch.linspace(
            -float(lateral_limit.item()),
            float(lateral_limit.item()),
            q_grid_size,
            device=features.device,
            dtype=features.dtype,
        )
        actions = torch.stack(
            (fixed_x.expand_as(lateral_grid), lateral_grid),
            dim=-1,
        )
        repeated_features = features.expand(q_grid_size, -1)
        q_values = torch.minimum(
            sac.critic_1(repeated_features, actions),
            sac.critic_2(repeated_features, actions),
        )
        best_index = int(torch.argmax(q_values).item())
        center_index = int(torch.argmin(torch.abs(lateral_grid)).item())
        nominal_normalized_y = torch.clamp(
            lateral_grid.new_tensor(float(nominal_y_m) / radius_y_m),
            min=-lateral_limit,
            max=lateral_limit,
        )
        nominal_index = int(
            torch.argmin(torch.abs(lateral_grid - nominal_normalized_y)).item()
        )

        result["counterfactuals"].append(
            {
                "nominal_y_m": float(nominal_y_m),
                "feature_delta_norm": float(
                    torch.linalg.vector_norm(
                        features - original_features
                    ).item()
                ),
                "actor_raw_mean": raw_mean[0].detach().cpu().tolist(),
                "actor_action_normalized": (
                    normalized_mean[0].detach().cpu().tolist()
                ),
                "decoded_foothold_xy_m": (
                    decoded_xy_m[0].detach().cpu().tolist()
                ),
                "best_q_foothold_y_m": float(
                    lateral_grid[best_index].item() * radius_y_m
                ),
                "best_q": float(q_values[best_index].item()),
                "center_q": float(q_values[center_index].item()),
                "nominal_q": float(q_values[nominal_index].item()),
                "q_min": float(q_values.min().item()),
                "q_max": float(q_values.max().item()),
                "q_scan_normalized_y_max": float(lateral_limit.item()),
            }
        )
    return result


def format_policy_sensitivity(result: dict) -> str:
    fields = [
        "[FOOTHOLD_POLICY_SENSITIVITY]",
        f"original_nominal_y={result['original_nominal_y_m']:+.5f}",
    ]
    for item in result["counterfactuals"]:
        fields.append(
            "nominal_y="
            f"{item['nominal_y_m']:+.5f} "
            f"feature_delta={item['feature_delta_norm']:.6g} "
            f"action_n={item['actor_action_normalized']} "
            f"decoded_xy_m={item['decoded_foothold_xy_m']}"
        )
    return " ".join(fields)


def format_q_sweep(result: dict, nominal_y_m: float) -> str:
    item = next(
        entry
        for entry in result["counterfactuals"]
        if abs(entry["nominal_y_m"] - nominal_y_m) < 1.0e-8
    )
    return (
        "[FOOTHOLD_Q_SWEEP] "
        f"nominal_y={item['nominal_y_m']:+.5f} "
        f"best_q_y={item['best_q_foothold_y_m']:+.5f} "
        f"best_q={item['best_q']:+.6f} "
        f"nominal_q={item['nominal_q']:+.6f} "
        f"center_q={item['center_q']:+.6f} "
        f"q_range=[{item['q_min']:+.6f},{item['q_max']:+.6f}] "
        f"scan_y_n=±{item['q_scan_normalized_y_max']:.5f}"
    )
