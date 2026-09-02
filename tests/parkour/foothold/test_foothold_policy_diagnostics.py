from __future__ import annotations

from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[3]
    / "scripts"
    / "instinct_rl"
    / "foothold_policy_diagnostics.py"
)


def _load_module():
    foothold_sac_name = "instinctlab.learning.foothold_sac"
    previous_foothold_sac = sys.modules.get(foothold_sac_name)
    foothold_sac = ModuleType(foothold_sac_name)

    def radial_squash(raw_action):
        return raw_action / torch.sqrt(
            1.0 + raw_action.square().sum(dim=-1, keepdim=True)
        )

    foothold_sac.radial_squash = radial_squash
    sys.modules[foothold_sac_name] = foothold_sac
    spec = importlib.util.spec_from_file_location(
        "foothold_policy_diagnostics",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_foothold_sac is None:
            del sys.modules[foothold_sac_name]
        else:
            sys.modules[foothold_sac_name] = previous_foothold_sac
    return module


OBS_SEGMENTS = OrderedDict(
    (
        ("proprioception", (2,)),
        ("nominal_foothold", (3,)),
        ("depth_image", (1,)),
    )
)


def _observation(nominal_y: float = 0.12) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 2.0, 0.15, nominal_y, -0.02, 7.0]],
        dtype=torch.float32,
    )


def test_replace_nominal_lateral_changes_only_named_component():
    module = _load_module()
    observation = _observation()

    replaced = module.replace_nominal_lateral(
        observation,
        OBS_SEGMENTS,
        -0.18,
    )

    assert torch.equal(
        observation,
        _observation(),
    ), "the rollout observation must remain untouched"
    expected = _observation()
    expected[0, 3] = -0.18
    assert torch.equal(replaced, expected)


def test_replace_nominal_lateral_rejects_missing_component():
    module = _load_module()

    with pytest.raises(KeyError, match="nominal_foothold"):
        module.replace_nominal_lateral(
            _observation(),
            OrderedDict((entry for entry in OBS_SEGMENTS.items() if entry[0] != "nominal_foothold")),
            -0.18,
        )


def test_replace_nominal_lateral_requires_xyz_nominal_component():
    module = _load_module()
    invalid_segments = OrderedDict(
        (
            ("proprioception", (2,)),
            ("nominal_foothold", (1,)),
            ("unused", (3,)),
        )
    )

    with pytest.raises(ValueError, match="three coordinates"):
        module.replace_nominal_lateral(
            _observation(),
            invalid_segments,
            -0.18,
        )


class _FakeActorCritic:
    obs_segments = OBS_SEGMENTS

    def planner_features(self, observations, *, detach_shared=True):
        assert detach_shared is True
        return observations.detach()

    def planner_distribution_from_features(self, features):
        nominal_y = features[:, 3]
        mean = torch.stack((torch.zeros_like(nominal_y), nominal_y), dim=-1)
        return torch.distributions.Normal(mean, torch.full_like(mean, 0.1))


class _NominalSeekingCritic(torch.nn.Module):
    def __init__(self, radius_y_m: float):
        super().__init__()
        self.radius_y_m = float(radius_y_m)

    def forward(self, features, actions):
        nominal_y = features[:, 3]
        action_y_m = actions[:, 1] * self.radius_y_m
        return -torch.square(action_y_m - nominal_y)


def test_diagnose_foothold_policy_detects_actor_and_q_nominal_sensitivity():
    module = _load_module()
    radius_y_m = 0.25
    critic = _NominalSeekingCritic(radius_y_m)
    sac = SimpleNamespace(critic_1=critic, critic_2=critic)

    result = module.diagnose_foothold_policy(
        _FakeActorCritic(),
        sac,
        _observation(nominal_y=0.12),
        radius_x_m=0.50,
        radius_y_m=radius_y_m,
        nominal_y_values_m=(-0.18, 0.18),
        q_grid_size=101,
    )

    assert result["original_nominal_y_m"] == pytest.approx(0.12)
    assert len(result["counterfactuals"]) == 2
    negative, positive = result["counterfactuals"]
    assert negative["nominal_y_m"] == pytest.approx(-0.18)
    assert positive["nominal_y_m"] == pytest.approx(0.18)
    assert negative["feature_delta_norm"] > 0.0
    assert positive["feature_delta_norm"] > 0.0
    assert negative["actor_action_normalized"][1] < 0.0
    assert positive["actor_action_normalized"][1] > 0.0
    assert negative["best_q_foothold_y_m"] == pytest.approx(-0.18, abs=0.006)
    assert positive["best_q_foothold_y_m"] == pytest.approx(0.18, abs=0.006)
    assert negative["nominal_q"] > negative["center_q"]
    assert positive["nominal_q"] > positive["center_q"]
    assert abs(negative["q_scan_normalized_y_max"]) <= 1.0
    assert abs(positive["q_scan_normalized_y_max"]) <= 1.0


def test_diagnostic_formatters_use_distinct_stable_prefixes():
    module = _load_module()
    radius_y_m = 0.25
    critic = _NominalSeekingCritic(radius_y_m)
    result = module.diagnose_foothold_policy(
        _FakeActorCritic(),
        SimpleNamespace(critic_1=critic, critic_2=critic),
        _observation(),
        radius_x_m=0.50,
        radius_y_m=radius_y_m,
        q_grid_size=51,
    )

    sensitivity = module.format_policy_sensitivity(result)
    q_line = module.format_q_sweep(result, nominal_y_m=-0.18)

    assert sensitivity.startswith("[FOOTHOLD_POLICY_SENSITIVITY]")
    assert "nominal_y=-0.18000" in sensitivity
    assert "nominal_y=+0.18000" in sensitivity
    assert q_line.startswith("[FOOTHOLD_Q_SWEEP]")
    assert "nominal_y=-0.18000" in q_line
    assert "best_q_y=" in q_line
