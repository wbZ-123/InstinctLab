from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_play_foothold_viz_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "instinct_rl"
        / "play_foothold_viz.py"
    )
    spec = importlib.util.spec_from_file_location("play_foothold_viz_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_foothold_marker_batch_contains_points_and_trajectory_samples():
    module = _load_play_foothold_viz_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([1]),
        swing_start_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        target_foothold_w=torch.tensor([[1.0, 0.0, 0.0]]),
        swing_reference_pos_w=torch.tensor([[0.5, 0.0, 0.2]]),
        actual_swing_foot_pos_w=torch.tensor([[0.45, 0.02, 0.18]]),
        swing_apex_height=torch.tensor([0.2]),
    )

    batch = module.build_foothold_marker_batch(
        data,
        env_id=0,
        trajectory_samples=5,
        swing_duration_s=0.8,
    )

    assert batch is not None
    assert batch.translations.shape == (9, 3)
    assert batch.marker_indices.tolist() == [
        module.MARKER_TARGET,
        module.MARKER_REFERENCE,
        module.MARKER_ACTUAL,
        module.MARKER_START,
        module.MARKER_TRAJECTORY,
        module.MARKER_TRAJECTORY,
        module.MARKER_TRAJECTORY,
        module.MARKER_TRAJECTORY,
        module.MARKER_TRAJECTORY,
    ]
    assert torch.allclose(batch.translations[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(batch.translations[1], torch.tensor([0.5, 0.0, 0.2]))
    assert torch.allclose(batch.translations[2], torch.tensor([0.45, 0.02, 0.18]))
    assert torch.allclose(batch.translations[3], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(batch.translations[4], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(batch.translations[-1], torch.tensor([1.0, 0.0, 0.0]))


def test_build_foothold_marker_batch_returns_none_when_required_data_is_missing():
    module = _load_play_foothold_viz_module()

    assert module.build_foothold_marker_batch(SimpleNamespace(), env_id=0) is None


def test_build_foothold_marker_batch_returns_none_when_not_active_swing():
    module = _load_play_foothold_viz_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([0]),
        swing_start_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        target_foothold_w=torch.tensor([[1.0, 0.0, 0.0]]),
        swing_reference_pos_w=torch.tensor([[0.5, 0.0, 0.2]]),
        actual_swing_foot_pos_w=torch.tensor([[0.45, 0.02, 0.18]]),
        swing_apex_height=torch.tensor([0.2]),
    )

    assert module.build_foothold_marker_batch(data, env_id=0) is None


def test_build_foothold_marker_batch_shows_unsafe_planner_proposal_during_hold():
    module = _load_play_foothold_viz_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([0]),
        learned_foothold_transaction_evaluated=torch.tensor([True]),
        learned_foothold_prepared_w=torch.tensor([[1.2, -0.3, 0.25]]),
        learned_foothold_safety_valid=torch.tensor([False]),
    )

    batch = module.build_foothold_marker_batch(data, env_id=0)

    assert batch is not None
    assert batch.translations.shape == (1, 3)
    assert batch.marker_indices.tolist() == [module.MARKER_TARGET]
    assert torch.allclose(
        batch.translations[0], torch.tensor([1.2, -0.3, 0.25])
    )
