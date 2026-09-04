import importlib.util
from pathlib import Path

import pytest
import torch


def _load_replay_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/learning/foothold_sac_replay.py"
    )
    spec = importlib.util.spec_from_file_location(
        "foothold_sac_replay_under_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_event_replay_add_and_sample_have_expected_shapes():
    module = _load_replay_module()
    replay = module.FootholdReplayBuffer(
        capacity=8,
        obs_dim=4,
        action_dim=2,
        device="cpu",
    )

    replay.add(
        torch.zeros(3, 4),
        torch.ones(3, 2),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.ones(3, 4),
        torch.tensor([False, True, False]),
    )

    assert len(replay) == 3
    batch = replay.sample(2)
    assert batch.obs.shape == (2, 4)
    assert batch.actions.shape == (2, 2)
    assert batch.rewards.shape == (2,)
    assert batch.next_obs.shape == (2, 4)
    assert batch.dones.shape == (2,)
    assert batch.nominal_safe.shape == (2,)
    assert torch.all(batch.actions == 1.0)


def test_event_replay_balances_safe_and_unsafe_branches():
    module = _load_replay_module()
    replay = module.FootholdReplayBuffer(16, 2, 2, "cpu")
    replay.add(
        torch.zeros(6, 2),
        torch.zeros(6, 2),
        torch.zeros(6),
        torch.zeros(6, 2),
        torch.zeros(6, dtype=torch.bool),
        nominal_safe=torch.tensor([True, True, True, True, False, False]),
    )
    batch = replay.sample(6, balanced_branches=True)
    assert int(batch.nominal_safe.sum()) == 3
    assert int((~batch.nominal_safe).sum()) == 3


def test_event_replay_is_circular_and_round_trips_state():
    module = _load_replay_module()
    replay = module.FootholdReplayBuffer(3, 1, 2, "cpu")
    for value in range(5):
        replay.add(
            torch.tensor([[float(value)]]),
            torch.zeros(1, 2),
            torch.tensor([float(value)]),
            torch.tensor([[float(value + 1)]]),
            torch.tensor([value == 4]),
        )

    assert len(replay) == 3
    restored = module.FootholdReplayBuffer(3, 1, 2, "cpu")
    restored.load_state_dict(replay.state_dict())
    assert len(restored) == 3
    assert restored.position == replay.position
    assert restored.size == replay.size
    torch.testing.assert_close(restored.rewards, replay.rewards)
    torch.testing.assert_close(
        torch.sort(replay.rewards).values,
        torch.tensor([2.0, 3.0, 4.0]),
    )


def test_event_replay_rejects_invalid_shapes_and_empty_sampling():
    module = _load_replay_module()
    replay = module.FootholdReplayBuffer(4, 3, 2, "cpu")

    with pytest.raises(ValueError, match="capacity"):
        module.FootholdReplayBuffer(0, 3, 2, "cpu")
    with pytest.raises(ValueError, match="obs_dim"):
        module.FootholdReplayBuffer(4, 0, 2, "cpu")
    with pytest.raises(ValueError, match="batch_size"):
        replay.sample(1)
    with pytest.raises(ValueError, match="obs"):
        replay.add(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.zeros(1, 3),
            torch.zeros(1, dtype=torch.bool),
        )
