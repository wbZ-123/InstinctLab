import importlib.util
from pathlib import Path

import pytest
import torch


def _load_storage_types():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/learning/foothold_rollout_storage.py"
    )
    spec = importlib.util.spec_from_file_location(
        "foothold_rollout_storage_for_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FootholdRolloutStorage, module.FootholdTransition


def _make_transition(FootholdTransition):
    transition = FootholdTransition()
    transition.observations = torch.zeros(2, 3)
    transition.critic_observations = torch.zeros(2, 3)
    transition.actions = torch.zeros(2, 31)
    transition.rewards = torch.zeros(2, 2)
    transition.dones = torch.zeros(2, dtype=torch.long)
    transition.values = torch.zeros(2, 2)
    transition.actions_log_prob = torch.zeros(2)
    transition.action_mean = torch.zeros(2, 31)
    transition.action_sigma = torch.ones(2, 31)
    transition.foothold_action_event = torch.tensor([True, False])
    transition.foothold_nominal_safe_event = torch.tensor([True, False])
    transition.foothold_nominal_unsafe_event = torch.tensor([False, False])
    return transition


def test_storage_keeps_event_mask_aligned_with_transition():
    FootholdRolloutStorage, FootholdTransition = _load_storage_types()
    storage = FootholdRolloutStorage(
        2,
        1,
        [3],
        [3],
        [31],
        num_rewards=2,
        device="cpu",
    )

    storage.add_transitions(_make_transition(FootholdTransition))

    assert storage.foothold_action_event[0].tolist() == [True, False]
    assert storage.foothold_nominal_safe_event[0].tolist() == [True, False]
    assert storage.foothold_nominal_unsafe_event[0].tolist() == [False, False]


def test_minibatch_keeps_event_mask_aligned_with_selected_rows():
    FootholdRolloutStorage, FootholdTransition = _load_storage_types()
    storage = FootholdRolloutStorage(
        2,
        2,
        [3],
        [3],
        [31],
        num_rewards=2,
        device="cpu",
    )
    first = _make_transition(FootholdTransition)
    storage.add_transitions(first)
    second = _make_transition(FootholdTransition)
    second.foothold_action_event = torch.tensor([False, True])
    second.foothold_nominal_safe_event = torch.tensor([False, False])
    second.foothold_nominal_unsafe_event = torch.tensor([False, True])
    storage.add_transitions(second)

    minibatch = storage.get_minibatch_from_selection(
        torch.tensor([1, 0]),
        torch.tensor([1, 0]),
    )

    assert minibatch.foothold_action_event.dtype == torch.bool
    assert minibatch.foothold_action_event.tolist() == [True, True]
    assert minibatch.foothold_nominal_safe_event.tolist() == [False, True]
    assert minibatch.foothold_nominal_unsafe_event.tolist() == [True, False]


def test_storage_rejects_missing_event_mask():
    FootholdRolloutStorage, FootholdTransition = _load_storage_types()
    storage = FootholdRolloutStorage(
        2,
        1,
        [3],
        [3],
        [31],
        num_rewards=2,
        device="cpu",
    )
    transition = _make_transition(FootholdTransition)
    transition.foothold_action_event = None

    with pytest.raises(ValueError, match="event"):
        storage.add_transitions(transition)


def test_storage_rejects_missing_nominal_branch_masks():
    FootholdRolloutStorage, FootholdTransition = _load_storage_types()
    storage = FootholdRolloutStorage(
        2,
        1,
        [3],
        [3],
        [31],
        num_rewards=2,
        device="cpu",
    )
    transition = _make_transition(FootholdTransition)
    transition.foothold_nominal_unsafe_event = None

    with pytest.raises(ValueError, match="nominal unsafe"):
        storage.add_transitions(transition)


def test_storage_rejects_non_boolean_event_mask():
    FootholdRolloutStorage, FootholdTransition = _load_storage_types()
    storage = FootholdRolloutStorage(
        2,
        1,
        [3],
        [3],
        [31],
        num_rewards=2,
        device="cpu",
    )
    transition = _make_transition(FootholdTransition)
    transition.foothold_action_event = torch.tensor([1, 0])

    with pytest.raises(TypeError, match="bool"):
        storage.add_transitions(transition)


@pytest.mark.parametrize(
    "safe_mask, unsafe_mask, error",
    [
        (torch.tensor([1, 0]), torch.tensor([False, False]), "bool"),
        (torch.tensor([True]), torch.tensor([False, False]), "shape"),
        (torch.tensor([True, False]), torch.tensor([True, False]), "overlap"),
        (torch.tensor([False, False]), torch.tensor([False, False]), "union"),
        (torch.tensor([False, False]), torch.tensor([True, True]), "union"),
    ],
)
def test_storage_rejects_invalid_nominal_branch_masks(
    safe_mask,
    unsafe_mask,
    error,
):
    FootholdRolloutStorage, FootholdTransition = _load_storage_types()
    storage = FootholdRolloutStorage(
        2,
        1,
        [3],
        [3],
        [31],
        num_rewards=2,
        device="cpu",
    )
    transition = _make_transition(FootholdTransition)
    transition.foothold_nominal_safe_event = safe_mask
    transition.foothold_nominal_unsafe_event = unsafe_mask

    with pytest.raises((TypeError, ValueError), match=error):
        storage.add_transitions(transition)


def test_advantages_are_normalized_independently_per_reward_group():
    FootholdRolloutStorage, FootholdTransition = _load_storage_types()
    storage = FootholdRolloutStorage(
        2,
        1,
        [3],
        [3],
        [31],
        num_rewards=2,
        device="cpu",
    )
    transition = _make_transition(FootholdTransition)
    transition.rewards = torch.tensor(
        [[1.0, 100.0], [3.0, 300.0]],
    )
    storage.add_transitions(transition)

    storage.compute_returns(
        last_values=torch.zeros(2, 2),
        gamma=0.0,
        lam=0.0,
    )

    torch.testing.assert_close(
        storage.advantages.mean(dim=(0, 1)),
        torch.zeros(2),
        atol=1.0e-6,
        rtol=0.0,
    )
