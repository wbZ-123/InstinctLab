import ast
from pathlib import Path

import pytest
import torch


def _load_foothold_event_from_generation():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py"
    )
    module = ast.parse(module_path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "foothold_event_from_generation"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(module_path), "exec"), namespace)
    return namespace["foothold_event_from_generation"]


def test_generation_change_marks_only_consumed_transitions():
    foothold_event_from_generation = _load_foothold_event_from_generation()
    before = torch.tensor([4, 7, 9], dtype=torch.int64)
    after = torch.tensor([5, 7, 10], dtype=torch.int64)

    assert foothold_event_from_generation(before, after).tolist() == [
        True,
        False,
        True,
    ]


def test_generation_is_monotonic_across_environment_reset():
    foothold_event_from_generation = _load_foothold_event_from_generation()
    before = torch.tensor([11], dtype=torch.int64)
    after_reset = torch.tensor([12], dtype=torch.int64)

    assert foothold_event_from_generation(before, after_reset).item()


def test_generation_decrease_is_rejected_as_reset_corruption():
    foothold_event_from_generation = _load_foothold_event_from_generation()

    with pytest.raises(ValueError, match="monotonic"):
        foothold_event_from_generation(
            torch.tensor([11], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )


def test_generation_shape_mismatch_is_rejected():
    foothold_event_from_generation = _load_foothold_event_from_generation()

    with pytest.raises(ValueError, match="shapes"):
        foothold_event_from_generation(
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([1, 2], dtype=torch.int64),
        )


def test_generation_requires_int64():
    foothold_event_from_generation = _load_foothold_event_from_generation()

    with pytest.raises(TypeError, match="int64"):
        foothold_event_from_generation(
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        )
