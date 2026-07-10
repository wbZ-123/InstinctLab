from dataclasses import fields
from collections.abc import Sequence
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_foothold_planner_data_class():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "foothold_planner_data_for_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FootholdPlannerData


def _load_foothold_planner_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/sensors/foothold_planner/foothold_planner.py"
    )
    source = module_path.read_text()
    start = source.index("def _clear_safe_target_event_buffers")
    end = source.index("class FootholdPlanner")
    namespace = {
        "FootholdPlannerData": object,
        "Sequence": Sequence,
        "torch": torch,
    }
    exec(source[start:end], namespace)
    return SimpleNamespace(
        clear_safe_target_event_buffers=namespace[
            "_clear_safe_target_event_buffers"
        ]
    )


def test_foothold_planner_data_exposes_clearance_debug_fields():
    FootholdPlannerData = _load_foothold_planner_data_class()
    field_names = {field.name for field in fields(FootholdPlannerData)}

    assert {
        "default_swing_reference_pos_w",
        "default_swing_apex_height",
        "swing_apex_height",
        "swing_clearance_safe",
        "swing_clearance_penetration",
    }.issubset(field_names)


def test_clear_safe_target_event_buffers_resets_event_fields_only_for_selected_envs():
    module = _load_foothold_planner_module()
    data = SimpleNamespace(
        safe_target_search_performed=torch.ones(3, dtype=torch.bool),
        safe_target_final_valid=torch.zeros(3, dtype=torch.bool),
        safe_target_used_fallback=torch.ones(3, dtype=torch.bool),
        safe_target_score=torch.ones(3),
        safe_target_nominal_inside_ellipse=torch.zeros(3, dtype=torch.bool),
        safe_target_nominal_obstacle_safe=torch.zeros(3, dtype=torch.bool),
        safe_target_nominal_valid=torch.zeros(3, dtype=torch.bool),
        safe_target_candidate_count=torch.ones(3) * 32.0,
        safe_target_candidate_inside_ellipse_count=torch.ones(3) * 24.0,
        safe_target_candidate_obstacle_safe_count=torch.ones(3) * 8.0,
        safe_target_candidate_valid_count=torch.ones(3) * 4.0,
    )

    module.clear_safe_target_event_buffers(data, torch.tensor([0, 2]))

    torch.testing.assert_close(
        data.safe_target_search_performed,
        torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(
        data.safe_target_final_valid,
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        data.safe_target_used_fallback,
        torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(data.safe_target_score, torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(
        data.safe_target_nominal_inside_ellipse,
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        data.safe_target_nominal_obstacle_safe,
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        data.safe_target_nominal_valid,
        torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(
        data.safe_target_candidate_count, torch.tensor([0.0, 32.0, 0.0])
    )
    torch.testing.assert_close(
        data.safe_target_candidate_inside_ellipse_count,
        torch.tensor([0.0, 24.0, 0.0]),
    )
    torch.testing.assert_close(
        data.safe_target_candidate_obstacle_safe_count,
        torch.tensor([0.0, 8.0, 0.0]),
    )
    torch.testing.assert_close(
        data.safe_target_candidate_valid_count,
        torch.tensor([0.0, 4.0, 0.0]),
    )
