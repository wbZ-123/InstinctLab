from dataclasses import fields
import importlib.util
from pathlib import Path


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
