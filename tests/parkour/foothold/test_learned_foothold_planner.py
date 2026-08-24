import importlib.util
import inspect
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import torch
import instinctlab_foothold.learned_target as learned_target

from instinctlab_foothold.learned_target import (
    LEARNED_FOOTHOLD_ROUTE_REASON_ENDPOINT_UNSAFE,
    LEARNED_FOOTHOLD_ROUTE_REASON_GEOMETRIC_INVALID,
    LEARNED_FOOTHOLD_ROUTE_REASON_PREFLIGHT_UNSAFE,
    LEARNED_FOOTHOLD_ROUTE_REASON_POSTCHECK_INVALID,
    LEARNED_FOOTHOLD_ROUTE_REASON_RECOVERY,
    LEARNED_FOOTHOLD_ROUTE_REASON_SUCCESS,
    LEARNED_FOOTHOLD_ROUTE_REASON_TRANSACTION_INVALIDATED,
    classify_learned_foothold_route,
    finalize_learned_foothold_route_outcome,
    LearnedFootholdPreparation,
    clear_learned_foothold_buffers,
    decode_normalized_foothold,
    learned_foothold_event_masks,
    learned_foothold_swing_ready,
    learned_foothold_transaction_ready,
    nominal_foothold_prepare_mask,
    lock_prepared_learned_foothold,
    prepare_learned_foothold_target,
    reachable_ellipse_usage,
    reframe_cached_world_foothold,
    route_nominal_and_learned_footholds,
    store_learned_foothold_preparation,
)


def test_learned_route_reason_classifies_the_first_failed_gate():
    reason = classify_learned_foothold_route(
        recovery_step=torch.tensor([True, False, False, False, False, False]),
        transaction_evaluated=torch.tensor(
            [True, False, True, True, True, True]
        ),
        learned_geometric_valid=torch.tensor(
            [True, True, False, True, True, True]
        ),
        learned_safety_valid=torch.tensor(
            [True, True, True, False, True, True]
        ),
        preflight_ready=torch.tensor(
            [True, True, True, True, True, True]
        ),
        preflight_safe=torch.tensor(
            [True, True, True, True, False, True]
        ),
        route_use_learned=torch.tensor(
            [False, False, False, False, False, True]
        ),
    )

    assert reason.tolist() == [
        LEARNED_FOOTHOLD_ROUTE_REASON_RECOVERY,
        LEARNED_FOOTHOLD_ROUTE_REASON_TRANSACTION_INVALIDATED,
        LEARNED_FOOTHOLD_ROUTE_REASON_GEOMETRIC_INVALID,
        LEARNED_FOOTHOLD_ROUTE_REASON_ENDPOINT_UNSAFE,
        LEARNED_FOOTHOLD_ROUTE_REASON_PREFLIGHT_UNSAFE,
        LEARNED_FOOTHOLD_ROUTE_REASON_SUCCESS,
    ]


def test_learned_route_reason_is_reclassified_after_final_lock_check():
    outcome = finalize_learned_foothold_route_outcome(
        initial_outcome=torch.tensor([
            LEARNED_FOOTHOLD_ROUTE_REASON_SUCCESS,
            LEARNED_FOOTHOLD_ROUTE_REASON_SUCCESS,
            LEARNED_FOOTHOLD_ROUTE_REASON_ENDPOINT_UNSAFE,
        ]),
        route_initial_executable=torch.tensor([True, True, False]),
        final_planner_valid=torch.tensor([True, False, False]),
    )
    assert outcome.tolist() == [
        LEARNED_FOOTHOLD_ROUTE_REASON_SUCCESS,
        LEARNED_FOOTHOLD_ROUTE_REASON_POSTCHECK_INVALID,
        LEARNED_FOOTHOLD_ROUTE_REASON_ENDPOINT_UNSAFE,
    ]


def test_hold_preflight_uses_prepared_learned_target_before_swing_lock():
    target = learned_target.select_preflight_target_w(
        route_use_learned=torch.tensor([True, False]),
        learned_prepared_w=torch.tensor(
            [[1.0, 2.0, 0.3], [3.0, 4.0, 0.5]]
        ),
        nominal_target_w=torch.tensor(
            [[10.0, 20.0, 0.3], [30.0, 40.0, 0.5]]
        ),
    )

    torch.testing.assert_close(
        target,
        torch.tensor([[1.0, 2.0, 0.3], [30.0, 40.0, 0.5]]),
    )


def test_nominal_foothold_waits_for_confirmed_hold_contact():
    mask = nominal_foothold_prepare_mask(
        hold=torch.tensor([True, True, True, False]),
        hold_contact_ready=torch.tensor([False, True, True, True]),
        nominal_ready=torch.tensor([False, False, True, False]),
        startup_hold=torch.tensor([False, False, False, False]),
    )
    assert mask.tolist() == [False, True, False, False]


def test_reachable_ellipse_usage_reports_lateral_overflow():
    usage = reachable_ellipse_usage(
        torch.tensor([[0.0, 0.30]]),
        radius_x=0.42,
        radius_y=0.25,
    )
    assert usage.item() > 1.0


def test_safe_nominal_locks_one_unsafe_learned_proposal_for_the_hold_transaction():
    ready = learned_foothold_transaction_ready(
        nominal_route_ready=torch.tensor([True, True, False]),
        transaction_evaluated=torch.tensor([True, True, True]),
        learned_prepared_valid=torch.tensor([True, True, True]),
        learned_geometric_valid=torch.tensor([True, True, True]),
        learned_safety_valid=torch.tensor([False, True, False]),
    )

    # First row keeps the unsafe action's PPO penalty but uses the safe
    # nominal instead of replacing the action again in the same HOLD.
    # A safe learned endpoint also completes the proposal transaction.  A
    # trajectory failure later falls back to nominal without resampling the
    # high-level action.
    assert ready.tolist() == [True, True, False]


def _load_foothold_planner_data_class():
    path = (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "sensors"
        / "foothold_planner"
        / "foothold_planner_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "learned_foothold_planner_data_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FootholdPlannerData


def _planner_source_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "sensors"
        / "foothold_planner"
        / name
    )


def _parkour_env_cfg_source_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "tasks"
        / "parkour"
        / "config"
        / "parkour_env_cfg.py"
    )


def _parkour_agent_cfg_source_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "tasks"
        / "parkour"
        / "config"
        / "g1"
        / "agents"
        / "instinct_rl_amp_cfg.py"
    )


def test_learned_planner_config_is_opt_in_without_duplicate_meter_limits():
    cfg_text = _planner_source_path("foothold_planner_cfg.py").read_text()

    assert "enable_learned_foothold: bool = False" in cfg_text
    assert "learned_foothold_x_range" not in cfg_text
    assert "learned_foothold_y_range" not in cfg_text
    assert "learned_foothold_step_height_limit_m" not in cfg_text


def test_learned_planner_enables_the_existing_autonomous_recovery_baseline():
    cfg_text = _parkour_env_cfg_source_path().read_text()
    learned_enable = cfg_text.index("    def enable_learned_foothold_planner(self)")
    recovery_enable = cfg_text.index("    def enable_contact_adaptive_recovery(")
    learned_block = cfg_text[learned_enable:recovery_enable]

    assert "_FOOTHOLD_RECOVERY_STABILITY_SEED_PATH" in cfg_text
    assert "if not self.scene.foothold_planner.enable_contact_adaptive_recovery:" in learned_block
    assert "self.enable_contact_adaptive_recovery(" in learned_block


def test_late_touchdown_search_is_limited_to_existing_touchdown_z_tolerance():
    planner_text = _planner_source_path("foothold_planner.py").read_text()

    assert "late_search_exhausted = late_elapsed >= self.cfg.overdue_s" in planner_text
    assert "self.cfg.max_foothold_step_height_m / late_speed" not in planner_text


def test_learned_recovery_cannot_create_or_execute_an_analytic_recovery_step():
    planner_text = _planner_source_path("foothold_planner.py").read_text()
    learned_block = planner_text[
        planner_text.index("if self.cfg.enable_learned_foothold:") :
        planner_text.index("else:\n                flat_result = sample_flat_targets", planner_text.index("if self.cfg.enable_learned_foothold:"))
    ]

    assert "make_recovery_foothold_target" not in learned_block
    assert "recovery_step.zero_()" in planner_text


def test_environment_enables_nominal_observation_only_with_learned_action():
    cfg_text = _parkour_env_cfg_source_path().read_text()

    assert "learned_foothold: mdp.LearnedFootholdActionCfg | None = None" in cfg_text
    assert "def enable_learned_foothold_planner(self)" in cfg_text
    assert "self.actions.learned_foothold = mdp.LearnedFootholdActionCfg(" in cfg_text
    assert "include_nominal_foothold" not in cfg_text
    assert "nominal_foothold: ObsTerm | None = None" in cfg_text
    assert "self.observations.policy.nominal_foothold = ObsTerm(" in cfg_text
    assert "self.observations.critic.nominal_foothold = ObsTerm(" in cfg_text
    assert cfg_text.count("func=mdp.nominal_foothold_observation") == 2
    assert cfg_text.count("history_length=1") >= 2
    assert cfg_text.count('params={"action_name": "joint_pos"}') == 2
    assert "class LearnedFootholdPlanningRewards:" in cfg_text
    assert (
        "foothold_planning: LearnedFootholdPlanningRewards | None = None"
        in cfg_text
    )
    assert (
        "self.rewards.foothold_planning = "
        "LearnedFootholdPlanningRewards()"
        in cfg_text
    )
    assert "self.rewards.rewards.learned_foothold_planning =" not in cfg_text


def test_learned_planner_routes_amp_reward_only_to_execution_reward_group():
    cfg_text = _parkour_agent_cfg_source_path().read_text()

    assert "auxiliary_reward_per_env_reward_coefs = [1.0]" in cfg_text
    assert "def enable_event_gated_foothold_ppo(" in cfg_text
    assert (
        "self.algorithm.auxiliary_reward_per_env_reward_coefs = [1.0, 0.0]"
        in cfg_text
    )


def test_agent_receives_reachability_radii_instead_of_hard_coding_them():
    cfg_text = _parkour_agent_cfg_source_path().read_text()

    assert "def enable_event_gated_foothold_ppo(" in cfg_text
    assert "reachability_radii_m: tuple[float, float]" in cfg_text
    assert "foothold_reachability_radii_m = (0.42, 0.25)" not in cfg_text
    assert (
        "self.algorithm.foothold_reachability_radii_m = "
        "reachability_radii_m"
        in cfg_text
    )


def test_planner_initializes_and_clears_learned_target_buffers():
    planner_text = _planner_source_path("foothold_planner.py").read_text()

    assert "self._data.learned_foothold_prepared_f = torch.zeros(" in planner_text
    assert "self._data.learned_foothold_locked = torch.zeros(" in planner_text
    assert "clear_learned_foothold_buffers(self._data, reset_env_ids)" in planner_text


def test_preflight_world_target_is_not_recomputed_after_swing_starts():
    """A HOLD-approved transaction must preserve its exact world target."""

    planner_text = _planner_source_path("foothold_planner.py").read_text()
    terrain_block = planner_text[
        planner_text.index("terrain_query = self._query_target_terrain_height_at_xy_w(") :
        planner_text.index(
            "self._data.target_foothold_f[new_swing_env_ids] = target_foothold_f"
        )
    ]

    # The cached target already had its world-frame height queried and its
    # clearance trajectory accepted during HOLD.  A duplicate query after the
    # state transition could change a locked target if the ray result changes.
    assert "apply_terrain_mask = ~use_preflight" in terrain_block
    assert "if torch.any(apply_terrain_mask).item():" in terrain_block


def test_preflight_is_run_when_a_hold_proposal_is_available_not_at_timeout():
    """The proposal and its clearance result must share one PPO event."""

    planner_text = _planner_source_path("foothold_planner.py").read_text()
    preflight_block = planner_text[
        planner_text.index("# Preflight the complete swing transaction") :
        planner_text.index("if self.cfg.enable_learned_foothold:\n            # A failed HOLD preflight")
    ]

    assert "preflight_window = (" in preflight_block
    assert "& hold_contact_ready" in preflight_block
    assert "hold_elapsed_s" not in preflight_block


def test_preflight_is_cached_until_a_new_hold_proposal_replaces_it():
    planner_text = _planner_source_path("foothold_planner.py").read_text()
    preflight_block = planner_text[
        planner_text.index("# Preflight the complete swing transaction") :
        planner_text.index("if self.cfg.enable_learned_foothold:\n            # A failed HOLD preflight")
    ]

    assert "& ~self._data.swing_preflight_ready[selected_env_ids]" in preflight_block
    preparation_block = planner_text[
        planner_text.index("if torch.any(prepare_learned).item():") :
        planner_text.index("swing_ready = torch.ones_like")
    ]
    assert "self._data.swing_preflight_ready[prepare_env_ids] = False" in preparation_block


def test_reward_event_pulse_and_hold_transaction_latch_are_not_reused():
    planner_text = _planner_source_path("foothold_planner.py").read_text()
    update_prefix = planner_text[
        planner_text.index("self._data.planner_valid[env_ids] = True") :
        planner_text.index("_clear_safe_target_event_buffers", planner_text.index(
            "self._data.planner_valid[env_ids] = True"
        ))
    ]
    assert "self._data.learned_foothold_evaluated[env_ids] = False" in update_prefix
    assert "learned_foothold_transaction_evaluated" not in update_prefix

    hold_block = planner_text[
        planner_text.index("prepare_learned, _ = learned_foothold_event_masks(") :
        planner_text.index("# Preflight the complete swing transaction")
    ]
    assert "learned_foothold_transaction_evaluated" in hold_block
    assert (
        "learned_evaluated=self._data.learned_foothold_evaluated"
        not in hold_block
    )


def test_failed_learned_preflight_is_visible_for_one_reward_cycle():
    planner_text = _planner_source_path("foothold_planner.py").read_text()
    failed_block = planner_text[
        planner_text.index("failed_preflight = preflight_env_ids") :
        planner_text.index("failed_preflight_mask = (", planner_text.index(
            "failed_preflight = preflight_env_ids"
        ))
    ]
    assert (
        "self._data.learned_foothold_prepared_valid["
        in failed_block
    )
    assert "swing_preflight_ready[failed_preflight] = False" not in failed_block


def test_failed_preflight_switches_to_nominal_on_the_next_hold_update():
    planner_text = _planner_source_path("foothold_planner.py").read_text()
    preflight_block = planner_text[
        planner_text.index("# Preflight the complete swing transaction") :
        planner_text.index("preflight_candidate = (")
    ]
    assert "retry_failed_preflight" in preflight_block
    assert (
        "self._data.swing_preflight_ready[\n"
        "                    retry_preflight_env_ids\n"
        "                ] = False"
        in preflight_block
    )


def test_failed_learned_preflight_does_not_fail_while_nominal_fallback_exists():
    planner_text = _planner_source_path("foothold_planner.py").read_text()
    result_block = planner_text[
        planner_text.index("failed_preflight = preflight_env_ids") :
        planner_text.index("swing_ready = swing_ready", planner_text.index(
            "failed_preflight = preflight_env_ids"
        ))
    ]
    assert "learned_fallback_available" in result_block
    assert "failed_without_fallback" in result_block
    assert "& ~learned_fallback_available" in result_block


def test_explicit_planning_failure_discards_transaction_on_next_update():
    planner_text = _planner_source_path("foothold_planner.py").read_text()
    discard_block = planner_text[
        planner_text.index("discard_pending_plan =") :
        planner_text.index("nominal_ready_before_update =", planner_text.index(
            "discard_pending_plan ="
        ))
    ]
    assert (
        "self._data.planning_failure[\n"
        "            selected_env_ids\n"
        "        ].clone()"
        in discard_block
    )
    assert "clear_learned_foothold_buffers" in discard_block


def test_planner_data_declares_prepared_and_locked_learned_target_buffers():
    data_class = _load_foothold_planner_data_class()
    field_names = {field.name for field in fields(data_class)}

    assert {
        "learned_foothold_enabled",
        "learned_foothold_action_normalized",
        "learned_foothold_decoded_f",
        "learned_foothold_prepared_f",
        "learned_foothold_prepared_w",
        "learned_foothold_prepared_valid",
        "learned_foothold_locked",
        "learned_foothold_target_f",
        "learned_foothold_target_w",
        "learned_foothold_used",
        "learned_foothold_height_valid",
        "learned_foothold_geometric_valid",
        "learned_foothold_safety_valid",
        "learned_foothold_evaluated",
        "learned_foothold_transaction_evaluated",
        "learned_foothold_route_event",
        "learned_foothold_route_use_nominal",
        "learned_foothold_route_use_learned",
        "learned_foothold_route_initial_executable",
        "learned_foothold_route_outcome",
        "learned_foothold_safety_score",
        "learned_foothold_penetrating_point_count",
        "learned_foothold_penetrating_point_ratio",
        "learned_foothold_total_penetration_depth",
        "nominal_foothold_prepared",
        "nominal_feasible_velocity_f",
        "nominal_curriculum_residual_f",
        "nominal_curriculum_radius_f",
        "nominal_curriculum_usage",
        "nominal_foothold_w",
        "nominal_geometric_valid",
        "nominal_safety_valid",
        "nominal_safety_score",
        "swing_preflight_safe",
        "swing_preflight_ready",
    }.issubset(field_names)


def test_clear_learned_foothold_buffers_resets_selected_environments_only():
    data = SimpleNamespace(
        learned_foothold_action_normalized=torch.ones(3, 2),
        learned_foothold_decoded_f=torch.ones(3, 3),
        learned_foothold_prepared_f=torch.ones(3, 3),
        learned_foothold_prepared_w=torch.ones(3, 3),
        learned_foothold_prepared_valid=torch.ones(3, dtype=torch.bool),
        learned_foothold_locked=torch.ones(3, dtype=torch.bool),
        learned_foothold_target_f=torch.ones(3, 3),
        learned_foothold_target_w=torch.ones(3, 3),
        learned_foothold_used=torch.ones(3, dtype=torch.bool),
        learned_foothold_height_valid=torch.ones(3, dtype=torch.bool),
        learned_foothold_geometric_valid=torch.ones(3, dtype=torch.bool),
        learned_foothold_safety_valid=torch.ones(3, dtype=torch.bool),
        learned_foothold_evaluated=torch.ones(3, dtype=torch.bool),
        learned_foothold_transaction_evaluated=torch.ones(
            3, dtype=torch.bool
        ),
        learned_foothold_event_generation=torch.tensor(
            [7, 11, 13],
            dtype=torch.int64,
        ),
        learned_foothold_route_event=torch.ones(3, dtype=torch.bool),
        learned_foothold_route_use_nominal=torch.ones(3, dtype=torch.bool),
        learned_foothold_route_use_learned=torch.ones(3, dtype=torch.bool),
        learned_foothold_route_initial_executable=torch.ones(
            3, dtype=torch.bool
        ),
        learned_foothold_route_outcome=torch.full(
            (3,), 5, dtype=torch.long
        ),
        learned_foothold_safety_score=torch.ones(3),
        learned_foothold_penetrating_point_count=torch.ones(3),
        learned_foothold_penetrating_point_ratio=torch.ones(3),
        learned_foothold_total_penetration_depth=torch.ones(3),
    )

    clear_learned_foothold_buffers(data, torch.tensor([0, 2]))

    torch.testing.assert_close(
        data.learned_foothold_event_generation,
        torch.tensor([7, 11, 13], dtype=torch.int64),
    )

    for name in (
        "learned_foothold_action_normalized",
        "learned_foothold_decoded_f",
        "learned_foothold_prepared_f",
        "learned_foothold_prepared_w",
        "learned_foothold_target_f",
        "learned_foothold_target_w",
        "learned_foothold_safety_score",
        "learned_foothold_penetrating_point_count",
        "learned_foothold_penetrating_point_ratio",
        "learned_foothold_total_penetration_depth",
    ):
        value = getattr(data, name)
        torch.testing.assert_close(value[0], torch.zeros_like(value[0]))
        torch.testing.assert_close(value[1], torch.ones_like(value[1]))
        torch.testing.assert_close(value[2], torch.zeros_like(value[2]))

    for name in (
        "learned_foothold_prepared_valid",
        "learned_foothold_locked",
        "learned_foothold_used",
        "learned_foothold_height_valid",
        "learned_foothold_geometric_valid",
        "learned_foothold_safety_valid",
        "learned_foothold_evaluated",
        "learned_foothold_transaction_evaluated",
        "learned_foothold_route_event",
        "learned_foothold_route_use_nominal",
        "learned_foothold_route_use_learned",
        "learned_foothold_route_initial_executable",
    ):
        assert getattr(data, name).tolist() == [False, True, False]
    assert data.learned_foothold_route_outcome.tolist() == [0, 5, 0]


def test_decode_normalized_foothold_preserves_points_inside_unit_disk():
    normalized = torch.tensor([[0.5, -0.4]])

    target_f = decode_normalized_foothold(
        normalized,
        radius_x=0.42,
        radius_y=0.25,
    )

    torch.testing.assert_close(target_f, torch.tensor([[0.21, -0.10]]))


def test_decode_normalized_foothold_radially_projects_to_shared_ellipse():
    normalized = torch.tensor([[1.0, 1.0]])

    target_f = decode_normalized_foothold(
        normalized,
        radius_x=0.42,
        radius_y=0.25,
    )

    ellipse_usage = (
        (target_f[:, 0] / 0.42).square()
        + (target_f[:, 1] / 0.25).square()
    )
    torch.testing.assert_close(ellipse_usage, torch.ones_like(ellipse_usage))


def test_event_masks_prepare_in_confirmed_hold_and_lock_only_on_new_swing():
    prepare, lock = learned_foothold_event_masks(
        hold=torch.tensor([True, True, False, False]),
        hold_contact_ready=torch.tensor([True, False, True, True]),
        nominal_ready=torch.tensor([True, True, True, True]),
        new_swing=torch.tensor([False, False, True, False]),
        enable=True,
    )

    assert prepare.tolist() == [True, False, False, False]
    assert lock.tolist() == [False, False, True, False]


def test_event_masks_are_empty_when_learned_planner_is_disabled():
    prepare, lock = learned_foothold_event_masks(
        hold=torch.tensor([True, False]),
        hold_contact_ready=torch.tensor([True, True]),
        nominal_ready=torch.tensor([True, True]),
        new_swing=torch.tensor([False, True]),
        enable=False,
    )

    assert prepare.tolist() == [False, False]
    assert lock.tolist() == [False, False]


def test_hold_does_not_consume_learned_action_before_nominal_was_published():
    prepare, _ = learned_foothold_event_masks(
        hold=torch.tensor([True, True]),
        hold_contact_ready=torch.tensor([True, True]),
        nominal_ready=torch.tensor([False, True]),
        new_swing=torch.tensor([False, False]),
        enable=True,
    )

    assert prepare.tolist() == [False, True]


def test_normal_route_prefers_geometrically_valid_learned_even_when_nominal_safe():
    route = route_nominal_and_learned_footholds(
        nominal_geometric_valid=torch.tensor([True, True, False, True]),
        nominal_safety_valid=torch.tensor([True, False, False, False]),
        learned_prepared=torch.tensor([True, True, True, True]),
        learned_geometric_valid=torch.tensor([True, True, True, False]),
    )

    assert route.use_nominal.tolist() == [False, False, False, False]
    assert route.use_learned.tolist() == [True, True, True, False]
    assert route.executable.tolist() == [True, True, True, False]


def test_normal_route_falls_back_to_safe_nominal_when_learned_is_unavailable():
    route = route_nominal_and_learned_footholds(
        nominal_geometric_valid=torch.tensor([True, True, True]),
        nominal_safety_valid=torch.tensor([True, True, False]),
        learned_prepared=torch.tensor([False, True, False]),
        learned_geometric_valid=torch.tensor([True, False, True]),
    )

    assert route.use_nominal.tolist() == [True, True, False]
    assert route.use_learned.tolist() == [False, False, False]
    assert route.executable.tolist() == [True, True, False]


def test_normal_swing_waits_for_transaction_evaluation_before_safe_nominal():
    ready = learned_foothold_swing_ready(
        nominal_route_ready=torch.tensor([True, True, False]),
        transaction_evaluated=torch.tensor([False, True, True]),
        learned_prepared_valid=torch.tensor([False, False, True]),
        learned_geometric_valid=torch.tensor([False, False, True]),
        learned_safety_valid=torch.tensor([False, False, True]),
        recovery_step=torch.tensor([False, False, False]),
    )

    assert ready.tolist() == [False, True, True]


def test_safe_learned_proposal_remains_swing_ready_after_event_pulse_clears():
    ready = learned_foothold_swing_ready(
        nominal_route_ready=torch.tensor([False]),
        transaction_evaluated=torch.tensor([True]),
        learned_prepared_valid=torch.tensor([True]),
        learned_geometric_valid=torch.tensor([True]),
        learned_safety_valid=torch.tensor([True]),
        recovery_step=torch.tensor([False]),
    )

    assert ready.tolist() == [True]


def test_recovery_swing_does_not_wait_for_or_use_learned_evaluation():
    ready = learned_foothold_swing_ready(
        nominal_route_ready=torch.tensor([True, False]),
        transaction_evaluated=torch.tensor([False, True]),
        learned_prepared_valid=torch.tensor([False, True]),
        learned_geometric_valid=torch.tensor([False, True]),
        learned_safety_valid=torch.tensor([False, True]),
        recovery_step=torch.tensor([True, True]),
    )

    assert ready.tolist() == [True, False]


def test_execution_route_executes_geometric_unsafe_learned_but_blocks_recovery():
    route = route_nominal_and_learned_footholds(
        nominal_geometric_valid=torch.tensor([False, False]),
        nominal_safety_valid=torch.tensor([False, False]),
        learned_prepared=torch.tensor([True, True]),
        learned_geometric_valid=torch.tensor([True, True]),
        learned_safety_valid=torch.tensor([False, False]),
        recovery_step=torch.tensor([False, True]),
    )

    # Danger-cylinder safety remains a PPO event signal, but an unsafe target
    # cannot enter a locked swing transaction.
    assert route.use_nominal.tolist() == [False, False]
    assert route.use_learned.tolist() == [False, False]
    assert route.executable.tolist() == [False, False]


def test_lock_requires_execution_safety_even_when_geometry_is_valid():
    data = SimpleNamespace(
        learned_foothold_prepared_valid=torch.tensor([True, True]),
        learned_foothold_locked=torch.tensor([False, False]),
        learned_foothold_used=torch.tensor([False, False]),
        learned_foothold_target_f=torch.zeros(2, 3),
        learned_foothold_target_w=torch.zeros(2, 3),
        learned_foothold_prepared_f=torch.ones(2, 3),
        learned_foothold_prepared_w=torch.ones(2, 3),
    )

    use = lock_prepared_learned_foothold(
        data=data,
        env_ids=torch.tensor([0, 1]),
        safety_valid=torch.tensor([True, False]),
    )

    assert use.tolist() == [True, False]
    assert data.learned_foothold_locked.tolist() == [True, False]


def test_recovery_route_uses_geometric_analytic_target_even_if_soft_unsafe():
    route = route_nominal_and_learned_footholds(
        nominal_geometric_valid=torch.tensor([True]),
        nominal_safety_valid=torch.tensor([False]),
        learned_prepared=torch.tensor([True]),
        learned_geometric_valid=torch.tensor([True]),
        learned_safety_valid=torch.tensor([False]),
        recovery_step=torch.tensor([True]),
    )

    assert route.use_nominal.tolist() == [False]
    assert route.use_learned.tolist() == [False]
    assert route.executable.tolist() == [False]


def test_prepare_target_queries_world_xy_and_applies_existing_height_limit():
    queried = []

    def terrain_query(points_xy_w):
        queried.append(points_xy_w.clone())
        return torch.tensor([0.55, 0.80]), torch.tensor([True, True])

    result = prepare_learned_foothold_target(
        normalized_action=torch.tensor([[0.5, 0.0], [0.5, 0.0]]),
        origin_w=torch.tensor([[1.0, 2.0, 0.40], [3.0, 4.0, 0.40]]),
        yaw_w=torch.zeros(2),
        radius_x=0.40,
        radius_y=0.20,
        max_step_height_m=0.25,
        terrain_height_query_w=terrain_query,
    )

    torch.testing.assert_close(
        queried[0],
        torch.tensor([[1.2, 2.0], [3.2, 4.0]]),
    )
    torch.testing.assert_close(
        result.target_f,
        torch.tensor([[0.2, 0.0, 0.15], [0.2, 0.0, 0.40]]),
    )
    assert result.height_valid.tolist() == [True, True]
    assert result.geometric_valid.tolist() == [True, False]


def test_cached_world_target_does_not_move_when_support_frame_drifts():
    cached_target_w = torch.tensor([[1.20, 2.10, 0.50]])

    target_f, geometric_valid = reframe_cached_world_foothold(
        target_w=cached_target_w,
        current_origin_w=torch.tensor([[1.05, 1.95, 0.40]]),
        current_yaw_w=torch.tensor([torch.pi / 2.0]),
        radius_x=0.42,
        radius_y=0.25,
        max_step_height_m=0.25,
    )

    torch.testing.assert_close(
        target_f,
        torch.tensor([[0.15, -0.15, 0.10]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert geometric_valid.tolist() == [True]


def test_cached_world_target_is_rejected_after_support_frame_drift_makes_it_unreachable():
    _, geometric_valid = reframe_cached_world_foothold(
        target_w=torch.tensor([[1.40, 2.00, 0.70]]),
        current_origin_w=torch.tensor([[1.00, 2.00, 0.40]]),
        current_yaw_w=torch.zeros(1),
        radius_x=0.30,
        radius_y=0.16,
        max_step_height_m=0.25,
    )

    assert geometric_valid.tolist() == [False]


def test_learned_target_does_not_use_an_uncalibrated_lateral_hard_gate():
    def terrain_query(points_xy_w):
        return torch.zeros(points_xy_w.shape[0]), torch.ones(
            points_xy_w.shape[0], dtype=torch.bool
        )

    result = prepare_learned_foothold_target(
        normalized_action=torch.tensor([[0.2, -0.2], [0.2, 0.2]]),
        origin_w=torch.zeros(2, 3),
        yaw_w=torch.zeros(2),
        radius_x=0.42,
        radius_y=0.25,
        max_step_height_m=0.25,
        terrain_height_query_w=terrain_query,
    )

    assert result.geometric_valid.tolist() == [True, True]


def test_learned_target_geometry_api_has_no_uncalibrated_lateral_gate():
    preparation_params = inspect.signature(
        prepare_learned_foothold_target
    ).parameters
    reframe_params = inspect.signature(
        reframe_cached_world_foothold
    ).parameters

    assert "swing_side" not in preparation_params
    assert "min_lateral_separation" not in preparation_params
    assert "swing_side" not in reframe_params
    assert "min_lateral_separation" not in reframe_params


def test_cached_world_target_rechecks_geometry_without_lateral_hard_gate():
    _, geometric_valid = reframe_cached_world_foothold(
        target_w=torch.tensor([[0.20, -0.08, 0.0]]),
        current_origin_w=torch.zeros(1, 3),
        current_yaw_w=torch.zeros(1),
        radius_x=0.42,
        radius_y=0.25,
        max_step_height_m=0.25,
    )

    assert geometric_valid.tolist() == [True]


def test_soft_unsafe_target_is_recorded_for_learning_but_not_locked():
    data = SimpleNamespace(
        learned_foothold_decoded_f=torch.zeros(2, 3),
        learned_foothold_prepared_f=torch.zeros(2, 3),
        learned_foothold_prepared_w=torch.zeros(2, 3),
        learned_foothold_prepared_valid=torch.zeros(2, dtype=torch.bool),
        learned_foothold_locked=torch.zeros(2, dtype=torch.bool),
        learned_foothold_target_f=torch.zeros(2, 3),
        learned_foothold_target_w=torch.zeros(2, 3),
        learned_foothold_used=torch.zeros(2, dtype=torch.bool),
        learned_foothold_height_valid=torch.zeros(2, dtype=torch.bool),
        learned_foothold_geometric_valid=torch.zeros(2, dtype=torch.bool),
        learned_foothold_safety_valid=torch.zeros(2, dtype=torch.bool),
        learned_foothold_evaluated=torch.zeros(2, dtype=torch.bool),
        learned_foothold_transaction_evaluated=torch.zeros(
            2, dtype=torch.bool
        ),
        learned_foothold_safety_score=torch.zeros(2),
        learned_foothold_penetrating_point_count=torch.zeros(2),
        learned_foothold_penetrating_point_ratio=torch.zeros(2),
        learned_foothold_total_penetration_depth=torch.zeros(2),
    )
    preparation = LearnedFootholdPreparation(
        decoded_f=torch.tensor([[0.2, 0.1, 0.0], [0.2, -0.1, 0.0]]),
        target_f=torch.tensor([[0.2, 0.1, 0.1], [0.2, -0.1, 0.1]]),
        target_w=torch.tensor([[1.2, 2.1, 0.5], [3.2, 3.9, 0.5]]),
        height_valid=torch.tensor([True, True]),
        geometric_valid=torch.tensor([True, True]),
    )

    store_learned_foothold_preparation(
        data=data,
        env_ids=torch.tensor([0, 1]),
        preparation=preparation,
        safety_valid=torch.tensor([True, False]),
        safety_score=torch.tensor([1.0, -0.5]),
        penetrating_point_count=torch.tensor([0.0, 3.0]),
        penetrating_point_ratio=torch.tensor([0.0, 0.5]),
        total_penetration_depth=torch.tensor([0.0, 0.03]),
    )
    used = lock_prepared_learned_foothold(
        data=data,
        env_ids=torch.tensor([0, 1]),
        safety_valid=data.learned_foothold_safety_valid,
    )

    assert used.tolist() == [True, False]
    assert data.learned_foothold_evaluated.tolist() == [True, True]
    assert data.learned_foothold_transaction_evaluated.tolist() == [True, True]
    assert data.learned_foothold_prepared_valid.tolist() == [True, True]
    assert data.learned_foothold_locked.tolist() == [True, False]
    assert data.learned_foothold_used.tolist() == [True, False]
    torch.testing.assert_close(
        data.learned_foothold_target_f[0],
        preparation.target_f[0],
    )
    # The unsafe proposal is still kept for its PPO penalty, but it is never
    # permitted to enter the locked swing transaction.
    torch.testing.assert_close(
        data.learned_foothold_prepared_f[1],
        preparation.target_f[1],
    )
    torch.testing.assert_close(
        data.learned_foothold_target_f[1],
        torch.zeros_like(data.learned_foothold_target_f[1]),
    )
