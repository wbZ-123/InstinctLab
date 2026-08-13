from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command

    def get_command(self, command_name: str) -> torch.Tensor:
        assert command_name == "base_velocity"
        return self.command


class _FakeScene:
    def __init__(self, sensors=None, robot=None):
        self.sensors = sensors or {}
        self._items = {}
        if robot is not None:
            self._items["robot"] = robot

    def __getitem__(self, name: str):
        return self._items[name]


class _FakePlanner:
    def __init__(self, data):
        self._data = data
        self.desired_velocity = None
        self.flat_target_curriculum_scale = None
        self.flat_target_curriculum_scale_calls = 0

    def set_desired_velocity(self, desired_velocity_f: torch.Tensor) -> None:
        self.desired_velocity = desired_velocity_f.clone()

    def set_flat_target_curriculum_scale(self, scale) -> None:
        self.flat_target_curriculum_scale = torch.as_tensor(scale).clone()
        self.flat_target_curriculum_scale_calls += 1

    @property
    def data(self):
        assert self.desired_velocity is not None
        return self._data


def _load_foothold_reward_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "source"
        / "instinctlab"
        / "instinctlab"
        / "envs"
        / "mdp"
        / "rewards"
        / "foothold.py"
    )
    spec = importlib.util.spec_from_file_location(
        "foothold_reward_under_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_locomotion_readiness_curriculum(cfg_text: str):
    assert '"curriculum_start_scale": 0.0' in cfg_text
    assert '"curriculum_end_scale": 1.00' in cfg_text or '"curriculum_end_scale": 1.0' in cfg_text
    assert '"curriculum_ramp_steps": 0' in cfg_text
    assert '"curriculum_gate": "locomotion_readiness"' in cfg_text
    assert (
        '"curriculum_min_episode_length": '
        "_FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH"
    ) in cfg_text
    assert (
        '"curriculum_full_episode_length": '
        "_FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH"
    ) in cfg_text
    assert '"curriculum_full_episode_length": 300' not in cfg_text
    assert '"reward_curriculum_full_episode_length": 300' not in cfg_text
    assert '"curriculum_velocity_command_name": "base_velocity"' in cfg_text
    assert '"curriculum_velocity_start_score": 0.4' in cfg_text
    assert '"curriculum_velocity_full_score": 0.7' in cfg_text


def test_foothold_curriculum_length_has_one_reward_source_of_truth():
    foothold = _load_foothold_reward_module()

    assert foothold.FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH == 100.0
    assert foothold.FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH == 700.0


def test_learned_foothold_safety_reward_is_event_gated_and_bounded():
    foothold = _load_foothold_reward_module()
    planner_data = SimpleNamespace(
        learned_foothold_evaluated=torch.tensor([True, False, True]),
        learned_foothold_geometric_valid=torch.tensor([True, True, False]),
        learned_foothold_safety_score=torch.tensor([1.0, -0.5, 0.2]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.learned_foothold_safety_event_reward(env)

    torch.testing.assert_close(
        reward,
        torch.tensor([1.0, 0.0, -1.0]),
    )


def test_stabilization_mask_prefers_contact_adaptive_flag_and_has_legacy_fallback():
    foothold = _load_foothold_reward_module()

    data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 1]),
        stabilization_active=torch.tensor([True, False, False]),
        recovery_step_active=torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(
        foothold.foothold_stabilization_mask(data),
        torch.tensor([True, False, False]),
    )

    legacy_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        recovery_step_active=torch.tensor([True, False]),
    )
    torch.testing.assert_close(
        foothold.foothold_stabilization_mask(legacy_data),
        torch.tensor([True, False]),
    )


def test_recovery_reward_mask_pauses_task_pressure_only_for_active_envs():
    foothold = _load_foothold_reward_module()
    data = SimpleNamespace(
        gait_mode=torch.tensor([8, 8, 1]),
        stabilization_active=torch.tensor([True, False, False]),
    )

    torch.testing.assert_close(
        foothold.mask_recovery_reward(
            torch.tensor([2.0, 3.0, 4.0]),
            data,
        ),
        torch.tensor([0.0, 3.0, 4.0]),
    )


def test_recovery_masked_velocity_wrapper_preserves_upstream_values(monkeypatch):
    foothold = _load_foothold_reward_module()
    upstream_mdp = types.ModuleType("isaaclab.envs.mdp")
    upstream_mdp.track_lin_vel_xy_exp = lambda env, **kwargs: torch.tensor(
        [1.0, 2.0]
    )
    isaaclab_envs = types.ModuleType("isaaclab.envs")
    isaaclab_envs.mdp = upstream_mdp
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.envs = isaaclab_envs
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", isaaclab_envs)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", upstream_mdp)

    data = SimpleNamespace(
        gait_mode=torch.tensor([8, 1]),
        stabilization_active=torch.tensor([True, False]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=data)},
        )
    )

    torch.testing.assert_close(
        foothold.track_lin_vel_xy_exp_recovery_masked(
            env,
            std=0.5,
            command_name="base_velocity",
        ),
        torch.tensor([0.0, 2.0]),
    )


def test_recovery_masked_feet_air_time_uses_project_parkour_reward(monkeypatch):
    foothold = _load_foothold_reward_module()
    upstream_mdp = types.ModuleType("instinctlab.tasks.parkour.mdp")
    upstream_mdp.feet_air_time = lambda env, **kwargs: torch.tensor([1.5, 2.5])
    parkour_tasks = types.ModuleType("instinctlab.tasks.parkour")
    parkour_tasks.mdp = upstream_mdp
    instinctlab_tasks = types.ModuleType("instinctlab.tasks")
    instinctlab_tasks.parkour = parkour_tasks
    instinctlab = types.ModuleType("instinctlab")
    instinctlab.tasks = instinctlab_tasks
    monkeypatch.setitem(sys.modules, "instinctlab", instinctlab)
    monkeypatch.setitem(sys.modules, "instinctlab.tasks", instinctlab_tasks)
    monkeypatch.setitem(sys.modules, "instinctlab.tasks.parkour", parkour_tasks)
    monkeypatch.setitem(sys.modules, "instinctlab.tasks.parkour.mdp", upstream_mdp)

    data = SimpleNamespace(
        gait_mode=torch.tensor([8, 1]),
        stabilization_active=torch.tensor([True, False]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=data)},
        )
    )

    torch.testing.assert_close(
        foothold.feet_air_time_recovery_masked(
            env,
            command_name="base_velocity",
            vel_threshold=0.15,
            sensor_cfg=object(),
        ),
        torch.tensor([0.0, 2.5]),
    )


def test_stabilization_masks_foothold_and_learned_event_rewards():
    foothold = _load_foothold_reward_module()
    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        planner_valid=torch.tensor([True, True]),
        actual_swing_foot_pos_w=torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]),
        swing_reference_pos_w=torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]),
        learned_foothold_evaluated=torch.tensor([True, True]),
        learned_foothold_geometric_valid=torch.tensor([True, True]),
        learned_foothold_safety_score=torch.tensor([1.0, 1.0]),
        stabilization_active=torch.tensor([True, False]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={"foothold_planner": SimpleNamespace(data=planner_data)},
        )
    )

    torch.testing.assert_close(
        foothold.foothold_swing_tracking_exp(env),
        torch.tensor([0.0, 1.0]),
    )
    torch.testing.assert_close(
        foothold.learned_foothold_safety_event_reward(env),
        torch.tensor([0.0, 1.0]),
    )


def test_learned_planning_reward_keeps_safe_nominal_and_scores_unsafe_nominal():
    foothold = _load_foothold_reward_module()
    planner_data = SimpleNamespace(
        learned_foothold_evaluated=torch.tensor(
            [True, True, True, True, False]
        ),
        learned_foothold_geometric_valid=torch.tensor(
            [True, True, True, False, True]
        ),
        learned_foothold_safety_valid=torch.tensor(
            [True, True, False, False, True]
        ),
        learned_foothold_safety_score=torch.tensor(
            [0.0, 0.0, -0.4, 0.8, 1.0]
        ),
        nominal_geometric_valid=torch.tensor(
            [True, True, True, True, True]
        ),
        nominal_safety_valid=torch.tensor(
            [True, True, False, False, False]
        ),
        raw_unclipped_foothold_f=torch.tensor(
            [
                    [0.20, 0.18, 0.0],
                    [0.20, 0.18, 0.0],
                    [0.20, 0.18, 0.0],
                    [0.20, 0.18, 0.0],
                    [0.20, 0.18, 0.0],
            ]
        ),
        learned_foothold_decoded_f=torch.tensor(
            [
                    [0.20, 0.18, 0.0],
                    [0.60, 0.18, 0.0],
                    [0.25, 0.08, 0.0],
                    [0.25, 0.08, 0.0],
                    [0.25, 0.08, 0.0],
            ]
        ),
        nominal_feasible_velocity_f=torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        swing_side=torch.tensor([0, 0, 0, 0, 0]),
        swing_preflight_ready=torch.tensor([False, False, False, True, False]),
        swing_preflight_safe=torch.tensor([True, True, True, False, True]),
        stabilization_active=torch.zeros(5, dtype=torch.bool),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.learned_foothold_planning_event_reward(
        env,
        reachability_radius_x=0.4,
        reachability_radius_y=0.2,
        velocity_lookahead_s=0.10,
        nominal_step_width_m=0.18,
        velocity_std=0.5,
    )

    assert reward[0].item() == 1.0
    assert 0.0 <= reward[1].item() < 0.01
    torch.testing.assert_close(reward[2], torch.tensor(-0.4))
    assert reward[3].item() == -1.0
    assert reward[4].item() == 0.0


def test_swing_tracking_rewards_reference_match_and_masks_non_swing():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 0]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.3],
                [1.0, 2.0, 0.3],
            ]
        ),
        swing_reference_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.3],
                [2.0, 2.0, 0.3],
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_swing_tracking_exp(env, std=0.2)

    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0]))


def test_swing_tracking_syncs_base_velocity_before_reading_planner_data():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1]),
        actual_swing_foot_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
        swing_reference_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
    )
    planner = _FakePlanner(planner_data)
    command = torch.tensor([[0.5, -0.1, 0.2]])
    env = SimpleNamespace(
        command_manager=_FakeCommandManager(command),
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": planner,
            }
        ),
    )

    reward = foothold.foothold_swing_tracking_exp(env, std=0.2)

    torch.testing.assert_close(planner.desired_velocity, command)
    torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_swing_tracking_can_be_disabled_until_locomotion_readiness_gate_opens():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1]),
        planner_valid=torch.tensor([True]),
        actual_swing_foot_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
        swing_reference_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(root_lin_vel_b=torch.tensor([[0.0, 0.0, 0.0]]))
    )
    env = SimpleNamespace(
        common_step_counter=100000,
        episode_length_buf=torch.tensor([10]),
        command_manager=_FakeCommandManager(torch.tensor([[0.6, 0.0, 0.0]])),
        scene=_FakeScene(
            sensors={"foothold_planner": _FakePlanner(planner_data)},
            robot=robot,
        ),
    )

    reward = foothold.foothold_swing_tracking_exp(
        env,
        std=0.2,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=100,
        curriculum_full_episode_length=300,
        curriculum_velocity_command_name="base_velocity",
        curriculum_velocity_std=0.5,
        curriculum_velocity_start_score=0.4,
        curriculum_velocity_full_score=0.7,
    )

    torch.testing.assert_close(reward, torch.tensor([0.0]))


def test_locomotion_readiness_curriculum_uses_episode_score_not_velocity_gate():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1]),
        planner_valid=torch.tensor([True]),
        actual_swing_foot_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
        swing_reference_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(root_lin_vel_b=torch.tensor([[0.0, 0.0, 0.0]]))
    )
    env = SimpleNamespace(
        common_step_counter=100000,
        episode_length_buf=torch.tensor([320]),
        command_manager=_FakeCommandManager(torch.tensor([[0.8, 0.0, 0.0]])),
        scene=_FakeScene(
            sensors={"foothold_planner": _FakePlanner(planner_data)},
            robot=robot,
        ),
    )

    reward = foothold.foothold_swing_tracking_exp(
        env,
        std=0.2,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=100,
        curriculum_full_episode_length=300,
        curriculum_velocity_command_name="base_velocity",
        curriculum_velocity_std=0.5,
        curriculum_velocity_start_score=0.4,
        curriculum_velocity_full_score=0.7,
    )

    torch.testing.assert_close(reward, torch.tensor([1.0]))


def test_locomotion_readiness_curriculum_is_per_env_not_mean_current_age():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 1]),
        planner_valid=torch.tensor([True, True, True]),
        actual_swing_foot_pos_w=torch.tensor(
            [[1.0, 2.0, 0.3], [1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]
        ),
        swing_reference_pos_w=torch.tensor(
            [[1.0, 2.0, 0.3], [1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]
        ),
    )
    planner = _FakePlanner(planner_data)
    env = SimpleNamespace(
        common_step_counter=100000,
        episode_length_buf=torch.tensor([50, 200, 360]),
        command_manager=_FakeCommandManager(
            torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
        ),
        scene=_FakeScene(sensors={"foothold_planner": planner}),
    )

    reward = foothold.foothold_swing_tracking_exp(
        env,
        std=0.2,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=100,
        curriculum_full_episode_length=300,
    )

    expected = torch.tensor([0.0, 0.5, 1.0])
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(planner.flat_target_curriculum_scale, expected)


def test_reward_curriculum_syncs_planner_scale_once_per_step_for_same_gate():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        planner_valid=torch.tensor([True, True]),
        actual_swing_foot_pos_w=torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]),
        swing_reference_pos_w=torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]),
        foot_contact=torch.tensor([[False, True], [True, False]]),
        swing_side=torch.tensor([0, 1]),
        phase=torch.tensor([0.3, 0.3]),
        swing_has_lifted=torch.tensor([True, True]),
    )
    planner = _FakePlanner(planner_data)
    env = SimpleNamespace(
        common_step_counter=100,
        episode_length_buf=torch.tensor([300, 300]),
        command_manager=_FakeCommandManager(torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])),
        scene=_FakeScene(sensors={"foothold_planner": planner}),
    )

    curriculum_kwargs = dict(
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=0,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=100,
        curriculum_full_episode_length=300,
    )
    foothold.foothold_swing_tracking_exp(env, std=0.2, **curriculum_kwargs)
    foothold.foothold_no_liftoff_indicator(env, **curriculum_kwargs)

    assert planner.flat_target_curriculum_scale_calls == 1
    torch.testing.assert_close(planner.flat_target_curriculum_scale, torch.tensor([1.0, 1.0]))


def test_reward_curriculum_can_be_explicitly_overridden_for_play_evaluation():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        planner_valid=torch.tensor([True, True]),
        actual_swing_foot_pos_w=torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]),
        swing_reference_pos_w=torch.tensor([[1.0, 2.0, 0.3], [1.0, 2.0, 0.3]]),
    )
    planner = _FakePlanner(planner_data)
    env = SimpleNamespace(
        common_step_counter=0,
        episode_length_buf=torch.tensor([0, 0]),
        foothold_reward_curriculum_override_scale=1.0,
        command_manager=_FakeCommandManager(torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])),
        scene=_FakeScene(sensors={"foothold_planner": planner}),
    )

    reward = foothold.foothold_swing_tracking_exp(
        env,
        std=0.2,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72000,
        curriculum_gate="locomotion_readiness",
    )

    torch.testing.assert_close(reward, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(planner.flat_target_curriculum_scale, torch.tensor([1.0, 1.0]))


def test_locomotion_readiness_curriculum_is_not_blocked_by_step_ramp_when_episode_is_long():
    foothold = _load_foothold_reward_module()

    env = SimpleNamespace(
        common_step_counter=0,
        episode_length_buf=torch.tensor([50, 950]),
        num_envs=2,
    )

    scale = foothold.foothold_reward_curriculum_scale(
        env,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=240000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=200,
        curriculum_full_episode_length=900,
    )

    torch.testing.assert_close(scale, torch.tensor([0.0, 1.0]))


def test_locomotion_readiness_curriculum_remembers_recent_completed_episode_after_reset():
    foothold = _load_foothold_reward_module()

    env = SimpleNamespace(
        common_step_counter=1,
        episode_length_buf=torch.tensor([950]),
        num_envs=1,
    )

    foothold.foothold_reward_curriculum_scale(
        env,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=240000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=200,
        curriculum_full_episode_length=900,
    )
    env.common_step_counter = 2
    env.episode_length_buf = torch.tensor([3])
    scale_after_reset = foothold.foothold_reward_curriculum_scale(
        env,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=240000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=200,
        curriculum_full_episode_length=900,
    )

    torch.testing.assert_close(scale_after_reset, torch.tensor([1.0]))


def test_touchdown_tracking_rewards_touchdown_confirm_with_continuous_error():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([3, 1, 3]),
        touchdown_accepted=torch.tensor([False, True, False]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [0.4, -0.2, 0.1],
                [0.4, -0.2, 0.1],
                [0.5, -0.2, 0.1],
            ]
        ),
        target_foothold_w=torch.tensor(
            [
                [0.4, -0.2, 0.1],
                [0.4, -0.2, 0.1],
                [0.4, -0.2, 0.1],
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_touchdown_tracking_exp(env, std=0.2)

    expected_error_reward = torch.exp(torch.tensor(-0.01 / 0.04))
    torch.testing.assert_close(
        reward,
        torch.tensor([1.0, 0.0, expected_error_reward.item()]),
    )


def test_swing_contact_indicator_flags_contact_after_liftoff_grace():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 2, 2, 0]),
        swing_side=torch.tensor([0, 0, 1, 1, 0]),
        phase=torch.tensor([0.05, 0.20, 0.20, 0.20, 0.20]),
        foot_contact=torch.tensor(
            [
                [True, True],   # left swing, still in grace window
                [True, True],   # left swing, late enough: bad
                [True, True],   # right swing, late enough: bad
                [True, False],  # right swing, lifted: ok
                [True, True],   # not swing mode
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    indicator = foothold.foothold_swing_contact_indicator(
        env,
        min_phase=0.20,
    )

    torch.testing.assert_close(
        indicator,
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]),
    )


def test_swing_contact_indicator_uses_training_progress_curriculum_scale():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        swing_side=torch.tensor([0, 0]),
        phase=torch.tensor([0.30, 0.30]),
        foot_contact=torch.tensor([[True, True], [True, True]]),
    )
    env = SimpleNamespace(
        common_step_counter=36000,
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        ),
    )

    indicator = foothold.foothold_swing_contact_indicator(
        env,
        min_phase=0.20,
        curriculum_start_scale=0.50,
        curriculum_end_scale=1.00,
        curriculum_ramp_steps=72000,
    )

    torch.testing.assert_close(
        indicator,
        torch.tensor([0.75, 0.75]),
    )


def test_locomotion_readiness_gate_keeps_foothold_penalty_off_before_stable_walking():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        swing_side=torch.tensor([0, 0]),
        phase=torch.tensor([0.30, 0.30]),
        foot_contact=torch.tensor([[True, True], [True, True]]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        )
    )
    env = SimpleNamespace(
        common_step_counter=100000,
        episode_length_buf=torch.tensor([10, 20]),
        command_manager=_FakeCommandManager(torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])),
        scene=_FakeScene(
            sensors={"foothold_planner": SimpleNamespace(data=planner_data)},
            robot=robot,
        ),
    )

    indicator = foothold.foothold_swing_contact_indicator(
        env,
        min_phase=0.20,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=100,
        curriculum_full_episode_length=300,
        curriculum_velocity_command_name="base_velocity",
        curriculum_velocity_std=0.5,
        curriculum_velocity_start_score=0.4,
        curriculum_velocity_full_score=0.7,
    )

    torch.testing.assert_close(indicator, torch.tensor([0.0, 0.0]))


def test_locomotion_readiness_gate_enables_foothold_penalty_after_stable_velocity_tracking():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        swing_side=torch.tensor([0, 0]),
        phase=torch.tensor([0.30, 0.30]),
        foot_contact=torch.tensor([[True, True], [True, True]]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([[0.58, 0.0, 0.0], [0.60, 0.0, 0.0]]),
        )
    )
    env = SimpleNamespace(
        common_step_counter=100000,
        episode_length_buf=torch.tensor([320, 360]),
        command_manager=_FakeCommandManager(torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])),
        scene=_FakeScene(
            sensors={"foothold_planner": SimpleNamespace(data=planner_data)},
            robot=robot,
        ),
    )

    indicator = foothold.foothold_swing_contact_indicator(
        env,
        min_phase=0.20,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=100,
        curriculum_full_episode_length=300,
        curriculum_velocity_command_name="base_velocity",
        curriculum_velocity_std=0.5,
        curriculum_velocity_start_score=0.4,
        curriculum_velocity_full_score=0.7,
    )

    torch.testing.assert_close(indicator, torch.tensor([1.0, 1.0]))


def test_reward_curriculum_scale_indicator_exposes_gate_value_for_logging():
    foothold = _load_foothold_reward_module()

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([[0.58, 0.0, 0.0], [0.60, 0.0, 0.0]]),
        )
    )
    env = SimpleNamespace(
        common_step_counter=100000,
        episode_length_buf=torch.tensor([320, 360]),
        command_manager=_FakeCommandManager(torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])),
        scene=_FakeScene(robot=robot),
    )

    scale = foothold.foothold_reward_curriculum_scale(
        env,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72000,
        curriculum_gate="locomotion_readiness",
        curriculum_min_episode_length=100,
        curriculum_full_episode_length=300,
        curriculum_velocity_command_name="base_velocity",
        curriculum_velocity_std=0.5,
        curriculum_velocity_start_score=0.4,
        curriculum_velocity_full_score=0.7,
    )

    torch.testing.assert_close(scale, torch.tensor([1.0, 1.0]))


def test_foothold_curriculum_reward_terms_use_explicit_manager_parameters():
    foothold = _load_foothold_reward_module()

    reward_functions = [
        foothold.foothold_reward_curriculum_scale,
        foothold.foothold_swing_tracking_exp,
        foothold.foothold_touchdown_tracking_exp,
        foothold.foothold_swing_contact_indicator,
        foothold.foothold_no_liftoff_indicator,
        foothold.foothold_swing_height_under_error_l1,
        foothold.foothold_swing_xy_error_l2,
        foothold.foothold_touchdown_xy_error_l2,
        foothold.foothold_touchdown_z_error_l1,
        foothold.foothold_gait_anomaly_indicator,
        foothold.foothold_recovery_indicator,
        foothold.foothold_clearance_penetration_l1,
        foothold.foothold_touchdown_accepted_indicator,
    ]

    for func in reward_functions:
        signature = inspect.signature(func)
        assert all(
            parameter.kind != inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ), f"{func.__name__} uses **kwargs, which IsaacLab managers treat as a mandatory parameter"


def test_no_liftoff_indicator_flags_swing_without_confirmed_liftoff():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 2, 2, 0]),
        phase=torch.tensor([0.20, 0.35, 0.35, 0.50, 0.50]),
        swing_has_lifted=torch.tensor([False, False, True, False, False]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    indicator = foothold.foothold_no_liftoff_indicator(
        env,
        min_phase=0.35,
    )

    torch.testing.assert_close(
        indicator,
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
    )


def test_gait_anomaly_indicator_reaches_full_scale_after_curriculum_ramp():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([4, 8, 1]),
        planner_valid=torch.tensor([True, True, False]),
    )
    env = SimpleNamespace(
        common_step_counter=100000,
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        ),
    )

    torch.testing.assert_close(
        foothold.foothold_gait_anomaly_indicator(
            env,
            curriculum_start_scale=0.50,
            curriculum_end_scale=1.00,
            curriculum_ramp_steps=72000,
        ),
        torch.tensor([1.0, 0.0, 1.0]),
    )


def test_swing_height_under_error_penalizes_only_below_reference_height():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 2, 1, 0]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [0.0, 0.0, 0.10],  # below reference by 0.05
                [0.0, 0.0, 0.16],  # above reference: no penalty
                [0.0, 0.0, 0.15],  # exactly reference: no penalty
                [0.0, 0.0, 0.05],  # non-swing: masked
            ]
        ),
        swing_reference_pos_w=torch.tensor(
            [
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 0.15],
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    error = foothold.foothold_swing_height_under_error_l1(
        env,
        max_error_m=0.04,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.04, 0.0, 0.0, 0.0]),
    )


def test_swing_xy_error_l2_penalizes_planar_reference_distance():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 2, 0]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.10],
                [1.0, 2.0, 0.20],
                [1.0, 2.0, 0.30],
            ]
        ),
        swing_reference_pos_w=torch.tensor(
            [
                [1.3, 2.4, 0.50],  # xy distance 0.5
                [1.0, 2.0, 0.00],  # xy distance 0.0
                [2.0, 2.0, 0.30],  # non-swing: masked
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    error = foothold.foothold_swing_xy_error_l2(
        env,
        max_error_m=0.30,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.30, 0.0, 0.0]),
    )


def test_swing_continuous_errors_are_disabled_when_planner_is_invalid():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        planner_valid=torch.tensor([True, False]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [0.0, 0.0, 0.10],
                [0.0, 0.0, 0.10],
            ]
        ),
        swing_reference_pos_w=torch.tensor(
            [
                [0.3, 0.4, 0.30],
                [0.3, 0.4, 0.30],
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    xy_error = foothold.foothold_swing_xy_error_l2(env, max_error_m=0.30)
    height_error = foothold.foothold_swing_height_under_error_l1(
        env, max_error_m=0.25
    )

    torch.testing.assert_close(xy_error, torch.tensor([0.30, 0.0]))
    torch.testing.assert_close(height_error, torch.tensor([0.20, 0.0]))


def test_swing_xy_error_l2_increases_weight_late_in_swing():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 1, 1]),
        phase=torch.tensor([0.40, 0.50, 0.65, 0.80]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [0.10, 0.0, 0.0],
                [0.10, 0.0, 0.0],
                [0.10, 0.0, 0.0],
                [0.10, 0.0, 0.0],
            ]
        ),
        swing_reference_pos_w=torch.zeros((4, 3)),
        planner_valid=torch.tensor([True, True, True, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    error = foothold.foothold_swing_xy_error_l2(
        env,
        max_error_m=0.30,
        late_phase_start=0.50,
        late_phase_full=0.80,
        late_phase_max_scale=2.0,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.10, 0.10, 0.15, 0.20]),
    )


def test_touchdown_xy_error_l2_scores_late_valid_swing_only():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 2, 5, 0]),
        phase=torch.tensor([0.40, 0.70, 0.80, 1.0, 1.0]),
        touchdown_xy_error=torch.tensor([0.01, 0.02, 0.05, 0.30, 0.40]),
        planner_valid=torch.tensor([True, True, True, True, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    error = foothold.foothold_touchdown_xy_error_l2(
        env,
        min_phase=0.65,
        target_tolerance_m=0.02,
        zero_score_m=0.05,
        max_penalty_m=0.05,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
    )


def test_touchdown_xy_error_l2_is_bounded_between_positive_one_and_negative_one():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 1, 1, 1, 1]),
        phase=torch.tensor([0.80, 0.80, 0.80, 0.80, 0.80, 0.80]),
        touchdown_xy_error=torch.tensor([0.01, 0.02, 0.035, 0.10, 0.30, 0.80]),
        planner_valid=torch.tensor([True, True, True, True, True, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    error = foothold.foothold_touchdown_xy_error_l2(
        env,
        min_phase=0.65,
        target_tolerance_m=0.02,
        zero_score_m=0.05,
        max_penalty_m=0.05,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([1.0, 1.0, 0.5, -1.0, -1.0, -1.0]),
    )


def test_touchdown_z_error_l1_penalizes_late_valid_swing_only():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1, 2, 5, 3]),
        phase=torch.tensor([0.40, 0.70, 0.80, 1.0, 1.0]),
        touchdown_z_error=torch.tensor([0.50, 0.20, 0.10, 0.30, 0.40]),
        planner_valid=torch.tensor([True, True, True, True, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    error = foothold.foothold_touchdown_z_error_l1(
        env,
        min_phase=0.65,
        max_error_m=0.20,
    )

    torch.testing.assert_close(
        error,
        torch.tensor([0.0, 0.20, 0.10, 0.0, 0.0]),
    )


def test_swing_tracking_is_disabled_when_planner_is_invalid():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([1, 1]),
        planner_valid=torch.tensor([True, False]),
        actual_swing_foot_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.3],
                [1.0, 2.0, 0.3],
            ]
        ),
        swing_reference_pos_w=torch.tensor(
            [
                [1.0, 2.0, 0.3],
                [1.0, 2.0, 0.3],
            ]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_swing_tracking_exp(env, std=0.2)

    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0]))


def test_parkour_reward_cfg_enables_no_liftoff_penalty():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_no_liftoff_indicator = RewTerm(" in cfg_text
    assert "func=mdp.foothold_no_liftoff_indicator" in cfg_text
    assert "_FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    assert "weight=-1.8 * _FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    assert '"min_phase": 0.35' in cfg_text
    _assert_locomotion_readiness_curriculum(cfg_text)


def test_parkour_reward_cfg_penalizes_swing_contact_after_grace():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_swing_contact_indicator = RewTerm(" in cfg_text
    assert "func=mdp.foothold_swing_contact_indicator" in cfg_text
    assert "weight=-1.2 * _FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    assert '"min_phase": 0.20' in cfg_text
    _assert_locomotion_readiness_curriculum(cfg_text)


def test_parkour_reward_cfg_enables_swing_height_under_error_penalty():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_swing_height_under_error_l1 = RewTerm(" in cfg_text
    assert "func=mdp.foothold_swing_height_under_error_l1" in cfg_text
    assert "weight=-3.0 * _FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    assert '"max_error_m": 0.25' in cfg_text
    _assert_locomotion_readiness_curriculum(cfg_text)


def test_parkour_reward_cfg_enables_swing_xy_error_penalty():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_swing_xy_error_l2 = RewTerm(" in cfg_text
    assert "func=mdp.foothold_swing_xy_error_l2" in cfg_text
    assert "weight=-1.5 * _FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    assert '"max_error_m": 0.30' in cfg_text
    assert '"late_phase_start": 0.50' in cfg_text
    assert '"late_phase_full": 0.80' in cfg_text
    assert '"late_phase_max_scale": 2.0' in cfg_text
    _assert_locomotion_readiness_curriculum(cfg_text)


def test_parkour_reward_cfg_enables_touchdown_continuous_error_penalties():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_touchdown_xy_error_l2 = RewTerm(" in cfg_text
    assert "func=mdp.foothold_touchdown_xy_error_l2" in cfg_text
    assert "weight=1.0 * _FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    assert "foothold_touchdown_z_error_l1 = RewTerm(" in cfg_text
    assert "func=mdp.foothold_touchdown_z_error_l1" in cfg_text
    assert '"min_phase": 0.65' in cfg_text
    assert '"target_tolerance_m": 0.02' in cfg_text
    assert '"zero_score_m": 0.05' in cfg_text
    assert '"max_penalty_m": 0.05' in cfg_text
    assert '"max_error_m": 0.20' in cfg_text


def test_parkour_reward_cfg_penalizes_clearance_penetration_without_safe_bonus():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_clearance_safe_indicator = RewTerm(" in cfg_text
    assert "func=instinct_mdp.foothold_clearance_safe_indicator" in cfg_text
    assert "foothold_clearance_penetration_l1 = RewTerm(" in cfg_text
    assert "func=instinct_mdp.foothold_clearance_penetration_l1" in cfg_text
    assert "foothold_clearance_safe_indicator = RewTerm(\n        func=instinct_mdp.foothold_clearance_safe_indicator,\n        weight=0.0" in cfg_text
    assert "weight=-4.0 * _FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    assert '"max_penetration_m": 0.15' in cfg_text


def test_parkour_reward_cfg_uses_unified_anomaly_penalty_and_keeps_diagnostics():
    repo_root = Path(__file__).resolve().parents[3]
    cfg_text = (
        repo_root
        / "source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
    ).read_text()

    assert "foothold_gait_anomaly_indicator = RewTerm(" in cfg_text
    assert "func=mdp.foothold_gait_anomaly_indicator" in cfg_text
    assert "weight=-2.0 * _FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    _assert_locomotion_readiness_curriculum(cfg_text)
    assert "foothold_early_contact_indicator = RewTerm(" in cfg_text
    assert "foothold_overdue_indicator = RewTerm(" in cfg_text
    assert "foothold_stance_lost_indicator = RewTerm(" in cfg_text
    assert "foothold_hold_contact_lost_indicator = RewTerm(" in cfg_text
    assert "foothold_recovery_indicator = RewTerm(" in cfg_text
    assert "weight=-0.3 * _FOOTHOLD_REWARD_WEIGHT_SCALE" in cfg_text
    _assert_locomotion_readiness_curriculum(cfg_text)
    assert "foothold_plan_invalid_indicator = RewTerm(" in cfg_text


def test_foothold_diagnostic_indicators_expose_gait_state_events():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        touchdown_accepted=torch.tensor(
            [False, False, False, True, False, False, False, False, False, False]
        ),
        planner_valid=torch.tensor(
            [True, True, True, True, True, True, True, False, True, True]
        ),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    torch.testing.assert_close(
        foothold.foothold_swing_mode_indicator(env),
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_reset_mode_indicator(env),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_left_swing_mode_indicator(env),
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_right_swing_mode_indicator(env),
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_touchdown_confirm_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_early_contact_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_overdue_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_stance_lost_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_hold_contact_lost_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_recovery_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_touchdown_accepted_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        foothold.foothold_plan_invalid_indicator(env),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )


def test_foothold_gait_anomaly_indicator_max_pools_failure_modes():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        gait_mode=torch.tensor([0, 1, 4, 5, 6, 7, 8, 3, 9]),
        planner_valid=torch.tensor([True, False, True, True, True, True, True, True, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    torch.testing.assert_close(
        foothold.foothold_gait_anomaly_indicator(env),
        torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
    )


def test_foothold_clearance_safe_indicator_returns_float_mask():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        swing_clearance_safe=torch.tensor([True, False, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_clearance_safe_indicator(env)

    expected = torch.tensor([1.0, 0.0, 1.0])
    torch.testing.assert_close(reward, expected)

def test_foothold_clearance_penetration_l1_returns_penetration_depth():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        swing_clearance_penetration=torch.tensor([0.0, 0.02, 0.15]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_clearance_penetration_l1(
        env,
        max_penetration_m=0.10,
    )

    expected = torch.tensor([0.0, 0.02, 0.10])
    torch.testing.assert_close(reward, expected)


def test_foothold_clearance_penetration_l1_is_curriculum_gated_when_unready():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        swing_clearance_penetration=torch.tensor([0.02, 0.10]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_clearance_penetration_l1(
        env,
        max_penetration_m=0.10,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72_000,
        curriculum_gate="locomotion_readiness",
    )

    torch.testing.assert_close(reward, torch.zeros(2))


def test_foothold_touchdown_accepted_indicator_is_curriculum_gated_when_unready():
    foothold = _load_foothold_reward_module()

    planner_data = SimpleNamespace(
        touchdown_accepted=torch.tensor([True, False, True]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "foothold_planner": SimpleNamespace(data=planner_data),
            }
        )
    )

    reward = foothold.foothold_touchdown_accepted_indicator(
        env,
        curriculum_start_scale=0.0,
        curriculum_end_scale=1.0,
        curriculum_ramp_steps=72_000,
        curriculum_gate="locomotion_readiness",
    )

    torch.testing.assert_close(reward, torch.zeros(3))
