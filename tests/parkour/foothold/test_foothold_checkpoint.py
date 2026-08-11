import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace


def _load_checkpoint_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root
        / "source/instinctlab/instinctlab/learning/foothold_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location(
        "foothold_checkpoint_for_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_states():
    source = {
        "encoder.weight": torch.arange(6.0).reshape(2, 3),
        "actor.gate.0.weight": torch.arange(20.0).reshape(4, 5),
        "actor.experts.0.0.weight": torch.arange(20.0).reshape(4, 5),
        "actor.experts.0.2.weight": torch.arange(116.0).reshape(29, 4),
        "actor.experts.0.2.bias": torch.arange(29.0),
        "critic.gate.0.weight": torch.arange(20.0).reshape(4, 5),
        "critic.experts.0.0.weight": torch.arange(20.0).reshape(4, 5),
        "critic.experts.0.2.weight": torch.arange(4.0).reshape(1, 4),
        "critic.experts.0.2.bias": torch.arange(1.0),
        "std": torch.arange(29.0) + 1.0,
    }
    destination = {
        "encoder.weight": torch.full((2, 3), -10.0),
        "actor.gate.0.weight": torch.full((4, 8), -11.0),
        "actor.experts.0.0.weight": torch.full((4, 8), -12.0),
        "actor.experts.0.2.weight": torch.full((29, 4), -13.0),
        "actor.experts.0.2.bias": torch.full((29,), -14.0),
        "critics.0.gate.0.weight": torch.full((4, 8), -15.0),
        "critics.0.experts.0.0.weight": torch.full((4, 8), -16.0),
        "critics.0.experts.0.2.weight": torch.full((1, 4), -17.0),
        "critics.0.experts.0.2.bias": torch.full((1,), -18.0),
        "critics.1.gate.0.weight": torch.full((4, 8), -19.0),
        "critics.1.experts.0.0.weight": torch.full((4, 8), -20.0),
        "critics.1.experts.0.2.weight": torch.full((1, 4), -21.0),
        "critics.1.experts.0.2.bias": torch.full((1,), -22.0),
        "foothold_actor.0.weight": torch.full((4, 8), -23.0),
        "foothold_actor.0.bias": torch.full((4,), -24.0),
        # Planner-only depth features are intentionally fresh when starting
        # from a legacy motor checkpoint.
        "foothold_depth_encoder.features.0.weight": torch.full(
            (8, 4, 3, 3), -27.0
        ),
        "motor_std": torch.full((29,), -25.0),
        "foothold_std": torch.full((2,), -26.0),
    }
    return source, destination


def _fake_input_column_maps():
    mapping = [0, 1, 3, 4, 6]
    return {
        "actor": mapping,
        "critics.0": mapping,
    }


def test_observation_expansion_adds_only_current_nominal_foothold():
    module = _load_checkpoint_module()

    expansion = module.learned_foothold_policy_input_expansion(
        nominal_foothold_dim=3,
    )

    assert expansion == 3


def test_legacy_input_column_map_preserves_each_temporal_frame():
    module = _load_checkpoint_module()
    destination_segments = OrderedDict(
        (
            ("prefix", (2,)),
            ("foothold_planner", (6,)),
            ("actions", (6,)),
            ("parallel_latent_0_depth_encoder", (2,)),
        )
    )

    column_map = module.build_legacy_input_column_map(
        destination_segments,
        temporal_appends={
            "foothold_planner": (1, 2),
            "actions": (1, 2),
        },
    )

    # Both temporal terms contain two 3D frames. Their legacy frame width is
    # two, so the appended third coordinate of each frame is skipped.
    assert column_map == [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15]


def test_legacy_input_column_map_skips_wholly_new_component():
    module = _load_checkpoint_module()
    destination_segments = OrderedDict(
        (
            ("prefix", (2,)),
            ("nominal_foothold", (3,)),
            ("actions", (6,)),
        )
    )

    column_map = module.build_legacy_input_column_map(
        destination_segments,
        temporal_appends={},
        new_components={"nominal_foothold"},
    )

    assert column_map == [0, 1, 5, 6, 7, 8, 9, 10]


def test_migration_copies_motor_actor_and_execution_critic_only():
    module = _load_checkpoint_module()
    source, destination = _fake_states()

    migrated, report = module.migrate_foothold_model_state(
        source,
        destination,
        motor_action_dim=29,
        input_column_maps=_fake_input_column_maps(),
        foothold_normalized_std=(0.05 / 0.42, 0.05 / 0.25),
    )

    torch.testing.assert_close(
        migrated["actor.experts.0.2.weight"],
        source["actor.experts.0.2.weight"],
    )
    torch.testing.assert_close(
        migrated["critics.0.experts.0.2.weight"],
        source["critic.experts.0.2.weight"],
    )
    torch.testing.assert_close(
        migrated["critics.1.experts.0.2.weight"],
        destination["critics.1.experts.0.2.weight"],
    )
    torch.testing.assert_close(
        migrated["motor_std"],
        source["std"],
    )
    torch.testing.assert_close(
        migrated["foothold_std"],
        torch.tensor([0.05 / 0.42, 0.05 / 0.25]),
    )
    torch.testing.assert_close(
        migrated["foothold_actor.0.weight"],
        destination["foothold_actor.0.weight"],
    )
    assert "foothold_actor.0.weight" in report.initialized
    assert "foothold_depth_encoder.features.0.weight" in report.initialized
    assert "critics.1.experts.0.2.weight" in report.initialized
    assert report.unexpected == []


def test_migration_preserves_new_observation_columns_as_initialized():
    module = _load_checkpoint_module()
    source, destination = _fake_states()

    migrated, _ = module.migrate_foothold_model_state(
        source,
        destination,
        motor_action_dim=29,
        input_column_maps=_fake_input_column_maps(),
        foothold_normalized_std=(0.1, 0.2),
    )

    torch.testing.assert_close(
        migrated["actor.gate.0.weight"][
            :,
            _fake_input_column_maps()["actor"],
        ],
        source["actor.gate.0.weight"],
    )
    torch.testing.assert_close(
        migrated["actor.gate.0.weight"][:, [2, 5, 7]],
        torch.zeros(4, 3),
    )
    torch.testing.assert_close(
        migrated["critics.0.experts.0.0.weight"][
            :,
            _fake_input_column_maps()["critics.0"],
        ],
        source["critic.experts.0.0.weight"],
    )


def test_migration_rejects_unapproved_shape_mismatch():
    module = _load_checkpoint_module()
    source, destination = _fake_states()
    source["encoder.weight"] = torch.zeros(3, 3)

    with pytest.raises(ValueError, match="encoder.weight"):
        module.migrate_foothold_model_state(
            source,
            destination,
            motor_action_dim=29,
            input_column_maps=_fake_input_column_maps(),
            foothold_normalized_std=(0.1, 0.2),
        )


def test_migration_rejects_unconsumed_source_parameter():
    module = _load_checkpoint_module()
    source, destination = _fake_states()
    source["unexpected.weight"] = torch.ones(1)

    with pytest.raises(ValueError, match="unexpected.weight"):
        module.migrate_foothold_model_state(
            source,
            destination,
            motor_action_dim=29,
            input_column_maps=_fake_input_column_maps(),
            foothold_normalized_std=(0.1, 0.2),
        )


def test_migration_rejects_shared_head_checkpoint():
    module = _load_checkpoint_module()
    source, destination = _fake_states()
    source["std"] = torch.ones(31)

    with pytest.raises(ValueError, match="shared-head"):
        module.migrate_foothold_model_state(
            source,
            destination,
            motor_action_dim=29,
            input_column_maps=_fake_input_column_maps(),
            foothold_normalized_std=(0.1, 0.2),
        )


class _FakeStateModule:
    def __init__(self, state):
        self._state = state
        self.loaded = None
        self.strict = None

    def state_dict(self):
        return self._state

    def load_state_dict(self, state, strict=True):
        self.loaded = state
        self.strict = strict

    motor_std = torch.ones(29)


def test_runner_initialization_loads_model_and_discriminator_without_resume(
    tmp_path,
):
    module = _load_checkpoint_module()
    source, destination = _fake_states()
    checkpoint_path = tmp_path / "model_30000.pt"
    torch.save(
        {
            "model_state_dict": source,
            "discriminator": {"weight": torch.tensor([3.0])},
            "optimizer_state_dict": {
                "state": {"must_not_load": True},
                "param_groups": [{"lr": 2.5e-5}],
            },
            "iter": 30000,
        },
        checkpoint_path,
    )
    actor_critic = _FakeStateModule(destination)
    discriminator = _FakeStateModule(
        {"weight": torch.tensor([-1.0])}
    )
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            actor_critic=actor_critic,
            discriminator=discriminator,
            optimizer=SimpleNamespace(param_groups=[{"lr": 1.0e-3}]),
            learning_rate=1.0e-3,
        ),
        current_learning_iteration=99,
    )

    report = module.initialize_runner_from_legacy_checkpoint(
        runner,
        checkpoint_path,
        motor_action_dim=29,
        input_column_maps=_fake_input_column_maps(),
        foothold_normalized_std=(0.1, 0.2),
    )

    assert report.unexpected == []
    assert actor_critic.strict is True
    assert discriminator.strict is True
    assert runner.current_learning_iteration == 0
    assert report.source_learning_rate == 2.5e-5
    assert runner.alg.learning_rate == 2.5e-5
    assert runner.alg.optimizer.param_groups[0]["lr"] == 2.5e-5
