from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "instinct_rl"
        / "play_step_terrain.py"
    )
    spec = importlib.util.spec_from_file_location("play_step_terrain_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _terrain(name: str, proportion: float):
    return SimpleNamespace(
        name=name,
        proportion=proportion,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
        perlin_cfg=SimpleNamespace(noise_scale=0.05),
    )


def test_configure_step_only_terrain_keeps_only_selected_stairs():
    module = _load_module()
    terrain_generator = SimpleNamespace(
        sub_terrains={
            "perlin_rough": _terrain("perlin_rough", 0.05),
            "pyramid_stairs": _terrain("pyramid_stairs", 0.15),
            "boxes": _terrain("boxes", 0.10),
        },
        num_rows=4,
        num_cols=10,
        curriculum=True,
    )
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(
            terrain=SimpleNamespace(
                terrain_generator=terrain_generator,
                max_init_terrain_level=5,
            ),
            num_envs=10,
        )
    )

    summary = module.configure_step_only_terrain(
        env_cfg,
        terrain_name="pyramid_stairs",
        num_rows=1,
        num_cols=1,
        terrain_level=0,
        disable_perlin=True,
    )

    assert summary["terrain_name"] == "pyramid_stairs"
    assert terrain_generator.sub_terrains["pyramid_stairs"].proportion == 1.0
    assert terrain_generator.sub_terrains["perlin_rough"].proportion == 0.0
    assert terrain_generator.sub_terrains["boxes"].proportion == 0.0
    assert terrain_generator.sub_terrains["pyramid_stairs"].wall_prob == [0.0, 0.0, 0.0, 0.0]
    assert terrain_generator.sub_terrains["pyramid_stairs"].perlin_cfg.noise_scale == 0.0
    assert terrain_generator.num_rows == 1
    assert terrain_generator.num_cols == 1
    assert terrain_generator.curriculum is False
    assert env_cfg.scene.terrain.max_init_terrain_level == 0


def test_configure_step_only_terrain_can_keep_up_and_down_stairs():
    module = _load_module()
    terrain_generator = SimpleNamespace(
        sub_terrains={
            "perlin_rough": _terrain("perlin_rough", 0.05),
            "pyramid_stairs": _terrain("pyramid_stairs", 0.15),
            "pyramid_stairs_inv": _terrain("pyramid_stairs_inv", 0.15),
            "boxes": _terrain("boxes", 0.10),
        },
        num_rows=4,
        num_cols=10,
        curriculum=True,
    )
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(
            terrain=SimpleNamespace(
                terrain_generator=terrain_generator,
                max_init_terrain_level=5,
            ),
            num_envs=10,
        )
    )

    summary = module.configure_step_only_terrain(
        env_cfg,
        terrain_name="pyramid_stairs,pyramid_stairs_inv",
        num_rows=1,
        num_cols=2,
    )

    assert summary["terrain_names"] == ["pyramid_stairs", "pyramid_stairs_inv"]
    assert terrain_generator.sub_terrains["pyramid_stairs"].proportion == 1.0
    assert terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion == 1.0
    assert terrain_generator.sub_terrains["perlin_rough"].proportion == 0.0
    assert terrain_generator.sub_terrains["boxes"].proportion == 0.0
    assert terrain_generator.num_cols == 2


def test_configure_step_only_terrain_rejects_unknown_terrain():
    module = _load_module()
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(
            terrain=SimpleNamespace(
                terrain_generator=SimpleNamespace(
                    sub_terrains={"pyramid_stairs": _terrain("pyramid_stairs", 0.15)}
                )
            )
        )
    )

    try:
        module.configure_step_only_terrain(env_cfg, terrain_name="boxes")
    except ValueError as exc:
        assert "boxes" in str(exc)
        assert "pyramid_stairs" in str(exc)
    else:
        raise AssertionError("expected unknown step terrain to raise ValueError")


def test_configure_step_play_visuals_hides_leg_volume_debug():
    module = _load_module()
    leg_volume_points = SimpleNamespace(debug_vis=True)
    env_cfg = SimpleNamespace(
        scene=SimpleNamespace(leg_volume_points=leg_volume_points)
    )

    summary = module.configure_step_play_visuals(
        env_cfg,
        leg_volume_debug_vis=False,
    )

    assert leg_volume_points.debug_vis is False
    assert summary == {"leg_volume_points_debug_vis": False}


def test_configure_free_world_camera_starts_near_robot_without_asset_lock():
    module = _load_module()
    viewer = SimpleNamespace(
        eye=[4.0, 0.75, 1.0],
        lookat=[0.0, 0.75, 0.0],
        origin_type="asset_root",
        asset_name="robot",
    )
    env_cfg = SimpleNamespace(viewer=viewer)

    summary = module.configure_free_world_camera(
        env_cfg,
        eye=(3.0, 1.5, 1.1),
        lookat=(0.0, 0.0, 0.6),
    )

    assert viewer.origin_type == "world"
    assert viewer.asset_name is None
    assert viewer.eye == [3.0, 1.5, 1.1]
    assert viewer.lookat == [0.0, 0.0, 0.6]
    assert summary == {
        "origin_type": "world",
        "asset_name": None,
        "eye": [3.0, 1.5, 1.1],
        "lookat": [0.0, 0.0, 0.6],
    }
