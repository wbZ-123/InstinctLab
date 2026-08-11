from __future__ import annotations


def configure_step_only_terrain(
    env_cfg,
    *,
    terrain_name: str = "pyramid_stairs",
    num_rows: int = 1,
    num_cols: int = 1,
    terrain_level: int = 0,
    disable_perlin: bool = True,
) -> dict[str, object]:
    """Restrict a parkour play config to one stair-like terrain family.

    This is intentionally a play-time diagnostic transform, not a training
    config change. It keeps the existing task and policy intact while ensuring
    the terrain generator cannot sample rough, gap, box, or slope sub-terrains.
    """

    terrain_cfg = env_cfg.scene.terrain
    terrain_generator = terrain_cfg.terrain_generator
    sub_terrains = terrain_generator.sub_terrains
    terrain_names = [
        name.strip() for name in terrain_name.split(",") if name.strip()
    ]
    if not terrain_names:
        raise ValueError("At least one step terrain name must be provided.")

    unknown_names = [name for name in terrain_names if name not in sub_terrains]
    if unknown_names:
        available = ", ".join(sorted(sub_terrains.keys()))
        raise ValueError(
            f"Unknown step terrain '{', '.join(unknown_names)}'. "
            f"Available terrains: {available}"
        )

    for name, sub_terrain_cfg in sub_terrains.items():
        sub_terrain_cfg.proportion = 1.0 if name in terrain_names else 0.0
        if hasattr(sub_terrain_cfg, "wall_prob"):
            sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]

    if disable_perlin:
        for name in terrain_names:
            selected = sub_terrains[name]
            if hasattr(selected, "perlin_cfg"):
                selected.perlin_cfg.noise_scale = 0.0

    terrain_generator.num_rows = int(num_rows)
    terrain_generator.num_cols = int(num_cols)
    terrain_generator.curriculum = False
    terrain_cfg.max_init_terrain_level = int(terrain_level)

    return {
        "terrain_name": terrain_name,
        "terrain_names": terrain_names,
        "num_rows": terrain_generator.num_rows,
        "num_cols": terrain_generator.num_cols,
        "terrain_level": terrain_cfg.max_init_terrain_level,
        "disable_perlin": bool(disable_perlin),
    }


def configure_step_play_visuals(
    env_cfg,
    *,
    leg_volume_debug_vis: bool = False,
) -> dict[str, object]:
    """Disable dense foot-volume debug geometry for clearer stair diagnostics."""

    leg_volume_points = getattr(env_cfg.scene, "leg_volume_points", None)
    if leg_volume_points is not None:
        leg_volume_points.debug_vis = bool(leg_volume_debug_vis)

    return {"leg_volume_points_debug_vis": bool(leg_volume_debug_vis)}


def configure_free_world_camera(
    env_cfg,
    *,
    eye: tuple[float, float, float] = (3.0, 1.5, 1.1),
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.6),
) -> dict[str, object]:
    """Place the viewer near the robot without locking it to the robot asset."""

    viewer = env_cfg.viewer
    viewer.origin_type = "world"
    viewer.asset_name = None
    viewer.eye = [float(value) for value in eye]
    viewer.lookat = [float(value) for value in lookat]

    return {
        "origin_type": viewer.origin_type,
        "asset_name": viewer.asset_name,
        "eye": viewer.eye,
        "lookat": viewer.lookat,
    }
