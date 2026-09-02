from __future__ import annotations

import torch


FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH = 100.0
FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH = 700.0
FOOTHOLD_NOMINAL_DEVIATION_TOLERANCE_M = 0.02


def _foothold_planner_data(env, sensor_name: str):
    return env.scene.sensors[sensor_name].data


def foothold_stabilization_mask(data) -> torch.Tensor:
    """Return environments in which foothold rewards must be paused.

    Contact-adaptive recovery exposes ``stabilization_active``.  The legacy
    field is kept as a fallback so older checkpoints/configurations retain
    their previous behavior when the new planner mode is disabled.
    """

    active = getattr(data, "stabilization_active", None)
    if active is None:
        active = getattr(data, "recovery_step_active", None)
    if active is None:
        gait_mode = getattr(data, "gait_mode", None)
        if gait_mode is not None:
            return torch.zeros_like(gait_mode, dtype=torch.bool)
        # Small unit-test fakes and legacy event-only data may not expose a
        # gait mode; infer the batch shape from the first tensor field.
        for value in vars(data).values():
            if torch.is_tensor(value):
                return torch.zeros_like(value, dtype=torch.bool)
        raise AttributeError(
            "Planner data must expose gait_mode or a tensor batch field."
        )
    return active.bool()


def _mask_stabilization_reward(value: torch.Tensor, data) -> torch.Tensor:
    return torch.where(
        foothold_stabilization_mask(data),
        torch.zeros_like(value),
        value,
    )


def mask_recovery_reward(value: torch.Tensor, data) -> torch.Tensor:
    """Pause a planner-specific reward while the motor policy self-stabilizes.

    This is deliberately a project-local wrapper.  The upstream locomotion
    reward functions remain unchanged.  Only terms that explicitly opt into
    this wrapper are suppressed for environments whose foothold sensor reports
    autonomous RECOVERY; the normal command-tracking terms stay active.
    """

    return _mask_stabilization_reward(value, data)


def no_fly(
    env,
    sensor_cfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return one only when neither selected foot has ground contact."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    force_history = env.scene.sensors[
        sensor_cfg.name
    ].data.net_forces_w_history
    selected_forces = force_history[:, :, sensor_cfg.body_ids]
    foot_contact = (
        torch.linalg.vector_norm(selected_forces, dim=-1).amax(dim=1)
        > threshold
    )
    return (~torch.any(foot_contact, dim=-1)).to(force_history.dtype)


def _recovery_masked_upstream_reward(
    env,
    upstream_name: str,
    *,
    sensor_name: str,
    module_name: str = "isaaclab.envs.mdp",
    **kwargs,
) -> torch.Tensor:
    # Import lazily so unit tests of this lightweight module do not need to
    # initialize the full Isaac Sim/PXR stack.
    if module_name == "isaaclab.envs.mdp":
        from isaaclab.envs import mdp as upstream_mdp
    elif module_name == "instinctlab.tasks.parkour.mdp":
        from instinctlab.tasks.parkour import mdp as upstream_mdp
    else:
        raise ValueError(f"Unsupported recovery reward module: {module_name}")

    upstream = getattr(upstream_mdp, upstream_name)
    value = upstream(env, **kwargs)
    data = _foothold_planner_data(env, sensor_name)
    return mask_recovery_reward(value, data)


def track_lin_vel_xy_exp_recovery_masked(
    env,
    std: float,
    command_name: str,
    sensor_name: str = "foothold_planner",
    asset_cfg=None,
) -> torch.Tensor:
    kwargs = {"std": std, "command_name": command_name}
    if asset_cfg is not None:
        kwargs["asset_cfg"] = asset_cfg
    return _recovery_masked_upstream_reward(
        env,
        "track_lin_vel_xy_exp",
        sensor_name=sensor_name,
        **kwargs,
    )


def feet_air_time_recovery_masked(
    env,
    command_name: str,
    vel_threshold: float,
    sensor_cfg,
    sensor_name: str = "foothold_planner",
) -> torch.Tensor:
    # ``feet_air_time`` is the project's parkour reward (it is not exported
    # by IsaacLab's base mdp module), so delegate to the parkour namespace.
    return _recovery_masked_upstream_reward(
        env,
        "feet_air_time",
        sensor_name=sensor_name,
        module_name="instinctlab.tasks.parkour.mdp",
        command_name=command_name,
        vel_threshold=vel_threshold,
        sensor_cfg=sensor_cfg,
    )


def track_ang_vel_z_exp_recovery_masked(
    env,
    std: float,
    command_name: str,
    sensor_name: str = "foothold_planner",
    asset_cfg=None,
) -> torch.Tensor:
    kwargs = {"std": std, "command_name": command_name}
    if asset_cfg is not None:
        kwargs["asset_cfg"] = asset_cfg
    return _recovery_masked_upstream_reward(
        env,
        "track_ang_vel_z_exp",
        sensor_name=sensor_name,
        **kwargs,
    )


def heading_error_recovery_masked(
    env,
    command_name: str,
    sensor_name: str = "foothold_planner",
) -> torch.Tensor:
    return _recovery_masked_upstream_reward(
        env,
        "heading_error",
        sensor_name=sensor_name,
        module_name="instinctlab.tasks.parkour.mdp",
        command_name=command_name,
    )


def dont_wait_recovery_masked(
    env,
    command_name: str,
    sensor_name: str = "foothold_planner",
    asset_cfg=None,
) -> torch.Tensor:
    kwargs = {"command_name": command_name}
    if asset_cfg is not None:
        kwargs["asset_cfg"] = asset_cfg
    return _recovery_masked_upstream_reward(
        env,
        "dont_wait",
        sensor_name=sensor_name,
        module_name="instinctlab.tasks.parkour.mdp",
        **kwargs,
    )


def stand_still_recovery_masked(
    env,
    command_name: str,
    sensor_name: str = "foothold_planner",
    asset_cfg=None,
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    kwargs = {
        "command_name": command_name,
        "threshold": threshold,
        "offset": offset,
    }
    if asset_cfg is not None:
        kwargs["asset_cfg"] = asset_cfg
    return _recovery_masked_upstream_reward(
        env,
        "stand_still",
        sensor_name=sensor_name,
        module_name="instinctlab.tasks.parkour.mdp",
        **kwargs,
    )


def _sync_desired_velocity_command(
    env,
    sensor_name: str,
    command_name: str | None,
) -> None:
    if command_name is None or not hasattr(env, "command_manager"):
        return

    planner = env.scene.sensors[sensor_name]
    planner.set_desired_velocity(
        env.command_manager.get_command(command_name),
    )


def _is_swing_mode(gait_mode: torch.Tensor) -> torch.Tensor:
    return (gait_mode == 1) | (gait_mode == 2)


def _is_touchdown_confirm_mode(gait_mode: torch.Tensor) -> torch.Tensor:
    return gait_mode == 3


def _planner_valid(data) -> torch.Tensor:
    value = getattr(data, "planner_valid", None)
    if value is None:
        return torch.ones_like(data.gait_mode, dtype=torch.bool)
    return value.bool()


def _is_valid_swing_mode(data) -> torch.Tensor:
    return _is_swing_mode(data.gait_mode) & _planner_valid(data)


def _is_valid_touchdown_confirm_mode(data) -> torch.Tensor:
    return _is_touchdown_confirm_mode(data.gait_mode) & _planner_valid(data)


def _is_late_touchdown_tracking_mode(
    data,
    min_phase: float,
) -> torch.Tensor:
    return _is_valid_swing_mode(data) & (data.phase >= min_phase)


def learned_foothold_safety_event_reward(
    env,
    sensor_name: str = "foothold_planner",
) -> torch.Tensor:
    """Reward only control steps where a learned foothold was evaluated."""

    data = _foothold_planner_data(env, sensor_name)
    event = data.learned_foothold_evaluated.bool()
    geometric_valid = data.learned_foothold_geometric_valid.bool()
    score = data.learned_foothold_safety_score.clamp(-1.0, 1.0)
    score = torch.where(
        geometric_valid,
        score,
        torch.full_like(score, -1.0),
    )
    return _mask_stabilization_reward(
        torch.where(event, score, torch.zeros_like(score)),
        data,
    )


def _normalized_nominal_deviation_cost(
    delta_xy: torch.Tensor,
    *,
    reachability_radius_x: float,
    reachability_radius_y: float,
) -> torch.Tensor:
    """Return the bounded cost corresponding to nominal-distance reward."""

    return (
        1.0
        - _nominal_deviation_reward(
            delta_xy,
            reachability_radius_x=reachability_radius_x,
            reachability_radius_y=reachability_radius_y,
        )
    ).clamp(0.0, 2.0) / 2.0


def _nominal_excess_deviation_cost(
    delta_xy: torch.Tensor,
    *,
    reachability_radius_x: float,
    reachability_radius_y: float,
    tolerance_m: float = FOOTHOLD_NOMINAL_DEVIATION_TOLERANCE_M,
) -> torch.Tensor:
    """Return zero through the 2 cm deadband and a bounded excess cost."""
    if delta_xy.shape[-1] != 2:
        raise ValueError("delta_xy must have two coordinates.")
    if (
        reachability_radius_x <= 0.0
        or reachability_radius_y <= 0.0
        or tolerance_m < 0.0
    ):
        raise ValueError("reachability radii must be positive.")

    radii = delta_xy.new_tensor((reachability_radius_x, reachability_radius_y))
    distance_m = torch.linalg.vector_norm(delta_xy, dim=-1)
    safe_distance = distance_m.clamp_min(torch.finfo(delta_xy.dtype).eps)
    direction = delta_xy / safe_distance.unsqueeze(-1)
    directional_radius = 1.0 / torch.sqrt(
        (direction / radii).square().sum(dim=-1)
    )
    directional_radius = torch.where(
        distance_m > 0.0,
        directional_radius,
        torch.full_like(
            directional_radius,
            min(reachability_radius_x, reachability_radius_y),
        ),
    )
    excess = (distance_m - tolerance_m).clamp_min(0.0)
    available = (directional_radius - tolerance_m).clamp_min(
        torch.finfo(delta_xy.dtype).eps
    )
    return (excess / available).clamp(0.0, 1.0)


def _nominal_deviation_reward(
    delta_xy: torch.Tensor,
    *,
    reachability_radius_x: float,
    reachability_radius_y: float,
    tolerance_m: float = FOOTHOLD_NOMINAL_DEVIATION_TOLERANCE_M,
) -> torch.Tensor:
    """Score nominal deviation with a 2 cm deadband and a +1 match bonus."""

    if delta_xy.shape[-1] != 2:
        raise ValueError("delta_xy must have two coordinates.")
    if (
        reachability_radius_x <= 0.0
        or reachability_radius_y <= 0.0
        or tolerance_m < 0.0
    ):
        raise ValueError(
            "reachability radii must be positive and tolerance non-negative."
        )
    radii = delta_xy.new_tensor(
        (reachability_radius_x, reachability_radius_y)
    )
    distance_m = torch.linalg.vector_norm(delta_xy, dim=-1)
    safe_distance = distance_m.clamp_min(torch.finfo(delta_xy.dtype).eps)
    direction = delta_xy / safe_distance.unsqueeze(-1)
    directional_radius = 1.0 / torch.sqrt(
        (direction / radii).square().sum(dim=-1)
    )
    directional_radius = torch.where(
        distance_m > 0.0,
        directional_radius,
        torch.full_like(
            directional_radius,
            min(reachability_radius_x, reachability_radius_y),
        ),
    )
    tolerance = delta_xy.new_tensor(tolerance_m)
    inside_tolerance = distance_m <= tolerance
    positive = 1.0 - distance_m / tolerance.clamp_min(
        torch.finfo(delta_xy.dtype).eps
    )
    negative = -(
        (distance_m - tolerance)
        / (directional_radius - tolerance).clamp_min(
            torch.finfo(delta_xy.dtype).eps
        )
    ).clamp(0.0, 1.0)
    return torch.where(inside_tolerance, positive, negative).clamp(-1.0, 1.0)


def _signed_command_progress_score(
    learned_velocity: torch.Tensor,
    desired_velocity: torch.Tensor,
) -> torch.Tensor:
    """Score progress along a frozen command, including reverse motion.

    The normalized projection is +1 for the desired velocity, 0 for no
    progress, and negative for motion in the opposite direction.  A zero
    command has no direction to score, so it returns a neutral value instead
    of manufacturing a direction from numerical noise.
    """

    if learned_velocity.shape != desired_velocity.shape:
        raise ValueError("learned and desired velocities must share shape.")
    if learned_velocity.shape[-1] != 2:
        raise ValueError("command velocities must contain two coordinates.")
    desired_norm_sq = torch.sum(torch.square(desired_velocity), dim=-1)
    projection = torch.sum(learned_velocity * desired_velocity, dim=-1)
    eps = torch.finfo(learned_velocity.dtype).eps
    score = (projection / desired_norm_sq.clamp_min(eps)).clamp(-1.0, 1.0)
    return torch.where(
        desired_norm_sq > eps,
        score,
        torch.zeros_like(score),
    )


def _reasonable_step_score(
    learned_velocity: torch.Tensor,
    desired_velocity: torch.Tensor,
) -> torch.Tensor:
    """Prefer the command-predicted step, not arbitrarily large progress."""
    if learned_velocity.shape != desired_velocity.shape:
        raise ValueError("learned and desired velocities must share shape.")
    if learned_velocity.shape[-1] != 2:
        raise ValueError("command velocities must contain two coordinates.")

    desired_norm_sq = torch.sum(torch.square(desired_velocity), dim=-1)
    projection_ratio = torch.sum(
        learned_velocity * desired_velocity,
        dim=-1,
    ) / desired_norm_sq.clamp_min(torch.finfo(learned_velocity.dtype).eps)
    score = (1.0 - torch.abs(projection_ratio - 1.0)).clamp(-1.0, 1.0)
    return torch.where(
        desired_norm_sq > torch.finfo(learned_velocity.dtype).eps,
        score,
        torch.zeros_like(score),
    )


def learned_foothold_planning_event_reward(
    env,
    sensor_name: str = "foothold_planner",
    reachability_radius_x: float = 0.42,
    reachability_radius_y: float = 0.25,
    velocity_lookahead_s: float = 0.10,
    nominal_step_width_m: float = 0.18,
    velocity_std: float = 0.5,
    safety_margin_reference_m: float = 0.04,
    # Kept for compatibility with older callers.  The branch-specific
    # objective no longer mixes these scalar weights into the safe-nominal
    # case.
    safe_nominal_weight: float | None = None,
    unsafe_nominal_weight: float | None = None,
) -> torch.Tensor:
    """Score learned footholds with branch-specific objectives.

    A safe nominal target is already the desired answer, so a safe learned
    target is scored only by its bounded distance to that nominal target.  A
    learned target that penetrates an obstacle receives only safety-driven
    punishment, with optional one-sided penalties for excessive deviation or
    reverse motion.  Progress and step-shape rewards are reserved for safe
    corrections to an unsafe nominal target. ``velocity_std`` and the legacy
    nominal weights remain accepted for checkpoint/config compatibility.
    """

    if (
        reachability_radius_x <= 0.0
        or reachability_radius_y <= 0.0
        or velocity_lookahead_s <= 0.0
        or nominal_step_width_m < 0.0
        or velocity_std <= 0.0
        or safety_margin_reference_m <= 0.0
        or (
            safe_nominal_weight is not None
            and not 0.0 <= safe_nominal_weight <= 1.0
        )
        or (
            unsafe_nominal_weight is not None
            and not 0.0 <= unsafe_nominal_weight <= 1.0
        )
    ):
        raise ValueError("planner reward scales and weights must be valid.")

    data = _foothold_planner_data(env, sensor_name)
    event = data.learned_foothold_evaluated.bool()
    delta_xy = (
        data.learned_foothold_decoded_f[:, :2]
        - data.raw_unclipped_foothold_f[:, :2]
    )
    nominal_affinity = _nominal_deviation_reward(
        delta_xy,
        reachability_radius_x=reachability_radius_x,
        reachability_radius_y=reachability_radius_y,
    ).clamp_min(0.0)
    nominal_excess_cost = _nominal_excess_deviation_cost(
        delta_xy,
        reachability_radius_x=reachability_radius_x,
        reachability_radius_y=reachability_radius_y,
    )
    nominal_term = nominal_affinity - nominal_excess_cost

    runtime_lookahead_s = getattr(data, "velocity_lookahead_s", None)
    if runtime_lookahead_s is None:
        lookahead_s = torch.full_like(
            data.nominal_feasible_velocity_f[:, 0],
            velocity_lookahead_s,
        )
    else:
        lookahead_s = runtime_lookahead_s.to(
            device=delta_xy.device,
            dtype=delta_xy.dtype,
        )
        if lookahead_s.shape != delta_xy.shape[:-1]:
            raise ValueError("velocity_lookahead_s data must match the batch.")
        fallback = torch.full_like(lookahead_s, velocity_lookahead_s)
        lookahead_s = torch.where(
            torch.isfinite(lookahead_s) & (lookahead_s > 0.0),
            lookahead_s,
            fallback,
        )

    side_sign = torch.where(
        data.swing_side == 0,
        torch.ones_like(data.swing_side, dtype=delta_xy.dtype),
        -torch.ones_like(data.swing_side, dtype=delta_xy.dtype),
    )
    learned_velocity = torch.stack(
        (
            data.learned_foothold_decoded_f[:, 0] / lookahead_s,
            (
                data.learned_foothold_decoded_f[:, 1]
                - side_sign * nominal_step_width_m
            )
            / lookahead_s,
        ),
        dim=-1,
    )
    signed_command_progress = _signed_command_progress_score(
        learned_velocity,
        data.nominal_feasible_velocity_f[:, :2],
    )
    reasonable_step = _reasonable_step_score(
        learned_velocity,
        data.nominal_feasible_velocity_f[:, :2],
    )

    learned_safety = data.learned_foothold_safety_score.clamp(
        -1.0,
        1.0,
    )
    learned_safety = torch.where(
        data.learned_foothold_geometric_valid.bool(),
        learned_safety,
        torch.full_like(learned_safety, -1.0),
    )
    preflight_known = getattr(
        data,
        "swing_preflight_ready",
        torch.zeros_like(event),
    ).bool()
    preflight_safe = getattr(
        data,
        "swing_preflight_safe",
        torch.ones_like(event),
    ).bool()
    execution_safe = ~preflight_known | preflight_safe
    learned_geometry_valid = data.learned_foothold_geometric_valid.bool()
    learned_safety_valid = data.learned_foothold_safety_valid.bool()
    nominal_safe = (
        data.nominal_geometric_valid.bool()
        & data.nominal_safety_valid.bool()
    )
    safety_margin = getattr(data, "learned_foothold_safety_margin_score", None)
    minimum_clearance = getattr(
        data,
        "learned_foothold_minimum_signed_clearance",
        None,
    )
    if safety_margin is None and minimum_clearance is not None:
        safety_margin = torch.clamp(
            minimum_clearance / safety_margin_reference_m,
            min=-1.0,
            max=1.0,
        )
    if safety_margin is None:
        # Older rollout buffers have no signed-clearance field. Their
        # penetration score is still a conservative negative fallback. A
        # non-penetrating legacy buffer has no outside-distance information,
        # so retain its previous fully-clear interpretation.
        safety_margin = torch.where(
            learned_safety < 0.0,
            learned_safety,
            torch.ones_like(learned_safety),
        )
    safety_margin = safety_margin.to(
        device=learned_safety.device,
        dtype=learned_safety.dtype,
    )
    safety_margin = torch.nan_to_num(
        safety_margin,
        nan=-1.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)
    penetrating_count = getattr(
        data,
        "learned_foothold_penetrating_point_count",
        None,
    )
    if penetrating_count is None:
        penetrating = learned_safety < 0.0
    else:
        penetrating = penetrating_count > 0.0
    # A failed obstacle gate is never allowed to look like a clear target just
    # because a legacy buffer omitted its per-point penetration statistics.
    penetrating = penetrating | ~learned_safety_valid
    # A safe nominal target has no reason to be optimized for a different
    # direction or step shape.  Its learned counterpart should simply copy it.
    unsafe_nominal_clear_score = (
        0.45 * safety_margin
        + 0.30 * signed_command_progress
        + 0.20 * reasonable_step
        + 0.05 * nominal_term
    )

    # Unsafe learned proposals must never obtain a positive score from making
    # forward progress.  Keep safety dominant and retain only one-sided
    # shaping penalties, so excessive deviation or reverse motion cannot be
    # mistaken for a successful correction.
    excessive_deviation_penalty = nominal_excess_cost.clamp(0.0, 1.0)
    reverse_motion_penalty = (-signed_command_progress).clamp(0.0, 1.0)
    penetrating_raw_score = (
        0.85 * safety_margin
        - 0.10 * excessive_deviation_penalty
        - 0.05 * reverse_motion_penalty
    )
    raw_score = torch.where(
        penetrating,
        penetrating_raw_score,
        torch.where(nominal_safe, nominal_term, unsafe_nominal_clear_score),
    )
    score = raw_score.clamp(-1.0, 1.0)
    score = torch.where(
        penetrating,
        torch.minimum(score, torch.full_like(score, -0.05)),
        score,
    )
    score = torch.where(
        ~execution_safe | ~learned_geometry_valid,
        torch.full_like(learned_safety, -1.0),
        score,
    )
    return _mask_stabilization_reward(
        torch.where(event, score, torch.zeros_like(score)),
        data,
    )


def _swing_foot_contact(data) -> torch.Tensor:
    env_ids = torch.arange(
        data.foot_contact.shape[0],
        device=data.foot_contact.device,
    )
    swing_side = data.swing_side.long().clamp(0, 1)
    return data.foot_contact[env_ids, swing_side].bool()


def _is_mode(
    gait_mode: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    return (gait_mode == mode).float()


def _readiness_episode_length(
    env,
    episode_length: torch.Tensor,
    ema_alpha: float = 0.20,
) -> torch.Tensor:
    """Return per-env readiness length from current age and completed-episode EMA.

    ``episode_length_buf`` is the age of the currently running episode.  It
    resets to a small value after an environment terminates, so using it alone
    makes a good walker temporarily look unready right after reset.  We keep a
    lightweight EMA of recently completed episode lengths on the environment and
    use the larger of current age and that EMA as the readiness signal.
    """

    current = episode_length.detach().to(dtype=torch.float32)
    previous = getattr(env, "_foothold_curriculum_previous_episode_length", None)
    ema = getattr(env, "_foothold_curriculum_completed_episode_length_ema", None)
    has_ema = getattr(env, "_foothold_curriculum_has_completed_episode_length_ema", None)
    last_update_step = getattr(env, "_foothold_curriculum_last_update_step", None)
    step = getattr(env, "common_step_counter", None)

    if previous is None or previous.shape != current.shape or previous.device != current.device:
        previous = current.clone()
        ema = torch.zeros_like(current)
        has_ema = torch.zeros_like(current, dtype=torch.bool)
        last_update_step = None

    if ema is None or ema.shape != current.shape or ema.device != current.device:
        ema = torch.zeros_like(current)
        has_ema = torch.zeros_like(current, dtype=torch.bool)

    if has_ema is None or has_ema.shape != current.shape or has_ema.device != current.device:
        has_ema = torch.zeros_like(current, dtype=torch.bool)

    should_update = True
    if step is not None and last_update_step is not None:
        try:
            should_update = int(step) != int(last_update_step)
        except (TypeError, ValueError):
            should_update = True

    if should_update:
        reset_mask = current < previous
        if torch.any(reset_mask):
            completed = previous
            updated_ema = torch.where(
                has_ema,
                (1.0 - float(ema_alpha)) * ema + float(ema_alpha) * completed,
                completed,
            )
            ema = torch.where(reset_mask, updated_ema, ema)
            has_ema = has_ema | reset_mask

        previous = current.clone()
        env._foothold_curriculum_previous_episode_length = previous
        env._foothold_curriculum_completed_episode_length_ema = ema
        env._foothold_curriculum_has_completed_episode_length_ema = has_ema
        env._foothold_curriculum_last_update_step = step

    ema_or_current = torch.where(has_ema, ema, current)
    return torch.maximum(current, ema_or_current)


def _reward_curriculum_scale(
    env,
    start_scale: float,
    end_scale: float,
    ramp_steps: int,
    gate: str | None = None,
    min_episode_length: float = 100.0,
    full_episode_length: float = 300.0,
    velocity_command_name: str = "base_velocity",
    velocity_std: float = 0.5,
    velocity_start_score: float = 0.4,
    velocity_full_score: float = 0.7,
    asset_name: str = "robot",
) -> float | torch.Tensor:
    episode_length_buf = getattr(env, "episode_length_buf", None)
    device = (
        episode_length_buf.device
        if isinstance(episode_length_buf, torch.Tensor)
        else torch.device("cpu")
    )
    override_scale = getattr(env, "foothold_reward_curriculum_override_scale", None)
    if override_scale is not None:
        override = torch.as_tensor(override_scale, dtype=torch.float32, device=device)
        if override.ndim == 0 and isinstance(episode_length_buf, torch.Tensor):
            override = override.expand(episode_length_buf.shape[0])
        return override.clamp(0.0, 1.0)

    if gate is None or gate == "none":
        if ramp_steps <= 0:
            return torch.tensor(float(end_scale), device=device)
        step = getattr(env, "common_step_counter", 0)
        if isinstance(step, torch.Tensor):
            step_tensor = step.detach().to(device=device, dtype=torch.float32)
        else:
            step_tensor = torch.tensor(float(step), device=device)
        progress = (step_tensor / float(ramp_steps)).clamp(0.0, 1.0)
        return float(start_scale) + (float(end_scale) - float(start_scale)) * progress

    if gate != "locomotion_readiness":
        raise ValueError(f"Unsupported foothold reward curriculum gate: {gate}")

    if episode_length_buf is None:
        num_envs = int(getattr(env, "num_envs", 1))
        return torch.zeros((num_envs,), dtype=torch.float32, device=device)

    episode_length = _readiness_episode_length(
        env,
        episode_length_buf.to(device=device, dtype=torch.float32),
    )
    episode_den = max(float(full_episode_length) - float(min_episode_length), 1e-6)
    episode_score = (
        (episode_length - float(min_episode_length)) / episode_den
    ).clamp(0.0, 1.0)

    del (
        velocity_command_name,
        velocity_std,
        velocity_start_score,
        velocity_full_score,
        asset_name,
    )
    return float(end_scale) * episode_score


def _sync_flat_target_curriculum_scale(
    env,
    sensor_name: str | None,
    scale: float | torch.Tensor,
    cache_key: tuple | None = None,
) -> None:
    if sensor_name is None or not hasattr(env, "scene"):
        return
    sensors = getattr(env.scene, "sensors", None)
    if sensors is None or sensor_name not in sensors:
        return
    set_scale = getattr(sensors[sensor_name], "set_flat_target_curriculum_scale", None)
    if set_scale is None:
        return

    if cache_key is not None:
        step = getattr(env, "common_step_counter", None)
        sync_key = (sensor_name, step, cache_key)
        if getattr(env, "_foothold_curriculum_last_synced_key", None) == sync_key:
            return
        env._foothold_curriculum_last_synced_key = sync_key

    if isinstance(scale, torch.Tensor):
        set_scale(scale.detach())
    else:
        set_scale(float(scale))


def _apply_reward_curriculum(
    value: torch.Tensor,
    env,
    curriculum_start_scale: float,
    curriculum_end_scale: float,
    curriculum_ramp_steps: int,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
    sensor_name: str | None = None,
) -> torch.Tensor:
    scale = _reward_curriculum_scale(
        env,
        start_scale=curriculum_start_scale,
        end_scale=curriculum_end_scale,
        ramp_steps=curriculum_ramp_steps,
        gate=curriculum_gate,
        min_episode_length=curriculum_min_episode_length,
        full_episode_length=curriculum_full_episode_length,
        velocity_command_name=curriculum_velocity_command_name,
        velocity_std=curriculum_velocity_std,
        velocity_start_score=curriculum_velocity_start_score,
        velocity_full_score=curriculum_velocity_full_score,
        asset_name=curriculum_asset_name,
    )
    cache_key = (
        float(curriculum_start_scale),
        float(curriculum_end_scale),
        int(curriculum_ramp_steps),
        curriculum_gate,
        float(curriculum_min_episode_length),
        float(curriculum_full_episode_length),
        curriculum_velocity_command_name,
        float(curriculum_velocity_std),
        float(curriculum_velocity_start_score),
        float(curriculum_velocity_full_score),
        curriculum_asset_name,
    )
    _sync_flat_target_curriculum_scale(env, sensor_name, scale, cache_key)
    return value * scale


def foothold_reward_curriculum_scale(
    env,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
) -> torch.Tensor:
    """Expose the current foothold reward curriculum scale for logging."""
    scale = _reward_curriculum_scale(
        env,
        start_scale=curriculum_start_scale,
        end_scale=curriculum_end_scale,
        ramp_steps=curriculum_ramp_steps,
        gate=curriculum_gate,
        min_episode_length=curriculum_min_episode_length,
        full_episode_length=curriculum_full_episode_length,
        velocity_command_name=curriculum_velocity_command_name,
        velocity_std=curriculum_velocity_std,
        velocity_start_score=curriculum_velocity_start_score,
        velocity_full_score=curriculum_velocity_full_score,
        asset_name=curriculum_asset_name,
    )
    if isinstance(scale, torch.Tensor):
        if scale.ndim == 0:
            episode_length_buf = getattr(env, "episode_length_buf", None)
            if hasattr(env, "num_envs"):
                num_envs = int(env.num_envs)
            elif episode_length_buf is not None:
                num_envs = int(episode_length_buf.shape[0])
            else:
                num_envs = 1
            return scale.expand(num_envs)
        return scale

    episode_length_buf = getattr(env, "episode_length_buf", None)
    device = getattr(episode_length_buf, "device", None)
    if device is None:
        device = torch.device("cpu")
    if hasattr(env, "num_envs"):
        num_envs = int(env.num_envs)
    elif episode_length_buf is not None:
        num_envs = int(episode_length_buf.shape[0])
    else:
        num_envs = 1
    return torch.full((num_envs,), float(scale), device=device)


def foothold_swing_tracking_exp(
    env,
    sensor_name: str = "foothold_planner",
    command_name: str | None = "base_velocity",
    std: float = 0.05,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Reward swing foot center tracking of the planner reference path."""
    _sync_desired_velocity_command(env, sensor_name, command_name)
    data = _foothold_planner_data(env, sensor_name)

    position_error = data.actual_swing_foot_pos_w - data.swing_reference_pos_w
    squared_error = torch.sum(torch.square(position_error), dim=-1)
    reward = torch.exp(-squared_error / (std * std))

    return _apply_reward_curriculum(
        _mask_stabilization_reward(
            reward * _is_valid_swing_mode(data).float(),
            data,
        ),
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_swing_velocity_tracking_exp(
    env,
    sensor_name: str = "foothold_planner",
    command_name: str | None = "base_velocity",
    std: float = 0.05,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Track the analytic swing velocity using position-equivalent error.

    Multiplying velocity error by the locked swing duration converts it to the
    position error it would accumulate over one swing.  This reuses the same
    five-centimeter bandwidth as position tracking without introducing a
    separate, arbitrary velocity threshold.
    """
    _sync_desired_velocity_command(env, sensor_name, command_name)
    data = _foothold_planner_data(env, sensor_name)
    velocity_error = (
        data.actual_swing_foot_vel_w - data.swing_reference_vel_w
    )
    equivalent_position_error = torch.linalg.vector_norm(
        velocity_error,
        dim=-1,
    ) * data.swing_duration_s
    reward = torch.exp(
        -torch.square(equivalent_position_error) / (std * std)
    )

    return _apply_reward_curriculum(
        _mask_stabilization_reward(
            reward * _is_valid_swing_mode(data).float(),
            data,
        ),
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_touchdown_tracking_exp(
    env,
    sensor_name: str = "foothold_planner",
    command_name: str | None = "base_velocity",
    std: float = 0.10,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Reward touchdown-confirm foot placement near the planner target."""
    _sync_desired_velocity_command(env, sensor_name, command_name)
    data = _foothold_planner_data(env, sensor_name)

    position_error = data.actual_swing_foot_pos_w - data.target_foothold_w
    squared_error = torch.sum(torch.square(position_error), dim=-1)
    reward = torch.exp(-squared_error / (std * std))

    return _apply_reward_curriculum(
        _mask_stabilization_reward(
            reward * _is_valid_touchdown_confirm_mode(data).float(),
            data,
        ),
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_swing_contact_indicator(
    env,
    sensor_name: str = "foothold_planner",
    min_phase: float = 0.20,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return 1 when the planned swing foot is still in contact too late.

    The small phase grace avoids penalizing the first few frames of a newly
    planned swing, where the foot may not have lifted yet.
    """
    data = _foothold_planner_data(env, sensor_name)
    swing_contact = _swing_foot_contact(data)
    after_liftoff_grace = data.phase >= min_phase
    indicator = (
        _is_swing_mode(data.gait_mode)
        & after_liftoff_grace
        & swing_contact
    ).float()
    return _apply_reward_curriculum(
        indicator,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_no_liftoff_indicator(
    env,
    sensor_name: str = "foothold_planner",
    min_phase: float = 0.35,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return 1 when swing has progressed but the foot never lifted.

    This is different from swing contact: it uses the planner's confirmed
    liftoff latch, so brief contact noise after a real liftoff is not counted
    as "never lifted".
    """
    data = _foothold_planner_data(env, sensor_name)
    after_liftoff_deadline = data.phase >= min_phase
    indicator = (
        _is_swing_mode(data.gait_mode)
        & after_liftoff_deadline
        & ~data.swing_has_lifted.bool()
    ).float()
    return _apply_reward_curriculum(
        indicator,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_swing_height_under_error_l1(
    env,
    sensor_name: str = "foothold_planner",
    max_error_m: float = 0.25,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return positive height deficit below the swing reference.

    Only under-shooting the reference height is penalized.  Overshooting is
    left to the base tracking/regularization terms, because the current failure
    mode is dragging the swing foot too low.
    """
    data = _foothold_planner_data(env, sensor_name)
    height_deficit = (
        data.swing_reference_pos_w[:, 2]
        - data.actual_swing_foot_pos_w[:, 2]
    ).clamp(min=0.0, max=max_error_m)
    penalty = height_deficit * _is_valid_swing_mode(data).float()
    return _apply_reward_curriculum(
        penalty,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_swing_xy_error_l2(
    env,
    sensor_name: str = "foothold_planner",
    max_error_m: float = 0.30,
    late_phase_start: float = 0.50,
    late_phase_full: float = 0.80,
    late_phase_max_scale: float = 2.0,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return planar distance from swing foot to the reference trajectory."""
    data = _foothold_planner_data(env, sensor_name)
    xy_error = torch.linalg.norm(
        (
            data.actual_swing_foot_pos_w[:, :2]
            - data.swing_reference_pos_w[:, :2]
        ),
        dim=-1,
    ).clamp_max(max_error_m)
    phase = getattr(data, "phase", None)
    if phase is None:
        phase_scale = torch.ones_like(xy_error)
    else:
        phase_window = max(late_phase_full - late_phase_start, 1.0e-6)
        phase_alpha = ((phase - late_phase_start) / phase_window).clamp(0.0, 1.0)
        phase_scale = 1.0 + phase_alpha * (late_phase_max_scale - 1.0)
    penalty = xy_error * phase_scale * _is_valid_swing_mode(data).float()
    return _apply_reward_curriculum(
        penalty,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_touchdown_xy_error_l2(
    env,
    sensor_name: str = "foothold_planner",
    min_phase: float = 0.65,
    target_tolerance_m: float = 0.02,
    zero_score_m: float = 0.05,
    max_penalty_m: float = 0.25,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return bounded touchdown XY placement score during the landing part of swing.

    The score is intentionally bounded to keep reward shaping balanced:

    - ``+1`` inside the target tolerance.
    - linearly decreases to ``0`` at ``zero_score_m``.
    - becomes negative past ``zero_score_m`` and saturates at ``-1`` after
      ``max_penalty_m`` additional error.
    """
    data = _foothold_planner_data(env, sensor_name)
    active = _is_late_touchdown_tracking_mode(
        data,
        min_phase,
    )
    positive_width = max(zero_score_m - target_tolerance_m, 1.0e-6)
    positive_score = (
        (zero_score_m - data.touchdown_xy_error) / positive_width
    ).clamp(0.0, 1.0)
    negative_alpha = (
        (data.touchdown_xy_error - zero_score_m).clamp_min(0.0)
        / max(max_penalty_m, 1.0e-6)
    ).clamp_max(1.0)
    negative_score = -torch.square(negative_alpha)
    score = torch.where(
        data.touchdown_xy_error <= zero_score_m,
        positive_score,
        negative_score,
    )
    score = score * active.float()
    return _apply_reward_curriculum(
        score,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_touchdown_z_error_l1(
    env,
    sensor_name: str = "foothold_planner",
    min_phase: float = 0.65,
    max_error_m: float = 0.20,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return touchdown height error during the landing part of swing."""
    data = _foothold_planner_data(env, sensor_name)
    active = _is_late_touchdown_tracking_mode(
        data,
        min_phase,
    )
    penalty = data.touchdown_z_error.clamp_max(max_error_m) * active.float()
    return _apply_reward_curriculum(
        penalty,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_swing_mode_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner state machine is in left/right swing."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_swing_mode(data.gait_mode).float()


def foothold_reset_mode_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner state machine is holding after reset."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 0)


def foothold_left_swing_mode_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner state machine is in left swing."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 1)


def foothold_right_swing_mode_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner state machine is in right swing."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 2)


def foothold_touchdown_confirm_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner accepted touchdown and swapped support."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_touchdown_confirm_mode(data.gait_mode).float()


def foothold_early_contact_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner reports early swing-foot contact."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 4)


def foothold_overdue_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the swing phase is overdue."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 5)


def foothold_stance_lost_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the support foot loses confirmed contact."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 6)


def foothold_hold_contact_lost_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when HOLD cannot establish stable double support."""
    data = _foothold_planner_data(env, sensor_name)
    return _is_mode(data.gait_mode, 9)


def foothold_gait_anomaly_indicator(
    env,
    sensor_name: str = "foothold_planner",
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return 1 for any planner gait anomaly, max-pooled across reasons."""
    data = _foothold_planner_data(env, sensor_name)
    failure_mode = (
        (data.gait_mode == 4)
        | (data.gait_mode == 5)
        | (data.gait_mode == 6)
        | (data.gait_mode == 7)
        | (data.gait_mode == 9)
    )
    planner_invalid = ~data.planner_valid.bool()
    indicator = (failure_mode | planner_invalid).float()
    return _apply_reward_curriculum(
        indicator,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_recovery_indicator(
    env,
    sensor_name: str = "foothold_planner",
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return 1 when the planner is in recovery after a gait failure."""
    data = _foothold_planner_data(env, sensor_name)
    indicator = _is_mode(data.gait_mode, 8)
    return _apply_reward_curriculum(
        indicator,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_clearance_safe_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the adjusted swing trajectory is clearance-safe."""
    data = _foothold_planner_data(env, sensor_name)
    return data.swing_clearance_safe.float()


def foothold_clearance_penetration_l1(
    env,
    sensor_name: str = "foothold_planner",
    max_penetration_m: float = 0.15,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return swing trajectory penetration depth into edge obstacles."""
    data = _foothold_planner_data(env, sensor_name)
    value = data.swing_clearance_penetration.clamp_max(max_penetration_m)
    return _apply_reward_curriculum(
        value,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_touchdown_accepted_indicator(
    env,
    sensor_name: str = "foothold_planner",
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = FOOTHOLD_CURRICULUM_MIN_EPISODE_LENGTH,
    curriculum_full_episode_length: float = FOOTHOLD_CURRICULUM_FULL_EPISODE_LENGTH,
    curriculum_velocity_command_name: str = "base_velocity",
    curriculum_velocity_std: float = 0.5,
    curriculum_velocity_start_score: float = 0.4,
    curriculum_velocity_full_score: float = 0.7,
    curriculum_asset_name: str = "robot",
):
    """Return 1 when the current swing has physically touched down."""
    data = _foothold_planner_data(env, sensor_name)
    value = data.touchdown_accepted.float()
    return _apply_reward_curriculum(
        value,
        env,
        curriculum_start_scale,
        curriculum_end_scale,
        curriculum_ramp_steps,
        curriculum_gate,
        curriculum_min_episode_length,
        curriculum_full_episode_length,
        curriculum_velocity_command_name,
        curriculum_velocity_std,
        curriculum_velocity_start_score,
        curriculum_velocity_full_score,
        curriculum_asset_name,
        sensor_name,
    )


def foothold_plan_invalid_indicator(
    env,
    sensor_name: str = "foothold_planner",
):
    """Return 1 when the planner reports no valid foothold plan."""
    data = _foothold_planner_data(env, sensor_name)
    return (~data.planner_valid).float()
