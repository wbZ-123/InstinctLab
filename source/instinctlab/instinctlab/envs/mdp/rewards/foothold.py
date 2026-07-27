from __future__ import annotations

import torch


def _foothold_planner_data(env, sensor_name: str):
    return env.scene.sensors[sensor_name].data


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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    std: float = 0.15,
    curriculum_start_scale: float = 1.0,
    curriculum_end_scale: float = 1.0,
    curriculum_ramp_steps: int = 0,
    curriculum_gate: str | None = None,
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
        reward * _is_valid_swing_mode(data).float(),
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
        reward * _is_valid_touchdown_confirm_mode(data).float(),
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
    curriculum_min_episode_length: float = 100.0,
    curriculum_full_episode_length: float = 300.0,
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
