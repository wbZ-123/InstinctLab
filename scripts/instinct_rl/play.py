"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import subprocess
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument(
    "--no_terminate", action="store_true", default=False, help="Do not remove termination conditions in simulation."
)
parser.add_argument(
    "--aux_reward",
    action="store_true",
    default=False,
    help="Whether to assign auxiliary rewards to each of the env's reward term.",
)
parser.add_argument(
    "--print_foothold_debug",
    action="store_true",
    default=False,
    help="Print command and foothold planner debug values while playing.",
)
parser.add_argument(
    "--print_foothold_debug_interval",
    type=int,
    default=50,
    help="Print foothold debug values every N play steps.",
)
parser.add_argument(
    "--print_foothold_debug_env_id",
    type=int,
    default=0,
    help="Primary environment id for reset debug and foothold marker visualization.",
)
parser.add_argument(
    "--print_foothold_debug_env_ids",
    type=str,
    default=None,
    help="Comma-separated environment ids to print, or 'all'. Defaults to --print_foothold_debug_env_id.",
)
parser.add_argument(
    "--print_foothold_marker_debug",
    action="store_true",
    default=False,
    help="Print extra foothold marker and reference-tracking debug values while playing.",
)
parser.add_argument(
    "--print_foothold_debug_on_anomaly",
    action="store_true",
    default=False,
    help="Print foothold debug only when an anomaly or large tracking error is detected.",
)
parser.add_argument(
    "--print_foothold_debug_on_plan_event",
    action="store_true",
    default=False,
    help="Also print foothold debug on fresh safe-target planning events, even between interval samples.",
)
parser.add_argument(
    "--print_reset_debug",
    action="store_true",
    default=False,
    help="Print termination/reset reason for --print_foothold_debug_env_id when that environment resets.",
)
parser.add_argument(
    "--show_foothold_debug_markers",
    action="store_true",
    default=False,
    help="Show foothold target, swing reference, actual swing foot, and reference trajectory markers in play.",
)
parser.add_argument(
    "--foothold_debug_trajectory_samples",
    type=int,
    default=12,
    help="Number of marker samples used for the displayed foothold swing reference trajectory.",
)
parser.add_argument(
    "--foothold_curriculum_scale_override",
    type=float,
    default=None,
    help=(
        "Override foothold reward/planner curriculum scale during play. "
        "Use 1.0 to evaluate a checkpoint with the full foothold planner curriculum."
    ),
)
parser.add_argument(
    "--foothold_step_terrain_only",
    action="store_true",
    default=False,
    help="Restrict play terrain generation to a single stair terrain family for foothold diagnostics.",
)
parser.add_argument(
    "--foothold_step_terrain_name",
    type=str,
    default="pyramid_stairs",
    help="Stair terrain family used by --foothold_step_terrain_only.",
)
parser.add_argument(
    "--foothold_step_terrain_rows",
    type=int,
    default=1,
    help="Number of terrain rows when --foothold_step_terrain_only is enabled.",
)
parser.add_argument(
    "--foothold_step_terrain_cols",
    type=int,
    default=1,
    help="Number of terrain columns when --foothold_step_terrain_only is enabled.",
)
parser.add_argument(
    "--foothold_step_terrain_level",
    type=int,
    default=0,
    help="Initial terrain level when --foothold_step_terrain_only is enabled.",
)
# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from instinct_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401
from instinctlab.learning import register_event_gated_foothold_algorithm
from instinctlab.managers.reward_manager import MultiRewardManager
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg
from play_debug import (
    build_foothold_debug_payload,
    build_reset_debug_payload,
    capture_reset_debug_snapshot,
    format_foothold_debug_line,
    format_learned_foothold_debug_line,
    format_reset_debug_line,
    is_foothold_debug_anomaly,
    is_foothold_debug_plan_event,
    is_learned_foothold_debug_event,
)
from play_curriculum import (
    load_checkpoint_foothold_curriculum_scale,
    load_recorded_foothold_curriculum_scale,
)
from play_foothold_viz import make_foothold_visualizer, update_foothold_visualizer
from play_learned_config import configure_learned_foothold_play
from play_step_terrain import (
    configure_free_world_camera,
    configure_step_only_terrain,
    configure_step_play_visuals,
)


def _read_nested_config_value(obj, path):
    """Read a nested value from configclass objects or loaded yaml dictionaries."""
    value = obj
    for name in path:
        if isinstance(value, dict):
            value = value.get(name, None)
        else:
            value = getattr(value, name, None)
        if value is None:
            return None
    return value


def _print_foothold_config_debug(prefix: str, cfg_or_env) -> None:
    """Print planner config values that are easy to confuse during play debugging."""
    if hasattr(cfg_or_env, "unwrapped"):
        try:
            foothold_cfg = cfg_or_env.unwrapped.scene.sensors["foothold_planner"].cfg
            startup_hold_s = getattr(foothold_cfg, "startup_hold_s", None)
            control_dt_s = getattr(foothold_cfg, "control_dt_s", None)
            print(
                f"[PLAY_CONFIG] {prefix} source=runtime_sensor "
                f"startup_hold_s={startup_hold_s} control_dt_s={control_dt_s}",
                flush=True,
            )
            return
        except Exception as exc:
            print(
                f"[PLAY_CONFIG] {prefix} failed to read runtime foothold config: {exc}",
                flush=True,
            )
            return

    startup_hold_s = _read_nested_config_value(
        cfg_or_env,
        ("scene", "foothold_planner", "startup_hold_s"),
    )
    control_dt_s = _read_nested_config_value(
        cfg_or_env,
        ("scene", "foothold_planner", "control_dt_s"),
    )
    print(
        f"[PLAY_CONFIG] {prefix} source=env_cfg "
        f"startup_hold_s={startup_hold_s} control_dt_s={control_dt_s}",
        flush=True,
    )

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()




def _parse_debug_env_ids(value: str | None, fallback_env_id: int, num_envs: int) -> list[int]:
    if value is None or value == "":
        env_ids = [int(fallback_env_id)]
    elif value.strip().lower() in {"all", "*"}:
        env_ids = list(range(int(num_envs)))
    else:
        env_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    unique_env_ids = []
    for env_id in env_ids:
        if env_id < 0 or env_id >= num_envs:
            raise ValueError(f"Debug env id {env_id} is outside [0, {num_envs}).")
        if env_id not in unique_env_ids:
            unique_env_ids.append(env_id)
    return unique_env_ids


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.foothold_step_terrain_only:
        step_terrain_summary = configure_step_only_terrain(
            env_cfg,
            terrain_name=args_cli.foothold_step_terrain_name,
            num_rows=args_cli.foothold_step_terrain_rows,
            num_cols=args_cli.foothold_step_terrain_cols,
            terrain_level=args_cli.foothold_step_terrain_level,
            disable_perlin=True,
        )
        print(f"[PLAY_TERRAIN] step_only={step_terrain_summary}", flush=True)
        visual_summary = configure_step_play_visuals(
            env_cfg,
            leg_volume_debug_vis=False,
        )
        print(f"[PLAY_VISUALS] step_diagnostic={visual_summary}", flush=True)
        camera_summary = configure_free_world_camera(env_cfg)
        print(f"[PLAY_CAMERA] free_world={camera_summary}", flush=True)
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    saved_agent_cfg = None
    saved_agent_cfg_path = os.path.join(log_dir, "params", "agent.yaml")
    if agent_cfg.load_run is not None and os.path.isfile(saved_agent_cfg_path):
        saved_agent_cfg = load_yaml(saved_agent_cfg_path)

    if args_cli.env_cfg:
        env_cfg = load_yaml(os.path.join(log_dir, "params", "env.yaml"))

    learned_foothold_play = False
    if saved_agent_cfg is not None:
        learned_foothold_play = configure_learned_foothold_play(
            env_cfg,
            saved_agent_cfg,
            register_algorithm=register_event_gated_foothold_algorithm,
        )
    if args_cli.agent_cfg or learned_foothold_play:
        if saved_agent_cfg is None:
            raise RuntimeError(
                f"Saved agent configuration not found: {saved_agent_cfg_path}"
            )
        agent_cfg_dict = saved_agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()
    print(
        "[PLAY_CONFIG] learned_foothold="
        f"{learned_foothold_play} agent_config_source="
        f"{'checkpoint' if args_cli.agent_cfg or learned_foothold_play else 'task_default'}",
        flush=True,
    )

    if args_cli.print_foothold_debug or args_cli.print_reset_debug:
        _print_foothold_config_debug("before_gym_make", env_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.print_foothold_debug or args_cli.print_reset_debug:
        _print_foothold_config_debug("after_gym_make", env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    foothold_curriculum_scale_override = args_cli.foothold_curriculum_scale_override
    if foothold_curriculum_scale_override is None and agent_cfg.load_run is not None:
        foothold_curriculum_scale_override = (
            load_checkpoint_foothold_curriculum_scale(resume_path)
        )
        if foothold_curriculum_scale_override is not None:
            print(
                "[INFO] Loaded foothold reward/planner curriculum scale "
                "from checkpoint for play: "
                f"{foothold_curriculum_scale_override:.3f}"
            )
    if foothold_curriculum_scale_override is None and agent_cfg.load_run is not None:
        foothold_curriculum_scale_override = load_recorded_foothold_curriculum_scale(
            log_dir,
        )
        if foothold_curriculum_scale_override is not None:
            print(
                "[INFO] Loaded recorded foothold reward/planner curriculum scale "
                f"for play: {foothold_curriculum_scale_override:.3f}"
            )
        else:
            print(
                "[INFO] No recorded foothold curriculum scale report found for play; "
                "using the environment's runtime curriculum state."
            )

    if foothold_curriculum_scale_override is not None:
        env.unwrapped.foothold_reward_curriculum_override_scale = float(
            foothold_curriculum_scale_override
        )
        print(
            "[INFO] Overriding foothold reward/planner curriculum scale during play: "
            f"{foothold_curriculum_scale_override:.3f}"
        )

    # react to custom play arguments
    if args_cli.no_terminate:
        # NOTE: This is only applicable with shadowing task
        env.unwrapped.termination_manager._term_cfgs = [
            env.unwrapped.termination_manager._term_cfgs[
                env.unwrapped.termination_manager._term_names.index("dataset_exhausted")
            ]
        ]
        env.unwrapped.termination_manager._term_names = ["dataset_exhausted"]

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.export_as_onnx(obs, export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    debug_env_ids = _parse_debug_env_ids(
        args_cli.print_foothold_debug_env_ids,
        args_cli.print_foothold_debug_env_id,
        env.unwrapped.num_envs,
    )
    if args_cli.print_foothold_debug or args_cli.print_reset_debug:
        print(f"[PLAY_DEBUG_CONFIG] debug_env_ids={debug_env_ids}", flush=True)
    timestep = 0
    foothold_visualizer = make_foothold_visualizer() if args_cli.show_foothold_debug_markers else None
    foothold_visualizer_warned = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            zero_act_active = timestep < args_cli.zero_act_until
            reset_debug_snapshots = {}
            if args_cli.print_reset_debug:
                for env_id in debug_env_ids:
                    reset_debug_snapshots[env_id] = capture_reset_debug_snapshot(
                        env.unwrapped,
                        env_id=env_id,
                    )
            if zero_act_active:
                actions[:] = 0.0
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
        timestep += 1

        if args_cli.print_reset_debug:
            for env_id in debug_env_ids:
                try:
                    done = bool(dones[env_id].detach().cpu().item())
                except Exception:
                    done = False
                if done:
                    try:
                        payload = build_reset_debug_payload(
                            env.unwrapped,
                            env_id=env_id,
                            done=done,
                            pre_step_snapshot=reset_debug_snapshots.get(env_id),
                        )
                        print(format_reset_debug_line(timestep, payload), flush=True)
                    except Exception as exc:
                        print(
                            f"[RESET_DEBUG] step={timestep} env_id={env_id} failed to read reset debug: {exc}",
                            flush=True,
                        )

        print_foothold_debug = (
            args_cli.print_foothold_debug or args_cli.print_foothold_marker_debug
        )
        if print_foothold_debug:
            interval_tick = timestep % max(args_cli.print_foothold_debug_interval, 1) == 0
            for env_id in debug_env_ids:
                try:
                    payload = build_foothold_debug_payload(
                        env.unwrapped,
                        env_id=env_id,
                        actions=actions,
                    )
                    plan_event = (
                        args_cli.print_foothold_debug_on_plan_event
                        and is_foothold_debug_plan_event(payload)
                    )
                    learned_event = is_learned_foothold_debug_event(payload)
                    if learned_event:
                        print(
                            format_learned_foothold_debug_line(
                                timestep,
                                payload,
                            ),
                            flush=True,
                        )
                    if not interval_tick and not plan_event:
                        continue
                    if (
                        not args_cli.print_foothold_debug_on_anomaly
                        or is_foothold_debug_anomaly(payload)
                        or plan_event
                    ):
                        print(
                            format_foothold_debug_line(
                                timestep,
                                payload,
                                zero_act_active=zero_act_active,
                            ),
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"[PLAY_DEBUG] step={timestep} env_id={env_id} failed to read foothold debug: {exc}",
                        flush=True,
                    )

        if foothold_visualizer is not None:
            try:
                foothold_sensor = env.unwrapped.scene.sensors["foothold_planner"]
                foothold_cfg = getattr(foothold_sensor, "cfg", None)
                swing_duration_s = float(
                    getattr(foothold_cfg, "swing_duration_s", 0.8)
                )
                update_foothold_visualizer(
                    foothold_visualizer,
                    foothold_sensor.data,
                    env_id=args_cli.print_foothold_debug_env_id,
                    trajectory_samples=args_cli.foothold_debug_trajectory_samples,
                    swing_duration_s=swing_duration_s,
                )
            except Exception as exc:
                if not foothold_visualizer_warned:
                    print(
                        f"[PLAY_DEBUG] step={timestep} failed to update foothold markers: {exc}",
                        flush=True,
                    )
                    foothold_visualizer_warned = True

        # override reward terms if auxiliary reward is enabled
        if args_cli.aux_reward:
            # NOTE: This is only applicable when reward_term has `.reward` to be overridden
            aux_rewards = ppo_runner.alg.compute_auxiliary_reward(infos["observations"])
            for aux_reward_name, aux_reward in aux_rewards.items():
                aux_term_cfg = env.unwrapped.reward_manager.get_term_cfg(aux_reward_name)  # type: ignore
                aux_term_cfg.func.reward[:] = aux_reward * getattr(ppo_runner.alg, aux_reward_name + "_coef", 1.0)

        # exit the loop if video_length is meet
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
