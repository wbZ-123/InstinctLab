"""Smoke test for the foothold planner sensor.

Run from the repository root with:

    ../IsaacLab/isaaclab.sh -p tests/parkour/foothold/smoke_foothold_planner.py \
        --headless \
        --task Instinct-Parkour-Target-Amp-G1-Play-v0 \
        --num_envs 1

This is intentionally not a pytest test: it launches Isaac Sim and verifies
that the planner sensor can be created, reset, stepped, and queried.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.getcwd(), "source", "instinctlab"))

print("[SMOKE] script started", flush=True)

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="Instinct-Parkour-Target-Amp-G1-Play-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[SMOKE] launching Isaac app", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[SMOKE] Isaac app launched", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import instinctlab.tasks.parkour.config.g1  # noqa: E402,F401

print("[SMOKE] imports after app completed", flush=True)


def main() -> None:
    print("[SMOKE] parsing env cfg", flush=True)
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    print("[SMOKE] env cfg parsed", type(env_cfg).__name__, flush=True)

    # The G1 parkour AMP task depends on AMASS motion-reference files.
    # This smoke test only verifies the foothold planner sensor, so it
    # disables those AMP-only pieces to avoid requiring local dataset paths.
    print(
        "[SMOKE] disabling AMP motion reference for foothold-only smoke",
        flush=True,
    )
    env_cfg.scene.motion_reference = None
    env_cfg.observations.amp_reference = None
    env_cfg.terminations.dataset_exhausted = None

    print("[SMOKE] creating gym env", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    print("[SMOKE] sensors:", sorted(unwrapped.scene.sensors.keys()), flush=True)

    planner = unwrapped.scene.sensors["foothold_planner"]
    print(
        "[SMOKE] planner virtual obstacles:",
        list(getattr(planner, "_virtual_obstacles", {}).keys()),
        flush=True,
    )
    print("[SMOKE] planner:", type(planner).__name__, flush=True)

    print("[SMOKE] resetting env", flush=True)
    obs, _ = env.reset()

    print(
        "[SMOKE] observation groups:",
        list(obs.keys()),
        flush=True,
    )

    if "policy" in obs:
        print(
            "[SMOKE] policy obs keys:",
            list(obs["policy"].keys())
            if isinstance(obs["policy"], dict)
            else type(obs["policy"]).__name__,
            flush=True,
        )
        if isinstance(obs["policy"], dict) and "foothold_planner" in obs["policy"]:
            print(
                "[SMOKE] policy foothold_planner obs shape:",
                tuple(obs["policy"]["foothold_planner"].shape),
                flush=True,
            )
            print(
                "[SMOKE] policy foothold_planner obs sample:",
                obs["policy"]["foothold_planner"][:1].detach().cpu().tolist(),
                flush=True,
            )

    if "critic" in obs:
        print(
            "[SMOKE] critic obs keys:",
            list(obs["critic"].keys())
            if isinstance(obs["critic"], dict)
            else type(obs["critic"]).__name__,
            flush=True,
        )
        if isinstance(obs["critic"], dict) and "foothold_planner" in obs["critic"]:
            print(
                "[SMOKE] critic foothold_planner obs shape:",
                tuple(obs["critic"]["foothold_planner"].shape),
                flush=True,
            )

    print(
        "[SMOKE] planner desired velocity will be synced from base_velocity",
        flush=True,
    )

    for step in range(40):
        actions = torch.zeros(
            unwrapped.num_envs,
            unwrapped.action_manager.total_action_dim,
            device=unwrapped.device,
        )
        env.step(actions)

        data = planner.data
        print(
            "[SMOKE]",
            "step=",
            step,
            "mode=",
            data.gait_mode[:1].detach().cpu().tolist(),
            "phase=",
            data.phase[:1].detach().cpu().tolist(),
            "command=",
            unwrapped.command_manager.get_command("base_velocity")[
                :1
            ].detach().cpu().tolist(),
            "contact=",
            data.foot_contact[:1].detach().cpu().tolist(),
            "target=",
            data.target_foothold_w[:1].detach().cpu().tolist(),
            "target_f=",
            data.target_foothold_f[:1].detach().cpu().tolist(),
            "feasible_velocity=",
            data.feasible_velocity_f[:1].detach().cpu().tolist(),
            "touchdown=",
            data.touchdown_accepted[:1].detach().cpu().tolist(),
            flush=True,
        )

    env.close()
    print("[SMOKE] env closed", flush=True)


try:
    try:
        main()
    except BaseException:
        print("[SMOKE] exception caught before app close:", flush=True)
        traceback.print_exc()
        raise
finally:
    print("[SMOKE] closing Isaac app", flush=True)
    simulation_app.close()
    print("[SMOKE] script finished", flush=True)
