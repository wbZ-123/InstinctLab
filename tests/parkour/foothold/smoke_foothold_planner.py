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

    monitor_terms = unwrapped.monitor_manager.active_terms
    print("[SMOKE] monitor terms:", sorted(monitor_terms.keys()), flush=True)

    monitor = monitor_terms["foothold_planner"]
    monitor_log = monitor.get_log(is_episode=False)
    print("[SMOKE] foothold monitor:", monitor_log, flush=True)

    assert "clearance_safe_fraction" in monitor_log
    assert "plan_invalid_fraction" in monitor_log
    assert all(
        bool(torch.isfinite(value).all())
        for value in monitor_log.values()
    )

    print(
        "[SMOKE] observation groups:",
        list(obs.keys()),
        flush=True,
    )

    def report_tensor(name: str, tensor: torch.Tensor):
        finite = torch.isfinite(tensor)
        print(
            f"[NAN_PROBE] {name}: "
            f"shape={tuple(tensor.shape)} "
            f"finite={bool(finite.all())} "
            f"nan={int(torch.isnan(tensor).sum().item())} "
            f"inf={int(torch.isinf(tensor).sum().item())} "
            f"min={tensor[finite].min().item() if finite.any() else 'NA'} "
            f"max={tensor[finite].max().item() if finite.any() else 'NA'}",
            flush=True,
        )
        if not finite.all():
            bad_idx = (~finite).nonzero()[:20].detach().cpu().tolist()
            print(f"[NAN_PROBE] {name} bad_idx first20={bad_idx}", flush=True)

    for group_name, group_obs in obs.items():
        if isinstance(group_obs, dict):
            for obs_name, value in group_obs.items():
                if torch.is_tensor(value):
                    report_tensor(f"{group_name}.{obs_name}", value)
        elif torch.is_tensor(group_obs):
            report_tensor(group_name, group_obs)

    planner = env.unwrapped.scene.sensors["foothold_planner"]
    data = planner.data

    for field_name in [
        "target_foothold_w",
        "target_foothold_f",
        "feasible_velocity_f",
        "swing_reference_pos_w",
        "default_swing_reference_pos_w",
        "swing_apex_height",
        "default_swing_apex_height",
        "swing_clearance_penetration",
        "phase",
    ]:
        value = getattr(data, field_name, None)
        if torch.is_tensor(value):
            report_tensor(f"planner.{field_name}", value)

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
            "[SMOKE] clearance default_apex=",
            data.default_swing_apex_height[:1].detach().cpu().tolist(),
            "adjusted_apex=",
            data.swing_apex_height[:1].detach().cpu().tolist(),
            "safe=",
            data.swing_clearance_safe[:1].detach().cpu().tolist(),
            "penetration=",
            data.swing_clearance_penetration[:1].detach().cpu().tolist(),
            "raw_target_f=",
            data.raw_unclipped_foothold_f[:1].detach().cpu().tolist()
                if data.raw_unclipped_foothold_f is not None
                else None,
            "safe_search=",
            data.safe_target_search_performed[:1].detach().cpu().tolist()
                if data.safe_target_search_performed is not None
                else None,
            "safe_final_valid=",
            data.safe_target_final_valid[:1].detach().cpu().tolist()
                if data.safe_target_final_valid is not None
                else None,
            "safe_fallback=",
            data.safe_target_used_fallback[:1].detach().cpu().tolist()
                if data.safe_target_used_fallback is not None
                else None,
            "safe_score=",
            data.safe_target_score[:1].detach().cpu().tolist()
                if data.safe_target_score is not None
                else None,
            "safe_nominal_valid=",
            data.safe_target_nominal_valid[:1].detach().cpu().tolist()
                if data.safe_target_nominal_valid is not None
                else None,
            "safe_candidate_valid_count=",
            data.safe_target_candidate_valid_count[:1].detach().cpu().tolist()
                if data.safe_target_candidate_valid_count is not None
                else None,
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
