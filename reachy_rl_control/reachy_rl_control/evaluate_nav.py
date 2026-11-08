#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import time
from collections import Counter

import numpy as np
import rclpy
from stable_baselines3 import PPO, SAC

from reachy_rl_control.environment_nav import ReachyNavEnv
from reachy_rl_control.nav_interface import ReachyNavInterface
from reachy_rl_control.nav_tasks import make_task_sampler


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate a V3 Reachy navigation policy."
    )
    ap.add_argument("--algo", choices=["ppo", "sac"], required=True)
    ap.add_argument("--world", choices=["hospital", "bookstore"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    return ap.parse_args()


def bootstrap_ci(values, n_boot=2000, seed=2026):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return [None, None]

    rng = np.random.default_rng(seed)

    means = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)

    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def safe_mean(values):
    values = [
        float(x) for x in values
        if x is not None and math.isfinite(float(x))
    ]

    if not values:
        return None

    return float(np.mean(values))


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(args.model)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.out)),
        exist_ok=True,
    )

    csv_path = args.out + ".csv"
    summary_path = args.out + "_summary.json"

    rclpy.init()

    robot = None
    env = None
    rows = []

    start_wall = time.monotonic()

    try:
        robot = ReachyNavInterface()
        robot.start()

        print("[eval] Waiting for navigation interface...")

        if not robot.wait_until_ready(timeout=30.0):
            raise RuntimeError(
                "Navigation interface not ready. "
                "Check /clock, /odom, /scan and /set_entity_state."
            )

        print("[eval] Navigation interface ready.")

        sampler = make_task_sampler(f"{args.world}_eval")

        env = ReachyNavEnv(
            robot=robot,
            task_sampler=sampler,
            control_duration=0.1,
            max_steps=args.max_steps,
            check_goal_pose=True,
            verbose=True,
        )

        if args.algo == "ppo":
            model = PPO.load(
                args.model,
                env=env,
                device="cpu",
            )
        elif args.algo == "sac":
            model = SAC.load(
                args.model,
                env=env,
                device="cpu",
            )
        else:
            raise ValueError(args.algo)

        print(
            f"[eval] Evaluating {args.episodes} episodes "
            f"deterministically..."
        )

        for episode in range(1, args.episodes + 1):

            obs, reset_info = env.reset(
                seed=args.seed if episode == 1 else None
            )

            done = False
            episode_return = 0.0
            steps = 0
            step_clearances = []
            final_info = None

            while not done:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = env.step(action)

                episode_return += float(reward)
                steps += 1

                step_clearances.append(
                    float(info["min_lidar_distance"])
                )

                done = bool(terminated or truncated)
                final_info = info

            reason = final_info["termination_reason"]

            reference = float(
                final_info["reference_path_length"]
            )

            path_length = float(
                final_info["path_length"]
            )

            # Path efficiency is meaningful only when the goal is reached.
            # Use an SPL-style denominator so efficiency is always in [0, 1].
            if (
                reason == "goal"
                and math.isfinite(reference)
                and reference > 0.0
                and path_length > 1e-9
            ):
                path_efficiency = (
                    reference / max(reference, path_length)
                )
            else:
                path_efficiency = float("nan")

            row = {
                "episode": episode,
                "termination_reason": reason,
                "success": int(reason == "goal"),
                "collision": int(reason == "collision"),
                "fall": int(reason == "fall"),
                "timeout": int(reason == "timeout"),

                "steps": steps,
                "return": episode_return,
                "sim_time": float(final_info["sim_time"]),

                "path_length": path_length,
                "reference_path_length": reference,
                "straight_line_distance": float(
                    final_info["straight_line_distance"]
                ),
                "path_efficiency": path_efficiency,

                "min_clearance": float(
                    final_info["episode_min_lidar_distance"]
                ),
                "mean_step_clearance": (
                    float(np.mean(step_clearances))
                    if step_clearances else float("nan")
                ),

                "near_miss": int(
                    bool(final_info["near_miss_occurred"])
                ),

                "steps_below_0_20": int(
                    final_info["episode_steps_below_preferred"]
                ),
                "steps_below_0_10": int(
                    final_info["episode_steps_unsafe"]
                ),
                "steps_below_0_05": int(
                    final_info["episode_steps_near_miss"]
                ),

                "fraction_below_0_20": (
                    final_info["episode_steps_below_preferred"] / steps
                ),
                "fraction_below_0_10": (
                    final_info["episode_steps_unsafe"] / steps
                ),
                "fraction_below_0_05": (
                    final_info["episode_steps_near_miss"] / steps
                ),

                "contact_source": (
                    final_info["contact_source"] or ""
                ),
                "contact_link": (
                    final_info["contact_link"] or ""
                ),
                "contact_other": (
                    final_info["contact_other"] or ""
                ),
            }

            rows.append(row)

            print(
                f"[eval] {episode}/{args.episodes} "
                f"{reason:9s} "
                f"steps={steps:3d} "
                f"return={episode_return:+.1f} "
                f"min_clearance={row['min_clearance']:.3f}"
            )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

        outcomes = Counter(
            r["termination_reason"] for r in rows
        )

        success = [r["success"] for r in rows]
        collision = [r["collision"] for r in rows]
        timeout = [r["timeout"] for r in rows]
        fall = [r["fall"] for r in rows]
        near_miss = [r["near_miss"] for r in rows]

        min_clearance = [
            r["min_clearance"] for r in rows
        ]

        efficiencies = [
            r["path_efficiency"]
            for r in rows
            if math.isfinite(r["path_efficiency"])
        ]

        summary = {
            "status": "completed",
            "algo": args.algo,
            "world": args.world,
            "model": args.model,
            "episodes": len(rows),
            "max_steps": args.max_steps,
            "deterministic": True,

            "outcomes": dict(outcomes),

            "success_rate": safe_mean(success),
            "success_rate_95ci": bootstrap_ci(success),

            "collision_rate": safe_mean(collision),
            "collision_rate_95ci": bootstrap_ci(collision),

            "timeout_rate": safe_mean(timeout),
            "timeout_rate_95ci": bootstrap_ci(timeout),

            "fall_rate": safe_mean(fall),
            "fall_rate_95ci": bootstrap_ci(fall),

            "near_miss_rate": safe_mean(near_miss),
            "near_miss_rate_95ci": bootstrap_ci(near_miss),

            "mean_min_clearance": safe_mean(min_clearance),
            "mean_min_clearance_95ci": bootstrap_ci(min_clearance),

            "mean_path_efficiency": safe_mean(efficiencies),
            "mean_path_efficiency_95ci": bootstrap_ci(efficiencies),

            "mean_steps": safe_mean(
                [r["steps"] for r in rows]
            ),

            "mean_return": safe_mean(
                [r["return"] for r in rows]
            ),

            "wall_seconds": (
                time.monotonic() - start_wall
            ),

            "csv": csv_path,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n===== EVALUATION COMPLETE =====")
        print(json.dumps(summary, indent=2))

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        if robot is not None:
            try:
                robot.shutdown()
            except Exception:
                pass

            try:
                robot.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
