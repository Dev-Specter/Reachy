#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time
from collections import Counter

import numpy as np
import rclpy
from std_srvs.srv import Trigger
from stable_baselines3 import PPO, SAC

from reachy_rl_control.environment_nav import ReachyNavEnv
from reachy_rl_control.nav_interface import ReachyNavInterface
from reachy_rl_control.nav_tasks import make_task_sampler, FixedTaskSampler, load_tasks


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate a V3 Reachy navigation policy."
    )
    ap.add_argument("--algo", choices=["ppo", "sac", "mbpo", "ur_mbpo"], required=True)
    ap.add_argument(
        "--world",
        choices=[
            "hospital",
            "bookstore",
            "hospital_dynamic_train",
            "hospital_dynamic_eval",
        ],
        required=True,
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--eval-set",
        default=None,
        help="Optional fixed task-set JSON. When supplied, tasks are loaded "
             "from this file instead of <world>_eval.",
    )
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


def reset_pedestrian_phase(node, timeout=2.0):
    """
    Reset dynamic pedestrian trajectory phase if the service exists.

    Static-world evaluation is unchanged when the service is absent.
    """
    client = node.create_client(Trigger, "/pedestrian/reset_phase")

    if not client.wait_for_service(timeout_sec=0.05):
        return False

    future = client.call_async(Trigger.Request())

    deadline = time.monotonic() + float(timeout)

    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.001)

    if not future.done():
        try:
            future.cancel()
        except Exception:
            pass
        return False

    result = future.result()

    success = bool(
        result is not None
        and result.success
    )

    if success:
        message = str(
            getattr(result, "message", "")
        )

        active_mode = None
        phase = None

        for part in message.split(","):
            part = part.strip()

            if part.startswith("active="):
                active_mode = (
                    part.split("=", 1)[1].strip()
                )

            elif part.startswith("phase="):
                value = (
                    part.split("=", 1)[1]
                    .strip()
                    .rstrip("s")
                )

                try:
                    phase = float(value)
                except ValueError:
                    phase = None

        node.last_pedestrian_motion = active_mode
        node.last_pedestrian_phase = phase

    return success


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

        if args.eval_set is not None:
            if not os.path.exists(args.eval_set):
                raise FileNotFoundError(args.eval_set)

            tasks = load_tasks(args.eval_set)

            if args.episodes > len(tasks):
                raise ValueError(
                    f"--episodes={args.episodes} exceeds fixed eval-set size "
                    f"{len(tasks)}: {args.eval_set}"
                )

            sampler = FixedTaskSampler(tasks)

            print(
                f"[eval] Using fixed task set: {args.eval_set} "
                f"({len(tasks)} tasks)"
            )
        else:
            if args.world in (
                "hospital_dynamic_train",
                "hospital_dynamic_eval",
            ):
                sampler = make_task_sampler(
                    args.world
                )
            else:
                sampler = make_task_sampler(
                    f"{args.world}_eval"
                )

        fixed_tasks = (
            sampler.tasks
            if isinstance(sampler, FixedTaskSampler)
            else None
        )

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
        elif args.algo in ("sac", "mbpo", "ur_mbpo"):
            # SAC, MBPO and UR-MBPO all deploy the learned SAC policy.
            # Dynamics ensembles are training-time components only.
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

            task_index = (
                (episode - 1) % len(fixed_tasks)
                if fixed_tasks
                else None
            )

            obs, reset_info = env.reset(
                seed=args.seed if episode == 1 else None,
                options=(
                    {"task": fixed_tasks[task_index]}
                    if fixed_tasks
                    else None
                ),
            )

            pedestrian_phase_reset = reset_pedestrian_phase(robot)

            if pedestrian_phase_reset:
                # Allow one short simulation interval for the proxy controller
                # to apply the phase-0 position before the first policy action.
                t_phase = robot.sim_time()
                robot.wait_sim_until(t_phase + 0.1)

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

                if steps < 20:
                    print(
                        f"[ACTION] step={steps + 1} "
                        f"shape={np.asarray(action).shape} "
                        f"raw={np.asarray(action)}"
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

            # -------------------------------------------------
            # Scenario labels for structured dynamic analysis
            # -------------------------------------------------

            robot_task_group = (
                final_info.get("robot_task_group") or ""
            )

            robot_geometry = (
                final_info.get("robot_geometry") or ""
            )

            robot_exposure = (
                final_info.get("robot_exposure") or ""
            )

            # FixedTaskSampler does not carry last_group.
            # For structured held-out evaluation, recover the
            # exact group from the frozen eval-set filename.
            if (
                not robot_task_group
                and args.eval_set
            ):
                eval_path = Path(args.eval_set)
                stem = eval_path.stem

                if (
                    eval_path.parent.name == "eval"
                    and eval_path.parent.parent.name == "dynamic"
                    and eval_path.parent.parent.parent.name == "hospital"
                ):
                    robot_task_group = stem

                    for geometry in (
                        "straight",
                        "turning",
                        "complex",
                    ):
                        geometry_prefix = geometry + "_"

                        if robot_task_group.startswith(
                            geometry_prefix
                        ):
                            robot_geometry = geometry
                            robot_exposure = (
                                robot_task_group[
                                    len(geometry_prefix):
                                ]
                            )
                            break

            row = {
                "episode": episode,
                "task_index": (
                    ""
                    if task_index is None
                    else task_index
                ),

                "robot_task_group": robot_task_group,
                "robot_geometry": robot_geometry,
                "robot_exposure": robot_exposure,
                "human_motion": (
                    final_info.get("human_motion") or ""
                ),
                "human_phase": (
                    final_info.get("human_phase")
                    if final_info.get("human_phase") is not None
                    else ""
                ),

                "termination_reason": reason,
                "success": int(reason == "goal"),
                "collision": int(reason == "collision"),
                "human_collision": int(reason == "human_collision"),
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

                "human_near_miss": int(
                    reason != "human_collision"
                    and bool(
                        final_info.get(
                            "human_near_miss_occurred",
                            False,
                        )
                    )
                ),
                "min_human_clearance": (
                    float(final_info["episode_min_human_clearance"])
                    if final_info.get("episode_min_human_clearance") is not None
                    else float("nan")
                ),
                "steps_human_near_miss": int(
                    final_info.get("episode_steps_human_near_miss", 0)
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
        human_collision = [r["human_collision"] for r in rows]
        timeout = [r["timeout"] for r in rows]
        fall = [r["fall"] for r in rows]
        near_miss = [r["near_miss"] for r in rows]
        human_near_miss = [r["human_near_miss"] for r in rows]

        min_clearance = [
            r["min_clearance"] for r in rows
        ]

        min_human_clearance = [
            r["min_human_clearance"]
            for r in rows
            if math.isfinite(r["min_human_clearance"])
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

            "human_collision_rate": safe_mean(human_collision),
            "human_collision_rate_95ci": bootstrap_ci(human_collision),

            "timeout_rate": safe_mean(timeout),
            "timeout_rate_95ci": bootstrap_ci(timeout),

            "fall_rate": safe_mean(fall),
            "fall_rate_95ci": bootstrap_ci(fall),

            "near_miss_rate": safe_mean(near_miss),
            "near_miss_rate_95ci": bootstrap_ci(near_miss),

            "human_near_miss_rate": safe_mean(human_near_miss),
            "human_near_miss_rate_95ci": bootstrap_ci(human_near_miss),

            "mean_min_clearance": safe_mean(min_clearance),
            "mean_min_clearance_95ci": bootstrap_ci(min_clearance),

            "mean_min_human_clearance": (
                safe_mean(min_human_clearance)
                if min_human_clearance else None
            ),
            "mean_min_human_clearance_95ci": (
                bootstrap_ci(min_human_clearance)
                if min_human_clearance else None
            ),

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
