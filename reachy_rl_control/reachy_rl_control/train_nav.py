#!/usr/bin/env python3

"""
V3 goal-navigation training entry point.

Current implementation:
    PPO only.

Once the PPO end-to-end smoke test passes, SAC/MBPO/UR-MBPO will be added
without changing the validated navigation environment.
"""

import argparse
import csv
import json
import os
import time
from collections import Counter

import rclpy

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from reachy_rl_control.environment_nav import ReachyNavEnv
from reachy_rl_control.nav_interface import ReachyNavInterface
from reachy_rl_control.nav_tasks import make_task_sampler


INFO_KEYWORDS = (
    "termination_reason",
    "success",
    "collision",
    "contact_source",
    "contact_link",
    "contact_other",
    "fell_over",
    "episode_min_lidar_distance",
    "near_miss_occurred",
    "episode_steps_below_preferred",
    "episode_steps_unsafe",
    "episode_steps_near_miss",
    "path_length",
    "straight_line_distance",
    "reference_path_length",
    "sim_time",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a V3 Reachy goal-navigation policy."
    )

    parser.add_argument(
        "--algo",
        choices=["ppo", "sac"],
        required=True,
        help="Algorithm to train.",
    )

    parser.add_argument(
        "--world",
        choices=["hospital", "bookstore"],
        required=True,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)

    return parser.parse_args()


def steps_label(steps):
    if steps >= 1000 and steps % 1000 == 0:
        return f"{steps // 1000}k"
    return str(steps)


def read_monitor_summary(path):
    if not os.path.exists(path):
        return 0, {}

    with open(path, newline="") as f:
        rows = list(
            csv.DictReader(
                line for line in f
                if not line.startswith("#")
            )
        )

    outcomes = Counter(
        row.get("termination_reason", "")
        for row in rows
        if row.get("termination_reason", "")
    )

    return len(rows), dict(outcomes)


def main():
    args = parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")

    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")

    label = steps_label(args.steps)

    run_name = (
        f"{args.world}_{args.algo}_seed{args.seed}_{label}"
    )

    if args.tag:
        run_name += f"_{args.tag}"

    run_dir = os.path.join(
        "/opt/reachy_ws/rl_runs",
        run_name,
    )

    if os.path.exists(run_dir):
        raise FileExistsError(
            f"Run directory already exists:\n{run_dir}\n"
            "Rename or remove it before starting a new run."
        )

    os.makedirs(run_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir)

    monitor_path = os.path.join(
        run_dir,
        "train.monitor.csv",
    )

    config = {
        "algo": args.algo,
        "world": args.world,
        "seed": args.seed,
        "requested_steps": args.steps,
        "max_episode_steps": args.max_steps,

        "environment": "ReachyNavEnv V3",
        "task_sampler": args.world,

        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "gamma": 0.99,
        },

        "sac": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 500,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        },

        "checkpoint_every": args.checkpoint_every,
        "tag": args.tag,
    }

    with open(
        os.path.join(run_dir, "config.json"),
        "w",
    ) as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("Reachy V3 navigation training")
    print("=" * 70)
    print(f"Algorithm       : {args.algo}")
    print(f"World           : {args.world}")
    print(f"Seed            : {args.seed}")
    print(f"Requested steps : {args.steps}")
    print(f"Max ep. steps   : {args.max_steps}")
    print(f"Output          : {run_dir}")
    print("=" * 70)

    rclpy.init()

    robot = None
    env = None
    model = None
    start_wall = time.monotonic()

    try:
        # ------------------------------------------------------------
        # ROS interface
        # ------------------------------------------------------------

        robot = ReachyNavInterface()
        robot.start()

        print("\n[train] Waiting for navigation interface...")

        if not robot.wait_until_ready(timeout=30.0):
            raise RuntimeError(
                "Reachy navigation interface did not become ready within "
                "30 seconds. Check /clock, /odom, /scan and "
                "/set_entity_state."
            )

        print("[train] Navigation interface ready.")

        # ------------------------------------------------------------
        # V3 task sampler + environment
        # ------------------------------------------------------------

        sampler = make_task_sampler(args.world)

        base_env = ReachyNavEnv(
            robot=robot,
            task_sampler=sampler,
            control_duration=0.1,
            max_steps=args.max_steps,
            check_goal_pose=True,
            verbose=True,
        )

        env = Monitor(
            base_env,
            filename=monitor_path,
            info_keywords=INFO_KEYWORDS,
        )

        # ------------------------------------------------------------
        # PPO
        # ------------------------------------------------------------

        if args.algo == "ppo":
            print("\n[train] Creating PPO model...")

            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                gamma=0.99,
                verbose=1,
                seed=args.seed,
            )

        elif args.algo == "sac":
            print("\n[train] Creating SAC model...")

            model = SAC(
                policy="MlpPolicy",
                env=env,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=500,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                verbose=1,
                seed=args.seed,
            )

        else:
            raise ValueError(
                f"Algorithm not implemented yet: {args.algo}"
            )

        callback = None

        if args.checkpoint_every > 0:
            callback = CheckpointCallback(
                save_freq=args.checkpoint_every,
                save_path=checkpoint_dir,
                name_prefix="model",
                save_replay_buffer=False,
                save_vecnormalize=False,
                verbose=1,
            )

        # ------------------------------------------------------------
        # Train
        # ------------------------------------------------------------

        print(
            f"\n[train] Starting {args.algo.upper()} for "
            f"{args.steps} requested timesteps..."
        )

        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            progress_bar=False,
        )

        # PPO may finish at the end of a rollout, so num_timesteps can be
        # slightly larger than the requested number.
        actual_steps = int(model.num_timesteps)

        final_path = os.path.join(
            run_dir,
            "final_model",
        )

        model.save(final_path)

        wall_seconds = time.monotonic() - start_wall

        episodes, outcomes = read_monitor_summary(
            monitor_path
        )

        summary = {
            "status": "completed",
            "algo": args.algo,
            "world": args.world,
            "seed": args.seed,
            "requested_timesteps": args.steps,
            "timesteps": actual_steps,
            "episodes": episodes,
            "outcomes": outcomes,
            "wall_seconds": wall_seconds,
            "final_model": final_path + ".zip",
        }

        with open(
            os.path.join(run_dir, "summary.json"),
            "w",
        ) as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("[train] TRAINING COMPLETED")
        print(f"[train] requested timesteps : {args.steps}")
        print(f"[train] actual timesteps    : {actual_steps}")
        print(f"[train] episodes            : {episodes}")
        print(f"[train] outcomes            : {outcomes}")
        print(f"[train] model               : {final_path}.zip")
        print("=" * 70)

    except Exception as exc:
        wall_seconds = time.monotonic() - start_wall

        failure = {
            "status": "failed",
            "algo": args.algo,
            "world": args.world,
            "seed": args.seed,
            "requested_timesteps": args.steps,
            "wall_seconds": wall_seconds,
            "error": repr(exc),
        }

        try:
            with open(
                os.path.join(run_dir, "summary.json"),
                "w",
            ) as f:
                json.dump(failure, f, indent=2)
        except Exception:
            pass

        print(f"\n[train] ERROR: {exc}")
        raise

    finally:
        print("\n[train] Stopping Reachy safely...")

        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(
                    "[train] Warning while closing environment:",
                    exc,
                )

        if robot is not None:
            try:
                robot.shutdown()
            except Exception as exc:
                print(
                    "[train] Warning while shutting down ROS interface:",
                    exc,
                )

            try:
                robot.destroy_node()
            except Exception:
                pass

        if rclpy.ok():
            rclpy.shutdown()

        print("[train] Training process closed.")


if __name__ == "__main__":
    main()
