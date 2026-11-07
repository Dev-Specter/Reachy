#!/usr/bin/env python3
"""
Smoke test for the V3 navigation environment (no learning).

Policies:
    scripted : turn toward the goal, drive when roughly aligned
               (checks reward/goal logic: should mostly reach the goal)
    random   : uniform random actions
               (checks collisions, timeouts and repeated resets)

Run inside Docker with the simulation already up (sim_up.sh), e.g.:

    PYTHONPATH=/opt/reachy_ws/src/reachy_rl_control:$PYTHONPATH \
    python3 -m reachy_rl_control.smoke_nav_env --tasks hospital_smoke --policy scripted --episodes 5
"""

import argparse
import math
import time

import numpy as np
import rclpy

from reachy_rl_control.environment_nav import OBS_DIM, ReachyNavEnv
from reachy_rl_control.nav_interface import ReachyNavInterface
from reachy_rl_control.nav_tasks import make_task_sampler


def scripted_action(observation: np.ndarray) -> np.ndarray:
    bearing = math.atan2(float(observation[3]), float(observation[4]))
    w = float(np.clip(2.5 * bearing, -1.0, 1.0))
    v = 1.0 if abs(bearing) < 0.35 else (0.3 if abs(bearing) < 0.8 else 0.0)
    return np.array([v, w], dtype=np.float32)


def check_observation(env, observation, where):
    if observation.shape != (OBS_DIM,):
        raise AssertionError(f"{where}: shape {observation.shape} != ({OBS_DIM},)")
    if not np.all(np.isfinite(observation)):
        raise AssertionError(f"{where}: non-finite observation {observation}")
    if not env.observation_space.contains(observation):
        raise AssertionError(f"{where}: observation outside observation_space")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="hospital_smoke")
    parser.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rclpy.init()
    robot = ReachyNavInterface()
    robot.start()
    env = None

    try:
        print("Waiting for /clock, /odom, /scan and /set_entity_state ...")
        if not robot.wait_until_ready(timeout=30.0):
            raise RuntimeError("Simulation not ready. Is sim_up.sh READY?")

        env = ReachyNavEnv(
            robot=robot,
            task_sampler=make_task_sampler(args.tasks),
            control_duration=0.1,
            max_steps=args.max_steps,
        )
        print(f"Observation space: {env.observation_space.shape} | Action space: {env.action_space.shape}")

        rng = np.random.default_rng(args.seed)
        results = []
        total_steps = 0
        wall_start = time.monotonic()

        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed if episode == 0 else None)
            check_observation(env, observation, "reset")

            episode_return = 0.0
            episode_wall = time.monotonic()

            while True:
                if args.policy == "scripted":
                    action = scripted_action(observation)
                else:
                    action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

                observation, reward, terminated, truncated, info = env.step(action)
                check_observation(env, observation, f"step {info['step']}")
                if not math.isfinite(reward):
                    raise AssertionError(f"non-finite reward {reward}")

                episode_return += reward
                total_steps += 1

                if terminated or truncated:
                    break

            results.append({
                "reason": info["termination_reason"],
                "steps": info["step"],
                "return": episode_return,
                "sim_s": info["sim_time"],
                "wall_s": time.monotonic() - episode_wall,
                "min_clear": info["episode_min_lidar_distance"],
                "path": info["path_length"],
                "straight": info["straight_line_distance"],
                "contact": info["contact_link"],
            })

        wall_total = time.monotonic() - wall_start

        print("\n================ SMOKE TEST SUMMARY ================")
        print(f"tasks={args.tasks} policy={args.policy} episodes={args.episodes}")
        print(f"{'#':>2} {'reason':>9} {'steps':>5} {'return':>8} {'sim_s':>6} {'wall_s':>6} "
              f"{'min_clr':>7} {'path':>5} {'line':>5}  contact")
        for i, r in enumerate(results, 1):
            print(f"{i:>2} {r['reason']:>9} {r['steps']:>5} {r['return']:>8.2f} {r['sim_s']:>6.1f} "
                  f"{r['wall_s']:>6.1f} {r['min_clear']:>7.3f} {r['path']:>5.2f} {r['straight']:>5.2f}  "
                  f"{r['contact'] or '-'}")
        reasons = [r["reason"] for r in results]
        print("\noutcomes:", {k: reasons.count(k) for k in sorted(set(reasons))})
        print(f"total env steps: {total_steps} in {wall_total:.1f} s wall "
              f"-> {total_steps / max(wall_total, 1e-6):.1f} steps/s (including resets)")
        print("All observation/reward checks passed.")

    finally:
        if env is not None:
            env.close()
        robot.shutdown()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
