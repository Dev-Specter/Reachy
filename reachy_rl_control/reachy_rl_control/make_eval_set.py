#!/usr/bin/env python3
"""
Generate a fixed, Gazebo-validated evaluation task set for one world.

Candidates are drawn from FreeSpaceTaskSampler with a fixed seed (2-8 m
shortest free-space path by default). Each candidate is run through the
environment's own reset check (goal and start teleport: no body/arm contact,
edge clearance >= 0.10 m, upright). The first N candidates that pass are
kept, in order. Every rejected candidate is recorded with its reason, so the
set is reproducible and auditable.

Why validation is needed: the map is probed at yaw 0, but tasks use random
headings, and at some headings an arm can reach under an overhang (e.g. a
desk top) that the LiDAR cannot see.

Run inside Docker with the world up (sim_up.sh), e.g.:

    PYTHONPATH=/opt/reachy_ws/src/reachy_rl_control:$PYTHONPATH \
    python3 -m reachy_rl_control.make_eval_set --name bookstore \
        --map /opt/reachy_ws/rl_sim/maps/bookstore_free.json \
        --out /opt/reachy_ws/rl_sim/maps/bookstore_eval.json
"""

import argparse
from dataclasses import asdict

import numpy as np
import rclpy

from reachy_rl_control.environment_nav import ReachyNavEnv
from reachy_rl_control.nav_interface import ReachyNavInterface
from reachy_rl_control.nav_tasks import FreeSpaceTaskSampler, save_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-distance", type=float, default=2.0)
    ap.add_argument("--max-distance", type=float, default=8.0)
    ap.add_argument("--max-candidates", type=int, default=200)
    args = ap.parse_args()

    sampler = FreeSpaceTaskSampler(args.map, args.min_distance, args.max_distance)
    rng = np.random.default_rng(args.seed)

    rclpy.init()
    robot = ReachyNavInterface()
    robot.start()
    env = None
    try:
        if not robot.wait_until_ready(timeout=30.0):
            raise RuntimeError("Simulation not ready. Is sim_up.sh READY?")
        env = ReachyNavEnv(robot, sampler, verbose=False)

        kept, rejected, tried = [], [], 0
        while len(kept) < args.n and tried < args.max_candidates:
            task = sampler.sample(rng)
            tried += 1
            try:
                env.reset(options={"task": task})
                kept.append(task)
            except RuntimeError as exc:
                reason = str(exc).split("Last failure: ")[-1]
                rejected.append({"candidate": tried, "task": asdict(task), "reason": reason})
                print(f"candidate {tried}: rejected ({reason})")

        if len(kept) < args.n:
            raise RuntimeError(f"Only {len(kept)} valid tasks after {tried} candidates.")

        save_tasks(args.out, kept, meta={
            "world": args.name, "map": args.map, "n": args.n, "seed": args.seed,
            "min_distance": args.min_distance, "max_distance": args.max_distance,
            "validated_in_gazebo": True, "candidates_tried": tried,
            "rejected": rejected,
        })
        lengths = np.array([t.reference_path_length for t in kept])
        print(f"\n{args.n} validated tasks -> {args.out}")
        print(f"candidates tried: {tried} | rejected: {len(rejected)}")
        print(f"path length min {lengths.min():.2f}  mean {lengths.mean():.2f}  max {lengths.max():.2f} m")

    finally:
        if env is not None:
            env.close()
        robot.shutdown()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
