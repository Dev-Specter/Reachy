#!/usr/bin/env python3
"""
Map the free space of a world for Reachy by probing it in Gazebo.

Starting from a known-free seed pose, a breadth-first flood fill visits a
regular grid. At each grid point Reachy is teleported there (yaw 0) and the
SAME validity test as an environment reset is applied:

    - no body or arm/head contact
    - edge clearance >= min_clearance (default 0.10 m = SAFE_DISTANCE)
    - upright, and not pushed away by physics (a push means overlap)

Only free points expand to their 8 neighbours, so every free point in the
output is connected to the seed: start/goal pairs sampled from it are
reachable at grid resolution. The default grid (0.2 m) and connectivity
clearance (0.05 m, the near-miss boundary) are chosen so that narrow and
angled aisles (~0.17 m side clearance in the Bookstore) are not missed;
task start/goal cells are filtered to >= 0.10 m by the task sampler. Because the real robot model is tested, arm
and tabletop overhangs are handled correctly.

Run inside Docker with the world up (sim_up.sh), e.g.:

    PYTHONPATH=/opt/reachy_ws/src/reachy_rl_control:$PYTHONPATH \
    python3 -m reachy_rl_control.probe_free_space --name hospital \
        --seed-x 0.37 --seed-y 1.07 --out /opt/reachy_ws/rl_sim/maps/hospital_free.json
"""

import argparse
import collections
import json
import math
import os
import time

import rclpy

from reachy_rl_control.nav_interface import ReachyNavInterface, scan_to_edge_clearance


def probe_point(robot, x, y, settle, window, max_tilt, max_push):
    robot.stop()
    robot.teleport(x, y, 0.0)
    robot.wait_sim(settle)
    robot.clear_contacts()
    t = robot.sim_time()
    robot.wait_sim(window)
    robot.wait_fresh(state_after=t, scan_after=t)

    state = robot.get_state()
    clearance = float(scan_to_edge_clearance(robot.get_scan(), 24, 3.0).min())
    contact = robot.get_contact()
    pushed = math.hypot(state.x - x, state.y - y)
    upright = abs(state.roll) < max_tilt and abs(state.pitch) < max_tilt

    reason = "ok"
    if contact is not None:
        reason = f"contact:{contact['link']}->{contact['other'].split('::')[0]}"
    elif not upright:
        reason = "tilted"
    elif pushed > max_push:
        reason = f"pushed {pushed:.2f} m"
    return reason, clearance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="world name, e.g. hospital / bookstore")
    ap.add_argument("--seed-x", type=float, required=True)
    ap.add_argument("--seed-y", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=float, default=0.2)
    ap.add_argument("--min-clearance", type=float, default=0.05,
                    help="clearance for the connectivity map (near-miss boundary); "
                         "task start/goal cells are filtered to >= 0.10 m later")
    ap.add_argument("--max-radius", type=float, default=20.0, help="stop expanding beyond this distance from the seed")
    ap.add_argument("--max-probes", type=int, default=40000)
    ap.add_argument("--settle", type=float, default=0.15, help="sim seconds after each teleport")
    ap.add_argument("--window", type=float, default=0.10, help="sim seconds watched for contacts")
    args = ap.parse_args()

    rclpy.init()
    robot = ReachyNavInterface()
    robot.start()

    g = args.grid
    cells = {}
    try:
        if not robot.wait_until_ready(timeout=30.0):
            raise RuntimeError("Simulation not ready. Is sim_up.sh READY?")

        queue = collections.deque([(0, 0)])
        queued = {(0, 0)}
        n_free = 0
        t_start = time.monotonic()

        while queue and len(cells) < args.max_probes:
            i, j = queue.popleft()
            x, y = args.seed_x + i * g, args.seed_y + j * g

            if math.hypot(i * g, j * g) > args.max_radius:
                cells[(i, j)] = {"i": i, "j": j, "x": x, "y": y, "free": False,
                                 "clearance": None, "reason": "outside max radius"}
                continue

            reason, clearance = probe_point(robot, x, y, args.settle, args.window,
                                            max_tilt=math.radians(5.0), max_push=0.05)
            free = reason == "ok" and clearance >= args.min_clearance
            if reason == "ok" and not free:
                reason = f"clearance {clearance:.2f}"
            cells[(i, j)] = {"i": i, "j": j, "x": round(x, 3), "y": round(y, 3), "free": free,
                             "clearance": round(clearance, 3), "reason": reason}

            if (i, j) == (0, 0) and not free:
                raise RuntimeError(f"Seed ({x:.2f}, {y:.2f}) is not free: {reason}")

            if free:
                n_free += 1
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nb = (i + di, j + dj)
                    if nb not in queued:
                        queued.add(nb)
                        queue.append(nb)

            if len(cells) % 100 == 0:
                rate = len(cells) / max(time.monotonic() - t_start, 1e-6)
                print(f"probed {len(cells):5d} | free {n_free:5d} | frontier {len(queue):4d} | "
                      f"{rate:.1f} probes/s", flush=True)

        elapsed = time.monotonic() - t_start
        hit_cap = bool(queue)

        free_cells = [c for c in cells.values() if c["free"]]
        out = {
            "world": args.name,
            "grid": g,
            "seed": [args.seed_x, args.seed_y],
            "min_clearance": args.min_clearance,
            "max_radius": args.max_radius,
            "settle": args.settle,
            "window": args.window,
            "probes": len(cells),
            "free_cells": len(free_cells),
            "free_area_m2": round(len(free_cells) * g * g, 2),
            "stopped_by_probe_cap": hit_cap,
            "radius_limited": any(c["reason"] == "outside max radius" for c in cells.values()),
            "elapsed_wall_s": round(elapsed, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cells": sorted(cells.values(), key=lambda c: (c["i"], c["j"])),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)

        print(f"\nDONE: {len(cells)} probes, {len(free_cells)} free cells "
              f"(~{out['free_area_m2']} m^2) in {elapsed:.0f} s -> {args.out}")
        if hit_cap:
            print("WARNING: stopped at --max-probes; frontier not exhausted.")
        if out["radius_limited"]:
            print(f"NOTE: free space reached the {args.max_radius} m radius limit "
                  "(the world may be open to the outside there).")

        # Text map (no dependencies): '.' free (>= 0.10 m), ':' free (0.05-0.10 m),
        # '#' blocked, ' ' not probed, 'S' seed. North (+y) is up.
        ii = [c["i"] for c in cells.values()]; jj = [c["j"] for c in cells.values()]
        lines = [f"{args.name}: grid {g} m, x from {args.seed_x + min(ii) * g:.2f} (left), "
                 f"y from {args.seed_y + max(jj) * g:.2f} (top). "
                 "'.' free >=0.10  ':' free 0.05-0.10  '#' blocked  'S' seed"]
        for j in range(max(jj), min(jj) - 1, -1):
            row = []
            for i in range(min(ii), max(ii) + 1):
                c = cells.get((i, j))
                if (i, j) == (0, 0):
                    row.append("S")
                elif c is None:
                    row.append(" ")
                elif not c["free"]:
                    row.append("#")
                else:
                    row.append("." if c["clearance"] >= 0.10 else ":")
            lines.append("".join(row).rstrip())
        txt = os.path.splitext(args.out)[0] + ".txt"
        with open(txt, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"text map -> {txt}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 9))
            blocked = [c for c in cells.values() if not c["free"]]
            ax.scatter([c["x"] for c in blocked], [c["y"] for c in blocked], s=8, c="tab:red", label="blocked")
            ax.scatter([c["x"] for c in free_cells], [c["y"] for c in free_cells], s=8,
                       c=[c["clearance"] for c in free_cells], cmap="viridis", vmin=0.1, vmax=1.5,
                       label="free (colour = clearance)")
            ax.plot(args.seed_x, args.seed_y, "k*", ms=14, label="seed")
            ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
            ax.set_title(f"{args.name}: {len(free_cells)} free cells @ {g} m")
            png = os.path.splitext(args.out)[0] + ".png"
            fig.savefig(png, dpi=120, bbox_inches="tight")
            print(f"plot -> {png}")
        except Exception as exc:
            print(f"(no plot: {exc})")

    finally:
        try:
            robot.stop()
            robot.teleport(args.seed_x, args.seed_y, 0.0)
        except Exception:
            pass
        robot.shutdown()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
