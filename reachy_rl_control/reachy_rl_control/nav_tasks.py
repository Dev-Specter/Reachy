#!/usr/bin/env python3
"""
Start/goal task sampling for the navigation environment.

A task is a start pose (x, y, yaw) and a goal point (x, y) in Gazebo world
coordinates. Samplers only PROPOSE tasks; the environment validates every
start and goal in Gazebo (teleport, then require no contact and enough LiDAR
clearance) and resamples when a proposal fails.

Samplers
    RegionTaskSampler     uniform in rectangles (used for the smoke tests)
    FreeSpaceTaskSampler  uniform over a probed free-space map
                          (probe_free_space.py); the start-goal distance band
                          is applied to the SHORTEST PATH through free space,
                          which is also stored as reference_path_length
    FixedTaskSampler      cycles a fixed list (evaluation sets)

Evaluation sets (fixed, seeded, identical for every algorithm):

    python3 -m reachy_rl_control.nav_tasks --map .../hospital_free.json \
        --n 50 --seed 2026 --out .../hospital_eval.json
"""

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Region = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax

MAPS_DIR = "/opt/reachy_ws/rl_sim/maps"


@dataclass(frozen=True)
class NavTask:
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    reference_path_length: float = float("nan")   # shortest free-space path (m), if known

    @property
    def straight_line_distance(self) -> float:
        return math.hypot(self.goal_x - self.start_x, self.goal_y - self.start_y)


# ---------------------------------------------------------------------------
# RECTANGLE SAMPLER (smoke tests)
# ---------------------------------------------------------------------------

def _area(region: Region) -> float:
    return max(1e-6, (region[1] - region[0]) * (region[3] - region[2]))


def _sample_point(rng: np.random.Generator, regions: Sequence[Region]):
    weights = np.array([_area(r) for r in regions], dtype=np.float64)
    region = regions[int(rng.choice(len(regions), p=weights / weights.sum()))]
    return (
        float(rng.uniform(region[0], region[1])),
        float(rng.uniform(region[2], region[3])),
    )


class RegionTaskSampler:
    """Uniform start/goal sampling inside rectangles, with a distance band."""

    def __init__(
        self,
        start_regions: List[Region],
        goal_regions: Optional[List[Region]] = None,
        min_distance: float = 1.5,
        max_distance: float = 6.0,
        random_start_yaw: bool = True,
        max_tries: int = 1000,
    ):
        self.start_regions = list(start_regions)
        self.goal_regions = list(goal_regions or start_regions)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.random_start_yaw = bool(random_start_yaw)
        self.max_tries = int(max_tries)

    def sample(self, rng: np.random.Generator) -> NavTask:
        for _ in range(self.max_tries):
            sx, sy = _sample_point(rng, self.start_regions)
            gx, gy = _sample_point(rng, self.goal_regions)
            d = math.hypot(gx - sx, gy - sy)
            if self.min_distance <= d <= self.max_distance:
                yaw = (
                    float(rng.uniform(-math.pi, math.pi))
                    if self.random_start_yaw
                    else math.atan2(gy - sy, gx - sx)
                )
                return NavTask(sx, sy, yaw, gx, gy)
        raise RuntimeError(
            "Could not sample a start/goal pair within the distance band; "
            "check the regions."
        )


# ---------------------------------------------------------------------------
# FREE-SPACE SAMPLER (probed map)
# ---------------------------------------------------------------------------

class FreeSpaceTaskSampler:
    """
    Sample start and goal uniformly from free cells of a probed map.

    Graph: every free cell of the map (the map's own connectivity clearance,
    0.05 m by default), 8-connected. Start and goal cells are restricted to
    cells with clearance >= endpoint_clearance (0.12 m: the 0.10 m reset
    threshold plus a 0.02 m margin, because the map was probed at yaw 0 and
    the measured edge clearance varies by ~0.01 m with heading), so tasks
    never begin or end in the unsafe band and pass the reset check. The distance band applies to
    the shortest path through the graph, stored as reference_path_length.
    """

    def __init__(
        self,
        map_path: str,
        min_distance: float = 2.0,
        max_distance: float = 8.0,
        endpoint_clearance: float = 0.12,
        max_tries: int = 2000,
    ):
        with open(map_path) as f:
            data = json.load(f)
        self.map_path = map_path
        self.grid = float(data["grid"])
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.max_tries = int(max_tries)

        self.cells = [c for c in data["cells"] if c["free"]]
        self.endpoints = [
            k for k, c in enumerate(self.cells)
            if c["clearance"] is not None and c["clearance"] >= endpoint_clearance
        ]
        if len(self.endpoints) < 2:
            raise ValueError(f"{map_path}: fewer than 2 usable start/goal cells.")
        self.index: Dict[Tuple[int, int], int] = {(c["i"], c["j"]): k for k, c in enumerate(self.cells)}

        self.neighbours: List[List[Tuple[int, float]]] = []
        for c in self.cells:
            i, j = c["i"], c["j"]
            nbs = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                k = self.index.get((i + di, j + dj))
                if k is not None:
                    nbs.append((k, self.grid))
            for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                k = self.index.get((i + di, j + dj))
                if k is not None:
                    nbs.append((k, self.grid * math.sqrt(2.0)))
            self.neighbours.append(nbs)

    def path_length(self, a: int, b: int) -> float:
        """Shortest free-space path length between cells a and b (Dijkstra)."""
        dist = {a: 0.0}
        heap = [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                return d
            if d > dist.get(u, math.inf) or d > self.max_distance:
                continue
            for v, w in self.neighbours[u]:
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return math.inf

    def sample(self, rng: np.random.Generator) -> NavTask:
        n = len(self.endpoints)
        for _ in range(self.max_tries):
            a = self.endpoints[int(rng.integers(n))]
            b = self.endpoints[int(rng.integers(n))]
            if a == b:
                continue
            ca, cb = self.cells[a], self.cells[b]
            if math.hypot(cb["x"] - ca["x"], cb["y"] - ca["y"]) > self.max_distance:
                continue
            d = self.path_length(a, b)
            if self.min_distance <= d <= self.max_distance:
                yaw = float(rng.uniform(-math.pi, math.pi))
                return NavTask(ca["x"], ca["y"], yaw, cb["x"], cb["y"], round(d, 3))
        raise RuntimeError(f"{self.map_path}: no task found in the {self.min_distance}-"
                           f"{self.max_distance} m path-length band.")


# ---------------------------------------------------------------------------
# FIXED SAMPLER (evaluation sets)
# ---------------------------------------------------------------------------

class FixedTaskSampler:
    """Cycle through a fixed list of tasks (for evaluation sets)."""

    def __init__(self, tasks: List[NavTask]):
        if not tasks:
            raise ValueError("FixedTaskSampler needs at least one task.")
        self.tasks = list(tasks)
        self.index = 0

    def sample(self, rng: np.random.Generator) -> NavTask:
        task = self.tasks[self.index % len(self.tasks)]
        self.index += 1
        return task


def save_tasks(path: str, tasks: List[NavTask], meta: Optional[dict] = None):
    with open(path, "w") as f:
        json.dump({"meta": meta or {}, "tasks": [asdict(t) for t in tasks]}, f, indent=1)


def load_tasks(path: str) -> List[NavTask]:
    with open(path) as f:
        return [NavTask(**t) for t in json.load(f)["tasks"]]


# ---------------------------------------------------------------------------
# NAMED TASK SETS
# ---------------------------------------------------------------------------
#
# *_smoke   : floor already driven over during validation (rectangles)
# <world>   : training tasks from the probed map, 2-8 m path length
# <world>_eval : fixed evaluation set (generate with this module's CLI)

def make_task_sampler(name: str, maps_dir: str = MAPS_DIR):
    if name == "hospital_smoke":
        return RegionTaskSampler(
            start_regions=[(0.40, 1.50, 0.95, 1.20)],
            goal_regions=[(3.00, 5.50, 0.95, 1.20)],
            min_distance=2.0,
            max_distance=5.5,
        )
    if name == "bookstore_smoke":
        return RegionTaskSampler(
            start_regions=[(0.00, 1.00, -0.03, 0.03)],
            goal_regions=[(2.50, 3.30, -0.03, 0.03)],
            min_distance=1.5,
            max_distance=3.3,
        )
    if name in ("hospital", "bookstore"):
        return FreeSpaceTaskSampler(f"{maps_dir}/{name}_free.json")
    if name in ("hospital_eval", "bookstore_eval"):
        return FixedTaskSampler(load_tasks(f"{maps_dir}/{name}.json"))
    raise ValueError(f"Unknown task set: {name}")


def main():
    ap = argparse.ArgumentParser(description="Generate a fixed, seeded evaluation task set.")
    ap.add_argument("--map", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-distance", type=float, default=2.0)
    ap.add_argument("--max-distance", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sampler = FreeSpaceTaskSampler(args.map, args.min_distance, args.max_distance)
    rng = np.random.default_rng(args.seed)
    tasks = [sampler.sample(rng) for _ in range(args.n)]
    save_tasks(args.out, tasks, meta={
        "map": args.map, "n": args.n, "seed": args.seed,
        "min_distance": args.min_distance, "max_distance": args.max_distance,
    })
    lengths = np.array([t.reference_path_length for t in tasks])
    print(f"{args.n} tasks -> {args.out} | path length min {lengths.min():.2f} "
          f"mean {lengths.mean():.2f} max {lengths.max():.2f} m")


if __name__ == "__main__":
    main()
