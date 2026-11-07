#!/usr/bin/env python3
"""
Start/goal task sampling for the navigation environment.

A task is a start pose (x, y, yaw) and a goal point (x, y) in Gazebo world
coordinates. Samplers only PROPOSE tasks; the environment validates every
start and goal in Gazebo (teleport, then require no contact and enough LiDAR
clearance) and resamples when a proposal fails.

Regions are axis-aligned rectangles (xmin, xmax, ymin, ymax).

The *_smoke regions below are deliberately narrow: they cover only floor that
Reachy has already driven over during simulation validation. Proper
world-wide regions will be defined from top-down views of each world.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Region = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


@dataclass(frozen=True)
class NavTask:
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float

    @property
    def straight_line_distance(self) -> float:
        return math.hypot(self.goal_x - self.start_x, self.goal_y - self.start_y)


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


# ---------------------------------------------------------------------------
# SMOKE-TEST REGIONS (floor already driven over during validation)
# ---------------------------------------------------------------------------
#
# Hospital: Reachy drove along y = 1.07 from x = 0.37 to x = 6.17 (wall contact),
#           right-side clearance ~0.77 m throughout.
# Bookstore: Reachy drove along y = 0.0 from x = 0.0 to x = 4.2 (chair contact),
#            left-side clearance only 0.17-0.20 m (a narrow aisle).

def make_task_sampler(name: str):
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
    raise ValueError(f"Unknown task set: {name}")
