#!/usr/bin/env python3
"""
V3 goal-reaching navigation environment for the Hospital and Bookstore worlds.

Task: drive Reachy's mobile base from a start pose to a goal point without
any collision, while keeping good LiDAR clearance. No path following.

Compatible with the existing training stack:
    - Gymnasium API, Env(robot=..., control_duration=0.1, max_steps=...)
    - same reward terms/scale as V1 (progress x10, -0.01/step, smoothness 0.05,
      collision -40, fall -30, timeout -5; proximity penalty bounded at
      0.05/step below 0.10 m edge clearance)
    - same info keys (termination_reason, min_lidar_distance,
      episode_min_lidar_distance, near_miss_occurred, success, ...)
    - env.lidar.get_min_distance() / get_normalized_distances() and
      env.world_waypoints for the existing wrappers
    - LiDAR block LAST, normalised by 3.0 m  -> lidar_start_index = 7

Differences from V1/V2 (all validated in Gazebo, see sim/README.md):
    - steps are timed in SIMULATION time (/clock), not wall time
    - collisions come from Gazebo contact sensors (body + arms/head)
    - LiDAR = clearance from the BASE EDGE (0.184 m offset corrected),
      24 x 15 deg sectors, only the mast sector (174-186 deg) masked
    - resets teleport via /set_entity_state; both the goal and the start
      pose are checked for contact and clearance before an episode starts
    - action is (v, w); no sideways motion

Observation (31):
     0  vx            measured forward speed (m/s)
     1  wz            measured yaw rate (rad/s)
     2  goal distance / 10 m, capped at 1
     3  sin(goal bearing)   bearing in the robot frame
     4  cos(goal bearing)
     5  previous v command / 0.30
     6  previous w command / 0.60
  7-30  24 LiDAR sectors: edge clearance / 3.0 m in [0, 1];
        sector k is centred on 15k deg (0 = ahead, counter-clockwise)

Action (2), each in [-1, 1]:
     0  v: +1 -> 0.30 m/s forward, -1 -> 0.05 m/s reverse (asymmetric)
     1  w: +/-1 -> +/-0.60 rad/s
"""

import math
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reachy_rl_control.nav_interface import (
    ReachyNavInterface,
    scan_to_edge_clearance,
)
from reachy_rl_control.nav_tasks import NavTask


# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------

# Action limits
V_MAX = 0.30
V_REVERSE = 0.05
W_MAX = 0.60

# Observation
NUM_LIDAR_SECTORS = 24
LIDAR_MAX_DISTANCE = 3.0
GOAL_DISTANCE_SCALE = 10.0
LIDAR_START_INDEX = 7
OBS_DIM = LIDAR_START_INDEX + NUM_LIDAR_SECTORS  # 31

# Task
GOAL_TOLERANCE = 0.30

# Clearance bands (EDGE clearance, metres) - identical in every world:
#   > 0.20      comfortably clear
#   0.10-0.20   acceptable (UR-MBPO may prefer more room)
#   0.05-0.10   unsafe proximity  -> bounded base penalty
#   < 0.05      near miss (metric)
#   contact     collision (terminal)
SAFE_DISTANCE = 0.10
NEAR_MISS_DISTANCE = 0.05
PREFERRED_CLEARANCE = 0.20

# Reward (same terms as V1; proximity penalty is now bounded and modest)
PROGRESS_SCALE = 10.0
STEP_PENALTY = 0.01
SMOOTHNESS_SCALE = 0.05
GOAL_REWARD = 25.0          # V1 final step gave +5 (waypoint) + 20 (goal)
COLLISION_PENALTY = 40.0
FALL_PENALTY = 30.0
TIMEOUT_PENALTY = 5.0
PROXIMITY_PENALTY_MAX = 0.05   # per step, reached only at zero clearance
RISK_PENALTY_MAX = 0.06        # per step, for UR-MBPO / risk-shaping wrappers

DEFAULT_MAX_STEPS = 500


def check_no_crash_incentive(max_steps: int, extra_per_step: float = 0.0):
    """
    Guarantee that finishing always beats crashing, for every route.

    Crashing at step t instead of finishing forgoes the remaining progress
    reward (>= 0) and the goal reward, adds the collision penalty, and saves
    only the per-step costs of the remaining steps. Hence

        R_success - R_crash(t) >= GOAL + COLLISION - c_max * (T_max - t) > 0

    whenever c_max * T_max < GOAL + COLLISION, with
    c_max = STEP_PENALTY + PROXIMITY_PENALTY_MAX + extra_per_step.

    The smoothness penalty is excluded: it is chosen by the policy and is zero
    for a constant command.
    """
    worst = (STEP_PENALTY + PROXIMITY_PENALTY_MAX + float(extra_per_step)) * int(max_steps)
    margin = GOAL_REWARD + COLLISION_PENALTY
    if worst >= margin:
        raise ValueError(
            f"Crash incentive: worst-case per-step costs over {max_steps} steps = {worst:.1f} "
            f">= goal + collision = {margin:.1f}. Reduce penalties or max_steps."
        )
    return worst, margin


# Fall detection
FALL_ANGLE_RAD = math.radians(30.0)

# Reset validation
RESET_SETTLE_TIME = 0.30          # sim seconds after a teleport
CONTACT_CHECK_TIME = 0.10         # sim seconds watched for contacts
START_MIN_CLEARANCE = 0.10
GOAL_MIN_CLEARANCE = 0.10
RESET_MAX_TILT_RAD = math.radians(5.0)
MAX_RESET_ATTEMPTS = 10


class LidarView:
    """Minimal LiDAR accessor expected by the existing wrappers."""

    def __init__(self):
        self.sector_distances = np.full(
            NUM_LIDAR_SECTORS, LIDAR_MAX_DISTANCE, dtype=np.float32
        )

    def update(self, sectors: np.ndarray):
        self.sector_distances = np.asarray(sectors, dtype=np.float32).copy()

    def get_distances(self) -> np.ndarray:
        return self.sector_distances.copy()

    def get_normalized_distances(self) -> np.ndarray:
        return np.clip(
            self.sector_distances / LIDAR_MAX_DISTANCE, 0.0, 1.0
        ).astype(np.float32)

    def get_min_distance(self) -> float:
        return float(np.min(self.sector_distances))


class ReachyNavEnv(gym.Env):

    metadata = {"render_modes": []}

    safe_distance = SAFE_DISTANCE
    near_miss_distance = NEAR_MISS_DISTANCE
    preferred_clearance = PREFERRED_CLEARANCE
    lidar_start_index = LIDAR_START_INDEX
    lidar_max_distance = LIDAR_MAX_DISTANCE

    def __init__(
        self,
        robot: ReachyNavInterface,
        task_sampler,
        control_duration: float = 0.1,
        max_steps: int = DEFAULT_MAX_STEPS,
        check_goal_pose: bool = True,
        verbose: bool = True,
    ):
        super().__init__()

        self.robot = robot
        self.task_sampler = task_sampler
        self.control_duration = float(control_duration)
        self.max_steps = int(max_steps)
        self.check_goal_pose = bool(check_goal_pose)
        self.verbose = bool(verbose)

        worst, margin = check_no_crash_incentive(self.max_steps)
        if self.verbose:
            print(f"Reward check: worst-case per-step costs {worst:.1f} < goal+collision {margin:.1f} "
                  f"over {self.max_steps} steps -> finishing always beats crashing.")

        self.lidar = LidarView()

        low = np.concatenate([
            np.full(LIDAR_START_INDEX, -np.inf, dtype=np.float32),
            np.zeros(NUM_LIDAR_SECTORS, dtype=np.float32),
        ])
        high = np.concatenate([
            np.full(LIDAR_START_INDEX, np.inf, dtype=np.float32),
            np.ones(NUM_LIDAR_SECTORS, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.episode_number = 0
        self.step_number = 0
        self.task: Optional[NavTask] = None

        self.previous_distance = 0.0
        self.previous_command = np.zeros(2, dtype=np.float32)   # physical (v, w)
        self.episode_min_lidar_distance = LIDAR_MAX_DISTANCE
        self.near_miss_occurred = False
        self.steps_below_preferred = 0
        self.steps_unsafe = 0
        self.steps_near_miss = 0
        self.path_length = 0.0
        self.episode_start_sim_time = 0.0
        self._last_xy = (0.0, 0.0)
        self._closed = False

    # -----------------------------------------------------------------------
    # COMPATIBILITY WITH EXISTING WRAPPERS
    # -----------------------------------------------------------------------

    @property
    def world_waypoints(self):
        if self.task is None:
            return []
        return [(self.task.start_x, self.task.start_y),
                (self.task.goal_x, self.task.goal_y)]

    @property
    def current_waypoint_index(self) -> int:
        return 1

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------

    def _read_sectors(self) -> np.ndarray:
        sectors = scan_to_edge_clearance(
            self.robot.get_scan(), NUM_LIDAR_SECTORS, LIDAR_MAX_DISTANCE
        )
        self.lidar.update(sectors)
        return sectors

    def _goal_geometry(self, state):
        dx = self.task.goal_x - state.x
        dy = self.task.goal_y - state.y
        distance = math.hypot(dx, dy)
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        bearing = math.atan2(local_y, local_x)
        return distance, bearing

    def _get_observation(self, state, sectors) -> np.ndarray:
        distance, bearing = self._goal_geometry(state)
        head = np.array([
            state.vx,
            state.wz,
            min(distance / GOAL_DISTANCE_SCALE, 1.0),
            math.sin(bearing),
            math.cos(bearing),
            self.previous_command[0] / V_MAX,
            self.previous_command[1] / W_MAX,
        ], dtype=np.float32)
        lidar = np.clip(sectors / LIDAR_MAX_DISTANCE, 0.0, 1.0).astype(np.float32)
        observation = np.concatenate([head, lidar]).astype(np.float32)
        if observation.shape != (OBS_DIM,):
            raise RuntimeError(f"Observation must be ({OBS_DIM},), got {observation.shape}.")
        return observation

    def _settle_and_check(self, min_clearance: float):
        """After a teleport: settle, then require no contact, clearance, upright."""
        r = self.robot
        r.stop()
        r.wait_sim(RESET_SETTLE_TIME)
        r.clear_contacts()
        t = r.sim_time()
        r.wait_sim(CONTACT_CHECK_TIME)
        r.wait_fresh(state_after=t, scan_after=t)

        state = r.get_state()
        sectors = self._read_sectors()
        contact = r.get_contact()

        if contact is not None:
            return False, state, sectors, f"contact with {contact['other']}"
        clearance = float(np.min(sectors))
        if clearance < min_clearance:
            return False, state, sectors, f"clearance {clearance:.2f} m < {min_clearance:.2f} m"
        if abs(state.roll) > RESET_MAX_TILT_RAD or abs(state.pitch) > RESET_MAX_TILT_RAD:
            return False, state, sectors, "not upright"
        return True, state, sectors, "ok"

    @staticmethod
    def _proximity_penalty(clearance: float) -> float:
        if clearance >= SAFE_DISTANCE:
            return 0.0
        closeness = float(np.clip((SAFE_DISTANCE - clearance) / SAFE_DISTANCE, 0.0, 1.0))
        return PROXIMITY_PENALTY_MAX * closeness

    # -----------------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        forced_task = options.get("task")

        self.robot.stop()

        accepted = False
        failure = "no attempt"
        state = sectors = None
        task = None

        for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
            task = forced_task if forced_task is not None else self.task_sampler.sample(self.np_random)

            if self.check_goal_pose:
                self.robot.teleport(task.goal_x, task.goal_y, task.start_yaw)
                ok, _, _, why = self._settle_and_check(GOAL_MIN_CLEARANCE)
                if not ok:
                    failure = f"goal ({task.goal_x:.2f}, {task.goal_y:.2f}): {why}"
                    if self.verbose:
                        print(f"[reset {attempt}/{MAX_RESET_ATTEMPTS}] rejected {failure}")
                    if forced_task is not None:
                        break
                    continue

            self.robot.teleport(task.start_x, task.start_y, task.start_yaw)
            ok, state, sectors, why = self._settle_and_check(START_MIN_CLEARANCE)
            if ok:
                accepted = True
                break

            failure = f"start ({task.start_x:.2f}, {task.start_y:.2f}): {why}"
            if self.verbose:
                print(f"[reset {attempt}/{MAX_RESET_ATTEMPTS}] rejected {failure}")
            if forced_task is not None:
                break

        if not accepted:
            raise RuntimeError(f"Could not obtain a valid reset. Last failure: {failure}")

        self.robot.clear_contacts()

        self.task = task
        self.episode_number += 1
        self.step_number = 0
        self.previous_command = np.zeros(2, dtype=np.float32)
        self.previous_distance, _ = self._goal_geometry(state)
        self.episode_min_lidar_distance = float(np.min(sectors))
        self.near_miss_occurred = bool(self.episode_min_lidar_distance < NEAR_MISS_DISTANCE)
        self.steps_below_preferred = 0
        self.steps_unsafe = 0
        self.steps_near_miss = 0
        self.path_length = 0.0
        self._last_xy = (state.x, state.y)
        self.episode_start_sim_time = self.robot.sim_time()

        if self.verbose:
            print(
                f"\nEpisode {self.episode_number} | start=({task.start_x:.2f}, {task.start_y:.2f}, "
                f"yaw {task.start_yaw:+.2f}) goal=({task.goal_x:.2f}, {task.goal_y:.2f}) "
                f"dist={self.previous_distance:.2f} m"
            )

        observation = self._get_observation(state, sectors)
        info = self._info(
            state=state, clearance=float(np.min(sectors)), progress=0.0,
            smoothness_penalty=0.0, proximity_penalty=0.0, contact=None,
            fell_over=False, termination_reason="reset",
        )
        return observation, info

    # -----------------------------------------------------------------------
    # STEP
    # -----------------------------------------------------------------------

    def step(self, action):
        if self.task is None:
            raise RuntimeError("step() called before reset().")

        self.step_number += 1

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        v = V_MAX * float(action[0]) if action[0] >= 0.0 else V_REVERSE * float(action[0])
        w = W_MAX * float(action[1])
        command = np.array([v, w], dtype=np.float32)

        # --- act for control_duration of SIMULATION time ---
        t0 = self.robot.sim_time()
        self.robot.send_velocity(v, w)
        self.robot.wait_sim_until(t0 + self.control_duration)
        self.robot.wait_fresh(
            state_after=t0 + 0.5 * self.control_duration,
            scan_after=t0,
        )

        state = self.robot.get_state()
        sectors = self._read_sectors()
        clearance = float(np.min(sectors))
        contact = self.robot.get_contact()

        # --- bookkeeping ---
        self.path_length += math.hypot(state.x - self._last_xy[0], state.y - self._last_xy[1])
        self._last_xy = (state.x, state.y)
        self.episode_min_lidar_distance = min(self.episode_min_lidar_distance, clearance)
        if clearance < PREFERRED_CLEARANCE:
            self.steps_below_preferred += 1
        if clearance < SAFE_DISTANCE:
            self.steps_unsafe += 1
        if clearance < NEAR_MISS_DISTANCE:
            self.steps_near_miss += 1
            self.near_miss_occurred = True

        distance, _ = self._goal_geometry(state)
        progress = self.previous_distance - distance

        # --- reward (V1 terms) ---
        reward = PROGRESS_SCALE * progress - STEP_PENALTY
        smoothness_penalty = SMOOTHNESS_SCALE * float(np.sum(np.square(command - self.previous_command)))
        reward -= smoothness_penalty
        proximity_penalty = self._proximity_penalty(clearance)
        reward -= proximity_penalty

        fell_over = abs(state.roll) >= FALL_ANGLE_RAD or abs(state.pitch) >= FALL_ANGLE_RAD

        terminated = False
        truncated = False
        reason = "running"

        if contact is not None:
            reward -= COLLISION_PENALTY
            terminated, reason = True, "collision"
        elif fell_over:
            reward -= FALL_PENALTY
            terminated, reason = True, "fall"
        elif distance <= GOAL_TOLERANCE:
            reward += GOAL_REWARD
            terminated, reason = True, "goal"
        elif self.step_number >= self.max_steps:
            reward -= TIMEOUT_PENALTY
            truncated, reason = True, "timeout"

        if terminated or truncated:
            self.robot.stop()
            if self.verbose:
                extra = f" | {contact['source']}:{contact['link']} -> {contact['other']}" if contact else ""
                print(
                    f"Episode {self.episode_number} ended: {reason} | steps={self.step_number} "
                    f"| goal dist={distance:.2f} m | min clearance={self.episode_min_lidar_distance:.3f} m{extra}"
                )

        self.previous_distance = distance
        self.previous_command = command

        observation = self._get_observation(state, sectors)
        info = self._info(
            state=state, clearance=clearance, progress=progress,
            smoothness_penalty=smoothness_penalty, proximity_penalty=proximity_penalty,
            contact=contact, fell_over=fell_over, termination_reason=reason,
        )
        return observation, float(reward), terminated, truncated, info

    # -----------------------------------------------------------------------
    # INFO
    # -----------------------------------------------------------------------

    def _info(self, state, clearance, progress, smoothness_penalty, proximity_penalty,
              contact, fell_over, termination_reason) -> Dict:
        distance, bearing = self._goal_geometry(state)
        return {
            "episode_number": self.episode_number,
            "step": self.step_number,
            "distance_to_goal": distance,
            "goal_bearing": bearing,
            "progress": progress,
            "smoothness_penalty": smoothness_penalty,
            "proximity_penalty": proximity_penalty,
            "min_lidar_distance": clearance,
            "episode_min_lidar_distance": self.episode_min_lidar_distance,
            "near_miss_occurred": self.near_miss_occurred,
            "clearance_band": (
                "clear" if clearance >= PREFERRED_CLEARANCE else
                "acceptable" if clearance >= SAFE_DISTANCE else
                "unsafe" if clearance >= NEAR_MISS_DISTANCE else "near_miss"
            ),
            "episode_steps_below_preferred": self.steps_below_preferred,
            "episode_steps_unsafe": self.steps_unsafe,
            "episode_steps_near_miss": self.steps_near_miss,
            "collision": contact is not None,
            "contact_source": None if contact is None else contact["source"],
            "contact_link": None if contact is None else contact["link"],
            "contact_other": None if contact is None else contact["other"],
            "fell_over": bool(fell_over),
            "roll": state.roll,
            "pitch": state.pitch,
            "x": state.x,
            "y": state.y,
            "yaw": state.yaw,
            "path_length": self.path_length,
            "sim_time": self.robot.sim_time() - self.episode_start_sim_time,
            "start": (self.task.start_x, self.task.start_y, self.task.start_yaw),
            "goal": (self.task.goal_x, self.task.goal_y),
            "straight_line_distance": self.task.straight_line_distance,
            "reference_path_length": getattr(self.task, "reference_path_length", float("nan")),
            "termination_reason": termination_reason,
            "success": termination_reason == "goal",
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.robot.stop()
        except Exception as exc:
            print("Warning while stopping Reachy:", exc)
