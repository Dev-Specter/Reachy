#!/usr/bin/env python3

import math
import time
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reachy_rl_control.reachy_gazebo_interface import ReachyGazeboInterface
from reachy_rl_control.trajectory import STRAIGHT_PATH, WAYPOINT_TOLERANCE


class ReachyGazeboEnv(gym.Env):
    """
    Gymnasium reinforcement-learning environment for Reachy's mobile base.

    Observation:
        [
            x,
            y,
            yaw,
            vx,
            vy,
            wz,
            goal_dx,
            goal_dy,
        ]

    Action:
        [
            vx,
            vy,
            omega,
        ]
    """

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        robot: ReachyGazeboInterface,
        control_duration: float = 0.1,
        max_steps: int = 300,
    ):
        super().__init__()

        self.robot = robot

        self.control_duration = control_duration
        self.max_steps = max_steps

        # ---------------------------------
        # ACTION SPACE
        # ---------------------------------

        # Reachy mobile base velocity commands:
        #
        # vx     forward/backward
        # vy     sideways
        # omega  rotation
        #
        self.action_space = spaces.Box(
            low=np.array(
                [-0.30, -0.30, -0.60],
                dtype=np.float32,
            ),
            high=np.array(
                [0.30, 0.30, 0.60],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # ---------------------------------
        # OBSERVATION SPACE
        # ---------------------------------

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

        # Episode information
        self.episode_number = 0
        self.step_number = 0

        # Trajectory information
        self.waypoints = []
        self.current_waypoint_index = 1

        # Reward information
        self.previous_distance = None
        self.previous_action = np.zeros(
            3,
            dtype=np.float32,
        )

        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0

    # ============================================================
    # TRAJECTORY
    # ============================================================

    def _create_world_trajectory(self) -> None:
        """
        Convert the predefined trajectory into Gazebo world coordinates.

        The trajectory begins at Reachy's reset position and follows
        Reachy's initial heading.
        """

        self.waypoints = []

        for path_x, path_y in STRAIGHT_PATH:

            world_x = (
                self.start_x
                + math.cos(self.start_yaw) * path_x
                - math.sin(self.start_yaw) * path_y
            )

            world_y = (
                self.start_y
                + math.sin(self.start_yaw) * path_x
                + math.cos(self.start_yaw) * path_y
            )

            self.waypoints.append(
                (world_x, world_y)
            )

    # ============================================================
    # GOAL
    # ============================================================

    def _get_current_goal(self) -> Tuple[float, float]:

        return self.waypoints[
            self.current_waypoint_index
        ]

    # ============================================================
    # OBSERVATION
    # ============================================================

    def _get_observation(self) -> np.ndarray:

        state = self.robot.get_state()

        if state is None:
            raise RuntimeError(
                "No odometry available."
            )

        goal_x, goal_y = self._get_current_goal()

        goal_dx = goal_x - state.x
        goal_dy = goal_y - state.y

        observation = np.array(
            [
                state.x,
                state.y,
                state.yaw,
                state.vx,
                state.vy,
                state.wz,
                goal_dx,
                goal_dy,
            ],
            dtype=np.float32,
        )

        return observation

    # ============================================================
    # DISTANCE TO GOAL
    # ============================================================

    def _distance_to_current_goal(self) -> float:

        state = self.robot.get_state()

        if state is None:
            raise RuntimeError(
                "No odometry available."
            )

        goal_x, goal_y = self._get_current_goal()

        return math.hypot(
            goal_x - state.x,
            goal_y - state.y,
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(seed=seed)

        self.episode_number += 1
        self.step_number = 0

        self.current_waypoint_index = 1

        self.previous_action = np.zeros(
            3,
            dtype=np.float32,
        )

        print(
            f"\nStarting RL episode "
            f"{self.episode_number}"
        )

        state = self.robot.reset_gazebo(
            timeout=5.0,
            settle_time=0.5,
        )

        self.start_x = state.x
        self.start_y = state.y
        self.start_yaw = state.yaw

        self._create_world_trajectory()

        self.previous_distance = (
            self._distance_to_current_goal()
        )

        observation = self._get_observation()

        info = {
            "waypoint": self.current_waypoint_index,
            "distance_to_goal": self.previous_distance,
        }

        return observation, info

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action):

        self.step_number += 1

        # Make sure action respects velocity limits
        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        vx = float(action[0])
        vy = float(action[1])
        omega = float(action[2])

        # ---------------------------------
        # APPLY ACTION
        # ---------------------------------

        self.robot.send_velocity(
            vx=vx,
            vy=vy,
            omega=omega,
        )

        time.sleep(
            self.control_duration
        )

        # ---------------------------------
        # DISTANCE / PROGRESS
        # ---------------------------------

        current_distance = (
            self._distance_to_current_goal()
        )

        progress = (
            self.previous_distance
            - current_distance
        )

        # Reward movement toward the waypoint
        reward = 10.0 * progress

        # Small penalty for taking another step
        reward -= 0.01

        # ---------------------------------
        # SMOOTHNESS PENALTY
        # ---------------------------------

        action_change = (
            action
            - self.previous_action
        )

        smoothness_penalty = (
            0.05
            * float(
                np.sum(
                    np.square(action_change)
                )
            )
        )

        reward -= smoothness_penalty

        self.previous_action = action.copy()

        # ---------------------------------
        # WAYPOINT / GOAL CHECK
        # ---------------------------------

        terminated = False
        waypoint_reached = False

        if current_distance <= WAYPOINT_TOLERANCE:

            waypoint_reached = True

            reward += 5.0

            # Is this the final waypoint?
            if (
                self.current_waypoint_index
                == len(self.waypoints) - 1
            ):

                reward += 20.0

                terminated = True

                self.robot.stop_for(
                    duration=0.5,
                    frequency=20.0,
                )

            else:

                self.current_waypoint_index += 1

                current_distance = (
                    self._distance_to_current_goal()
                )

        # ---------------------------------
        # TIME LIMIT
        # ---------------------------------

        truncated = (
            self.step_number
            >= self.max_steps
        )

        if truncated and not terminated:

            reward -= 5.0

            self.robot.stop_for(
                duration=0.5,
                frequency=20.0,
            )

        # ---------------------------------
        # PREPARE NEXT STEP
        # ---------------------------------

        self.previous_distance = (
            current_distance
        )

        observation = self._get_observation()

        info: Dict = {
            "episode": self.episode_number,
            "step": self.step_number,
            "waypoint": self.current_waypoint_index,
            "waypoint_reached": waypoint_reached,
            "distance_to_goal": current_distance,
            "progress": progress,
            "smoothness_penalty": smoothness_penalty,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self):

        self.robot.stop_for(
            duration=1.0,
            frequency=20.0,
        )
