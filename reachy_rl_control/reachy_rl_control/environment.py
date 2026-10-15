#!/usr/bin/env python3

import math
import time
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)
from reachy_rl_control.trajectory import (
    STRAIGHT_PATH,
    WAYPOINT_TOLERANCE,
)


# ---------------------------------------------------------
# SAFETY PARAMETERS
# ---------------------------------------------------------

# Reachy is considered to have fallen if either roll
# or pitch exceeds 30 degrees.
FALL_ANGLE_DEGREES = 30.0
FALL_ANGLE_RAD = math.radians(
    FALL_ANGLE_DEGREES
)

# Large negative reward for falling.
FALL_PENALTY = 30.0


class ReachyGazeboEnv(gym.Env):
    """
    Gymnasium reinforcement-learning environment for
    Reachy's mobile base in Gazebo.

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
            normalized_vx,
            normalized_vy,
            normalized_omega,
        ]

    Each action is in the range [-1, 1].

    The normalized values are converted internally to:

        vx     -> [-0.30, 0.30] m/s
        vy     -> [-0.30, 0.30] m/s
        omega  -> [-0.60, 0.60] rad/s
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        robot: ReachyGazeboInterface,
        control_duration: float = 0.1,
        max_steps: int = 300,
    ):
        super().__init__()

        self.robot = robot

        self.control_duration = float(
            control_duration
        )

        self.max_steps = int(
            max_steps
        )

        # -------------------------------------------------
        # OBSERVATION SPACE
        # -------------------------------------------------

        # Observation:
        #
        # x
        # y
        # yaw
        # vx
        # vy
        # wz
        # relative goal x
        # relative goal y

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

        # -------------------------------------------------
        # ACTION SPACE
        # -------------------------------------------------

        # Stable-Baselines3 recommends normalized actions.
        #
        # PPO/SAC output:
        #
        # [-1, 1]
        #
        # These are later converted into physical Reachy
        # velocity commands.

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # -------------------------------------------------
        # EPISODE STATE
        # -------------------------------------------------

        self.episode_number = 0
        self.step_number = 0

        self.world_waypoints: List[
            Tuple[float, float]
        ] = []

        # Index 0 is the starting position.
        # The first target is waypoint 1.
        self.current_waypoint_index = 1

        self.previous_distance = 0.0

        # This stores PHYSICAL velocity commands:
        #
        # [vx, vy, omega]
        #
        # not normalized PPO actions.
        self.previous_action = np.zeros(
            3,
            dtype=np.float32,
        )

        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0

    # -----------------------------------------------------
    # TRAJECTORY CONVERSION
    # -----------------------------------------------------

    def _create_world_trajectory(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
    ) -> List[Tuple[float, float]]:
        """
        Convert the predefined local trajectory into
        Gazebo world coordinates.

        STRAIGHT_PATH is defined relative to the robot's
        starting pose.
        """

        world_waypoints = []

        cos_yaw = math.cos(
            start_yaw
        )

        sin_yaw = math.sin(
            start_yaw
        )

        for local_x, local_y in STRAIGHT_PATH:

            world_x = (
                start_x
                + cos_yaw * local_x
                - sin_yaw * local_y
            )

            world_y = (
                start_y
                + sin_yaw * local_x
                + cos_yaw * local_y
            )

            world_waypoints.append(
                (
                    world_x,
                    world_y,
                )
            )

        return world_waypoints

    # -----------------------------------------------------
    # DISTANCE TO CURRENT WAYPOINT
    # -----------------------------------------------------

    def _distance_to_current_waypoint(
        self,
        state=None,
    ) -> float:
        """
        Return Euclidean distance between Reachy and the
        current target waypoint.
        """

        if state is None:
            state = self.robot.get_state()

        goal_x, goal_y = (
            self.world_waypoints[
                self.current_waypoint_index
            ]
        )

        dx = (
            goal_x
            - state.x
        )

        dy = (
            goal_y
            - state.y
        )

        return math.hypot(
            dx,
            dy,
        )

    # -----------------------------------------------------
    # OBSERVATION
    # -----------------------------------------------------

    def _get_observation(
        self,
        state=None,
    ) -> np.ndarray:
        """
        Build the eight-dimensional RL observation.
        """

        if state is None:
            state = self.robot.get_state()

        goal_x, goal_y = (
            self.world_waypoints[
                self.current_waypoint_index
            ]
        )

        goal_dx = (
            goal_x
            - state.x
        )

        goal_dy = (
            goal_y
            - state.y
        )

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

    # -----------------------------------------------------
    # RESET
    # -----------------------------------------------------

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

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

        # Reset Gazebo and wait for fresh odometry.
        self.robot.reset_gazebo()

        state = self.robot.get_state()

        # Store the initial pose.
        self.start_x = float(
            state.x
        )

        self.start_y = float(
            state.y
        )

        self.start_yaw = float(
            state.yaw
        )

        # Convert predefined local trajectory into world
        # coordinates using this reset pose.
        self.world_waypoints = (
            self._create_world_trajectory(
                self.start_x,
                self.start_y,
                self.start_yaw,
            )
        )

        self.previous_distance = (
            self._distance_to_current_waypoint(
                state
            )
        )

        observation = (
            self._get_observation(
                state
            )
        )

        info: Dict = {
            "episode_number":
                self.episode_number,

            "step":
                self.step_number,

            "waypoint":
                self.current_waypoint_index,

            "waypoint_reached":
                False,

            "distance_to_goal":
                self.previous_distance,

            "progress":
                0.0,

            "smoothness_penalty":
                0.0,

            "fell_over":
                False,

            "roll":
                float(state.roll),

            "pitch":
                float(state.pitch),

            "termination_reason":
                "reset",
        }

        return observation, info

    # -----------------------------------------------------
    # STEP
    # -----------------------------------------------------

    def step(
        self,
        action,
    ):
        self.step_number += 1

        # -------------------------------------------------
        # NORMALIZE / CLIP ACTION
        # -------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        # -------------------------------------------------
        # CONVERT NORMALIZED ACTION INTO PHYSICAL COMMANDS
        # -------------------------------------------------

        vx = (
            0.30
            * float(action[0])
        )

        vy = (
            0.30
            * float(action[1])
        )

        omega = (
            0.60
            * float(action[2])
        )

        physical_action = np.array(
            [
                vx,
                vy,
                omega,
            ],
            dtype=np.float32,
        )

        # -------------------------------------------------
        # SEND COMMAND TO REACHY
        # -------------------------------------------------

        self.robot.send_velocity(
            vx,
            vy,
            omega,
        )

        time.sleep(
            self.control_duration
        )

        state_after_action = (
            self.robot.get_state()
        )

        # -------------------------------------------------
        # DISTANCE / PROGRESS
        # -------------------------------------------------

        current_distance = (
            self._distance_to_current_waypoint(
                state_after_action
            )
        )

        progress = (
            self.previous_distance
            - current_distance
        )

        # -------------------------------------------------
        # BASE REWARD
        # -------------------------------------------------

        # Positive reward for moving closer to waypoint.
        reward = (
            10.0
            * progress
        )

        # Small time penalty encourages faster completion.
        reward -= 0.01

        # -------------------------------------------------
        # SMOOTHNESS PENALTY
        # -------------------------------------------------

        # Smoothness is calculated using PHYSICAL
        # velocity commands rather than normalized
        # RL actions.

        action_change = (
            physical_action
            - self.previous_action
        )

        smoothness_penalty = (
            0.05
            * float(
                np.sum(
                    np.square(
                        action_change
                    )
                )
            )
        )

        reward -= (
            smoothness_penalty
        )

        # -------------------------------------------------
        # FALL DETECTION
        # -------------------------------------------------

        roll = float(
            state_after_action.roll
        )

        pitch = float(
            state_after_action.pitch
        )

        fell_over = bool(
            abs(roll)
            >= FALL_ANGLE_RAD
            or
            abs(pitch)
            >= FALL_ANGLE_RAD
        )

        # -------------------------------------------------
        # TERMINATION FLAGS
        # -------------------------------------------------

        terminated = False
        truncated = False

        waypoint_reached = False

        termination_reason = (
            "running"
        )

        # -------------------------------------------------
        # FALL FAILURE
        # -------------------------------------------------

        if fell_over:

            reward -= (
                FALL_PENALTY
            )

            terminated = True

            termination_reason = (
                "fall"
            )

            print(
                "\nFALL DETECTED | "
                f"Roll: "
                f"{math.degrees(roll):.1f} deg | "
                f"Pitch: "
                f"{math.degrees(pitch):.1f} deg"
            )

            self.robot.stop_for(
                duration=0.5,
                frequency=20.0,
            )

        # -------------------------------------------------
        # WAYPOINT / GOAL CHECK
        # -------------------------------------------------

        elif (
            current_distance
            <= WAYPOINT_TOLERANCE
        ):

            waypoint_reached = True

            # Reward for reaching an intermediate waypoint.
            reward += 5.0

            # Check whether this is the final waypoint.
            if (
                self.current_waypoint_index
                == len(
                    self.world_waypoints
                ) - 1
            ):

                # Additional final-goal reward.
                reward += 20.0

                terminated = True

                termination_reason = (
                    "goal"
                )

                self.robot.stop_for(
                    duration=0.5,
                    frequency=20.0,
                )

            else:

                # Move target to the next waypoint.
                self.current_waypoint_index += 1

                current_distance = (
                    self._distance_to_current_waypoint(
                        state_after_action
                    )
                )

        # -------------------------------------------------
        # TIMEOUT
        # -------------------------------------------------

        if (
            self.step_number
            >= self.max_steps
            and not terminated
        ):

            truncated = True

            termination_reason = (
                "timeout"
            )

            reward -= 5.0

            self.robot.stop_for(
                duration=0.5,
                frequency=20.0,
            )

        # -------------------------------------------------
        # STORE VALUES FOR NEXT STEP
        # -------------------------------------------------

        self.previous_distance = (
            current_distance
        )

        self.previous_action = (
            physical_action.copy()
        )

        # Get fresh state after stopping if the episode
        # finished.
        if (
            terminated
            or truncated
        ):
            state_after_action = (
                self.robot.get_state()
            )

        observation = (
            self._get_observation(
                state_after_action
            )
        )

        # -------------------------------------------------
        # INFO
        # -------------------------------------------------

        info: Dict = {
            "episode_number":
                self.episode_number,

            "step":
                self.step_number,

            "waypoint":
                self.current_waypoint_index,

            "waypoint_reached":
                waypoint_reached,

            "distance_to_goal":
                current_distance,

            "progress":
                progress,

            "smoothness_penalty":
                smoothness_penalty,

            "fell_over":
                fell_over,

            "roll":
                roll,

            "pitch":
                pitch,

            "termination_reason":
                termination_reason,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # -----------------------------------------------------
    # CLOSE
    # -----------------------------------------------------

    def close(self):
        """
        Always stop Reachy when the environment closes.
        """

        self.robot.stop_for(
            duration=1.0,
            frequency=20.0,
        )
