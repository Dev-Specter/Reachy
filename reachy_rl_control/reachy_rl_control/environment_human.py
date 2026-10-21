#!/usr/bin/env python3

import math
import threading
import time
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rclpy.executors import (
    ExternalShutdownException,
    SingleThreadedExecutor,
)

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)

from reachy_rl_control.human_manager import (
    HumanManager,
)

from reachy_rl_control.lidar_processor import (
    LidarProcessor,
)

from reachy_rl_control.trajectory import (
    STRAIGHT_PATH,
    WAYPOINT_TOLERANCE,
)


# =========================================================
# SAFETY PARAMETERS
# =========================================================

FALL_ANGLE_DEGREES = 30.0

FALL_ANGLE_RAD = math.radians(
    FALL_ANGLE_DEGREES
)

FALL_PENALTY = 30.0

COLLISION_PENALTY = 40.0


# =========================================================
# RESET VALIDATION
# =========================================================

RESET_MAX_ATTEMPTS = 3

# Reachy must begin each episode upright.
RESET_MAX_ANGLE_DEGREES = 5.0

RESET_MAX_ANGLE_RAD = math.radians(
    RESET_MAX_ANGLE_DEGREES
)

# Reachy must also be nearly stationary before an
# RL episode is accepted as valid.
RESET_MAX_LINEAR_SPEED = 0.05

RESET_MAX_ANGULAR_SPEED = 0.10

# Additional settling time after reset_gazebo().
RESET_SETTLE_TIME = 0.75


# =========================================================
# HUMAN PROXIMITY PARAMETERS
# =========================================================

# Proximity penalty begins below this LiDAR clearance.
SAFE_DISTANCE = 0.75

# Used for near-miss logging.
NEAR_MISS_DISTANCE = 0.50

# Maximum per-step proximity penalty approaches this value.
PROXIMITY_PENALTY_SCALE = 1.0


# =========================================================
# LIDAR PARAMETERS
# =========================================================

NUM_LIDAR_SECTORS = 12

LIDAR_MAX_DISTANCE = 3.0


# =========================================================
# HUMAN CROSSING SCENARIO
# =========================================================

# The human pose is specified directly relative to the
# Gazebo model called "reachy".
#
# Reachy's local trajectory travels along +X.
#
# Human starts:
#
#     +1.0 m in Reachy's local X direction
#     -1.5 m in Reachy's local Y direction
#
# The human then moves across the trajectory in local +Y.

HUMAN_REFERENCE_FRAME = "reachy"

HUMAN_LOCAL_X = 1.0
HUMAN_LOCAL_Y = -1.5
HUMAN_Z = 0.9

HUMAN_LOCAL_VX = 0.0
HUMAN_LOCAL_VY = 0.35

HUMAN_OMEGA = 0.0


class ReachyHumanGazeboEnv(gym.Env):
    """
    Human-aware Gymnasium environment for Reachy's
    mobile base in Gazebo.

    Observation
    -----------

    20 values:

    [
        x,
        y,
        yaw,
        vx,
        vy,
        wz,
        goal_dx,
        goal_dy,

        lidar_sector_0,
        lidar_sector_1,
        ...
        lidar_sector_11,
    ]

    The first 8 values describe Reachy's state and
    current waypoint.

    The final 12 values are normalized LiDAR sector
    distances in [0, 1].

        0.0 = very close obstacle
        1.0 = 3 metres or farther

    Action
    ------

    [
        normalized_vx,
        normalized_vy,
        normalized_omega,
    ]

    Each action lies in [-1, 1].

    Physical action limits:

        vx     = +/- 0.30 m/s
        vy     = +/- 0.30 m/s
        omega  = +/- 0.60 rad/s

    Human scenario
    --------------

    Human spawn pose is specified relative to the Gazebo
    model "reachy":

        x = +1.0 m
        y = -1.5 m
        z = +0.9 m

    Human crossing velocity:

        vx = 0.0 m/s
        vy = +0.35 m/s

    Termination reasons:

        goal
        collision
        fall
        timeout
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
        # HUMAN + LIDAR ROS NODES
        # -------------------------------------------------

        self.human = HumanManager()

        self.lidar = LidarProcessor(
            num_sectors=NUM_LIDAR_SECTORS,
            max_distance=LIDAR_MAX_DISTANCE,
        )

        self._aux_executor = (
            SingleThreadedExecutor()
        )

        self._aux_executor.add_node(
            self.human
        )

        self._aux_executor.add_node(
            self.lidar
        )

        self._aux_spin_thread = (
            threading.Thread(
                target=self._spin_auxiliary_nodes,
                daemon=True,
            )
        )

        self._aux_spin_thread.start()

        # -------------------------------------------------
        # CHECK GAZEBO SERVICES
        # -------------------------------------------------

        if not self.human.wait_for_services(
            timeout=10.0
        ):
            raise RuntimeError(
                "HumanManager could not connect to "
                "Gazebo spawn/delete services."
            )

        # -------------------------------------------------
        # CHECK LIDAR
        # -------------------------------------------------

        if not self.lidar.wait_for_scan(
            timeout=10.0
        ):
            raise RuntimeError(
                "No /scan messages received."
            )

        # -------------------------------------------------
        # OBSERVATION SPACE
        # -------------------------------------------------

        observation_low = np.concatenate(
            (
                np.full(
                    8,
                    -np.inf,
                    dtype=np.float32,
                ),
                np.zeros(
                    NUM_LIDAR_SECTORS,
                    dtype=np.float32,
                ),
            )
        )

        observation_high = np.concatenate(
            (
                np.full(
                    8,
                    np.inf,
                    dtype=np.float32,
                ),
                np.ones(
                    NUM_LIDAR_SECTORS,
                    dtype=np.float32,
                ),
            )
        )

        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # ACTION SPACE
        # -------------------------------------------------

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

        # Waypoint 0 is the starting position.
        # Waypoint 1 is the first target.
        self.current_waypoint_index = 1

        self.previous_distance = 0.0

        # Physical action:
        #
        # [vx, vy, omega]
        self.previous_action = np.zeros(
            3,
            dtype=np.float32,
        )

        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0

        # -------------------------------------------------
        # HUMAN EPISODE STATE
        # -------------------------------------------------

        # These values are Reachy-relative coordinates,
        # not Gazebo world coordinates.
        self.human_spawn_x = (
            HUMAN_LOCAL_X
        )

        self.human_spawn_y = (
            HUMAN_LOCAL_Y
        )

        self.human_velocity_x = (
            HUMAN_LOCAL_VX
        )

        self.human_velocity_y = (
            HUMAN_LOCAL_VY
        )

        self.episode_min_lidar_distance = (
            LIDAR_MAX_DISTANCE
        )

        self.near_miss_occurred = False

        self._closed = False

        print(
            "\nHuman-aware Reachy environment ready."
        )

        print(
            "Observation size:",
            self.observation_space.shape,
        )

        print(
            "Human reference frame:",
            HUMAN_REFERENCE_FRAME,
        )

    # =====================================================
    # AUXILIARY ROS EXECUTOR
    # =====================================================

    def _spin_auxiliary_nodes(
        self,
    ):
        try:
            self._aux_executor.spin()

        except ExternalShutdownException:
            pass

    # =====================================================
    # TRAJECTORY
    # =====================================================

    def _create_world_trajectory(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
    ) -> List[Tuple[float, float]]:
        """
        Convert the predefined Reachy-local trajectory into
        Reachy's odometry coordinate system.

        This is used for trajectory tracking and reward.

        It is NOT used to determine the human's Gazebo
        spawn position.
        """

        world_waypoints = []

        cos_yaw = math.cos(
            start_yaw
        )

        sin_yaw = math.sin(
            start_yaw
        )

        for (
            local_x,
            local_y,
        ) in STRAIGHT_PATH:

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

    # =====================================================
    # DISTANCE TO CURRENT WAYPOINT
    # =====================================================

    def _distance_to_current_waypoint(
        self,
        state=None,
    ) -> float:

        if state is None:
            state = (
                self.robot.get_state()
            )

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

    # =====================================================
    # OBSERVATION
    # =====================================================

    def _get_observation(
        self,
        state=None,
    ) -> np.ndarray:

        if state is None:
            state = (
                self.robot.get_state()
            )

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

        robot_observation = np.array(
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

        lidar_observation = (
            self.lidar.get_normalized_distances()
        )

        observation = np.concatenate(
            (
                robot_observation,
                lidar_observation,
            )
        ).astype(
            np.float32
        )

        return observation

    # =====================================================
    # PROXIMITY PENALTY
    # =====================================================

    def _calculate_proximity_penalty(
        self,
        min_distance: float,
    ) -> float:

        if (
            min_distance
            >= SAFE_DISTANCE
        ):
            return 0.0

        normalized_closeness = (
            SAFE_DISTANCE
            - min_distance
        ) / SAFE_DISTANCE

        normalized_closeness = float(
            np.clip(
                normalized_closeness,
                0.0,
                1.0,
            )
        )

        return (
            PROXIMITY_PENALTY_SCALE
            * normalized_closeness
        )

    # =====================================================
    # START HUMAN CROSSING
    # =====================================================

    def _start_human_crossing(
        self,
    ):
        """
        Human is spawned with identity orientation relative
        to the Reachy Gazebo model.

        The crossing command is therefore kept as:

            vx = 0.0
            vy = +0.35

        so the human crosses Reachy's path along local +Y.
        """

        self.human_velocity_x = (
            HUMAN_LOCAL_VX
        )

        self.human_velocity_y = (
            HUMAN_LOCAL_VY
        )

        # Publish multiple times for reliability.
        for _ in range(5):

            self.human.send_velocity(
                vx=self.human_velocity_x,
                vy=self.human_velocity_y,
                omega=HUMAN_OMEGA,
            )

            time.sleep(
                0.02
            )

    # =====================================================
    # ROBOT RESET VALIDATION
    # =====================================================

    def _reset_robot_until_stable(
        self,
    ):
        """
        Reset Gazebo and accept the reset only when Reachy
        is both upright and approximately stationary.
        """

        for attempt in range(
            1,
            RESET_MAX_ATTEMPTS + 1,
        ):

            print(
                f"Resetting Reachy "
                f"(attempt {attempt}/"
                f"{RESET_MAX_ATTEMPTS})..."
            )

            self.robot.reset_gazebo()

            time.sleep(
                RESET_SETTLE_TIME
            )

            state = (
                self.robot.get_state()
            )

            if state is None:

                print(
                    "No valid state after reset."
                )

                continue

            roll = float(
                state.roll
            )

            pitch = float(
                state.pitch
            )

            vx = float(
                state.vx
            )

            vy = float(
                state.vy
            )

            wz = float(
                state.wz
            )

            attitude_stable = bool(
                abs(roll)
                <= RESET_MAX_ANGLE_RAD
                and
                abs(pitch)
                <= RESET_MAX_ANGLE_RAD
            )

            velocity_stable = bool(
                abs(vx)
                <= RESET_MAX_LINEAR_SPEED
                and
                abs(vy)
                <= RESET_MAX_LINEAR_SPEED
                and
                abs(wz)
                <= RESET_MAX_ANGULAR_SPEED
            )

            stable = bool(
                attitude_stable
                and
                velocity_stable
            )

            if stable:

                print(
                    "Reachy reset is stable | "
                    f"roll="
                    f"{math.degrees(roll):.2f} deg | "
                    f"pitch="
                    f"{math.degrees(pitch):.2f} deg | "
                    f"vx={vx:.3f} m/s | "
                    f"vy={vy:.3f} m/s | "
                    f"wz={wz:.3f} rad/s"
                )

                return state

            print(
                "WARNING: unstable reset | "
                f"roll="
                f"{math.degrees(roll):.2f} deg | "
                f"pitch="
                f"{math.degrees(pitch):.2f} deg | "
                f"vx={vx:.3f} m/s | "
                f"vy={vy:.3f} m/s | "
                f"wz={wz:.3f} rad/s"
            )

            # Stop any residual commanded motion before
            # attempting another Gazebo reset.
            self.robot.stop_for(
                duration=0.5,
                frequency=20.0,
            )

        raise RuntimeError(
            "Reachy could not obtain a stable "
            "upright and stationary reset after "
            f"{RESET_MAX_ATTEMPTS} attempts."
        )

    # =====================================================
    # RESET
    # =====================================================

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

        self.episode_min_lidar_distance = (
            LIDAR_MAX_DISTANCE
        )

        self.near_miss_occurred = False

        print(
            f"\nStarting HUMAN RL episode "
            f"{self.episode_number}"
        )

                # -------------------------------------------------
        # REMOVE PREVIOUS HUMAN BEFORE RESETTING REACHY
        # -------------------------------------------------

        # Stop the previous human first.
        self.human.stop()

        # Delete the previous human before /reset_world.
        #
        # This prevents the old human from being moved by
        # Gazebo reset physics into Reachy's arms/body while
        # Reachy itself is being reset.
        self.human.delete_human()

        time.sleep(
            0.25
        )

        # Any contact messages associated with the previous
        # episode must not carry into the new episode.
        self.human.clear_collision()

        # -------------------------------------------------
        # RESET REACHY / GAZEBO
        # -------------------------------------------------

        state = (
            self._reset_robot_until_stable()
        )

        # -------------------------------------------------
        # RESET REACHY / GAZEBO
        # -------------------------------------------------

        state = (
            self._reset_robot_until_stable()
        )

        self.start_x = float(
            state.x
        )

        self.start_y = float(
            state.y
        )

        self.start_yaw = float(
            state.yaw
        )

        # -------------------------------------------------
        # CREATE REACHY TRAJECTORY
        # -------------------------------------------------

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

        # -------------------------------------------------
        # HUMAN SPAWN PARAMETERS
        # -------------------------------------------------

        # These values are NOT transformed through /odom.
        #
        # Gazebo SpawnEntity interprets them relative to
        # the Gazebo entity named "reachy".

        self.human_spawn_x = (
            HUMAN_LOCAL_X
        )

        self.human_spawn_y = (
            HUMAN_LOCAL_Y
        )

        # -------------------------------------------------
        # RESET HUMAN RELATIVE TO REACHY
        # -------------------------------------------------

        self.human.reset_human(
            x=self.human_spawn_x,
            y=self.human_spawn_y,
            z=HUMAN_Z,
            reference_frame=(
                HUMAN_REFERENCE_FRAME
            ),
        )

        # -------------------------------------------------
        # WAIT FOR FRESH LIDAR
        # -------------------------------------------------

        self.lidar.scan_received = False

        if not self.lidar.wait_for_scan(
            timeout=5.0
        ):
            raise RuntimeError(
                "No fresh LiDAR scan received "
                "after human reset."
            )

        # -------------------------------------------------
        # INITIAL OBSERVATION
        # -------------------------------------------------

        observation = (
            self._get_observation(
                state
            )
        )

        initial_min_distance = (
            self.lidar.get_min_distance()
        )

        self.episode_min_lidar_distance = (
            initial_min_distance
        )

        # -------------------------------------------------
        # START HUMAN CROSSING
        # -------------------------------------------------

        self._start_human_crossing()

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

            "proximity_penalty":
                0.0,

            "min_lidar_distance":
                initial_min_distance,

            "episode_min_lidar_distance":
                self.episode_min_lidar_distance,

            "near_human":
                False,

            "near_miss_occurred":
                False,

            "collision":
                False,

            "fell_over":
                False,

            "roll":
                float(state.roll),

            "pitch":
                float(state.pitch),

            "human_spawn_x":
                self.human_spawn_x,

            "human_spawn_y":
                self.human_spawn_y,

            "human_reference_frame":
                HUMAN_REFERENCE_FRAME,

            "human_velocity_x":
                self.human_velocity_x,

            "human_velocity_y":
                self.human_velocity_y,

            "termination_reason":
                "reset",

            "success":
                False,
        }

        return (
            observation,
            info,
        )

    # =====================================================
    # STEP
    # =====================================================

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
        # CONVERT TO PHYSICAL VELOCITY
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
        # SEND REACHY ACTION
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
        # TRAJECTORY PROGRESS
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

        reward = (
            10.0
            * progress
        )

        reward -= 0.01

        # -------------------------------------------------
        # ACTION SMOOTHNESS
        # -------------------------------------------------

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
        # LIDAR / HUMAN PROXIMITY
        # -------------------------------------------------

        min_lidar_distance = (
            self.lidar.get_min_distance()
        )

        self.episode_min_lidar_distance = min(
            self.episode_min_lidar_distance,
            min_lidar_distance,
        )

        near_human = bool(
            min_lidar_distance
            < SAFE_DISTANCE
        )

        current_near_miss = bool(
            min_lidar_distance
            < NEAR_MISS_DISTANCE
        )

        if current_near_miss:

            self.near_miss_occurred = True

        proximity_penalty = (
            self._calculate_proximity_penalty(
                min_lidar_distance
            )
        )

        reward -= (
            proximity_penalty
        )

        # -------------------------------------------------
        # COLLISION
        # -------------------------------------------------

        collision = bool(
            self.human.collision_detected
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
        # COLLISION FAILURE
        # -------------------------------------------------

        if collision:

            reward -= (
                COLLISION_PENALTY
            )

            terminated = True

            termination_reason = (
                "collision"
            )

            print(
                "\nHUMAN COLLISION DETECTED"
            )

            self.robot.stop_for(
                duration=0.5,
                frequency=20.0,
            )

            self.human.stop()

        # -------------------------------------------------
        # FALL FAILURE
        # -------------------------------------------------

        elif fell_over:

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

            self.human.stop()

        # -------------------------------------------------
        # WAYPOINT / GOAL
        # -------------------------------------------------

        elif (
            current_distance
            <= WAYPOINT_TOLERANCE
        ):

            waypoint_reached = True

            # Intermediate waypoint reward.
            reward += 5.0

            if (
                self.current_waypoint_index
                == len(
                    self.world_waypoints
                ) - 1
            ):

                # Final trajectory completion reward.
                reward += 20.0

                terminated = True

                termination_reason = (
                    "goal"
                )

                self.robot.stop_for(
                    duration=0.5,
                    frequency=20.0,
                )

                self.human.stop()

            else:

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
            and
            not terminated
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

            self.human.stop()

        # -------------------------------------------------
        # STORE VALUES FOR NEXT STEP
        # -------------------------------------------------

        self.previous_distance = (
            current_distance
        )

        self.previous_action = (
            physical_action.copy()
        )

        # -------------------------------------------------
        # GET FRESH STATE IF EPISODE ENDED
        # -------------------------------------------------

        if (
            terminated
            or
            truncated
        ):

            state_after_action = (
                self.robot.get_state()
            )

        observation = (
            self._get_observation(
                state_after_action
            )
        )

        success = bool(
            termination_reason
            == "goal"
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

            "proximity_penalty":
                proximity_penalty,

            "min_lidar_distance":
                min_lidar_distance,

            "episode_min_lidar_distance":
                self.episode_min_lidar_distance,

            "near_human":
                near_human,

            "near_miss_occurred":
                self.near_miss_occurred,

            "collision":
                collision,

            "fell_over":
                fell_over,

            "roll":
                roll,

            "pitch":
                pitch,

            "human_spawn_x":
                self.human_spawn_x,

            "human_spawn_y":
                self.human_spawn_y,

            "human_reference_frame":
                HUMAN_REFERENCE_FRAME,

            "human_velocity_x":
                self.human_velocity_x,

            "human_velocity_y":
                self.human_velocity_y,

            "termination_reason":
                termination_reason,

            "success":
                success,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # =====================================================
    # CLOSE
    # =====================================================

    def close(
        self,
    ):

        if self._closed:

            return

        self._closed = True

        print(
            "\nClosing human-aware environment..."
        )

        try:

            self.robot.stop_for(
                duration=1.0,
                frequency=20.0,
            )

        except Exception as exc:

            print(
                "Warning while stopping Reachy:",
                exc,
            )

        try:

            self.human.stop()

            self.human.delete_human()

        except Exception as exc:

            print(
                "Warning while stopping human:",
                exc,
            )

        try:

            self._aux_executor.shutdown()

        except Exception:

            pass

        try:

            self.human.destroy_node()

        except Exception:

            pass

        try:

            self.lidar.destroy_node()

        except Exception:

            pass

        if (
            self._aux_spin_thread.is_alive()
        ):

            self._aux_spin_thread.join(
                timeout=1.0
            )

        print(
            "Human-aware environment closed."
        )
