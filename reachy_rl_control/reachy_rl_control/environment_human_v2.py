#!/usr/bin/env python3

"""
V2 human-aware Reachy environment.

This environment intentionally inherits all task behaviour from
the frozen V1 human-aware environment.

UNCHANGED:
    - action space
    - reward
    - waypoint rewards
    - collision handling
    - fall handling
    - human crossing scenario
    - proximity penalties
    - reset behaviour
    - termination rules
    - episode horizon

CHANGED:
    - observation representation only

V2 removes absolute world-frame x, y and yaw and replaces them
with robot/path-relative trajectory information.

Observation order (22 values):

    0   vx
    1   vy
    2   wz

    3   next_waypoint_local_x
    4   next_waypoint_local_y

    5   lookahead_local_x
    6   lookahead_local_y

    7   signed_cross_track_error
    8   heading_error
    9   remaining_path_length_capped

    10-21  normalized LiDAR sectors

The path used for reward/termination remains exactly the path
defined by the inherited environment.

Path-relative helper calculations interpolate along that same
polyline and therefore do NOT alter reward waypoint spacing.
"""

import numpy as np
from gymnasium import spaces

from reachy_rl_control.environment_human import (
    ReachyHumanGazeboEnv as _BaseReachyHumanGazeboEnv,
    NUM_LIDAR_SECTORS,
)

from reachy_rl_control.path_relative_features import (
    build_v2_observation,
    path_length,
)


class ReachyHumanGazeboEnv(
    _BaseReachyHumanGazeboEnv
):
    """
    Human-aware environment using the V2 path-relative
    observation representation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # -------------------------------------------------
        # Initialise the complete frozen V1 environment.
        # -------------------------------------------------

        super().__init__(
            *args,
            **kwargs,
        )

        # -------------------------------------------------
        # Replace ONLY the observation space.
        #
        # First 10 values:
        #     path-relative robot/path state
        #
        # Final 12:
        #     normalized LiDAR [0, 1]
        # -------------------------------------------------

        observation_low = np.concatenate(
            (
                np.full(
                    10,
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
                    10,
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

        # Latest path-relative metadata.
        #
        # These variables are for evaluation/logging only.
        # They do not affect reward or termination.

        self.v2_path_metadata = {}

    # =====================================================
    # V2 OBSERVATION
    # =====================================================

    def _get_observation(
        self,
        state=None,
    ) -> np.ndarray:

        if state is None:
            state = (
                self.robot.get_state()
            )

        lidar_observation = (
            self.lidar.get_normalized_distances()
        )

        observation, metadata = (
            build_v2_observation(
                robot_x=state.x,
                robot_y=state.y,
                robot_yaw=state.yaw,
                vx=state.vx,
                vy=state.vy,
                wz=state.wz,
                path=self.world_waypoints,
                lidar_normalized=lidar_observation,
            )
        )

        self.v2_path_metadata = metadata

        if observation.shape != (22,):
            raise RuntimeError(
                "V2 human observation must have "
                f"shape (22,), received "
                f"{observation.shape}."
            )

        return observation

    # =====================================================
    # PROGRESS METRICS
    # =====================================================

    def _add_v2_progress_info(
        self,
        info,
    ):
        """
        Add path progress to the Gymnasium info dictionary.

        This is evaluation metadata only.
        """

        if info is None:
            info = {}

        metadata = (
            self.v2_path_metadata
            if self.v2_path_metadata
            else {}
        )

        progress_distance = float(
            metadata.get(
                "progress_distance",
                0.0,
            )
        )

        progress_ratio = float(
            metadata.get(
                "progress_ratio",
                0.0,
            )
        )

        if self.world_waypoints:
            total_length = float(
                path_length(
                    self.world_waypoints
                )
            )
        else:
            total_length = 0.0

        info[
            "path_progress_distance"
        ] = progress_distance

        info[
            "path_progress_ratio"
        ] = progress_ratio

        info[
            "path_progress_percent"
        ] = (
            100.0
            * progress_ratio
        )

        info[
            "path_total_length"
        ] = total_length

        return info

    # =====================================================
    # RESET WRAPPER
    # =====================================================

    def reset(
        self,
        *args,
        **kwargs,
    ):
        observation, info = (
            super().reset(
                *args,
                **kwargs,
            )
        )

        info = (
            self._add_v2_progress_info(
                info
            )
        )

        return (
            observation,
            info,
        )

    # =====================================================
    # STEP WRAPPER
    # =====================================================

    def step(
        self,
        action,
    ):
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = super().step(
            action
        )

        info = (
            self._add_v2_progress_info(
                info
            )
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
