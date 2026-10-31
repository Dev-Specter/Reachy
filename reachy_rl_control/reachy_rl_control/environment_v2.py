#!/usr/bin/env python3

"""
V2 no-human Reachy Gazebo environment.

The frozen V1 environment remains unchanged.

V2 changes only the observation representation.

Observation order (10 values):

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
"""

import numpy as np
from gymnasium import spaces

from reachy_rl_control.environment import (
    ReachyGazeboEnv as _BaseReachyGazeboEnv,
)

from reachy_rl_control.path_relative_features import (
    compute_path_relative_features,
    path_length,
)


class ReachyGazeboEnv(
    _BaseReachyGazeboEnv
):
    """
    Reachy Gazebo environment using the V2
    path-relative observation representation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        # V2 no-human observation:
        #
        # 3 velocity values
        # +
        # 7 path-relative values
        # =
        # 10 values

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

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

        path_features, metadata = (
            compute_path_relative_features(
                robot_x=state.x,
                robot_y=state.y,
                robot_yaw=state.yaw,
                path=self.world_waypoints,
            )
        )

        observation = np.concatenate(
            (
                np.asarray(
                    [
                        state.vx,
                        state.vy,
                        state.wz,
                    ],
                    dtype=np.float32,
                ),
                path_features,
            )
        ).astype(
            np.float32,
            copy=False,
        )

        self.v2_path_metadata = metadata

        if observation.shape != (10,):
            raise RuntimeError(
                "V2 observation must have "
                f"shape (10,), received "
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
