#!/usr/bin/env python3

"""
V2.1 human-aware Reachy environment.

V2.1 preserves:
    - frozen V1 reward
    - frozen V1 termination rules
    - frozen V1 human scenario
    - frozen V1 reset logic
    - frozen V1 waypoint spacing
    - V2 22-dimensional observation
    - V2 LiDAR layout
    - V2 progress metadata

The only V2 -> V2.1 change is:

    V2:
        immediate target = interpolated point 0.25 m
        ahead of current projected path progress

    V2.1:
        immediate target = EXACTLY
        self.world_waypoints[self.current_waypoint_index]

This makes the immediate observed waypoint identical to the
waypoint used by the frozen reward logic.
"""

import numpy as np

from reachy_rl_control.environment_human_v2 import (
    ReachyHumanGazeboEnv as _V2ReachyHumanGazeboEnv,
)

from reachy_rl_control.path_relative_features import (
    compute_path_relative_features,
    world_vector_to_robot_frame,
)


class ReachyHumanGazeboEnv(
    _V2ReachyHumanGazeboEnv
):
    """
    V2.1 path-relative observation.

    Observation remains 22-dimensional:

        0   vx
        1   vy
        2   wz

        3   current reward waypoint local x
        4   current reward waypoint local y

        5   look-ahead local x
        6   look-ahead local y

        7   cross-track error
        8   heading error
        9   remaining path length capped

        10-21 LiDAR
    """

    def _get_observation(
        self,
        state=None,
    ) -> np.ndarray:

        if state is None:
            state = self.robot.get_state()

        if not self.world_waypoints:
            raise RuntimeError(
                "V2.1 observation requested before "
                "world trajectory was created."
            )

        # -------------------------------------------------
        # EXACT FROZEN REWARD TARGET
        # -------------------------------------------------
        #
        # The frozen reward/environment uses this same
        # current_waypoint_index.
        #
        # No 0.25 m resampling or alternate target is used.
        # -------------------------------------------------

        goal_x, goal_y = (
            self.world_waypoints[
                self.current_waypoint_index
            ]
        )

        reward_target_local = (
            world_vector_to_robot_frame(
                goal_x - state.x,
                goal_y - state.y,
                state.yaw,
            )
        )

        # -------------------------------------------------
        # PATH TRACKING FEATURES
        #
        # IMPORTANT:
        # self.world_waypoints is passed directly.
        #
        # This is the exact polyline used by the frozen
        # reward environment. It is NOT resampled.
        #
        # The helper interpolates only to obtain the
        # 0.8 m look-ahead point along that same polyline.
        # -------------------------------------------------

        (
            path_features,
            metadata,
        ) = compute_path_relative_features(
            robot_x=state.x,
            robot_y=state.y,
            robot_yaw=state.yaw,
            path=self.world_waypoints,
        )

        # compute_path_relative_features layout:
        #
        # 0 interpolated next x     <- intentionally ignored
        # 1 interpolated next y     <- intentionally ignored
        # 2 lookahead x
        # 3 lookahead y
        # 4 cross-track error
        # 5 heading error
        # 6 remaining distance capped

        lidar_observation = (
            self.lidar.get_normalized_distances()
        )

        observation = np.concatenate(
            (
                np.asarray(
                    [
                        state.vx,
                        state.vy,
                        state.wz,

                        reward_target_local[0],
                        reward_target_local[1],

                        path_features[2],
                        path_features[3],

                        path_features[4],
                        path_features[5],
                        path_features[6],
                    ],
                    dtype=np.float32,
                ),

                np.asarray(
                    lidar_observation,
                    dtype=np.float32,
                ),
            )
        ).astype(
            np.float32,
            copy=False,
        )

        if observation.shape != (22,):
            raise RuntimeError(
                "V2.1 observation must have "
                f"shape (22,), got "
                f"{observation.shape}."
            )

        # Preserve V2 progress metadata and add explicit
        # reward-target information for verification.

        metadata[
            "reward_waypoint_index"
        ] = int(
            self.current_waypoint_index
        )

        metadata[
            "reward_waypoint_world"
        ] = (
            float(goal_x),
            float(goal_y),
        )

        metadata[
            "reward_waypoint_local"
        ] = (
            float(reward_target_local[0]),
            float(reward_target_local[1]),
        )

        self.v2_path_metadata = metadata

        return observation
