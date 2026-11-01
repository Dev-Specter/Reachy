#!/usr/bin/env python3

import numpy as np

from reachy_rl_control.environment_v2 import (
    ReachyGazeboEnv as ReachyGazeboEnvV2,
)

from reachy_rl_control.path_relative_features import (
    compute_path_relative_features,
    world_vector_to_robot_frame,
)


class ReachyGazeboEnv(ReachyGazeboEnvV2):
    """
    No-human V2.1 environment.

    V2.1 keeps the V2 path-relative observation but replaces
    the interpolated immediate target with the exact waypoint
    currently used by the frozen reward environment.

    Observation layout:

        0   vx
        1   vy
        2   wz

        3   exact reward waypoint local x
        4   exact reward waypoint local y

        5   look-ahead local x
        6   look-ahead local y

        7   cross-track error
        8   heading error
        9   remaining path length capped

    Progress ratio remains evaluation metadata only.
    """

    def _get_observation(
        self,
        state=None,
    ) -> np.ndarray:

        if state is None:
            state = self.robot.get_state()

        if state is None:
            raise RuntimeError(
                "V2.1 observation requested without "
                "a valid robot state."
            )

        if not self.world_waypoints:
            raise RuntimeError(
                "V2.1 observation requested before "
                "world trajectory was created."
            )

        # -------------------------------------------------
        # EXACT FROZEN REWARD TARGET
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
        # REMAINING PATH-RELATIVE FEATURES
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

        # path_features:
        #
        # 0 interpolated immediate x  <- ignored
        # 1 interpolated immediate y  <- ignored
        # 2 lookahead x
        # 3 lookahead y
        # 4 cross-track error
        # 5 heading error
        # 6 remaining distance capped

        observation = np.asarray(
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
        )

        if observation.shape != (10,):
            raise RuntimeError(
                "No-human V2.1 observation must "
                f"have shape (10,), got "
                f"{observation.shape}."
            )

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
