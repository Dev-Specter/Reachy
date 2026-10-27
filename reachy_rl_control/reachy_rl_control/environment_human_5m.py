#!/usr/bin/env python3

"""
5 m straight-path human-aware Reachy environment.

This module deliberately reuses the frozen 2 m human-aware
environment and changes only the reference trajectory.

Everything else remains inherited unchanged from:

    reachy_rl_control.environment_human

This preserves the existing:

- 20-dimensional observation space
- 3-dimensional action space
- action scaling
- reward function
- human crossing behaviour
- LiDAR processing
- collision detection
- near-miss detection
- fall detection
- reset validation
- safety thresholds
- termination logic
- Gazebo interface
- control timing

The only experimental change is:

    2 m straight trajectory
        ->
    5 m straight trajectory
"""


import math

from reachy_rl_control.environment_human import (
    ReachyHumanGazeboEnv as _BaseReachyHumanGazeboEnv,
)


# ============================================================
# 5 M STRAIGHT REFERENCE PATH
# ============================================================
#
# Same 0.5 m waypoint spacing as the frozen 2 m experiment.
#
# Local coordinates are relative to Reachy's stable reset pose.
#
# The trajectory is transformed into the odometry/world frame
# using Reachy's initial x, y and yaw, exactly as in the frozen
# human-aware environment.
# ============================================================

STRAIGHT_PATH = [
    (0.0, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.5, 0.0),
    (2.0, 0.0),
    (2.5, 0.0),
    (3.0, 0.0),
    (3.5, 0.0),
    (4.0, 0.0),
    (4.5, 0.0),
    (5.0, 0.0),
]


# ============================================================
# 5 M HUMAN-AWARE ENVIRONMENT
# ============================================================

class ReachyHumanGazeboEnv(
    _BaseReachyHumanGazeboEnv
):
    """
    Human-aware Reachy Gazebo environment using a 5 m
    straight trajectory.

    All environment behaviour is inherited from the frozen
    2 m ReachyHumanGazeboEnv.

    Only `_create_world_trajectory()` is overridden so that
    the reference trajectory extends to 5 metres.
    """

    def _create_world_trajectory(
        self,
        start_x,
        start_y,
        start_yaw,
    ):
        """
        Transform the local 5 m straight trajectory into
        the odometry/world coordinate frame.

        Local waypoint:

            (local_x, local_y)

        is transformed using Reachy's stable reset pose:

            world_x =
                start_x
                + cos(start_yaw) * local_x
                - sin(start_yaw) * local_y

            world_y =
                start_y
                + sin(start_yaw) * local_x
                + cos(start_yaw) * local_y

        This keeps the trajectory relative to Reachy's
        actual starting pose rather than assuming that
        Gazebo always resets exactly at (0, 0, 0).
        """

        cos_yaw = math.cos(
            start_yaw
        )

        sin_yaw = math.sin(
            start_yaw
        )

        world_trajectory = []

        for (
            local_x,
            local_y,
        ) in STRAIGHT_PATH:

            world_x = (
                start_x
                +
                cos_yaw * local_x
                -
                sin_yaw * local_y
            )

            world_y = (
                start_y
                +
                sin_yaw * local_x
                +
                cos_yaw * local_y
            )

            world_trajectory.append(
                (
                    float(world_x),
                    float(world_y),
                )
            )

        return world_trajectory
