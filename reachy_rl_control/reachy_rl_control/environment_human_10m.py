#!/usr/bin/env python3

"""
10 m straight-path human-aware Reachy environment.

This module reuses the frozen human-aware Reachy environment
and changes only the reference trajectory.

Everything else remains inherited unchanged from:

    reachy_rl_control.environment_human

Therefore the following remain identical to the original
2 m human-aware experiment:

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

The only intentional experimental change is:

    2 m straight trajectory
        ->
    10 m straight trajectory
"""


import math

from reachy_rl_control.environment_human import (
    ReachyHumanGazeboEnv as _BaseReachyHumanGazeboEnv,
)


# ============================================================
# 10 M STRAIGHT REFERENCE PATH
# ============================================================
#
# Same 0.5 m waypoint spacing used in the frozen 2 m
# experiment and the 5 m generalisation experiment.
#
# Local coordinates are relative to Reachy's stable reset
# pose.
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
    (5.5, 0.0),
    (6.0, 0.0),
    (6.5, 0.0),
    (7.0, 0.0),
    (7.5, 0.0),
    (8.0, 0.0),
    (8.5, 0.0),
    (9.0, 0.0),
    (9.5, 0.0),
    (10.0, 0.0),
]


# ============================================================
# 10 M HUMAN-AWARE ENVIRONMENT
# ============================================================

class ReachyHumanGazeboEnv(
    _BaseReachyHumanGazeboEnv
):
    """
    Human-aware Reachy Gazebo environment using a 10 m
    straight reference trajectory.

    All behaviour is inherited from the frozen 2 m
    ReachyHumanGazeboEnv.

    Only `_create_world_trajectory()` is overridden.
    """

    def _create_world_trajectory(
        self,
        start_x,
        start_y,
        start_yaw,
    ):
        """
        Transform the local 10 m trajectory into the
        odometry/world coordinate frame.

        This uses Reachy's actual stable reset position and
        orientation, so the trajectory remains relative to
        Reachy's starting pose.
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
