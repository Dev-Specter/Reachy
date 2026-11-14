#!/usr/bin/env python3

import math

import gymnasium as gym
import numpy as np

from reachy_rl_control.environment_human import (
    SAFE_DISTANCE,
)


class LidarEmergencyStopWrapper(gym.Wrapper):
    """
    Fixed runtime LiDAR safety filter.

    The policy proposes its normal normalized action.

    If the current processed LiDAR clearance is below
    SAFE_DISTANCE, the whole action is replaced by zero for
    that control cycle.

    This wrapper does NOT:
      - retrain the policy
      - change observations
      - change rewards
      - change termination logic
      - change environment thresholds
    """

    def __init__(
        self,
        env,
        stop_distance=SAFE_DISTANCE,
    ):
        super().__init__(env)

        self.stop_distance = float(
            stop_distance
        )

        self.episode_interventions = 0
        self.total_interventions = 0

        print(
            "\nLiDAR emergency-stop safety filter enabled."
        )

        print(
            "  stop distance:",
            self.stop_distance,
            "m",
        )

        print(
            "  intervention action: [0, 0, 0]"
        )

    @property
    def world_waypoints(self):
        return self.env.unwrapped.world_waypoints

    def reset(
        self,
        **kwargs,
    ):
        self.episode_interventions = 0

        return self.env.reset(
            **kwargs
        )

    def step(
        self,
        action,
    ):
        base_env = self.env.unwrapped

        if not hasattr(
            base_env,
            "lidar",
        ):
            raise RuntimeError(
                "Safety filter could not access "
                "the environment LiDAR processor."
            )

        clearance_before = float(
            base_env.lidar.get_min_distance()
        )

        if not math.isfinite(
            clearance_before
        ):
            raise RuntimeError(
                "Safety filter received non-finite "
                f"LiDAR clearance: {clearance_before}"
            )

        proposed_action = np.asarray(
            action,
            dtype=np.float32,
        ).copy()

        triggered = bool(
            clearance_before
            < self.stop_distance
        )

        if triggered:
            applied_action = np.zeros_like(
                proposed_action,
                dtype=np.float32,
            )

            self.episode_interventions += 1
            self.total_interventions += 1

        else:
            applied_action = proposed_action

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            applied_action
        )

        info = dict(info)

        info[
            "safety_filter_triggered"
        ] = triggered

        info[
            "safety_filter_clearance_before"
        ] = clearance_before

        info[
            "episode_safety_filter_interventions"
        ] = self.episode_interventions

        info[
            "total_safety_filter_interventions"
        ] = self.total_interventions

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
