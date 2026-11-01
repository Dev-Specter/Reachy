#!/usr/bin/env python3

import math

import gymnasium as gym
import numpy as np

from reachy_rl_control.environment_human import (
    SAFE_DISTANCE,
)


class RealRiskRewardWrapper(gym.Wrapper):
    """
    Add the UR-MBPO-style clearance risk penalty to REAL
    environment rewards only.

    This wrapper does not alter:
      - observations
      - actions
      - termination
      - the frozen human environment
      - MBPO synthetic transitions
      - the MBPO dynamics model

    Risk:
        clip(
            (SAFE_DISTANCE - clearance)
            / SAFE_DISTANCE,
            0,
            1,
        )

    Added penalty:
        risk_scale * risk

    Experiment 3 uses:
        risk_scale = 1.0
    """

    def __init__(
        self,
        env,
        risk_scale=1.0,
    ):
        super().__init__(env)

        self.risk_scale = float(
            risk_scale
        )

        if self.risk_scale < 0.0:
            raise ValueError(
                "risk_scale must be >= 0."
            )

        if float(SAFE_DISTANCE) <= 0.0:
            raise ValueError(
                "SAFE_DISTANCE must be > 0."
            )

        print(
            "\nReal-risk reward wrapper enabled."
        )

        print(
            "  risk scale:",
            self.risk_scale,
        )

        print(
            "  safe clearance:",
            float(SAFE_DISTANCE),
            "m",
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
        ) = self.env.step(
            action
        )

        info = dict(info)

        if "min_lidar_distance" not in info:
            raise RuntimeError(
                "RealRiskRewardWrapper expected "
                "'min_lidar_distance' in environment info."
            )

        clearance = float(
            info["min_lidar_distance"]
        )

        if not math.isfinite(clearance):
            raise RuntimeError(
                "Non-finite min_lidar_distance received: "
                f"{clearance}"
            )

        risk = float(
            np.clip(
                (
                    float(SAFE_DISTANCE)
                    - clearance
                )
                / float(SAFE_DISTANCE),
                0.0,
                1.0,
            )
        )

        risk_penalty = (
            self.risk_scale
            * risk
        )

        base_reward = float(
            reward
        )

        modified_reward = (
            base_reward
            - risk_penalty
        )

        # Logging only.
        info[
            "base_environment_reward"
        ] = base_reward

        info[
            "real_risk_value"
        ] = risk

        info[
            "real_risk_penalty"
        ] = risk_penalty

        info[
            "reward_after_real_risk"
        ] = modified_reward

        return (
            observation,
            modified_reward,
            terminated,
            truncated,
            info,
        )
