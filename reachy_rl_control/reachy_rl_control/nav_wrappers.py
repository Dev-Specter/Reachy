#!/usr/bin/env python3
"""
Wrappers for the V3 navigation environment (environment_nav.ReachyNavEnv).

They mirror the existing frozen wrappers (risk_reward_wrapper.py and
safety_filter_wrapper.py), but use the V3 thresholds, which are EDGE
clearances identical in every world, and a bounded per-step risk penalty.

NavRiskRewardWrapper
    Adds the UR-MBPO-style preference for more room to REAL rewards:
        risk    = clip((PREFERRED_CLEARANCE - clearance) / PREFERRED_CLEARANCE, 0, 1)
        penalty = risk_penalty_max * risk          (<= 0.06 per step by default)
    At construction it verifies that, together with the base per-step costs,
    finishing an episode still always beats crashing (see
    environment_nav.check_no_crash_incentive).

NavEmergencyStopWrapper
    Fixed runtime safety filter: when the current edge clearance is below
    stop_distance (default SAFE_DISTANCE = 0.10 m) the action is replaced by
    zero velocity for that step.
"""

import math

import gymnasium as gym
import numpy as np

from reachy_rl_control.environment_nav import (
    PREFERRED_CLEARANCE,
    RISK_PENALTY_MAX,
    SAFE_DISTANCE,
    check_no_crash_incentive,
)


class NavRiskRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        preferred_clearance: float = PREFERRED_CLEARANCE,
        risk_penalty_max: float = RISK_PENALTY_MAX,
    ):
        super().__init__(env)
        self.preferred_clearance = float(preferred_clearance)
        self.risk_penalty_max = float(risk_penalty_max)
        if self.preferred_clearance <= 0.0 or self.risk_penalty_max < 0.0:
            raise ValueError("preferred_clearance must be > 0 and risk_penalty_max >= 0.")

        worst, margin = check_no_crash_incentive(
            env.unwrapped.max_steps, extra_per_step=self.risk_penalty_max
        )
        print(
            f"\nNav risk wrapper: preferred clearance {self.preferred_clearance:.2f} m, "
            f"max {self.risk_penalty_max:.3f}/step | worst case {worst:.1f} < {margin:.1f}"
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        clearance = float(info["min_lidar_distance"])
        if not math.isfinite(clearance):
            raise RuntimeError(f"Non-finite min_lidar_distance: {clearance}")

        risk = float(np.clip(
            (self.preferred_clearance - clearance) / self.preferred_clearance, 0.0, 1.0
        ))
        risk_penalty = self.risk_penalty_max * risk
        modified = float(reward) - risk_penalty

        info["base_environment_reward"] = float(reward)
        info["real_risk_value"] = risk
        info["real_risk_penalty"] = risk_penalty
        info["reward_after_real_risk"] = modified

        return observation, modified, terminated, truncated, info


class NavEmergencyStopWrapper(gym.Wrapper):

    def __init__(self, env, stop_distance: float = SAFE_DISTANCE):
        super().__init__(env)
        self.stop_distance = float(stop_distance)
        self.episode_interventions = 0
        self.total_interventions = 0
        print(f"\nNav emergency-stop filter: stop below {self.stop_distance:.2f} m edge clearance")

    @property
    def world_waypoints(self):
        return self.env.unwrapped.world_waypoints

    def reset(self, **kwargs):
        self.episode_interventions = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        clearance_before = float(self.env.unwrapped.lidar.get_min_distance())
        if not math.isfinite(clearance_before):
            raise RuntimeError(f"Non-finite LiDAR clearance: {clearance_before}")

        proposed = np.asarray(action, dtype=np.float32).copy()
        triggered = clearance_before < self.stop_distance
        if triggered:
            applied = np.zeros_like(proposed)
            self.episode_interventions += 1
            self.total_interventions += 1
        else:
            applied = proposed

        observation, reward, terminated, truncated, info = self.env.step(applied)
        info = dict(info)
        info["safety_filter_triggered"] = bool(triggered)
        info["safety_filter_clearance_before"] = clearance_before
        info["episode_safety_filter_interventions"] = self.episode_interventions
        info["total_safety_filter_interventions"] = self.total_interventions
        return observation, reward, terminated, truncated, info
