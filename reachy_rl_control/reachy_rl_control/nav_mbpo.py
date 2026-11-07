#!/usr/bin/env python3
"""
MBPO and UR-MBPO callbacks for the V3 navigation environment.

NavMBPOCallback    = mbpo_dynamics.MBPOCallback         + collision-terminal fix
NavURMBPOCallback  = ur_mbpo_dynamics.URMBPOCallback    + collision-terminal fix
                                                        + V3 observation layout
                                                        + V3 clearance definitions

Collision-terminal fix
    frozen code : model_done = reason in ("goal", "fall")
    V3          : model_done = reason in ("goal", "collision", "fall")
In V3 a collision ends the real episode, so the dynamics model must learn
that it is terminal; otherwise synthetic transitions continue "through"
obstacles. Timeouts remain non-terminal (artificial horizon). The frozen
files are unchanged so earlier results stay reproducible.

V3 UR-MBPO configuration (matches the real-environment definitions)
    LiDAR block      : indices 7-30 (24 sectors), normalised by 3.0 m; the
                       minimum predicted value x 3.0 IS the edge clearance
    safe_clearance   : 0.20 m  (PREFERRED_CLEARANCE)
    near_miss        : 0.05 m  (NEAR_MISS_DISTANCE)
    risk_penalty     : 0.06 per step at most (RISK_PENALTY_MAX)
    Risk shaping is applied to SYNTHETIC transitions only, as in the earlier
    UR-MBPO experiments; real rewards are the plain environment reward.
"""

import numpy as np

from reachy_rl_control.environment_nav import (
    GOAL_DISTANCE_SCALE,
    LIDAR_MAX_DISTANCE,
    LIDAR_START_INDEX,
    NEAR_MISS_DISTANCE,
    NUM_LIDAR_SECTORS,
    OBS_DIM,
    PREFERRED_CLEARANCE,
    RISK_PENALTY_MAX,
    V_MAX,
    W_MAX,
)
from reachy_rl_control.mbpo_dynamics import MBPOCallback
from reachy_rl_control.ur_mbpo_dynamics import URMBPOCallback

TERMINAL_REASONS = ("goal", "collision", "fall")


class CollisionTerminalMixin:
    """Replaces _store_real_transition so collisions are terminal for the model."""

    def _store_real_transition(self):
        actions = np.asarray(self.locals["actions"], dtype=np.float32)
        new_obs = np.asarray(self.locals["new_obs"], dtype=np.float32)
        rewards = np.asarray(self.locals["rewards"], dtype=np.float32)
        dones = np.asarray(self.locals["dones"])
        info = self.locals["infos"][0]

        if bool(dones[0]) and "terminal_observation" in info:
            next_obs = np.asarray(info["terminal_observation"], dtype=np.float32).copy()
        else:
            next_obs = new_obs[0].copy()

        model_done = info.get("termination_reason", "") in TERMINAL_REASONS

        self.real_buffer.append((
            self.previous_obs.copy(),
            actions[0].copy(),
            float(rewards[0]),
            next_obs,
            float(model_done),
        ))

        # After an automatic VecEnv reset, new_obs already holds the reset observation.
        self.previous_obs = new_obs[0].copy()


class NavMBPOCallback(CollisionTerminalMixin, MBPOCallback):
    pass


class NavURMBPOCallback(CollisionTerminalMixin, URMBPOCallback):

    def __init__(
        self,
        *args,
        uncertainty_threshold: float = 0.05,
        uncertainty_penalty_scale: float = 2.0,
        safe_clearance: float = PREFERRED_CLEARANCE,
        near_miss_clearance: float = NEAR_MISS_DISTANCE,
        lidar_max_distance: float = LIDAR_MAX_DISTANCE,
        risk_penalty_scale: float = RISK_PENALTY_MAX,
        lidar_start_index: int = LIDAR_START_INDEX,
        **kwargs,
    ):
        super().__init__(
            *args,
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_penalty_scale=uncertainty_penalty_scale,
            safe_clearance=safe_clearance,
            near_miss_clearance=near_miss_clearance,
            lidar_max_distance=lidar_max_distance,
            risk_penalty_scale=risk_penalty_scale,
            lidar_start_index=lidar_start_index,
            **kwargs,
        )

        if self.obs_dim != OBS_DIM:
            raise ValueError(f"V3 UR-MBPO expects observation dimension {OBS_DIM}, got {self.obs_dim}.")
        if self.lidar_start_index != LIDAR_START_INDEX:
            raise ValueError(f"V3 UR-MBPO expects LiDAR at index {LIDAR_START_INDEX}.")
        if self.num_lidar_sectors != NUM_LIDAR_SECTORS:
            raise ValueError(f"V3 UR-MBPO expects {NUM_LIDAR_SECTORS} LiDAR sectors.")

        # Typical ranges of the 7 non-LiDAR features, used to normalise
        # ensemble disagreement before averaging across dimensions.
        self.uncertainty_scales = np.ones(self.obs_dim, dtype=np.float32)
        self.uncertainty_scales[:LIDAR_START_INDEX] = np.array([
            V_MAX,                          # 0 vx (m/s)
            W_MAX,                          # 1 wz (rad/s)
            1.0 / GOAL_DISTANCE_SCALE,      # 2 goal distance / 10 m  -> 1 m
            1.0,                            # 3 sin(goal bearing)
            1.0,                            # 4 cos(goal bearing)
            1.0,                            # 5 previous v command (normalised)
            1.0,                            # 6 previous w command (normalised)
        ], dtype=np.float32)
        # LiDAR sectors keep scale 1.0 (already in [0, 1]).

    def _on_training_start(self):
        super()._on_training_start()
        worst_unc = self.uncertainty_penalty_scale * self.uncertainty_threshold
        print("\n======================================")
        print("V3 NAV UR-MBPO CONFIGURATION")
        print("======================================")
        print(f"Observation {self.obs_dim} | LiDAR index {self.lidar_start_index} "
              f"| sectors {self.num_lidar_sectors}")
        print(f"Synthetic risk: preferred clearance {self.safe_clearance:.2f} m, "
              f"max {self.risk_penalty_scale:.3f}/step")
        print(f"Synthetic uncertainty penalty: max {worst_unc:.3f}/step on accepted transitions")
        print("Collisions are terminal for the dynamics model.")
