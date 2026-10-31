#!/usr/bin/env python3

"""
UR-MBPO callback for the V2 path-relative observation.

The frozen V1 implementation in ur_mbpo_dynamics.py is not
modified.

V2 observation layout
---------------------

Observation dimension = 22

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

    10-21 normalized LiDAR sectors

The inherited uncertainty rejection, synthetic transition
generation, risk penalty and logging logic are unchanged.
"""

import numpy as np

from reachy_rl_control.ur_mbpo_dynamics import (
    URMBPOCallback as _BaseURMBPOCallback,
)


class URMBPOCallback(
    _BaseURMBPOCallback
):
    """
    V2 UR-MBPO callback.

    Only the observation interpretation and uncertainty
    normalization are changed.
    """

    def __init__(
        self,
        *args,
        lidar_start_index=10,
        **kwargs,
    ):

        super().__init__(
            *args,
            lidar_start_index=lidar_start_index,
            **kwargs,
        )

        # -------------------------------------------------
        # VALIDATE V2 OBSERVATION
        # -------------------------------------------------

        if self.obs_dim != 22:
            raise ValueError(
                "V2 UR-MBPO expects observation "
                f"dimension 22, received {self.obs_dim}."
            )

        if self.lidar_start_index != 10:
            raise ValueError(
                "V2 UR-MBPO expects LiDAR to start "
                f"at index 10, received "
                f"{self.lidar_start_index}."
            )

        if self.num_lidar_sectors != 12:
            raise ValueError(
                "V2 UR-MBPO expects 12 LiDAR sectors, "
                f"received {self.num_lidar_sectors}."
            )

        # -------------------------------------------------
        # V2 UNCERTAINTY NORMALISATION
        # -------------------------------------------------
        #
        # Indices 0-9 are the V2 non-LiDAR features.
        #
        # LiDAR indices 10-21 remain at scale 1.0 because
        # those values are already normalized to [0, 1].
        # -------------------------------------------------

        self.uncertainty_scales = np.ones(
            self.obs_dim,
            dtype=np.float32,
        )

        self.uncertainty_scales[:10] = np.array(
            [
                0.30,       # vx
                0.30,       # vy
                0.60,       # wz

                0.50,       # next waypoint local x
                0.50,       # next waypoint local y

                0.80,       # look-ahead local x
                0.80,       # look-ahead local y

                0.50,       # cross-track error

                np.pi,      # heading error

                2.00,       # capped remaining path length
            ],
            dtype=np.float32,
        )

    def _on_training_start(
        self,
    ):
        super()._on_training_start()

        print(
            "\n======================================"
        )
        print(
            "V2 PATH-RELATIVE UR-MBPO CONFIGURATION"
        )
        print(
            "======================================"
        )

        print(
            "Observation dimension:",
            self.obs_dim,
        )

        print(
            "LiDAR start index:",
            self.lidar_start_index,
        )

        print(
            "V2 uncertainty scales:"
        )

        print(
            self.uncertainty_scales
        )
