#!/usr/bin/env python3

import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import LaserScan


class LidarProcessor(Node):

    def __init__(
        self,
        num_sectors=12,
        max_distance=3.0,
    ):

        super().__init__("lidar_processor")

        self.num_sectors = int(
            num_sectors
        )

        self.max_distance = float(
            max_distance
        )

        # Reachy's own body permanently appears
        # in these 30-degree sectors.
        # Sector indexing is zero-based.
        self.self_masked_sectors = (
            5,
            6,
        )

        self.sector_distances = np.full(
            self.num_sectors,
            self.max_distance,
            dtype=np.float32,
        )

        self.scan_received = False

        self.subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self._scan_callback,
            10,
        )

        self.get_logger().info(
            "LiDAR processor started: "
            f"{self.num_sectors} sectors"
        )

    def _scan_callback(
        self,
        msg,
    ):

        ranges = np.asarray(
            msg.ranges,
            dtype=np.float32,
        )

        if ranges.size == 0:
            return

        # Replace NaN and infinity values with
        # the maximum useful sensing distance.
        ranges = np.where(
            np.isfinite(ranges),
            ranges,
            self.max_distance,
        )

        # Any value beyond our chosen useful
        # sensing distance is treated as max range.
        ranges = np.clip(
            ranges,
            float(msg.range_min),
            self.max_distance,
        )

        # Split the full 360-degree scan into
        # equal angular sectors.
        sectors = np.array_split(
            ranges,
            self.num_sectors,
        )

        distances = []

        for sector in sectors:

            if len(sector) == 0:

                distances.append(
                    self.max_distance
                )

            else:

                distances.append(
                    float(
                        np.min(sector)
                    )
                )

        # Remove permanent self-reflections caused
        # by Reachy's own body.
        for sector_index in (
            self.self_masked_sectors
        ):

            if (
                0
                <= sector_index
                < len(distances)
            ):

                distances[
                    sector_index
                ] = self.max_distance

        self.sector_distances = np.asarray(
            distances,
            dtype=np.float32,
        )

        self.scan_received = True

    def get_distances(self):

        return (
            self.sector_distances.copy()
        )

    def get_normalized_distances(self):

        normalized = (
            self.sector_distances
            / self.max_distance
        )

        return np.clip(
            normalized,
            0.0,
            1.0,
        ).astype(np.float32)

    def get_min_distance(self):

        return float(
            np.min(
                self.sector_distances
            )
        )

    def wait_for_scan(
        self,
        timeout=5.0,
    ):

        start_time = time.monotonic()

        while not self.scan_received:

            elapsed = (
                time.monotonic()
                - start_time
            )

            if elapsed > timeout:
                return False

            time.sleep(
                0.01
            )

        return True


def main(args=None):

    rclpy.init(
        args=args
    )

    node = LidarProcessor(
        num_sectors=12,
        max_distance=3.0,
    )

    executor = (
        SingleThreadedExecutor()
    )

    executor.add_node(
        node
    )

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )

    spin_thread.start()

    try:

        print(
            "\nWaiting for /scan..."
        )

        if not node.wait_for_scan(
            timeout=5.0
        ):

            raise RuntimeError(
                "No LiDAR scan received."
            )

        print(
            "LiDAR received.\n"
        )

        while rclpy.ok():

            distances = (
                node.get_distances()
            )

            normalized = (
                node.get_normalized_distances()
            )

            print(
                "LiDAR sectors (m):"
            )

            print(
                np.round(
                    distances,
                    2,
                )
            )

            print(
                "Normalized sectors:"
            )

            print(
                np.round(
                    normalized,
                    2,
                )
            )

            print(
                "Minimum distance:",
                f"{node.get_min_distance():.2f} m",
            )

            print()

            time.sleep(
                1.0
            )

    except KeyboardInterrupt:

        print(
            "\nLiDAR processor stopped."
        )

    finally:

        executor.shutdown()

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()

        spin_thread.join(
            timeout=1.0
        )


if __name__ == "__main__":
    main()
