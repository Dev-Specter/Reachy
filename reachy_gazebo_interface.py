#!/usr/bin/env python3

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


@dataclass(frozen=True)
class BaseState:
    """State of the Reachy mobile base obtained from /odom."""

    x: float
    y: float
    yaw: float

    vx: float
    vy: float
    wz: float


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion orientation to yaw angle in radians."""

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)


def angle_difference(a: float, b: float) -> float:
    """Return wrapped angle difference a - b in [-pi, pi]."""

    return math.atan2(math.sin(a - b), math.cos(a - b))


class ReachyGazeboInterface(Node):
    """
    Interface between Python control code and Reachy's Gazebo mobile base.

    Publishes:
        /cmd_vel    geometry_msgs/msg/Twist

    Subscribes:
        /odom       nav_msgs/msg/Odometry
    """

    def __init__(
        self,
        max_vxy: float = 0.30,
        max_wz: float = 0.60,
    ):
        super().__init__("reachy_gazebo_interface")

        # Conservative software limits for our development/testing.
        # These are project limits, not claimed hardware limits.
        self.max_vxy = max_vxy
        self.max_wz = max_wz

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_callback,
            10,
        )

        self._state_lock = threading.Lock()
        self._latest_state: Optional[BaseState] = None

        self.get_logger().info("Reachy Gazebo interface started.")
        self.get_logger().info("Publishing commands to /cmd_vel")
        self.get_logger().info("Reading odometry from /odom")

    def _odom_callback(self, msg: Odometry) -> None:
        """Store the latest odometry message."""

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist

        yaw = quaternion_to_yaw(
            q.x,
            q.y,
            q.z,
            q.w,
        )

        state = BaseState(
            x=p.x,
            y=p.y,
            yaw=yaw,
            vx=v.linear.x,
            vy=v.linear.y,
            wz=v.angular.z,
        )

        with self._state_lock:
            self._latest_state = state

    def get_state(self) -> Optional[BaseState]:
        """Return the newest mobile-base state."""

        with self._state_lock:
            return self._latest_state

    def wait_for_odometry(self, timeout: float = 5.0) -> bool:
        """Wait until at least one /odom message has been received."""

        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            if self.get_state() is not None:
                return True

            time.sleep(0.05)

        return False

    def send_velocity(
        self,
        vx: float,
        vy: float,
        omega: float,
    ) -> None:
        """
        Send planar velocity command.

        vx     : forward/backward velocity
        vy     : lateral velocity
        omega  : yaw angular velocity, rad/s
        """

        vx = max(-self.max_vxy, min(self.max_vxy, vx))
        vy = max(-self.max_vxy, min(self.max_vxy, vy))
        omega = max(-self.max_wz, min(self.max_wz, omega))

        msg = Twist()

        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(omega)

        self.cmd_vel_pub.publish(msg)

    def stop(self) -> None:
        """Explicitly send zero velocity to the base."""

        self.send_velocity(
            vx=0.0,
            vy=0.0,
            omega=0.0,
        )

    def command_for(
        self,
        vx: float,
        vy: float,
        omega: float,
        duration: float,
        frequency: float = 20.0,
    ) -> None:
        """
        Publish a velocity command repeatedly for a fixed duration.

        A zero command is always sent afterwards, even if Ctrl+C occurs.
        """

        period = 1.0 / frequency
        start_time = time.monotonic()

        try:
            while time.monotonic() - start_time < duration:
                self.send_velocity(vx, vy, omega)
                time.sleep(period)

        finally:
            # Send zero several times to make stopping reliable.
            for _ in range(5):
                self.stop()
                time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(
        description="Test the Reachy Gazebo mobile-base interface."
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=0.0,
        help="Forward velocity in m/s",
    )

    parser.add_argument(
        "--vy",
        type=float,
        default=0.0,
        help="Lateral velocity in m/s",
    )

    parser.add_argument(
        "--omega",
        type=float,
        default=0.0,
        help="Yaw velocity in rad/s",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Command duration in seconds",
    )

    args = parser.parse_args()

    rclpy.init()

    robot = ReachyGazeboInterface()

    executor = SingleThreadedExecutor()
    executor.add_node(robot)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )

    spin_thread.start()

    try:
        print("\nWaiting for /odom...")

        if not robot.wait_for_odometry(timeout=5.0):
            raise RuntimeError(
                "No /odom data received. Is Gazebo running?"
            )

        initial_state = robot.get_state()

        print("\nInitial state:")
        print(initial_state)

        print(
            f"\nCommanding:"
            f" vx={args.vx:.3f},"
            f" vy={args.vy:.3f},"
            f" omega={args.omega:.3f}"
            f" for {args.duration:.1f} seconds"
        )

        robot.command_for(
            vx=args.vx,
            vy=args.vy,
            omega=args.omega,
            duration=args.duration,
        )

        # Give odometry a moment to update after stopping.
        time.sleep(0.2)

        final_state = robot.get_state()

        print("\nFinal state:")
        print(final_state)

        if initial_state is not None and final_state is not None:
            dx = final_state.x - initial_state.x
            dy = final_state.y - initial_state.y
            dyaw = angle_difference(
                final_state.yaw,
                initial_state.yaw,
            )

            distance = math.hypot(dx, dy)

            print("\nMeasured change:")
            print(f"  dx       = {dx:.4f} m")
            print(f"  dy       = {dy:.4f} m")
            print(f"  distance = {distance:.4f} m")
            print(f"  d_yaw    = {dyaw:.4f} rad")
            print(
                f"  d_yaw    = {math.degrees(dyaw):.2f} degrees"
            )

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        print("\nStopping Reachy...")

        for _ in range(5):
            robot.stop()
            time.sleep(0.02)

        executor.shutdown()
        robot.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(timeout=1.0)

        print("Interface closed safely.")


if __name__ == "__main__":
    main()
