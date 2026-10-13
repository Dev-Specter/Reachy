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
from std_srvs.srv import Empty


# ============================================================
# Robot state
# ============================================================

@dataclass(frozen=True)
class BaseState:
    """State of the Reachy mobile base obtained from /odom."""

    x: float
    y: float
    yaw: float

    vx: float
    vy: float
    wz: float


# ============================================================
# Helper functions
# ============================================================

def quaternion_to_yaw(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:
    """Convert quaternion orientation to yaw angle in radians."""

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(
        siny_cosp,
        cosy_cosp,
    )


def angle_difference(
    a: float,
    b: float,
) -> float:
    """
    Return wrapped angular difference a - b
    in the range [-pi, pi].
    """

    return math.atan2(
        math.sin(a - b),
        math.cos(a - b),
    )


# ============================================================
# Reachy Gazebo interface
# ============================================================

class ReachyGazeboInterface(Node):
    """
    ROS 2 interface for Reachy's Gazebo mobile base.

    Publishes:
        /cmd_vel
            geometry_msgs/msg/Twist

    Subscribes:
        /odom
            nav_msgs/msg/Odometry

    Gazebo service:
        /reset_world
            std_srvs/srv/Empty
    """

    def __init__(
        self,
        max_vxy: float = 0.30,
        max_wz: float = 0.60,
        reset_service: str = "/reset_world",
    ):
        super().__init__(
            "reachy_gazebo_interface"
        )

        # ----------------------------------------------------
        # Software velocity limits
        # ----------------------------------------------------

        self.max_vxy = max_vxy
        self.max_wz = max_wz

        # ----------------------------------------------------
        # /cmd_vel publisher
        # ----------------------------------------------------

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        # ----------------------------------------------------
        # /odom subscriber
        # ----------------------------------------------------

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_callback,
            10,
        )

        # ----------------------------------------------------
        # Gazebo reset service
        # ----------------------------------------------------

        self.reset_service_name = reset_service

        self.reset_client = self.create_client(
            Empty,
            self.reset_service_name,
        )

        # ----------------------------------------------------
        # State storage
        # ----------------------------------------------------

        self._state_lock = threading.Lock()

        self._latest_state: Optional[BaseState] = None

        self._odom_sequence = 0

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        self.get_logger().info(
            "Reachy Gazebo interface started."
        )

        self.get_logger().info(
            "Publishing velocity commands to /cmd_vel"
        )

        self.get_logger().info(
            "Reading odometry from /odom"
        )

        self.get_logger().info(
            f"Reset service: {self.reset_service_name}"
        )

    # ========================================================
    # Odometry
    # ========================================================

    def _odom_callback(
        self,
        msg: Odometry,
    ) -> None:
        """Store the latest odometry state."""

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        velocity = msg.twist.twist

        yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )

        state = BaseState(
            x=position.x,
            y=position.y,
            yaw=yaw,

            vx=velocity.linear.x,
            vy=velocity.linear.y,
            wz=velocity.angular.z,
        )

        with self._state_lock:

            self._latest_state = state

            self._odom_sequence += 1

    def get_state(
        self,
    ) -> Optional[BaseState]:
        """Return the most recent odometry state."""

        with self._state_lock:

            return self._latest_state

    def get_odom_sequence(
        self,
    ) -> int:
        """Return number of received odometry messages."""

        with self._state_lock:

            return self._odom_sequence

    def wait_for_odometry(
        self,
        timeout: float = 5.0,
    ) -> bool:
        """
        Wait until at least one /odom message is received.
        """

        start_time = time.monotonic()

        while (
            time.monotonic() - start_time
            < timeout
        ):

            if self.get_state() is not None:

                return True

            time.sleep(0.05)

        return False

    # ========================================================
    # Velocity commands
    # ========================================================

    def send_velocity(
        self,
        vx: float,
        vy: float,
        omega: float,
    ) -> None:
        """
        Publish one mobile-base velocity command.

        vx:
            Forward/backward velocity in m/s.

        vy:
            Sideways velocity in m/s.

        omega:
            Yaw angular velocity in rad/s.
        """

        # Clamp commands to safe development limits.

        vx = max(
            -self.max_vxy,
            min(
                self.max_vxy,
                vx,
            ),
        )

        vy = max(
            -self.max_vxy,
            min(
                self.max_vxy,
                vy,
            ),
        )

        omega = max(
            -self.max_wz,
            min(
                self.max_wz,
                omega,
            ),
        )

        msg = Twist()

        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(omega)

        self.cmd_vel_pub.publish(msg)

    def stop(
        self,
    ) -> None:
        """
        Publish one zero-velocity command.
        """

        self.send_velocity(
            vx=0.0,
            vy=0.0,
            omega=0.0,
        )

    def stop_for(
        self,
        duration: float = 1.0,
        frequency: float = 20.0,
    ) -> None:
        """
        Repeatedly publish zero velocity.

        Reachy's Gazebo controller keeps the previous
        velocity command, so sending zero repeatedly
        provides reliable stopping.
        """

        if frequency <= 0.0:

            raise ValueError(
                "frequency must be greater than zero"
            )

        if duration < 0.0:

            raise ValueError(
                "duration cannot be negative"
            )

        period = 1.0 / frequency

        start_time = time.monotonic()

        while (
            time.monotonic() - start_time
            < duration
        ):

            self.stop()

            time.sleep(period)

    def command_for(
        self,
        vx: float,
        vy: float,
        omega: float,
        duration: float,
        frequency: float = 20.0,
    ) -> None:
        """
        Publish a velocity command for a fixed duration.

        When movement finishes, zero velocity is
        published repeatedly for one second.
        """

        if frequency <= 0.0:

            raise ValueError(
                "frequency must be greater than zero"
            )

        if duration < 0.0:

            raise ValueError(
                "duration cannot be negative"
            )

        period = 1.0 / frequency

        start_time = time.monotonic()

        try:

            while (
                time.monotonic() - start_time
                < duration
            ):

                self.send_velocity(
                    vx=vx,
                    vy=vy,
                    omega=omega,
                )

                time.sleep(period)

        finally:

            # Important:
            # Reachy's controller retains its last velocity.
            # Therefore explicitly command zero for 1 second.

            self.stop_for(
                duration=1.0,
                frequency=20.0,
            )

    # ========================================================
    # Gazebo reset
    # ========================================================

    def reset_gazebo(
        self,
        timeout: float = 5.0,
        settle_time: float = 0.5,
    ) -> BaseState:
        """
        Reset Gazebo and return fresh odometry.

        The ROS executor must be spinning in another
        thread while this method is executing.
        """

        self.get_logger().info(
            f"Resetting Gazebo using "
            f"{self.reset_service_name}"
        )

        # ----------------------------------------------------
        # Stop robot before resetting
        # ----------------------------------------------------

        self.stop_for(
            duration=0.5,
            frequency=20.0,
        )

        # ----------------------------------------------------
        # Wait for reset service
        # ----------------------------------------------------

        if not self.reset_client.wait_for_service(
            timeout_sec=timeout
        ):

            raise RuntimeError(
                f"Gazebo reset service "
                f"{self.reset_service_name} "
                f"is unavailable."
            )

        # ----------------------------------------------------
        # Send reset request
        # ----------------------------------------------------

        request = Empty.Request()

        future = self.reset_client.call_async(
            request
        )

        deadline = (
            time.monotonic()
            + timeout
        )

        # Background executor processes the response.

        while not future.done():

            if time.monotonic() >= deadline:

                raise TimeoutError(
                    "Timed out waiting for "
                    "Gazebo reset service."
                )

            time.sleep(0.01)

        # ----------------------------------------------------
        # Check reset result
        # ----------------------------------------------------

        exception = future.exception()

        if exception is not None:

            raise RuntimeError(
                f"Gazebo reset failed: "
                f"{exception}"
            )

        self.get_logger().info(
            "Gazebo reset service completed."
        )

        # Controller may remember an earlier velocity,
        # therefore send zero again after reset.

        self.stop_for(
            duration=0.5,
            frequency=20.0,
        )

        # ----------------------------------------------------
        # Wait for fresh odometry after reset
        # ----------------------------------------------------

        sequence_after_reset = (
            self.get_odom_sequence()
        )

        deadline = (
            time.monotonic()
            + timeout
        )

        while (
            time.monotonic()
            < deadline
        ):

            if (
                self.get_odom_sequence()
                > sequence_after_reset
            ):

                break

            time.sleep(0.01)

        else:

            raise TimeoutError(
                "No fresh /odom message "
                "received after reset."
            )

        # ----------------------------------------------------
        # Allow Gazebo physics to settle
        # ----------------------------------------------------

        if settle_time > 0.0:

            time.sleep(
                settle_time
            )

        state = self.get_state()

        if state is None:

            raise RuntimeError(
                "No valid odometry state "
                "after reset."
            )

        self.get_logger().info(
            "Gazebo reset completed successfully."
        )

        return state


# ============================================================
# Manual interface test
# ============================================================

def main(
    args=None,
):

    parser = argparse.ArgumentParser(
        description=(
            "Test Reachy's Gazebo "
            "mobile-base interface."
        )
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
        help="Sideways velocity in m/s",
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
        help="Movement duration in seconds",
    )

    parsed_args = parser.parse_args(
        args=args
    )

    # --------------------------------------------------------
    # Start ROS
    # --------------------------------------------------------

    rclpy.init()

    robot = ReachyGazeboInterface(
        reset_service="/reset_world"
    )

    # --------------------------------------------------------
    # Start executor in background thread
    # --------------------------------------------------------

    executor = SingleThreadedExecutor()

    executor.add_node(
        robot
    )

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )

    spin_thread.start()

    try:

        print(
            "\nWaiting for /odom..."
        )

        if not robot.wait_for_odometry(
            timeout=5.0
        ):

            raise RuntimeError(
                "No /odom data received. "
                "Is Reachy Gazebo running?"
            )

        # ----------------------------------------------------
        # Initial state
        # ----------------------------------------------------

        initial_state = robot.get_state()

        print(
            "\nInitial state:"
        )

        print(
            initial_state
        )

        # ----------------------------------------------------
        # Movement
        # ----------------------------------------------------

        print(
            "\nCommanding:"
            f" vx={parsed_args.vx:.3f},"
            f" vy={parsed_args.vy:.3f},"
            f" omega={parsed_args.omega:.3f},"
            f" duration="
            f"{parsed_args.duration:.1f}s"
        )

        robot.command_for(
            vx=parsed_args.vx,
            vy=parsed_args.vy,
            omega=parsed_args.omega,
            duration=parsed_args.duration,
        )

        # Give odometry time to update.

        time.sleep(
            0.2
        )

        # ----------------------------------------------------
        # Final state
        # ----------------------------------------------------

        final_state = robot.get_state()

        print(
            "\nFinal state:"
        )

        print(
            final_state
        )

        # ----------------------------------------------------
        # Calculate movement
        # ----------------------------------------------------

        if (
            initial_state is not None
            and
            final_state is not None
        ):

            dx = (
                final_state.x
                - initial_state.x
            )

            dy = (
                final_state.y
                - initial_state.y
            )

            dyaw = angle_difference(
                final_state.yaw,
                initial_state.yaw,
            )

            distance = math.hypot(
                dx,
                dy,
            )

            print(
                "\nMeasured change:"
            )

            print(
                f"  dx       = "
                f"{dx:.4f} m"
            )

            print(
                f"  dy       = "
                f"{dy:.4f} m"
            )

            print(
                f"  distance = "
                f"{distance:.4f} m"
            )

            print(
                f"  d_yaw    = "
                f"{dyaw:.4f} rad"
            )

            print(
                f"  d_yaw    = "
                f"{math.degrees(dyaw):.2f}°"
            )

    except KeyboardInterrupt:

        print(
            "\nKeyboard interrupt received."
        )

    finally:

        # ----------------------------------------------------
        # Always stop robot before shutting down
        # ----------------------------------------------------

        print(
            "\nStopping Reachy..."
        )

        robot.stop_for(
            duration=1.0,
            frequency=20.0,
        )

        executor.shutdown()

        robot.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()

        spin_thread.join(
            timeout=1.0
        )

        print(
            "Interface closed safely."
        )


if __name__ == "__main__":
    main()