#!/usr/bin/env python3

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Empty


# ---------------------------------------------------------
# ROBOT STATE
# ---------------------------------------------------------

@dataclass
class BaseState:
    x: float
    y: float
    yaw: float

    vx: float
    vy: float
    wz: float

    # Added for fall detection.
    roll: float = 0.0
    pitch: float = 0.0


# ---------------------------------------------------------
# QUATERNION HELPERS
# ---------------------------------------------------------

def quaternion_to_yaw(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:
    """
    Convert quaternion orientation to yaw angle.
    """

    siny_cosp = 2.0 * (
        w * z
        + x * y
    )

    cosy_cosp = 1.0 - 2.0 * (
        y * y
        + z * z
    )

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp,
    )

    return yaw


def quaternion_to_roll_pitch(
    x: float,
    y: float,
    z: float,
    w: float,
):
    """
    Convert quaternion orientation to roll and pitch.

    Returned values are in radians.
    """

    # -------------------------
    # Roll
    # -------------------------

    sinr_cosp = 2.0 * (
        w * x
        + y * z
    )

    cosr_cosp = 1.0 - 2.0 * (
        x * x
        + y * y
    )

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp,
    )

    # -------------------------
    # Pitch
    # -------------------------

    sinp = 2.0 * (
        w * y
        - z * x
    )

    # Protect asin against tiny numerical errors.
    sinp = max(
        -1.0,
        min(
            1.0,
            sinp,
        ),
    )

    pitch = math.asin(
        sinp
    )

    return roll, pitch


def angle_difference(
    target: float,
    current: float,
) -> float:
    """
    Return the shortest signed angular difference between
    target and current angles.
    """

    difference = (
        target
        - current
        + math.pi
    ) % (
        2.0
        * math.pi
    ) - math.pi

    return difference


# ---------------------------------------------------------
# REACHY GAZEBO INTERFACE
# ---------------------------------------------------------

class ReachyGazeboInterface(Node):

    def __init__(
        self,
        cmd_vel_topic: str = "/cmd_vel",
        odom_topic: str = "/odom",
        reset_service: str = "/reset_world",
        max_vx: float = 0.30,
        max_vy: float = 0.30,
        max_wz: float = 0.60,
    ):
        super().__init__(
            "reachy_gazebo_interface"
        )

        self.cmd_vel_topic = (
            cmd_vel_topic
        )

        self.odom_topic = (
            odom_topic
        )

        self.reset_service_name = (
            reset_service
        )

        # Safe command limits.
        self.max_vx = abs(
            float(max_vx)
        )

        self.max_vy = abs(
            float(max_vy)
        )

        self.max_wz = abs(
            float(max_wz)
        )

        # -------------------------------------------------
        # STATE STORAGE
        # -------------------------------------------------

        self._latest_state: Optional[
            BaseState
        ] = None

        self._last_odom_time: Optional[
            float
        ] = None

        # -------------------------------------------------
        # ROS PUBLISHER
        # -------------------------------------------------

        self.cmd_vel_publisher = (
            self.create_publisher(
                Twist,
                self.cmd_vel_topic,
                10,
            )
        )

        # -------------------------------------------------
        # ROS SUBSCRIBER
        # -------------------------------------------------

        self.odom_subscription = (
            self.create_subscription(
                Odometry,
                self.odom_topic,
                self._odom_callback,
                10,
            )
        )

        # -------------------------------------------------
        # RESET SERVICE CLIENT
        # -------------------------------------------------

        self.reset_client = (
            self.create_client(
                Empty,
                self.reset_service_name,
            )
        )

        # -------------------------------------------------
        # STARTUP LOGGING
        # -------------------------------------------------

        self.get_logger().info(
            "Reachy Gazebo interface started."
        )

        self.get_logger().info(
            f"Publishing velocity commands to "
            f"{self.cmd_vel_topic}"
        )

        self.get_logger().info(
            f"Reading odometry from "
            f"{self.odom_topic}"
        )

        self.get_logger().info(
            f"Reset service: "
            f"{self.reset_service_name}"
        )

    # -----------------------------------------------------
    # ODOMETRY CALLBACK
    # -----------------------------------------------------

    def _odom_callback(
        self,
        msg: Odometry,
    ):
        """
        Store the latest Reachy base state from /odom.
        """

        position = (
            msg.pose.pose.position
        )

        orientation = (
            msg.pose.pose.orientation
        )

        linear_velocity = (
            msg.twist.twist.linear
        )

        angular_velocity = (
            msg.twist.twist.angular
        )

        yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )

        roll, pitch = (
            quaternion_to_roll_pitch(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        )

        self._latest_state = (
            BaseState(
                x=float(
                    position.x
                ),

                y=float(
                    position.y
                ),

                yaw=float(
                    yaw
                ),

                vx=float(
                    linear_velocity.x
                ),

                vy=float(
                    linear_velocity.y
                ),

                wz=float(
                    angular_velocity.z
                ),

                roll=float(
                    roll
                ),

                pitch=float(
                    pitch
                ),
            )
        )

        self._last_odom_time = (
            time.monotonic()
        )

    # -----------------------------------------------------
    # STATE ACCESS
    # -----------------------------------------------------

    def get_state(
        self,
    ) -> BaseState:
        """
        Return the latest received Reachy base state.
        """

        if self._latest_state is None:

            raise RuntimeError(
                "No odometry has been received yet."
            )

        # Return a copy so external code cannot modify
        # the internally stored state by accident.
        state = self._latest_state

        return BaseState(
            x=state.x,
            y=state.y,
            yaw=state.yaw,
            vx=state.vx,
            vy=state.vy,
            wz=state.wz,
            roll=state.roll,
            pitch=state.pitch,
        )

    # -----------------------------------------------------
    # WAIT FOR ODOMETRY
    # -----------------------------------------------------

    def wait_for_odometry(
        self,
        timeout: float = 5.0,
    ) -> bool:
        """
        Wait until at least one /odom message has arrived.

        This assumes the node is being spun by an executor
        in another thread.
        """

        start_time = (
            time.monotonic()
        )

        while (
            time.monotonic()
            - start_time
            < timeout
        ):

            if (
                self._latest_state
                is not None
            ):
                return True

            time.sleep(
                0.05
            )

        return False

    # -----------------------------------------------------
    # VELOCITY LIMITING
    # -----------------------------------------------------

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # -----------------------------------------------------
    # SEND VELOCITY
    # -----------------------------------------------------

    def send_velocity(
        self,
        vx: float,
        vy: float,
        omega: float,
    ):
        """
        Publish a velocity command to Reachy's mobile base.
        """

        vx = self._clamp(
            float(vx),
            -self.max_vx,
            self.max_vx,
        )

        vy = self._clamp(
            float(vy),
            -self.max_vy,
            self.max_vy,
        )

        omega = self._clamp(
            float(omega),
            -self.max_wz,
            self.max_wz,
        )

        msg = Twist()

        msg.linear.x = (
            vx
        )

        msg.linear.y = (
            vy
        )

        msg.linear.z = (
            0.0
        )

        msg.angular.x = (
            0.0
        )

        msg.angular.y = (
            0.0
        )

        msg.angular.z = (
            omega
        )

        self.cmd_vel_publisher.publish(
            msg
        )

    # -----------------------------------------------------
    # STOP
    # -----------------------------------------------------

    def stop_for(
        self,
        duration: float = 1.0,
        frequency: float = 20.0,
    ):
        """
        Repeatedly publish zero velocity commands.

        Repeating the stop command makes stopping more
        reliable than sending only one zero Twist.
        """

        duration = max(
            0.0,
            float(duration),
        )

        frequency = max(
            1.0,
            float(frequency),
        )

        period = (
            1.0
            / frequency
        )

        end_time = (
            time.monotonic()
            + duration
        )

        while (
            time.monotonic()
            < end_time
        ):

            self.send_velocity(
                0.0,
                0.0,
                0.0,
            )

            time.sleep(
                period
            )

        # Final zero command.
        self.send_velocity(
            0.0,
            0.0,
            0.0,
        )

    # -----------------------------------------------------
    # COMMAND FOR FIXED DURATION
    # -----------------------------------------------------

    def command_for(
        self,
        vx: float,
        vy: float,
        omega: float,
        duration: float,
        frequency: float = 20.0,
    ):
        """
        Send a velocity command repeatedly for a specified
        time and then stop Reachy safely.
        """

        duration = max(
            0.0,
            float(duration),
        )

        frequency = max(
            1.0,
            float(frequency),
        )

        period = (
            1.0
            / frequency
        )

        end_time = (
            time.monotonic()
            + duration
        )

        try:

            while (
                time.monotonic()
                < end_time
            ):

                self.send_velocity(
                    vx,
                    vy,
                    omega,
                )

                time.sleep(
                    period
                )

        finally:

            self.stop_for(
                duration=0.5,
                frequency=20.0,
            )

    # -----------------------------------------------------
    # RESET GAZEBO
    # -----------------------------------------------------

    def reset_gazebo(
        self,
        service_timeout: float = 5.0,
        odom_timeout: float = 5.0,
        settle_time: float = 1.0,
    ) -> bool:
        """
        Safely stop Reachy, call /reset_world, and wait for
        fresh odometry after the reset.
        """

        # Stop before resetting.
        self.stop_for(
            duration=0.5,
            frequency=20.0,
        )

        self.get_logger().info(
            f"Resetting Gazebo using "
            f"{self.reset_service_name}"
        )

        # -------------------------------------------------
        # WAIT FOR RESET SERVICE
        # -------------------------------------------------

        if not self.reset_client.wait_for_service(
            timeout_sec=service_timeout
        ):

            raise RuntimeError(
                f"Reset service "
                f"{self.reset_service_name} "
                f"is not available."
            )

        # Remember current odometry timestamp so we can
        # wait for a fresh message after reset.
        old_odom_time = (
            self._last_odom_time
        )

        request = Empty.Request()

        future = (
            self.reset_client.call_async(
                request
            )
        )

        # Because this node normally has a background
        # executor spinning it, simply wait for the future.
        start_time = (
            time.monotonic()
        )

        while not future.done():

            if (
                time.monotonic()
                - start_time
                > service_timeout
            ):

                raise RuntimeError(
                    "Timed out waiting for "
                    "Gazebo reset service."
                )

            time.sleep(
                0.01
            )

        exception = (
            future.exception()
        )

        if exception is not None:

            raise RuntimeError(
                f"Gazebo reset failed: "
                f"{exception}"
            )

        self.get_logger().info(
            "Gazebo reset service completed."
        )

        # -------------------------------------------------
        # WAIT FOR FRESH ODOMETRY
        # -------------------------------------------------

        odom_start = (
            time.monotonic()
        )

        while (
            time.monotonic()
            - odom_start
            < odom_timeout
        ):

            if (
                self._latest_state
                is not None
            ):

                if old_odom_time is None:

                    break

                if (
                    self._last_odom_time
                    is not None
                    and
                    self._last_odom_time
                    > old_odom_time
                ):

                    break

            time.sleep(
                0.02
            )

        else:

            raise RuntimeError(
                "Timed out waiting for fresh "
                "odometry after Gazebo reset."
            )

        # Allow physics to settle.
        if settle_time > 0.0:

            time.sleep(
                settle_time
            )

        # Ensure the base remains stopped.
        self.stop_for(
            duration=0.2,
            frequency=20.0,
        )

        self.get_logger().info(
            "Gazebo reset completed successfully."
        )

        return True


# ---------------------------------------------------------
# COMMAND-LINE TEST
# ---------------------------------------------------------

def main(args=None):

    parser = argparse.ArgumentParser(
        description=(
            "Send a velocity command to Reachy's "
            "Gazebo mobile base."
        )
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=0.0,
        help=(
            "Forward/backward velocity in m/s."
        ),
    )

    parser.add_argument(
        "--vy",
        type=float,
        default=0.0,
        help=(
            "Sideways velocity in m/s."
        ),
    )

    parser.add_argument(
        "--omega",
        type=float,
        default=0.0,
        help=(
            "Angular velocity in rad/s."
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help=(
            "Command duration in seconds."
        ),
    )

    parsed_args, ros_args = (
        parser.parse_known_args(
            args=args
        )
    )

    rclpy.init(
        args=ros_args
    )

    node = ReachyGazeboInterface()

    try:

        print(
            "\nWaiting for /odom..."
        )

        # Since there is no background executor in this
        # standalone command-line test, spin manually.
        start_time = (
            time.monotonic()
        )

        while (
            node._latest_state is None
            and
            time.monotonic()
            - start_time
            < 5.0
        ):

            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )

        if (
            node._latest_state
            is None
        ):

            raise RuntimeError(
                "No /odom received. "
                "Is Gazebo running?"
            )

        state = node.get_state()

        print(
            "Odometry received."
        )

        print(
            f"Initial state: "
            f"x={state.x:.3f}, "
            f"y={state.y:.3f}, "
            f"yaw={state.yaw:.3f}, "
            f"roll={math.degrees(state.roll):.2f} deg, "
            f"pitch={math.degrees(state.pitch):.2f} deg"
        )

        print(
            "\nSending command:"
        )

        print(
            f"vx={parsed_args.vx:.3f}, "
            f"vy={parsed_args.vy:.3f}, "
            f"omega={parsed_args.omega:.3f}, "
            f"duration={parsed_args.duration:.2f}s"
        )

        node.command_for(
            vx=parsed_args.vx,
            vy=parsed_args.vy,
            omega=parsed_args.omega,
            duration=parsed_args.duration,
        )

        print(
            "\nCommand finished. Reachy stopped."
        )

    except KeyboardInterrupt:

        print(
            "\nInterrupted."
        )

    finally:

        node.stop_for(
            duration=1.0,
            frequency=20.0,
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
