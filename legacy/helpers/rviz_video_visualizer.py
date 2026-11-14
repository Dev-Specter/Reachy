#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    ReliabilityPolicy,
)

from reachy_rl_control.trajectory import (
    STRAIGHT_PATH,
)


class ReachyRvizVideoVisualizer(Node):
    """
    Passive visualization node for dissertation videos.

    Displays:
      - planned trajectory
      - actual trajectory
      - waypoint markers
      - live robot state text

    RViz itself displays /scan directly for LiDAR perception.

    This node does NOT command Reachy and does NOT modify
    the RL environment.
    """

    def __init__(self):

        super().__init__(
            "reachy_rviz_video_visualizer"
        )

        self.frame_id = "odom"

        # -------------------------------------------------
        # LATCHED QoS FOR STATIC VISUALS
        # -------------------------------------------------

        latched_qos = QoSProfile(
            depth=1
        )

        latched_qos.reliability = (
            ReliabilityPolicy.RELIABLE
        )

        latched_qos.durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
        )

        # -------------------------------------------------
        # PUBLISHERS
        # -------------------------------------------------

        self.planned_path_pub = (
            self.create_publisher(
                Path,
                "/reachy_viz/planned_path",
                latched_qos,
            )
        )

        self.actual_path_pub = (
            self.create_publisher(
                Path,
                "/reachy_viz/actual_path",
                10,
            )
        )

        self.waypoint_pub = (
            self.create_publisher(
                MarkerArray,
                "/reachy_viz/waypoints",
                latched_qos,
            )
        )

        self.state_text_pub = (
            self.create_publisher(
                Marker,
                "/reachy_viz/state_text",
                10,
            )
        )

        # -------------------------------------------------
        # ODOMETRY
        # -------------------------------------------------

        self.odom_sub = (
            self.create_subscription(
                Odometry,
                "/odom",
                self._odom_callback,
                20,
            )
        )

        # -------------------------------------------------
        # INTERNAL STATE
        # -------------------------------------------------

        self.start_x = None
        self.start_y = None
        self.start_yaw = None

        self.last_x = None
        self.last_y = None
        self.last_yaw = None

        self.actual_path = Path()

        self.actual_path.header.frame_id = (
            self.frame_id
        )

        self.get_logger().info(
            "Reachy RViz video visualizer started."
        )

        self.get_logger().info(
            "Listening to /odom."
        )

        self.get_logger().info(
            "RViz should display /scan directly."
        )

    # =====================================================
    # YAW
    # =====================================================

    @staticmethod
    def _quaternion_to_yaw(
        quaternion,
    ):

        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        siny_cosp = (
            2.0
            * (
                w * z
                +
                x * y
            )
        )

        cosy_cosp = (
            1.0
            -
            2.0
            * (
                y * y
                +
                z * z
            )
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp,
        )

    # =====================================================
    # ANGLE DIFFERENCE
    # =====================================================

    @staticmethod
    def _angle_difference(
        a,
        b,
    ):

        return math.atan2(
            math.sin(a - b),
            math.cos(a - b),
        )

    # =====================================================
    # CREATE QUATERNION FROM YAW
    # =====================================================

    @staticmethod
    def _set_yaw_quaternion(
        orientation,
        yaw,
    ):

        orientation.x = 0.0
        orientation.y = 0.0

        orientation.z = math.sin(
            yaw / 2.0
        )

        orientation.w = math.cos(
            yaw / 2.0
        )

    # =====================================================
    # RESET VISUALIZATION
    # =====================================================

    def _reset_visualization(
        self,
        x,
        y,
        yaw,
    ):

        self.start_x = float(x)
        self.start_y = float(y)
        self.start_yaw = float(yaw)

        self.actual_path = Path()

        self.actual_path.header.frame_id = (
            self.frame_id
        )

        self._publish_planned_path()

        self._publish_waypoints()

        self.get_logger().info(
            "Visualization trajectory reset | "
            f"x={x:.3f}, "
            f"y={y:.3f}, "
            f"yaw={math.degrees(yaw):.2f} deg"
        )

    # =====================================================
    # CREATE WORLD WAYPOINTS
    # =====================================================

    def _world_waypoints(
        self,
    ):

        cos_yaw = math.cos(
            self.start_yaw
        )

        sin_yaw = math.sin(
            self.start_yaw
        )

        world_waypoints = []

        for (
            local_x,
            local_y,
        ) in STRAIGHT_PATH:

            world_x = (
                self.start_x
                +
                cos_yaw * local_x
                -
                sin_yaw * local_y
            )

            world_y = (
                self.start_y
                +
                sin_yaw * local_x
                +
                cos_yaw * local_y
            )

            world_waypoints.append(
                (
                    world_x,
                    world_y,
                )
            )

        return world_waypoints

    # =====================================================
    # PLANNED PATH
    # =====================================================

    def _publish_planned_path(
        self,
    ):

        if self.start_x is None:
            return

        now = (
            self.get_clock()
            .now()
            .to_msg()
        )

        path = Path()

        path.header.frame_id = (
            self.frame_id
        )

        path.header.stamp = now

        for (
            x,
            y,
        ) in self._world_waypoints():

            pose = PoseStamped()

            pose.header.frame_id = (
                self.frame_id
            )

            pose.header.stamp = now

            pose.pose.position.x = float(
                x
            )

            pose.pose.position.y = float(
                y
            )

            pose.pose.position.z = 0.05

            self._set_yaw_quaternion(
                pose.pose.orientation,
                self.start_yaw,
            )

            path.poses.append(
                pose
            )

        self.planned_path_pub.publish(
            path
        )

    # =====================================================
    # WAYPOINT MARKERS
    # =====================================================

    def _publish_waypoints(
        self,
    ):

        if self.start_x is None:
            return

        now = (
            self.get_clock()
            .now()
            .to_msg()
        )

        markers = MarkerArray()

        world_waypoints = (
            self._world_waypoints()
        )

        for index, (
            x,
            y,
        ) in enumerate(
            world_waypoints
        ):

            # ---------------------------------------------
            # SPHERE
            # ---------------------------------------------

            marker = Marker()

            marker.header.frame_id = (
                self.frame_id
            )

            marker.header.stamp = now

            marker.ns = "waypoints"

            marker.id = index

            marker.type = Marker.SPHERE

            marker.action = Marker.ADD

            marker.pose.position.x = float(
                x
            )

            marker.pose.position.y = float(
                y
            )

            marker.pose.position.z = 0.10

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.12

            marker.color.r = 0.1
            marker.color.g = 1.0
            marker.color.b = 0.2
            marker.color.a = 1.0

            markers.markers.append(
                marker
            )

            # ---------------------------------------------
            # LABEL
            # ---------------------------------------------

            label = Marker()

            label.header.frame_id = (
                self.frame_id
            )

            label.header.stamp = now

            label.ns = "waypoint_labels"

            label.id = (
                100 + index
            )

            label.type = (
                Marker.TEXT_VIEW_FACING
            )

            label.action = Marker.ADD

            label.pose.position.x = float(
                x
            )

            label.pose.position.y = float(
                y
            )

            label.pose.position.z = 0.32

            label.pose.orientation.w = 1.0

            label.scale.z = 0.15

            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0

            label.text = (
                f"WP{index}"
            )

            markers.markers.append(
                label
            )

        self.waypoint_pub.publish(
            markers
        )

    # =====================================================
    # ROBOT STATE TEXT
    # =====================================================

    def _publish_state_text(
        self,
        msg,
        x,
        y,
        yaw,
    ):

        marker = Marker()

        marker.header.frame_id = (
            self.frame_id
        )

        marker.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        marker.ns = "robot_state"

        marker.id = 0

        marker.type = (
            Marker.TEXT_VIEW_FACING
        )

        marker.action = Marker.ADD

        marker.pose.position.x = float(
            x
        )

        marker.pose.position.y = float(
            y
        )

        marker.pose.position.z = 1.6

        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.16

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        vx = float(
            msg.twist.twist.linear.x
        )

        vy = float(
            msg.twist.twist.linear.y
        )

        wz = float(
            msg.twist.twist.angular.z
        )

        marker.text = (
            "REACHY STATE\n"
            f"x={x:+.2f} m   "
            f"y={y:+.2f} m\n"
            f"yaw={math.degrees(yaw):+.1f} deg\n"
            f"vx={vx:+.2f} m/s   "
            f"vy={vy:+.2f} m/s\n"
            f"wz={wz:+.2f} rad/s"
        )

        self.state_text_pub.publish(
            marker
        )

    # =====================================================
    # ACTUAL PATH
    # =====================================================

    def _append_actual_path(
        self,
        msg,
    ):

        pose = PoseStamped()

        pose.header.frame_id = (
            self.frame_id
        )

        pose.header.stamp = (
            msg.header.stamp
        )

        pose.pose.position.x = (
            msg.pose.pose.position.x
        )

        pose.pose.position.y = (
            msg.pose.pose.position.y
        )

        pose.pose.position.z = 0.08

        pose.pose.orientation.x = (
            msg.pose.pose.orientation.x
        )

        pose.pose.orientation.y = (
            msg.pose.pose.orientation.y
        )

        pose.pose.orientation.z = (
            msg.pose.pose.orientation.z
        )

        pose.pose.orientation.w = (
            msg.pose.pose.orientation.w
        )

        self.actual_path.header.stamp = (
            msg.header.stamp
        )

        self.actual_path.poses.append(
            pose
        )

        # Prevent unlimited growth if accidentally left
        # running for many episodes.
        if len(
            self.actual_path.poses
        ) > 3000:

            self.actual_path.poses = (
                self.actual_path.poses[
                    -3000:
                ]
            )

        self.actual_path_pub.publish(
            self.actual_path
        )

    # =====================================================
    # ODOM CALLBACK
    # =====================================================

    def _odom_callback(
        self,
        msg,
    ):

        x = float(
            msg.pose.pose.position.x
        )

        y = float(
            msg.pose.pose.position.y
        )

        yaw = (
            self._quaternion_to_yaw(
                msg.pose.pose.orientation
            )
        )

        # -------------------------------------------------
        # FIRST MESSAGE
        # -------------------------------------------------

        if self.start_x is None:

            self._reset_visualization(
                x,
                y,
                yaw,
            )

        # -------------------------------------------------
        # DETECT GAZEBO RESET
        # -------------------------------------------------
        #
        # Normal motion between odometry messages is far
        # smaller than 0.25 m or 40 degrees.
        #
        # A sudden jump therefore indicates /reset_world.
        # -------------------------------------------------

        elif (
            self.last_x is not None
        ):

            position_jump = math.hypot(
                x - self.last_x,
                y - self.last_y,
            )

            yaw_jump = abs(
                self._angle_difference(
                    yaw,
                    self.last_yaw,
                )
            )

            if (
                position_jump > 0.25
                or
                yaw_jump > math.radians(
                    40.0
                )
            ):

                self._reset_visualization(
                    x,
                    y,
                    yaw,
                )

        # -------------------------------------------------
        # PUBLISH LIVE VISUALS
        # -------------------------------------------------

        self._append_actual_path(
            msg
        )

        self._publish_state_text(
            msg,
            x,
            y,
            yaw,
        )

        self.last_x = x
        self.last_y = y
        self.last_yaw = yaw


def main(
    args=None,
):

    rclpy.init(
        args=args
    )

    node = (
        ReachyRvizVideoVisualizer()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()
