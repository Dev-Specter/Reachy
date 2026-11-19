#!/usr/bin/env python3
"""Publish the goal of a frozen task (or any x y) as RViz markers on /nav_goal_markers.

  python3 goal_marker.py --task-file /opt/reachy_ws/rl_sim/maps/hospital/static/eval.json --index 17
  python3 goal_marker.py --xy 4.37 -1.93 --label "goal"

In RViz add a MarkerArray display on /nav_goal_markers (Fixed Frame: odom).
Shows the goal (0.30 m success radius), the start pose, and a text label.
Markers are latched, so RViz shows them even if it is opened later.
"""
import argparse, json, math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def mk(i, kind, x, y, z, sx, sy, sz, rgba, frame, text="", yaw=0.0):
    m = Marker()
    m.header.frame_id = frame
    m.ns, m.id, m.type, m.action = "nav_goal", i, kind, Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
    m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    m.scale.x, m.scale.y, m.scale.z = sx, sy, sz
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    m.text = text
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--xy", type=float, nargs=2)
    ap.add_argument("--label", default=None)
    ap.add_argument("--frame", default="odom")
    a = ap.parse_args()

    start = None
    if a.task_file:
        d = json.load(open(a.task_file))
        t = (d["tasks"] if isinstance(d, dict) else d)[a.index]
        gx, gy = t["goal_x"], t["goal_y"]
        start = (t["start_x"], t["start_y"], t["start_yaw"])
        label = a.label or f"goal (task {a.index})"
    elif a.xy:
        gx, gy = a.xy
        label = a.label or "goal"
    else:
        ap.error("give --task-file/--index or --xy")

    rclpy.init()
    node = Node("nav_goal_marker")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(MarkerArray, "/nav_goal_markers", qos)
    arr = MarkerArray()
    arr.markers.append(mk(0, Marker.CYLINDER, gx, gy, 0.01, 0.60, 0.60, 0.02, (0.89, 0.70, 0.24, 0.6), a.frame))
    arr.markers.append(mk(1, Marker.SPHERE, gx, gy, 0.15, 0.15, 0.15, 0.15, (0.89, 0.70, 0.24, 1.0), a.frame))
    arr.markers.append(mk(2, Marker.TEXT_VIEW_FACING, gx, gy, 0.6, 0.0, 0.0, 0.25, (1, 1, 1, 1), a.frame, label))
    if start:
        arr.markers.append(mk(3, Marker.ARROW, start[0], start[1], 0.05, 0.5, 0.08, 0.08,
                              (1.0, 1.0, 1.0, 0.9), a.frame, yaw=start[2]))
    pub.publish(arr)
    node.get_logger().info(f"Published goal markers at ({gx:.2f}, {gy:.2f}) in frame '{a.frame}'. Ctrl+C to stop.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    (rclpy.ok() and rclpy.shutdown())


if __name__ == "__main__":
    main()
