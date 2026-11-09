#!/usr/bin/env python3
"""
ROS 2 interface for goal-reaching navigation in the Hospital and Bookstore
worlds (Reachy 2023, ROS 2 Humble, Gazebo Classic 11).

One node provides everything the environment needs:

    /cmd_vel               geometry_msgs/Twist          publish
    /odom                  nav_msgs/Odometry            subscribe
    /scan                  sensor_msgs/LaserScan        subscribe
    /clock                 rosgraph_msgs/Clock          subscribe
    /reachy/body_contact   gazebo_msgs/ContactsState    subscribe
    /reachy/arm_contact    gazebo_msgs/ContactsState    subscribe (deep queue)
    /set_entity_state      gazebo_msgs/SetEntityState   client

All waiting is done in Gazebo SIMULATION time, taken from /clock and from the
100 Hz /odom stamps (whichever is newer, since /clock is only ~3 Hz in sim
time), so behaviour does not depend on the real-time factor. Wall-clock timeouts only detect a stalled
simulator.

Geometry constants were measured in the validated simulation (sim/README.md).
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan


# ---------------------------------------------------------------------------
# VALIDATED GEOMETRY
# ---------------------------------------------------------------------------

LIDAR_X_OFFSET = 0.184          # LiDAR sits 0.184 m ahead of the base centre
BASE_RADIUS = 0.245             # base collision cylinder radius
SELF_MASK_DEG = (174.0, 186.0)  # mast (bar_link) self-hit at ~0.29 m behind


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def quaternion_to_rpy(x, y, z, w):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def scan_to_edge_clearance(
    msg: LaserScan,
    num_sectors: int = 24,
    max_distance: float = 3.0,
) -> np.ndarray:
    """
    Convert a LaserScan into per-sector clearance from the BASE EDGE.

    For a beam at angle a with range r, the hit point in the base frame is
    (0.184 + r cos a, r sin a); clearance = |hit| - 0.245.

    - Beams in the mast sector (174-186 deg) are ignored.
    - No return (inf / >= range_max) counts as clear (max_distance).
    - A finite return at or below range_min counts as 0.0 (dangerously close),
      never as clear.
    - Sector k is centred on 15k deg for 24 sectors (0 = straight ahead,
      counter-clockwise = left).
    """

    ranges = np.asarray(msg.ranges, dtype=np.float64)

    if ranges.size == 0:
        return np.full(num_sectors, max_distance, dtype=np.float32)

    angles = np.mod(
        msg.angle_min + np.arange(ranges.size) * msg.angle_increment,
        2.0 * np.pi,
    )

    finite = np.isfinite(ranges)
    valid = finite & (ranges > msg.range_min) & (ranges < msg.range_max)
    too_close = finite & (ranges <= msg.range_min)

    clearance = np.full(ranges.size, max_distance, dtype=np.float64)

    hit_x = LIDAR_X_OFFSET + ranges[valid] * np.cos(angles[valid])
    hit_y = ranges[valid] * np.sin(angles[valid])
    clearance[valid] = np.hypot(hit_x, hit_y) - BASE_RADIUS

    clearance[too_close] = 0.0

    mask_lo, mask_hi = np.radians(SELF_MASK_DEG[0]), np.radians(SELF_MASK_DEG[1])
    clearance[(angles >= mask_lo) & (angles <= mask_hi)] = max_distance

    clearance = np.clip(clearance, 0.0, max_distance)

    width = 2.0 * np.pi / num_sectors
    sector_index = (
        np.floor(np.mod(angles + 0.5 * width, 2.0 * np.pi) / width).astype(np.int64)
        % num_sectors
    )

    sectors = np.full(num_sectors, max_distance, dtype=np.float64)
    np.minimum.at(sectors, sector_index, clearance)

    return sectors.astype(np.float32)


@dataclass
class NavState:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float
    roll: float
    pitch: float
    stamp: float  # simulation time of the odometry message


# ---------------------------------------------------------------------------
# INTERFACE NODE
# ---------------------------------------------------------------------------

class ReachyNavInterface(Node):

    def __init__(
        self,
        entity_name: str = "reachy",
        v_min: float = -0.05,
        v_max: float = 0.30,
        w_max: float = 0.60,
    ):
        super().__init__("reachy_nav_interface")

        self.entity_name = entity_name
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.w_max = abs(float(w_max))

        self._lock = threading.Lock()

        self._state: Optional[NavState] = None
        self._scan: Optional[LaserScan] = None
        self._scan_stamp = -1.0
        self._sim_time: Optional[float] = None

        # Latched first contact since the last clear_contacts().
        self._contact: Optional[dict] = None
        self._contact_ignore_before = -1.0

        self._executor = None
        self._spin_thread = None

        sensor_qos = qos_profile_sensor_data
        # 17 arm/head sensors publish in bursts at the same sim instant;
        # a shallow queue silently drops most of them.
        contact_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.create_subscription(Odometry, "/odom", self._odom_callback, sensor_qos)
        self.create_subscription(LaserScan, "/scan", self._scan_callback, sensor_qos)
        self.create_subscription(Clock, "/clock", self._clock_callback, sensor_qos)
        self.create_subscription(
            ContactsState, "/reachy/body_contact",
            lambda m: self._contact_callback("body", m), contact_qos,
        )
        self.create_subscription(
            ContactsState, "/reachy/arm_contact",
            lambda m: self._contact_callback("arm", m), contact_qos,
        )

        self.set_state_client = self.create_client(SetEntityState, "/set_entity_state")

    # -----------------------------------------------------------------------
    # EXECUTOR
    # -----------------------------------------------------------------------

    def start(self):
        """Spin this node in a background thread."""
        if self._spin_thread is not None:
            return
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        def _spin():
            try:
                self._executor.spin()
            except ExternalShutdownException:
                pass

        self._spin_thread = threading.Thread(target=_spin, daemon=True)
        self._spin_thread.start()

    def shutdown(self):
        try:
            self.stop()
        except Exception:
            pass
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)

    # -----------------------------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------------------------

    def _advance_sim_time(self, stamp: float):
        # Call with self._lock held. Simulation time only moves forward.
        if self._sim_time is None or stamp > self._sim_time:
            self._sim_time = stamp

    def _clock_callback(self, msg: Clock):
        # gazebo_ros_init publishes /clock at only ~10 Hz WALL time (~3 Hz sim
        # time at RTF 3.3), so /odom stamps (100 Hz sim time) also drive the clock.
        with self._lock:
            self._advance_sim_time(_stamp_to_sec(msg.clock))

    def _odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        state = NavState(
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=float(yaw),
            vx=float(msg.twist.twist.linear.x),
            vy=float(msg.twist.twist.linear.y),
            wz=float(msg.twist.twist.angular.z),
            roll=float(roll),
            pitch=float(pitch),
            stamp=_stamp_to_sec(msg.header.stamp),
        )
        with self._lock:
            self._state = state
            self._advance_sim_time(state.stamp)

    def _scan_callback(self, msg: LaserScan):
        with self._lock:
            self._scan = msg
            self._scan_stamp = _stamp_to_sec(msg.header.stamp)

    def _contact_callback(self, source: str, msg: ContactsState):
        if not msg.states:
            return
        stamp = _stamp_to_sec(msg.header.stamp)
        with self._lock:
            if self._contact is not None or stamp <= self._contact_ignore_before:
                return
            first = msg.states[0]
            robot_side, other_side = first.collision1_name, first.collision2_name
            if other_side.startswith(self.entity_name + "::"):
                robot_side, other_side = other_side, robot_side
            self._contact = {
                "source": source,                 # "body" or "arm"
                "link": msg.header.frame_id,      # e.g. base_footprint, r_upper_arm
                "robot_collision": robot_side,
                "other": other_side,
                "sim_time": stamp,
            }

    # -----------------------------------------------------------------------
    # READINESS / TIME
    # -----------------------------------------------------------------------

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Wait (wall time) for /clock, /odom, /scan and /set_entity_state."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                ready = (
                    self._sim_time is not None
                    and self._state is not None
                    and self._scan is not None
                )
            if ready and self.set_state_client.wait_for_service(timeout_sec=0.1):
                return True
            time.sleep(0.05)
        return False

    def sim_time(self) -> float:
        with self._lock:
            if self._sim_time is None:
                raise RuntimeError("No /clock received yet.")
            return self._sim_time

    def wait_sim_until(self, target: float, wall_timeout: float = 10.0):
        deadline = time.monotonic() + wall_timeout
        while self.sim_time() < target:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Simulation clock stalled (no /clock progress). "
                    "Is gzserver running and unpaused?"
                )
            time.sleep(0.0005)

    def wait_sim(self, duration: float, wall_timeout: float = 10.0):
        self.wait_sim_until(self.sim_time() + float(duration), wall_timeout)

    def wait_fresh(self, state_after: float, scan_after: float, wall_timeout: float = 5.0):
        """Wait until odometry and scan are stamped at/after the given sim times."""
        deadline = time.monotonic() + wall_timeout
        while True:
            with self._lock:
                state_ok = self._state is not None and self._state.stamp >= state_after
                scan_ok = self._scan is not None and self._scan_stamp >= scan_after
            if state_ok and scan_ok:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"No fresh sensor data (odom ok={state_ok}, scan ok={scan_ok})."
                )
            time.sleep(0.0005)

    # -----------------------------------------------------------------------
    # SENSORS
    # -----------------------------------------------------------------------

    def get_state(self) -> NavState:
        with self._lock:
            if self._state is None:
                raise RuntimeError("No /odom received yet.")
            s = self._state
            return NavState(**s.__dict__)

    def get_scan(self) -> LaserScan:
        with self._lock:
            if self._scan is None:
                raise RuntimeError("No /scan received yet.")
            return self._scan

    def clear_contacts(self):
        """Forget latched contacts; ignore contact messages stamped up to now."""
        with self._lock:
            self._contact = None
            self._contact_ignore_before = (
                self._sim_time if self._sim_time is not None else -1.0
            )

    def get_contact(self) -> Optional[dict]:
        with self._lock:
            return None if self._contact is None else dict(self._contact)

    # -----------------------------------------------------------------------
    # COMMANDS
    # -----------------------------------------------------------------------

    def send_velocity(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = max(self.v_min, min(self.v_max, float(v)))
        msg.angular.z = max(-self.w_max, min(self.w_max, float(w)))
        self.cmd_vel_publisher.publish(msg)

    def stop(self):
        # planar_move keeps executing the last command, so always send zeros.
        for _ in range(3):
            self.cmd_vel_publisher.publish(Twist())

    def teleport(self, x: float, y: float, yaw: float, z: float = 0.02,
                 wall_timeout: float = 5.0, attempts: int = 4):
        """
        Teleport the robot with /set_entity_state, retrying lost or slow replies.

        Under heavy gzserver load a Humble service reply can occasionally be
        lost. Re-sending the same pose is idempotent, so the call is retried
        up to `attempts` times before giving up.
        """
        request = SetEntityState.Request()
        request.state.name = self.entity_name
        request.state.pose.position.x = float(x)
        request.state.pose.position.y = float(y)
        request.state.pose.position.z = float(z)
        request.state.pose.orientation.z = math.sin(0.5 * float(yaw))
        request.state.pose.orientation.w = math.cos(0.5 * float(yaw))
        request.state.reference_frame = "world"

        if not hasattr(self, "teleport_retries"):
            self.teleport_retries = 0

        last = "no attempt"
        for attempt in range(1, int(attempts) + 1):
            if not self.set_state_client.wait_for_service(timeout_sec=wall_timeout):
                last = "service not available"
            else:
                future = self.set_state_client.call_async(request)
                deadline = time.monotonic() + wall_timeout
                while not future.done() and time.monotonic() < deadline:
                    time.sleep(0.001)
                if future.done():
                    result = future.result()
                    if result is not None and result.success:
                        return
                    last = "success=False"
                else:
                    last = f"no reply within {wall_timeout:.0f} s"
                    try:
                        self.set_state_client.remove_pending_request(future)
                    except Exception:
                        pass
                    try:
                        future.cancel()
                    except Exception:
                        pass
            if attempt < attempts:
                self.teleport_retries += 1
                self.get_logger().warn(
                    f"/set_entity_state attempt {attempt}/{attempts} failed ({last}); retrying "
                    f"(total retries so far: {self.teleport_retries})"
                )
                time.sleep(0.2)

        raise RuntimeError(
            f"/set_entity_state failed after {attempts} attempts for '{self.entity_name}' ({last})."
        )
