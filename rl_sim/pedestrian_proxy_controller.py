#!/usr/bin/env python3

import math
import random

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from std_srvs.srv import Trigger


class PedestrianProxyController(Node):

    def __init__(self):
        super().__init__('pedestrian_proxy_controller')

        self.client = self.create_client(
            SetEntityState,
            '/set_entity_state'
        )

        if not self.client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('/set_entity_state unavailable')

        self.declare_parameter('phase_offset', 0.0)
        self.declare_parameter('motion_mode', 'seen')
        self.declare_parameter('trajectory_seed', 12345)

        trajectory_seed = int(
            self.get_parameter('trajectory_seed').value
        )

        self.rng = random.Random(trajectory_seed)

        self.motion_mode = 'seen'
        self.phase_offset = 0.0

        requested_mode = self._configure_motion_from_parameters()

        now = self.get_clock().now().nanoseconds / 1e9
        self.phase_origin = now - self.phase_offset

        self.get_logger().info(
            f'initial requested_mode={requested_mode}, '
            f'active_mode={self.motion_mode}, '
            f'phase={self.phase_offset:.2f}s, '
            f'trajectory_seed={trajectory_seed}'
        )

        self.reset_phase_service = self.create_service(
            Trigger,
            '/pedestrian/reset_phase',
            self.reset_phase_callback,
        )

        # Only one Gazebo request is allowed to be outstanding.
        self.last_future = None
        self.pending_pose = None

        # 10 Hz simulation-time target updates.
        self.timer = self.create_timer(
            0.10,
            self.update_proxy
        )

        self.get_logger().info(
            f'pedestrian controller started: '
            f'mode={self.motion_mode}, '
            f'phase_offset={self.phase_offset:.2f}'
        )

    @staticmethod
    def _validate_mode(mode):
        allowed = {
            'seen',
            'slow',
            'fast',
            'reverse',
            'stop_go',
            'l_shape',
            'curve',
            'diagonal',
            'train_mix',
        }

        if mode not in allowed:
            raise ValueError(
                f'Unknown motion_mode={mode!r}; '
                f'allowed={sorted(allowed)}'
            )

    def _configure_motion_from_parameters(self):

        requested_mode = str(
            self.get_parameter('motion_mode').value
        )

        self._validate_mode(requested_mode)

        if requested_mode == 'train_mix':

            self.motion_mode = self.rng.choice([
                'seen',
                'l_shape',
                'curve',
            ])

            self.phase_offset = self.rng.choice([
                0.0,
                2.0,
                4.0,
                6.0,
            ])

        else:

            self.motion_mode = requested_mode

            self.phase_offset = float(
                self.get_parameter('phase_offset').value
            )

        return requested_mode

    def reset_phase_callback(self, request, response):

        try:
            requested_mode = (
                self._configure_motion_from_parameters()
            )
        except ValueError as exc:
            response.success = False
            response.message = str(exc)
            return response

        now = self.get_clock().now().nanoseconds / 1e9

        self.phase_origin = (
            now - self.phase_offset
        )

        response.success = True
        response.message = (
            f'pedestrian reset: '
            f'requested={requested_mode}, '
            f'active={self.motion_mode}, '
            f'phase={self.phase_offset:.2f}s'
        )

        self.get_logger().info(response.message)

        return response

    @staticmethod
    def _straight_cycle(speed):
        travel = 4.0 / speed
        turn = 0.5
        return 2.0 * travel + 2.0 * turn

    def target_pose(self, elapsed):

        x = 1.8
        z = 0.85

        mode = self.motion_mode

        # ---------------------------------------------------------
        # Seen dynamics: original trajectory, exactly preserved.
        # ---------------------------------------------------------
        if mode == 'seen':

            speed = 0.5
            travel = 8.0
            cycle = 17.0
            phase = elapsed % cycle

            if phase <= travel:
                y = -1.0 + speed * phase
                yaw = math.pi / 2.0

            elif phase <= 8.5:
                y = 3.0
                yaw = -math.pi / 2.0

            elif phase <= 16.5:
                y = 3.0 - speed * (phase - 8.5)
                yaw = -math.pi / 2.0

            else:
                y = -1.0
                yaw = math.pi / 2.0

            return x, y, z, yaw

        # ---------------------------------------------------------
        # Slow / fast: same spatial trajectory, unseen speed.
        # ---------------------------------------------------------
        if mode in ('slow', 'fast'):

            speed = 0.3 if mode == 'slow' else 0.7

            travel = 4.0 / speed
            turn = 0.5
            cycle = 2.0 * travel + 2.0 * turn
            phase = elapsed % cycle

            if phase <= travel:
                y = -1.0 + speed * phase
                yaw = math.pi / 2.0

            elif phase <= travel + turn:
                y = 3.0
                yaw = -math.pi / 2.0

            elif phase <= 2.0 * travel + turn:
                y = (
                    3.0
                    - speed
                    * (phase - travel - turn)
                )
                yaw = -math.pi / 2.0

            else:
                y = -1.0
                yaw = math.pi / 2.0

            return x, y, z, yaw

        # ---------------------------------------------------------
        # Reverse: same speed but starts from the opposite end.
        # ---------------------------------------------------------
        if mode == 'reverse':

            speed = 0.5
            travel = 8.0
            cycle = 17.0
            phase = elapsed % cycle

            if phase <= travel:
                y = 3.0 - speed * phase
                yaw = -math.pi / 2.0

            elif phase <= 8.5:
                y = -1.0
                yaw = math.pi / 2.0

            elif phase <= 16.5:
                y = (
                    -1.0
                    + speed * (phase - 8.5)
                )
                yaw = math.pi / 2.0

            else:
                y = 3.0
                yaw = -math.pi / 2.0

            return x, y, z, yaw

        # ---------------------------------------------------------
        # L-shaped trajectory:
        #
        # Segment 1:
        #   (1.8, -1.0) -> (1.8, 1.0)
        #
        # Segment 2:
        #   (1.8, 1.0) -> (3.0, 1.0)
        #
        # Then reverse along the same path.
        #
        # Speed approximately 0.5 m/s.
        # ---------------------------------------------------------
        if mode == 'l_shape':

            phase = elapsed % 14.8

            # Upward vertical section: 2 m / 0.5 = 4 s
            if phase <= 4.0:
                x = 1.8
                y = -1.0 + 0.5 * phase
                yaw = math.pi / 2.0

            # Horizontal section: 1.2 m / 0.5 = 2.4 s
            elif phase <= 6.4:
                x = 1.8 + 0.5 * (phase - 4.0)
                y = 1.0
                yaw = 0.0

            # Short turn hold
            elif phase <= 6.9:
                x = 3.0
                y = 1.0
                yaw = math.pi

            # Horizontal return
            elif phase <= 9.3:
                x = 3.0 - 0.5 * (phase - 6.9)
                y = 1.0
                yaw = math.pi

            # Vertical return
            elif phase <= 13.3:
                x = 1.8
                y = 1.0 - 0.5 * (phase - 9.3)
                yaw = -math.pi / 2.0

            else:
                x = 1.8
                y = -1.0
                yaw = math.pi / 2.0

            return x, y, z, yaw

        # ---------------------------------------------------------
        # Curved trajectory:
        #
        # Quarter-circle arc centred near (1.8, 1.0),
        # radius = 2 m.
        #
        # Forward:
        #   approximately (1.8, -1.0) -> (3.8, 1.0)
        #
        # Then return along the same arc.
        # ---------------------------------------------------------
        if mode == 'curve':

            radius = 2.0

            # Arc length = pi*r/2 = pi m.
            # At ~0.5 m/s this takes ~6.28 s.
            travel = 2.0 * math.pi
            turn = 0.5
            cycle = 2.0 * travel + 2.0 * turn

            phase = elapsed % cycle

            if phase <= travel:

                u = phase / travel

                theta = (
                    -math.pi / 2.0
                    + u * math.pi / 2.0
                )

                x = 1.8 + radius * math.cos(theta)
                y = 1.0 + radius * math.sin(theta)

                # Tangent direction along increasing theta.
                yaw = theta + math.pi / 2.0

            elif phase <= travel + turn:

                x = 3.8
                y = 1.0
                yaw = math.pi

            elif phase <= 2.0 * travel + turn:

                u = (
                    phase - travel - turn
                ) / travel

                theta = (
                    0.0
                    - u * math.pi / 2.0
                )

                x = 1.8 + radius * math.cos(theta)
                y = 1.0 + radius * math.sin(theta)

                # Tangent direction along decreasing theta.
                yaw = theta - math.pi / 2.0

            else:

                x = 1.8
                y = -1.0
                yaw = math.pi / 2.0

            return x, y, z, yaw

        # ---------------------------------------------------------
        # Diagonal trajectory:
        #
        # (1.0, -1.0) -> (2.6, 3.0)
        #
        # This changes both x and y simultaneously and creates
        # oblique crossings instead of perpendicular crossings.
        # ---------------------------------------------------------
        if mode == 'diagonal':

            x0 = 1.0
            y0 = -1.0

            x1 = 2.6
            y1 = 3.0

            dx = x1 - x0
            dy = y1 - y0

            distance = math.hypot(dx, dy)
            speed = 0.5

            travel = distance / speed
            turn = 0.5
            cycle = 2.0 * travel + 2.0 * turn

            phase = elapsed % cycle

            forward_yaw = math.atan2(dy, dx)
            reverse_yaw = math.atan2(-dy, -dx)

            if phase <= travel:

                u = phase / travel

                x = x0 + u * dx
                y = y0 + u * dy
                yaw = forward_yaw

            elif phase <= travel + turn:

                x = x1
                y = y1
                yaw = reverse_yaw

            elif phase <= 2.0 * travel + turn:

                u = (
                    phase - travel - turn
                ) / travel

                x = x1 - u * dx
                y = y1 - u * dy
                yaw = reverse_yaw

            else:

                x = x0
                y = y0
                yaw = forward_yaw

            return x, y, z, yaw

        # ---------------------------------------------------------
        # Stop-go:
        #   - walk from -1.0 to 0.5 over 3 s
        #   - stop at y=0.5 for 2 s
        #   - continue to 3.0 over 5 s
        #   - return continuously
        # ---------------------------------------------------------
        if mode == 'stop_go':

            cycle = 19.0
            phase = elapsed % cycle

            if phase <= 3.0:
                y = -1.0 + 0.5 * phase
                yaw = math.pi / 2.0

            elif phase <= 5.0:
                y = 0.5
                yaw = math.pi / 2.0

            elif phase <= 10.0:
                y = (
                    0.5
                    + 0.5 * (phase - 5.0)
                )
                yaw = math.pi / 2.0

            elif phase <= 10.5:
                y = 3.0
                yaw = -math.pi / 2.0

            elif phase <= 18.5:
                y = (
                    3.0
                    - 0.5 * (phase - 10.5)
                )
                yaw = -math.pi / 2.0

            else:
                y = -1.0
                yaw = math.pi / 2.0

            return x, y, z, yaw

        raise RuntimeError(mode)

    def update_proxy(self):

        # Do not queue Gazebo service calls.
        if self.last_future is not None:

            if not self.last_future.done():
                return

            try:
                self.last_future.result()
            except Exception as exc:
                self.get_logger().warn(
                    f'/set_entity_state request failed: {exc}'
                )

            self.last_future = None
            self.pending_pose = None

        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self.phase_origin

        x, y, z, yaw = self.target_pose(elapsed)

        state = EntityState()
        state.name = 'pedestrian_proxy'

        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z

        state.pose.orientation.z = math.sin(
            yaw / 2.0
        )
        state.pose.orientation.w = math.cos(
            yaw / 2.0
        )

        state.reference_frame = 'world'

        req = SetEntityState.Request()
        req.state = state

        self.pending_pose = state
        self.last_future = self.client.call_async(req)


def main():

    rclpy.init()

    node = PedestrianProxyController()

    try:
        rclpy.spin(node)

    except (
        KeyboardInterrupt,
        ExternalShutdownException,
    ):
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
