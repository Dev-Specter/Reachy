#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState


class PedestrianProxyController(Node):

    def __init__(self):
        super().__init__('pedestrian_proxy_controller')


        self.client = self.create_client(
            SetEntityState,
            '/set_entity_state'
        )

        if not self.client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('/set_entity_state unavailable')

        self.cycle = 17.0
        self.last_future = None

        # 20 Hz motion updates
        self.timer = self.create_timer(0.05, self.update_proxy)

        self.get_logger().info(
            'pedestrian_proxy_controller started'
        )

    def update_proxy(self):

        # Gazebo /clock via ROS simulated time.
        t = self.get_clock().now().nanoseconds / 1e9
        phase = t % self.cycle

        x = 1.8
        z = 0.85

        if phase <= 8.0:
            y = -1.0 + 0.5 * phase
            yaw = math.pi / 2.0

        elif phase <= 8.5:
            y = 3.0
            yaw = -math.pi / 2.0

        elif phase <= 16.5:
            y = 3.0 - 0.5 * (phase - 8.5)
            yaw = -math.pi / 2.0

        else:
            y = -1.0
            yaw = math.pi / 2.0

        state = EntityState()
        state.name = 'pedestrian_proxy'

        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z

        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)

        state.reference_frame = 'world'

        req = SetEntityState.Request()
        req.state = state

        self.last_future = self.client.call_async(req)


def main():
    rclpy.init()
    node = PedestrianProxyController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
