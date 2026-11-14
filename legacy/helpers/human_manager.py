#!/usr/bin/env python3

import argparse
import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from ament_index_python.packages import (
    get_package_share_directory,
)

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import (
    SpawnEntity,
    DeleteEntity,
)


class HumanManager(Node):

    def __init__(self):

        super().__init__(
            "human_manager"
        )

        self.human_name = (
            "human_1"
        )

        # --------------------------------------------------
        # Gazebo services
        # --------------------------------------------------

        self.spawn_client = (
            self.create_client(
                SpawnEntity,
                "/spawn_entity",
            )
        )

        self.delete_client = (
            self.create_client(
                DeleteEntity,
                "/delete_entity",
            )
        )

        # --------------------------------------------------
        # Human velocity publisher
        # --------------------------------------------------

        self.velocity_publisher = (
            self.create_publisher(
                Twist,
                "/human_1/cmd_vel",
                10,
            )
        )

        # --------------------------------------------------
        # Human contact / collision subscriber
        # --------------------------------------------------

        self.contact_subscription = (
            self.create_subscription(
                ContactsState,
                "/human_1/contacts",
                self._contact_callback,
                10,
            )
        )

        self.collision_detected = False

        self.collision_pairs = []

        # --------------------------------------------------
        # Permanent human SDF
        # --------------------------------------------------

        package_share = (
            get_package_share_directory(
                "reachy_rl_control"
            )
        )

        self.sdf_path = (
            os.path.join(
                package_share,
                "models",
                "human_contact.sdf",
            )
        )

        if not os.path.isfile(
            self.sdf_path
        ):

            raise FileNotFoundError(
                f"Human SDF not found: "
                f"{self.sdf_path}"
            )

        with open(
            self.sdf_path,
            "r",
            encoding="utf-8",
        ) as sdf_file:

            self.human_sdf = (
                sdf_file.read()
            )

        self.get_logger().info(
            "Human model loaded from "
            f"{self.sdf_path}"
        )

    # ------------------------------------------------------
    # Contacts
    # ------------------------------------------------------

    def _contact_callback(
        self,
        msg,
    ):

        if len(
            msg.states
        ) == 0:

            return

        self.collision_detected = True

        for state in msg.states:

            pair = (
                state.collision1_name,
                state.collision2_name,
            )

            if pair not in (
                self.collision_pairs
            ):

                self.collision_pairs.append(
                    pair
                )

                self.get_logger().warn(
                    "HUMAN COLLISION DETECTED: "
                    f"{pair[0]} <-> "
                    f"{pair[1]}"
                )

    def clear_collision(
        self,
    ):

        self.collision_detected = False

        self.collision_pairs = []

    # ------------------------------------------------------
    # Future helper
    # ------------------------------------------------------

    def _wait_for_future(
        self,
        future,
        timeout=5.0,
    ):

        start = (
            time.monotonic()
        )

        while not future.done():

            if (
                time.monotonic()
                - start
                > timeout
            ):

                return None

            time.sleep(
                0.01
            )

        return future.result()

    # ------------------------------------------------------
    # Services
    # ------------------------------------------------------

    def wait_for_services(
        self,
        timeout=5.0,
    ):

        spawn_ready = (
            self.spawn_client.wait_for_service(
                timeout_sec=timeout
            )
        )

        delete_ready = (
            self.delete_client.wait_for_service(
                timeout_sec=timeout
            )
        )

        return (
            spawn_ready
            and delete_ready
        )

    def delete_human(
        self,
    ):

        request = (
            DeleteEntity.Request()
        )

        request.name = (
            self.human_name
        )

        future = (
            self.delete_client.call_async(
                request
            )
        )

        response = (
            self._wait_for_future(
                future,
                timeout=5.0,
            )
        )

        if response is None:

            self.get_logger().warn(
                "Delete human service "
                "timed out."
            )

            return False

        if response.success:

            self.get_logger().info(
                "Existing human deleted."
            )

            return True

        # This is normal if no human currently exists.
        self.get_logger().info(
            "No existing human "
            "needed deleting."
        )

        return False

    def spawn_human(
        self,
        x=1.0,
        y=-1.5,
        z=0.9,
        reference_frame="world",
    ):

        """
        Spawn the simulated human.

        x, y and z are interpreted relative to
        reference_frame.

        Examples:

            reference_frame="world"

        means x/y/z are Gazebo world coordinates.

            reference_frame="reachy"

        means x/y/z are relative to Reachy's
        Gazebo model frame.
        """

        request = (
            SpawnEntity.Request()
        )

        request.name = (
            self.human_name
        )

        request.xml = (
            self.human_sdf
        )

        request.robot_namespace = ""

        # --------------------------------------------------
        # IMPORTANT:
        #
        # SpawnEntity interprets initial_pose relative to
        # this Gazebo entity/frame.
        #
        # For the human-aware RL environment we will use:
        #
        #     reference_frame="reachy"
        #
        # so the crossing scenario does not depend on
        # Reachy's /odom coordinates.
        # --------------------------------------------------

        request.reference_frame = str(
            reference_frame
        )

        request.initial_pose.position.x = (
            float(x)
        )

        request.initial_pose.position.y = (
            float(y)
        )

        request.initial_pose.position.z = (
            float(z)
        )

        # Identity orientation relative to the requested
        # reference frame.
        request.initial_pose.orientation.x = (
            0.0
        )

        request.initial_pose.orientation.y = (
            0.0
        )

        request.initial_pose.orientation.z = (
            0.0
        )

        request.initial_pose.orientation.w = (
            1.0
        )

        future = (
            self.spawn_client.call_async(
                request
            )
        )

        response = (
            self._wait_for_future(
                future,
                timeout=5.0,
            )
        )

        if response is None:

            raise RuntimeError(
                "Spawn human service "
                "timed out."
            )

        if not response.success:

            raise RuntimeError(
                "Could not spawn human: "
                f"{response.status_message}"
            )

        self.get_logger().info(
            "Human spawned at "
            f"x={x:.2f}, "
            f"y={y:.2f}, "
            f"z={z:.2f} "
            f"relative to "
            f"'{reference_frame}'"
        )

        return True

    # ------------------------------------------------------
    # Human movement
    # ------------------------------------------------------

    def send_velocity(
        self,
        vx=0.0,
        vy=0.0,
        omega=0.0,
    ):

        msg = Twist()

        msg.linear.x = (
            float(vx)
        )

        msg.linear.y = (
            float(vy)
        )

        msg.angular.z = (
            float(omega)
        )

        self.velocity_publisher.publish(
            msg
        )

    def stop(
        self,
    ):

        # Publish several times for reliability.
        for _ in range(5):

            self.send_velocity(
                0.0,
                0.0,
                0.0,
            )

            time.sleep(
                0.02
            )

    def move_for(
        self,
        vx,
        vy,
        omega,
        duration,
        stop_on_collision=True,
        frequency=10.0,
    ):

        period = (
            1.0
            / frequency
        )

        start = (
            time.monotonic()
        )

        while (
            time.monotonic()
            - start
            < duration
        ):

            if (
                stop_on_collision
                and
                self.collision_detected
            ):

                self.get_logger().warn(
                    "Stopping human because "
                    "a collision was detected."
                )

                break

            self.send_velocity(
                vx,
                vy,
                omega,
            )

            time.sleep(
                period
            )

        self.stop()

    # ------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------

    def reset_human(
        self,
        x=1.0,
        y=-1.5,
        z=0.9,
        reference_frame="world",
    ):

        """
        Stop, delete and respawn the human.

        The supplied position is interpreted relative
        to reference_frame.
        """

        self.stop()

        self.delete_human()

        time.sleep(
            0.25
        )

        self.clear_collision()

        self.spawn_human(
            x=x,
            y=y,
            z=z,
            reference_frame=reference_frame,
        )

        # Give Gazebo time to initialise plugins.
        time.sleep(
            0.75
        )

        self.get_logger().info(
            "Human reset completed."
        )


def main(
    args=None,
):

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--x",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--y",
        type=float,
        default=-1.5,
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--vy",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--omega",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--reference-frame",
        type=str,
        default="world",
        help=(
            "Gazebo entity/frame used as the "
            "spawn reference. "
            "Examples: world, reachy"
        ),
    )

    parsed_args, ros_args = (
        parser.parse_known_args()
    )

    rclpy.init(
        args=ros_args
    )

    node = HumanManager()

    executor = (
        SingleThreadedExecutor()
    )

    executor.add_node(
        node
    )

    spin_thread = (
        threading.Thread(
            target=executor.spin,
            daemon=True,
        )
    )

    spin_thread.start()

    try:

        print(
            "\nWaiting for Gazebo services..."
        )

        if not node.wait_for_services():

            raise RuntimeError(
                "Gazebo spawn/delete services "
                "are unavailable."
            )

        print(
            "Gazebo services ready."
        )

        node.reset_human(
            x=parsed_args.x,
            y=parsed_args.y,
            reference_frame=(
                parsed_args.reference_frame
            ),
        )

        print(
            "\nHuman crossing test:"
        )

        print(
            f"reference_frame="
            f"{parsed_args.reference_frame}"
        )

        print(
            f"vx={parsed_args.vx:.2f}, "
            f"vy={parsed_args.vy:.2f}, "
            f"duration="
            f"{parsed_args.duration:.1f}s"
        )

        node.move_for(
            vx=parsed_args.vx,
            vy=parsed_args.vy,
            omega=parsed_args.omega,
            duration=parsed_args.duration,
        )

        print(
            "\nTest completed."
        )

        print(
            "Collision detected:",
            node.collision_detected,
        )

        if node.collision_pairs:

            print(
                "Collision pairs:"
            )

            for pair in (
                node.collision_pairs
            ):

                print(
                    f"  {pair[0]} "
                    f"<-> {pair[1]}"
                )

    except KeyboardInterrupt:

        print(
            "\nHuman test interrupted."
        )

    finally:

        node.stop()

        executor.shutdown()

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()

        spin_thread.join(
            timeout=1.0
        )


if __name__ == "__main__":
    main()
