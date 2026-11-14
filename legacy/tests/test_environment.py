#!/usr/bin/env python3

import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from reachy_rl_control.environment import ReachyGazeboEnv
from reachy_rl_control.reachy_gazebo_interface import ReachyGazeboInterface


def main(args=None):
    rclpy.init(args=args)

    robot = ReachyGazeboInterface(
        reset_service="/reset_world"
    )

    executor = SingleThreadedExecutor()
    executor.add_node(robot)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )
    spin_thread.start()

    env = None

    try:
        print("\nWaiting for /odom...")

        if not robot.wait_for_odometry(timeout=5.0):
            raise RuntimeError(
                "No /odom received. Is Gazebo running?"
            )

        print("Odometry received.")

        env = ReachyGazeboEnv(
            robot=robot,
            control_duration=1.0,
            control_frequency=20.0,
        )

        # First reset
        print("\nResetting environment...")

        observation = env.reset()

        print("\nObservation after reset:")
        print(observation)

        # Move forward
        print("\nMoving Reachy forward for 1 second...")

        observation = env.step(
            [0.15, 0.0, 0.0]
        )

        print("\nObservation after movement:")
        print(observation)

        # Reset again
        print("\nResetting environment again...")

        observation = env.reset()

        print("\nObservation after second reset:")
        print(observation)

        print("\nEnvironment test completed successfully.")

    except KeyboardInterrupt:
        print("\nTest interrupted.")

    except Exception as error:
        print(f"\nERROR: {error}")

    finally:
        print("\nStopping Reachy safely...")

        if env is not None:
            env.close()
        else:
            robot.stop_for(
                duration=1.0,
                frequency=20.0,
            )

        executor.shutdown()

        robot.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(timeout=1.0)

        print("Environment test closed.")


if __name__ == "__main__":
    main()
