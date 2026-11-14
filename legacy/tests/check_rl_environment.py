#!/usr/bin/env python3

import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
from stable_baselines3.common.env_checker import check_env

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
            control_duration=0.1,
            max_steps=250,
        )

        print("\nRunning Stable-Baselines3 environment checker...")
        print(
            "Reachy may move randomly during this test. "
            "This is expected in Gazebo."
        )

        check_env(
            env,
            warn=True,
            skip_render_check=True,
        )

        print(
            "\nEnvironment checker PASSED."
        )

    except KeyboardInterrupt:
        print("\nEnvironment check interrupted.")

    except Exception as error:
        print(f"\nEnvironment checker ERROR: {error}")
        raise

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

        print("Environment checker closed.")


if __name__ == "__main__":
    main()
