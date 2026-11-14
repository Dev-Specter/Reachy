#!/usr/bin/env python3

import threading

import numpy as np
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
            control_duration=0.1,
            max_steps=250,
        )

        # ------------------------------------------------
        # RESET TEST
        # ------------------------------------------------

        observation, info = env.reset()

        print("\n===== RESET TEST =====")

        print("Observation:")
        print(observation)

        print("\nObservation shape:")
        print(observation.shape)

        print("\nReset info:")
        print(info)

        print("\nAction space:")
        print(env.action_space)

        print("\nObservation space:")
        print(env.observation_space)

        if observation.shape != (8,):
            raise RuntimeError(
                f"Expected observation shape (8,), "
                f"got {observation.shape}"
            )

        if env.action_space.shape != (3,):
            raise RuntimeError(
                f"Expected action shape (3,), "
                f"got {env.action_space.shape}"
            )

        print("\nObservation/action space test passed.")

        # ------------------------------------------------
        # STEP TEST
        # ------------------------------------------------

        print("\n===== STEP TEST =====")

        action = np.array(
            [0.15, 0.0, 0.0],
            dtype=np.float32,
        )

        print(
            "\nApplying fixed forward action:",
            action,
        )

        terminated = False
        truncated = False

        for step in range(1, 251):

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            print(
                f"Step {step:02d} | "
                f"Reward: {reward:7.3f} | "
                f"Waypoint: {info['waypoint']} | "
                f"Distance: {info['distance_to_goal']:.3f} m | "
                f"Progress: {info['progress']:.4f} | "
                f"Terminated: {terminated} | "
                f"Truncated: {truncated}"
            )

            if terminated or truncated:
                print("\nEpisode finished.")
                break

        # ------------------------------------------------
        # FINAL INFORMATION
        # ------------------------------------------------

        print("\n===== FINAL RESULT =====")

        print("Final observation:")
        print(observation)

        print("\nFinal info:")
        print(info)

        print(
            "\nRL environment test completed successfully."
        )

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

        print("RL environment test closed.")


if __name__ == "__main__":
    main()
