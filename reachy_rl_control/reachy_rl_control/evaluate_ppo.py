#!/usr/bin/env python3

import os
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from stable_baselines3 import PPO

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

        # --------------------------------------------------
        # CREATE ENVIRONMENT
        # --------------------------------------------------

        env = ReachyGazeboEnv(
            robot=robot,
            control_duration=0.1,
            max_steps=250,
        )

        # --------------------------------------------------
        # MODEL PATH
        # --------------------------------------------------

        model_path = (
            "/opt/reachy_ws/src/"
            "reachy_rl_control/models/"
            "ppo_reachy_normalized_smoke.zip"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PPO model not found: {model_path}"
            )

        print("\nLoading PPO model...")
        print(model_path)

        model = PPO.load(model_path)

        print("PPO model loaded successfully.")

        # --------------------------------------------------
        # RESET ENVIRONMENT
        # --------------------------------------------------

        observation, info = env.reset()

        print("\n===== PPO EVALUATION =====")
        print("Running deterministic policy.")
        print("Maximum steps: 250")

        total_reward = 0.0

        terminated = False
        truncated = False

        # --------------------------------------------------
        # EVALUATION LOOP
        # --------------------------------------------------

        for step in range(1, 251):

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += reward

            print(
                f"Step {step:03d} | "
                f"Action: "
                f"[{action[0]:+.3f}, "
                f"{action[1]:+.3f}, "
                f"{action[2]:+.3f}] | "
                f"Reward: {reward:+.3f} | "
                f"Waypoint: {info['waypoint']} | "
                f"Distance: "
                f"{info['distance_to_goal']:.3f} m | "
                f"Terminated: {terminated} | "
                f"Truncated: {truncated}"
            )

            if terminated:
                print(
                    "\nPPO reached the final goal."
                )
                break

            if truncated:
                print(
                    "\nPPO reached the episode "
                    "step limit."
                )
                break

        # --------------------------------------------------
        # FINAL RESULTS
        # --------------------------------------------------

        print("\n===== PPO EVALUATION RESULT =====")

        print(
            f"Total reward: "
            f"{total_reward:.3f}"
        )

        print(
            f"Steps: "
            f"{info['step']}"
        )

        print(
            f"Final waypoint: "
            f"{info['waypoint']}"
        )

        print(
            f"Final distance to waypoint: "
            f"{info['distance_to_goal']:.3f} m"
        )

        print(
            f"Terminated: "
            f"{terminated}"
        )

        print(
            f"Truncated: "
            f"{truncated}"
        )

        print("\nFinal observation:")
        print(observation)

        if terminated:
            print(
                "\nPPO smoke model SUCCESSFULLY "
                "completed the trajectory."
            )
        else:
            print(
                "\nPPO smoke model did not yet "
                "complete the trajectory."
            )

            print(
                "This is normal after only "
                "~2000 training timesteps."
            )

    except KeyboardInterrupt:
        print("\nPPO evaluation interrupted.")

    except Exception as error:
        print(
            f"\nPPO EVALUATION ERROR: {error}"
        )
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

        print("PPO evaluation process closed.")


if __name__ == "__main__":
    main()
