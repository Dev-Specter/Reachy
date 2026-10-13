#!/usr/bin/env python3

import os
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

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

        # ----------------------------------------------------
        # CREATE RL ENVIRONMENT
        # ----------------------------------------------------

        env = ReachyGazeboEnv(
            robot=robot,
            control_duration=0.1,
            max_steps=250,
        )

        env = Monitor(env)

        # ----------------------------------------------------
        # MODEL DIRECTORY
        # ----------------------------------------------------

        model_directory = (
            "/opt/reachy_ws/src/"
            "reachy_rl_control/models"
        )

        os.makedirs(
            model_directory,
            exist_ok=True,
        )

        model_path = os.path.join(
            model_directory,
            "ppo_raechy_10k",
        )

        # ----------------------------------------------------
        # CREATE PPO MODEL
        # ----------------------------------------------------

        print("\nCreating PPO model...")

        model = PPO(
            policy="MlpPolicy",
            env=env,

            learning_rate=3e-4,

            n_steps=256,

            batch_size=64,

            gamma=0.99,

            verbose=1,

            seed=42,
        )

        print("\nStarting PPO smoke training...")
        print("Total timesteps: 10000")
        print(
            "Reachy will move automatically "
            "during training."
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.learn(
            total_timesteps=10000,
            progress_bar=False,
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model.save(model_path)

        print("\nPPO smoke training completed.")

        print(
            f"Model saved to:\n{model_path}.zip"
        )

    except KeyboardInterrupt:
        print("\nPPO training interrupted.")

    except Exception as error:
        print(f"\nPPO TRAINING ERROR: {error}")
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

        print("PPO training process closed.")


if __name__ == "__main__":
    main()
