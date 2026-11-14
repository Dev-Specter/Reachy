#!/usr/bin/env python3

import os
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from reachy_rl_control.environment import ReachyGazeboEnv
from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)


TOTAL_TIMESTEPS = 10000


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

        if not robot.wait_for_odometry(
            timeout=5.0
        ):
            raise RuntimeError(
                "No /odom received. "
                "Is Gazebo running?"
            )

        print("Odometry received.")

        # -------------------------------------------------
        # ENVIRONMENT
        # -------------------------------------------------

        env = ReachyGazeboEnv(
            robot=robot,
            control_duration=0.1,
            max_steps=250,
        )

        env = Monitor(env)

        # -------------------------------------------------
        # MODEL DIRECTORY
        # -------------------------------------------------

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
            "sac_reachy_10k",
        )

        # -------------------------------------------------
        # SAC
        # -------------------------------------------------

        print("\nCreating SAC model...")

        model = SAC(
            policy="MlpPolicy",
            env=env,

            learning_rate=3e-4,

            buffer_size=100000,

            learning_starts=500,

            batch_size=256,

            tau=0.005,

            gamma=0.99,

            train_freq=1,

            gradient_steps=1,

            ent_coef="auto",

            verbose=1,

            seed=42,
        )

        print("\nStarting SAC smoke training...")
        print(
            f"Total timesteps: "
            f"{TOTAL_TIMESTEPS}"
        )

        print(
            "Reachy will move automatically "
            "during training."
        )

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=False,
        )

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        model.save(
            model_path
        )

        print(
            "\nSAC smoke training completed."
        )

        print(
            f"Model saved to:\n"
            f"{model_path}.zip"
        )

    except KeyboardInterrupt:

        print(
            "\nSAC training interrupted."
        )

    except Exception as error:

        print(
            f"\nSAC TRAINING ERROR: "
            f"{error}"
        )

        raise

    finally:

        print(
            "\nStopping Reachy safely..."
        )

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

        spin_thread.join(
            timeout=1.0
        )

        print(
            "SAC training process closed."
        )


if __name__ == "__main__":
    main()
