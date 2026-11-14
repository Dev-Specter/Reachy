#!/usr/bin/env python3

import os
import threading

import rclpy
from rclpy.executors import (
    SingleThreadedExecutor,
)

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import (
    Monitor,
)

from reachy_rl_control.environment import (
    ReachyGazeboEnv,
)

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)

from reachy_rl_control.mbpo_dynamics import (
    MBPOCallback,
)


TOTAL_TIMESTEPS = 10000


def main(args=None):

    rclpy.init(args=args)

    robot = ReachyGazeboInterface(
        reset_service="/reset_world"
    )

    executor = (
        SingleThreadedExecutor()
    )

    executor.add_node(robot)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )

    spin_thread.start()

    env = None

    try:

        print(
            "\nWaiting for /odom..."
        )

        if not robot.wait_for_odometry(
            timeout=5.0
        ):
            raise RuntimeError(
                "No /odom received. "
                "Is Gazebo running?"
            )

        print(
            "Odometry received."
        )

        env = ReachyGazeboEnv(
            robot=robot,
            control_duration=0.1,
            max_steps=250,
        )

        env = Monitor(env)

        model_directory = (
            "/opt/reachy_ws/src/"
            "reachy_rl_control/models"
        )

        os.makedirs(
            model_directory,
            exist_ok=True,
        )

        policy_path = os.path.join(
            model_directory,
            "mbpo_reachy_10k",
        )

        dynamics_path = os.path.join(
            model_directory,
            "mbpo_dynamics_10k.pt",
        )

        print(
            "\nCreating MBPO policy..."
        )

        # MBPO uses SAC as the policy
        # optimizer.
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

        obs_dim = (
            env.observation_space
            .shape[0]
        )

        action_dim = (
            env.action_space
            .shape[0]
        )

        mbpo_callback = MBPOCallback(
            obs_dim=obs_dim,
            action_dim=action_dim,

            ensemble_size=5,

            model_warmup_steps=1000,

            model_train_frequency=500,

            model_gradient_steps=50,

            model_batch_size=256,

            synthetic_transitions=1000,

            real_buffer_size=50000,

            reward_scale=30.0,

            seed=42,

            verbose=1,
        )

        print(
            "\nStarting MBPO smoke training..."
        )

        print(
            f"Real environment timesteps: "
            f"{TOTAL_TIMESTEPS}"
        )

        print(
            "Reachy will move automatically."
        )

        model.learn(
            total_timesteps=(
                TOTAL_TIMESTEPS
            ),
            callback=mbpo_callback,
            progress_bar=False,
        )

        model.save(
            policy_path
        )

        mbpo_callback.save_dynamics(
            dynamics_path
        )

        print(
            "\nMBPO smoke training "
            "completed."
        )

        print(
            "Policy saved to:\n"
            f"{policy_path}.zip"
        )

        print(
            "Dynamics ensemble saved to:\n"
            f"{dynamics_path}"
        )

    except KeyboardInterrupt:

        print(
            "\nMBPO training interrupted."
        )

    except Exception as error:

        print(
            "\nMBPO TRAINING ERROR: "
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
            "MBPO training process closed."
        )


if __name__ == "__main__":
    main()
