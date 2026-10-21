#!/usr/bin/env python3

import os
import threading

import rclpy

from rclpy.executors import (
    ExternalShutdownException,
    SingleThreadedExecutor,
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)

from reachy_rl_control.environment_human import (
    ReachyHumanGazeboEnv,
)


TOTAL_TIMESTEPS = 2000

MODEL_PATH = (
    "/opt/reachy_ws/src/reachy_rl_control/models/"
    "ppo_human_smoke_fixed"
)


class HumanTrainingCallback(BaseCallback):

    def __init__(self):
        super().__init__()

        self.goals = 0
        self.collisions = 0
        self.falls = 0
        self.timeouts = 0
        self.near_misses = 0

    def _on_step(self) -> bool:

        infos = self.locals.get(
            "infos",
            [],
        )

        dones = self.locals.get(
            "dones",
            [],
        )

        for info, done in zip(
            infos,
            dones,
        ):

            if not done:
                continue

            reason = info.get(
                "termination_reason",
                "unknown",
            )

            if reason == "goal":
                self.goals += 1

            elif reason == "collision":
                self.collisions += 1

            elif reason == "fall":
                self.falls += 1

            elif reason == "timeout":
                self.timeouts += 1

            if info.get(
                "near_miss_occurred",
                False,
            ):
                self.near_misses += 1

            print(
                "\nEpisode finished during training | "
                f"reason={reason} | "
                f"min_distance="
                f"{info.get('episode_min_lidar_distance', 0.0):.3f} m"
            )

        return True

    def _on_training_end(self):

        print(
            "\n===== PPO HUMAN SMOKE SUMMARY ====="
        )

        print(
            "Goals:",
            self.goals,
        )

        print(
            "Collisions:",
            self.collisions,
        )

        print(
            "Falls:",
            self.falls,
        )

        print(
            "Timeouts:",
            self.timeouts,
        )

        print(
            "Episodes with near miss:",
            self.near_misses,
        )


def spin_robot_executor(
    executor,
):

    try:

        executor.spin()

    except ExternalShutdownException:

        pass


def main():

    rclpy.init()

    robot = ReachyGazeboInterface()

    robot_executor = (
        SingleThreadedExecutor()
    )

    robot_executor.add_node(
        robot
    )

    robot_thread = threading.Thread(
        target=spin_robot_executor,
        args=(robot_executor,),
        daemon=True,
    )

    robot_thread.start()

    env = None

    try:

        print(
            "\nWaiting for Reachy /odom..."
        )

        if not robot.wait_for_odometry(
            timeout=10.0
        ):

            raise RuntimeError(
                "No Reachy odometry received."
            )

        print(
            "Reachy odometry received."
        )

        print(
            "\nCreating human-aware environment..."
        )

        env = ReachyHumanGazeboEnv(
            robot=robot,
            control_duration=0.1,
            max_steps=250,
        )

        os.makedirs(
            os.path.dirname(
                MODEL_PATH
            ),
            exist_ok=True,
        )

        print(
            "\nCreating PPO model..."
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=42,
            device="cpu",
        )

        callback = (
            HumanTrainingCallback()
        )

        print(
            "\n=================================="
        )

        print(
            "STARTING PPO HUMAN SMOKE TRAINING"
        )

        print(
            "=================================="
        )

        print(
            "Total timesteps:",
            TOTAL_TIMESTEPS,
        )

        print(
            "Observation size: 20"
        )

        print(
            "Maximum episode steps: 250"
        )

        print(
            "Human crossing speed: 0.35 m/s"
        )

        print(
            "\nReachy will move automatically."
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
        )

        model.save(
            MODEL_PATH
        )

        print(
            "\nPPO human smoke training completed."
        )

        print(
            "Model saved to:"
        )

        print(
            MODEL_PATH + ".zip"
        )

    except KeyboardInterrupt:

        print(
            "\nTraining interrupted."
        )

    finally:

        print(
            "\nStopping training safely..."
        )

        if env is not None:

            try:

                env.close()

            except Exception as exc:

                print(
                    "Environment close warning:",
                    exc,
                )

        try:

            robot_executor.shutdown()

        except Exception:

            pass

        try:

            robot.destroy_node()

        except Exception:

            pass

        if rclpy.ok():

            rclpy.shutdown()

        if robot_thread.is_alive():

            robot_thread.join(
                timeout=1.0
            )

        print(
            "Training process closed."
        )


if __name__ == "__main__":
    main()
