#!/usr/bin/env python3

import os
import threading

import rclpy

from rclpy.executors import (
    SingleThreadedExecutor,
)

from stable_baselines3 import SAC

from stable_baselines3.common.callbacks import (
    BaseCallback,
)

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)

from reachy_rl_control.environment_human import (
    ReachyHumanGazeboEnv,
)


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

TOTAL_TIMESTEPS = 10000

MAX_STEPS = 250

CONTROL_DURATION = 0.1

SEED = 42

MODEL_DIRECTORY = (
    "/opt/reachy_ws/src/"
    "reachy_rl_control/models"
)

MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "sac_human_10k",
)


# =========================================================
# TRAINING STATISTICS CALLBACK
# =========================================================

class HumanTrainingCallback(
    BaseCallback
):
    """
    Record human-aware episode outcomes during SAC training.
    """

    def __init__(
        self,
        verbose=0,
    ):
        super().__init__(
            verbose=verbose
        )

        self.goals = 0
        self.collisions = 0
        self.falls = 0
        self.timeouts = 0
        self.near_miss_episodes = 0
        self.completed_episodes = 0

    def _on_step(
        self,
    ) -> bool:

        infos = self.locals.get(
            "infos",
            [],
        )

        dones = self.locals.get(
            "dones",
            [],
        )

        for (
            done,
            info,
        ) in zip(
            dones,
            infos,
        ):

            if not done:
                continue

            self.completed_episodes += 1

            reason = str(
                info.get(
                    "termination_reason",
                    "unknown",
                )
            )

            near_miss = bool(
                info.get(
                    "near_miss_occurred",
                    False,
                )
            )

            minimum_distance = float(
                info.get(
                    "episode_min_lidar_distance",
                    float("nan"),
                )
            )

            if reason == "goal":

                self.goals += 1

            elif reason == "collision":

                self.collisions += 1

            elif reason == "fall":

                self.falls += 1

            elif reason == "timeout":

                self.timeouts += 1

            if near_miss:

                self.near_miss_episodes += 1

            print(
                "\nEpisode finished during training | "
                f"reason={reason} | "
                f"min_distance="
                f"{minimum_distance:.3f} m"
            )

        return True


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # ROS
    # -----------------------------------------------------

    rclpy.init()

    robot = (
        ReachyGazeboInterface()
    )

    robot_executor = (
        SingleThreadedExecutor()
    )

    robot_executor.add_node(
        robot
    )

    robot_spin_thread = (
        threading.Thread(
            target=robot_executor.spin,
            daemon=True,
        )
    )

    robot_spin_thread.start()

    env = None

    try:

        # -------------------------------------------------
        # WAIT FOR REACHY
        # -------------------------------------------------

        print(
            "\nWaiting for Reachy /odom..."
        )

        if not robot.wait_for_odometry(
            timeout=10.0
        ):

            raise RuntimeError(
                "No Reachy /odom data received. "
                "Is Gazebo running?"
            )

        print(
            "Reachy odometry received."
        )

        # -------------------------------------------------
        # HUMAN-AWARE ENVIRONMENT
        # -------------------------------------------------

        print(
            "\nCreating human-aware environment..."
        )

        env = ReachyHumanGazeboEnv(
            robot=robot,
            control_duration=(
                CONTROL_DURATION
            ),
            max_steps=MAX_STEPS,
        )

        print(
            "\nObservation size:",
            env.observation_space.shape,
        )

        print(
            "Action size:",
            env.action_space.shape,
        )

        # -------------------------------------------------
        # SAC MODEL
        # -------------------------------------------------

        print(
            "\nCreating SAC model..."
        )

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

            seed=SEED,

            device="cpu",
        )

        callback = (
            HumanTrainingCallback()
        )

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        print(
            "\n=================================="
        )

        print(
            "STARTING SAC HUMAN 10K TRAINING"
        )

        print(
            "=================================="
        )

        print(
            f"Total timesteps: "
            f"{TOTAL_TIMESTEPS}"
        )

        print(
            "Observation size: 20"
        )

        print(
            f"Maximum episode steps: "
            f"{MAX_STEPS}"
        )

        print(
            "Human crossing speed: "
            "0.35 m/s"
        )

        print(
            f"Random seed: {SEED}"
        )

        print(
            "\nSAC hyperparameters:"
        )

        print(
            "  learning_rate   = 0.0003"
        )

        print(
            "  buffer_size     = 100000"
        )

        print(
            "  learning_starts = 500"
        )

        print(
            "  batch_size      = 256"
        )

        print(
            "  tau             = 0.005"
        )

        print(
            "  gamma           = 0.99"
        )

        print(
            "  train_freq      = 1"
        )

        print(
            "  gradient_steps  = 1"
        )

        print(
            "  ent_coef        = auto"
        )

        print(
            "\nReachy will move automatically."
        )

        model.learn(
            total_timesteps=(
                TOTAL_TIMESTEPS
            ),

            callback=callback,

            progress_bar=False,
        )

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        model.save(
            MODEL_PATH
        )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        print(
            "\n=================================="
        )

        print(
            "SAC HUMAN 10K TRAINING SUMMARY"
        )

        print(
            "=================================="
        )

        print(
            "Completed episodes:",
            callback.completed_episodes,
        )

        print(
            "Goals:",
            callback.goals,
        )

        print(
            "Collisions:",
            callback.collisions,
        )

        print(
            "Falls:",
            callback.falls,
        )

        print(
            "Timeouts:",
            callback.timeouts,
        )

        print(
            "Episodes with near miss:",
            callback.near_miss_episodes,
        )

        print(
            "\nSAC human training completed."
        )

        print(
            "Model saved to:"
        )

        print(
            MODEL_PATH + ".zip"
        )

    except KeyboardInterrupt:

        print(
            "\nSAC training interrupted."
        )

    finally:

        print(
            "\nStopping training safely..."
        )

        # -------------------------------------------------
        # CLOSE HUMAN ENVIRONMENT
        # -------------------------------------------------

        if env is not None:

            try:

                env.close()

            except Exception as exc:

                print(
                    "Environment close warning:",
                    exc,
                )

        # -------------------------------------------------
        # STOP ROBOT EXECUTOR
        # -------------------------------------------------

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

        if robot_spin_thread.is_alive():

            robot_spin_thread.join(
                timeout=1.0
            )

        print(
            "Training process closed."
        )


if __name__ == "__main__":
    main()
