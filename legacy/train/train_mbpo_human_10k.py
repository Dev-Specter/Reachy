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

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
)

from reachy_rl_control.environment_human import (
    ReachyHumanGazeboEnv,
)

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)

from reachy_rl_control.mbpo_dynamics import (
    MBPOCallback,
)


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

TOTAL_TIMESTEPS = 10000

MAX_STEPS = 250

CONTROL_DURATION = 0.1

SEED = 42


# =========================================================
# HUMAN TRAINING STATISTICS
# =========================================================

class HumanTrainingCallback(
    BaseCallback
):
    """
    Log episode outcomes separately from the MBPO callback.

    This does not modify training.
    It only records human-aware safety/task statistics.
    """

    def __init__(
        self,
        verbose=0,
    ):

        super().__init__(
            verbose=verbose
        )

        self.completed_episodes = 0

        self.goals = 0
        self.collisions = 0
        self.falls = 0
        self.timeouts = 0

        self.near_miss_episodes = 0

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

            minimum_clearance = float(
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
                "\nMBPO episode finished | "
                f"reason={reason} | "
                "minimum LiDAR clearance="
                f"{minimum_clearance:.3f} m"
            )

        return True


# =========================================================
# MAIN
# =========================================================

def main(
    args=None,
):

    # -----------------------------------------------------
    # ROS INITIALISATION
    # -----------------------------------------------------

    rclpy.init(
        args=args
    )

    robot = (
        ReachyGazeboInterface(
            reset_service="/reset_world"
        )
    )

    executor = (
        SingleThreadedExecutor()
    )

    executor.add_node(
        robot
    )

    spin_thread = (
        threading.Thread(
            target=executor.spin,
            daemon=True,
        )
    )

    spin_thread.start()

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
                "No /odom received. "
                "Is Gazebo running?"
            )

        print(
            "Odometry received."
        )

        # -------------------------------------------------
        # HUMAN-AWARE ENVIRONMENT
        # -------------------------------------------------

        print(
            "\nCreating human-aware MBPO environment..."
        )

        env = ReachyHumanGazeboEnv(
            robot=robot,
            control_duration=(
                CONTROL_DURATION
            ),
            max_steps=MAX_STEPS,
        )

        env = Monitor(
            env
        )

        print(
            "\nEnvironment ready."
        )

        print(
            "Observation size:",
            env.observation_space.shape,
        )

        print(
            "Action size:",
            env.action_space.shape,
        )

        # -------------------------------------------------
        # OUTPUT PATHS
        # -------------------------------------------------

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
            "mbpo_human_10k",
        )

        dynamics_path = os.path.join(
            model_directory,
            "mbpo_human_dynamics_10k.pt",
        )

        # -------------------------------------------------
        # SAC POLICY
        # -------------------------------------------------

        print(
            "\nCreating MBPO SAC policy..."
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

        # -------------------------------------------------
        # AUTOMATIC DIMENSIONS
        # -------------------------------------------------

        obs_dim = int(
            env.observation_space.shape[0]
        )

        action_dim = int(
            env.action_space.shape[0]
        )

        print(
            "\nMBPO dimensions:"
        )

        print(
            f"  observation dimension = "
            f"{obs_dim}"
        )

        print(
            f"  action dimension      = "
            f"{action_dim}"
        )

        if obs_dim != 20:

            raise RuntimeError(
                "Expected human-aware observation "
                f"dimension 20, received {obs_dim}."
            )

        if action_dim != 3:

            raise RuntimeError(
                "Expected action dimension 3, "
                f"received {action_dim}."
            )

        # -------------------------------------------------
        # MBPO DYNAMICS CALLBACK
        # -------------------------------------------------

        mbpo_callback = (
            MBPOCallback(
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

                seed=SEED,

                verbose=1,
            )
        )

        # -------------------------------------------------
        # SAFETY / OUTCOME LOGGING
        # -------------------------------------------------

        human_stats_callback = (
            HumanTrainingCallback()
        )

        callbacks = CallbackList(
            [
                mbpo_callback,
                human_stats_callback,
            ]
        )

        # -------------------------------------------------
        # PRINT CONFIGURATION
        # -------------------------------------------------

        print(
            "\n======================================"
        )

        print(
            "STARTING MBPO HUMAN 10K TRAINING"
        )

        print(
            "======================================"
        )

        print(
            f"Real environment timesteps: "
            f"{TOTAL_TIMESTEPS}"
        )

        print(
            f"Observation dimension: "
            f"{obs_dim}"
        )

        print(
            f"Action dimension: "
            f"{action_dim}"
        )

        print(
            f"Maximum episode steps: "
            f"{MAX_STEPS}"
        )

        print(
            "Control duration: "
            f"{CONTROL_DURATION:.1f} s"
        )

        print(
            "Human crossing speed: "
            "0.35 m/s"
        )

        print(
            f"Seed: {SEED}"
        )

        print(
            "\nMBPO configuration:"
        )

        print(
            "  ensemble_size          = 5"
        )

        print(
            "  model_warmup_steps     = 1000"
        )

        print(
            "  model_train_frequency  = 500"
        )

        print(
            "  model_gradient_steps   = 50"
        )

        print(
            "  model_batch_size       = 256"
        )

        print(
            "  synthetic_transitions  = 1000"
        )

        print(
            "  real_buffer_size       = 50000"
        )

        print(
            "  reward_scale           = 30.0"
        )

        print(
            "  synthetic horizon      = 1"
        )

        print(
            "\nReachy will move automatically."
        )

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        model.learn(
            total_timesteps=(
                TOTAL_TIMESTEPS
            ),

            callback=callbacks,

            progress_bar=False,
        )

        # -------------------------------------------------
        # SAVE SAC POLICY
        # -------------------------------------------------

        model.save(
            policy_path
        )

        # -------------------------------------------------
        # SAVE DYNAMICS ENSEMBLE
        # -------------------------------------------------

        mbpo_callback.save_dynamics(
            dynamics_path
        )

        # -------------------------------------------------
        # TRAINING SUMMARY
        # -------------------------------------------------

        print(
            "\n======================================"
        )

        print(
            "MBPO HUMAN 10K TRAINING SUMMARY"
        )

        print(
            "======================================"
        )

        print(
            "Completed episodes:",
            human_stats_callback.completed_episodes,
        )

        print(
            "Goals:",
            human_stats_callback.goals,
        )

        print(
            "Collisions:",
            human_stats_callback.collisions,
        )

        print(
            "Falls:",
            human_stats_callback.falls,
        )

        print(
            "Timeouts:",
            human_stats_callback.timeouts,
        )

        print(
            "Episodes with near miss:",
            human_stats_callback.near_miss_episodes,
        )

        print(
            "\nSynthetic transitions generated:",
            mbpo_callback.total_synthetic,
        )

        print(
            "\nMBPO human training completed."
        )

        print(
            "\nPolicy saved to:"
        )

        print(
            policy_path
            + ".zip"
        )

        print(
            "\nDynamics ensemble saved to:"
        )

        print(
            dynamics_path
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
            "\nStopping MBPO training safely..."
        )

        # -------------------------------------------------
        # CLOSE ENVIRONMENT
        # -------------------------------------------------

        if env is not None:

            try:

                env.close()

            except Exception as error:

                print(
                    "Environment close warning:",
                    error,
                )

        else:

            try:

                robot.stop_for(
                    duration=1.0,
                    frequency=20.0,
                )

            except Exception:

                pass

        # -------------------------------------------------
        # SHUTDOWN ROS
        # -------------------------------------------------

        try:

            executor.shutdown()

        except Exception:

            pass

        try:

            robot.destroy_node()

        except Exception:

            pass

        if rclpy.ok():

            rclpy.shutdown()

        if spin_thread.is_alive():

            spin_thread.join(
                timeout=1.0
            )

        print(
            "MBPO training process closed."
        )


if __name__ == "__main__":

    main()
