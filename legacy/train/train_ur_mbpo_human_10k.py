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

from reachy_rl_control.ur_mbpo_dynamics import (
    URMBPOCallback,
)


# =========================================================
# SMOKE TEST CONFIGURATION
# =========================================================

TOTAL_TIMESTEPS = 10000

MAX_STEPS = 250

CONTROL_DURATION = 0.1

SEED = 42


# =========================================================
# HUMAN EPISODE STATISTICS
# =========================================================

class HumanTrainingCallback(
    BaseCallback
):
    """
    Passive logging callback.

    It does not alter training.
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

        for done, info in zip(
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
                "\nUR-MBPO episode finished | "
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
            "\nCreating human-aware environment..."
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

        obs_dim = int(
            env.observation_space.shape[0]
        )

        action_dim = int(
            env.action_space.shape[0]
        )

        print(
            "\nEnvironment ready."
        )

        print(
            "Observation dimension:",
            obs_dim,
        )

        print(
            "Action dimension:",
            action_dim,
        )

        if obs_dim != 20:

            raise RuntimeError(
                "Expected human-aware "
                f"observation dimension 20, "
                f"received {obs_dim}."
            )

        if action_dim != 3:

            raise RuntimeError(
                "Expected action dimension 3, "
                f"received {action_dim}."
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
            "ur_mbpo_human_10k",
        )

        dynamics_path = os.path.join(
            model_directory,
            "ur_mbpo_human_dynamics_10k.pt",
        )

        # -------------------------------------------------
        # SAC POLICY
        # -------------------------------------------------

        print(
            "\nCreating UR-MBPO SAC policy..."
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
        # UR-MBPO CALLBACK
        # -------------------------------------------------

        ur_mbpo_callback = (
            URMBPOCallback(
                obs_dim=obs_dim,

                action_dim=action_dim,

                # -----------------------------------------
                # SAME MBPO PARAMETERS AS BASELINE
                # -----------------------------------------

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

                # -----------------------------------------
                # INITIAL UR-MBPO PARAMETERS
                # -----------------------------------------

                uncertainty_threshold=0.05,

                uncertainty_penalty_scale=2.0,

                safe_clearance=0.75,

                near_miss_clearance=0.50,

                lidar_max_distance=3.0,

                risk_penalty_scale=2.0,

                lidar_start_index=8,
            )
        )

        human_stats_callback = (
            HumanTrainingCallback()
        )

        callbacks = CallbackList(
            [
                ur_mbpo_callback,
                human_stats_callback,
            ]
        )

        # -------------------------------------------------
        # CONFIGURATION SUMMARY
        # -------------------------------------------------

        print(
            "\n======================================"
        )

        print(
            "UR-MBPO HUMAN SMOKE TEST"
        )

        print(
            "======================================"
        )

        print(
            "Real environment timesteps:",
            TOTAL_TIMESTEPS,
        )

        print(
            "Observation dimension:",
            obs_dim,
        )

        print(
            "Action dimension:",
            action_dim,
        )

        print(
            "Maximum episode steps:",
            MAX_STEPS,
        )

        print(
            "Seed:",
            SEED,
        )

        print(
            "\nBaseline MBPO parameters:"
        )

        print(
            "  ensemble_size         = 5"
        )

        print(
            "  model_warmup_steps    = 1000"
        )

        print(
            "  model_train_frequency = 500"
        )

        print(
            "  model_gradient_steps  = 50"
        )

        print(
            "  model_batch_size      = 256"
        )

        print(
            "  synthetic_transitions = 1000"
        )

        print(
            "  real_buffer_size      = 50000"
        )

        print(
            "  reward_scale          = 30.0"
        )

        print(
            "\nInitial UR parameters:"
        )

        print(
            "  uncertainty_threshold     = 0.05"
        )

        print(
            "  uncertainty_penalty_scale = 2.0"
        )

        print(
            "  safe_clearance             = 0.75 m"
        )

        print(
            "  near_miss_clearance        = 0.50 m"
        )

        print(
            "  risk_penalty_scale         = 2.0"
        )

        print(
            "\nThis is a diagnostic smoke run."
        )

        print(
            "Do not use its policy performance "
            "as a dissertation result."
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
        # SAVE SMOKE MODEL
        # -------------------------------------------------

        model.save(
            policy_path
        )

        ur_mbpo_callback.save_dynamics(
            dynamics_path
        )

        # -------------------------------------------------
        # ACCEPTANCE RATE
        # -------------------------------------------------

        total_candidates = (
            ur_mbpo_callback.total_candidates
        )

        total_accepted = (
            ur_mbpo_callback.total_accepted
        )

        total_rejected = (
            ur_mbpo_callback
            .total_uncertainty_rejected
        )

        if total_candidates > 0:

            acceptance_rate = (
                100.0
                *
                total_accepted
                /
                total_candidates
            )

        else:

            acceptance_rate = 0.0

        # -------------------------------------------------
        # FINAL SUMMARY
        # -------------------------------------------------

        print(
            "\n======================================"
        )

        print(
            "UR-MBPO SMOKE TEST SUMMARY"
        )

        print(
            "======================================"
        )

        print(
            "\nReal environment:"
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
            "\nSynthetic model data:"
        )

        print(
            "Candidates:",
            total_candidates,
        )

        print(
            "Accepted:",
            total_accepted,
        )

        print(
            "Rejected by uncertainty:",
            total_rejected,
        )

        print(
            "Acceptance rate:",
            f"{acceptance_rate:.1f}%",
        )

        print(
            "Risk-penalized accepted transitions:",
            ur_mbpo_callback.total_risk_penalized,
        )

        print(
            "Predicted near-miss accepted transitions:",
            ur_mbpo_callback
            .total_near_miss_predictions,
        )

        print(
            "\nLatest uncertainty:"
        )

        print(
            "Minimum:",
            f"{ur_mbpo_callback.last_min_uncertainty:.6f}",
        )

        print(
            "Mean:",
            f"{ur_mbpo_callback.last_mean_uncertainty:.6f}",
        )

        print(
            "Maximum:",
            f"{ur_mbpo_callback.last_max_uncertainty:.6f}",
        )

        print(
            "\nLatest predicted LiDAR clearance:"
        )

        print(
            "Mean:",
            f"{ur_mbpo_callback.last_mean_predicted_clearance:.3f} m",
        )

        print(
            "Minimum:",
            f"{ur_mbpo_callback.last_min_predicted_clearance:.3f} m",
        )

        print(
            "\nSmoke policy saved to:"
        )

        print(
            policy_path
            + ".zip"
        )

        print(
            "\nSmoke dynamics saved to:"
        )

        print(
            dynamics_path
        )

    except KeyboardInterrupt:

        print(
            "\nUR-MBPO smoke test interrupted."
        )

    except Exception as error:

        print(
            "\nUR-MBPO SMOKE TEST ERROR:"
        )

        print(
            error
        )

        raise

    finally:

        print(
            "\nStopping UR-MBPO smoke test safely..."
        )

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
            "UR-MBPO smoke process closed."
        )


if __name__ == "__main__":

    main()
