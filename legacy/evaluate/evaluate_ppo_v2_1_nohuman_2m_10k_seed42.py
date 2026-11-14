#!/usr/bin/env python3

import csv
import math
import os
import threading
import time

import numpy as np
import rclpy

from rclpy.executors import SingleThreadedExecutor
from stable_baselines3 import PPO

from reachy_rl_control.environment_v2_1 import (
    ReachyGazeboEnv,
)

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)


# =========================================================
# CONFIGURATION
# =========================================================

NUM_EPISODES = 20

MAX_STEPS = 250

CONTROL_DURATION = 0.1

MODEL_PATH = (
    "/opt/reachy_ws/src/reachy_rl_control/"
    "models/ppo_v2_1_nohuman_2m_10k_seed42.zip"
)

RESULTS_DIRECTORY = (
    "/opt/reachy_ws/src/results"
)

EPISODE_CSV_PATH = os.path.join(
    RESULTS_DIRECTORY,
    "ppo_v2_1_nohuman_2m_10k_seed42_20episodes.csv",
)

SUMMARY_CSV_PATH = os.path.join(
    RESULTS_DIRECTORY,
    "ppo_v2_1_nohuman_2m_10k_seed42_summary.csv",
)


# =========================================================
# GEOMETRY HELPERS
# =========================================================

def point_to_segment_distance(
    px,
    py,
    ax,
    ay,
    bx,
    by,
):

    ab_x = bx - ax
    ab_y = by - ay

    ap_x = px - ax
    ap_y = py - ay

    ab_length_squared = (
        ab_x * ab_x
        + ab_y * ab_y
    )

    if ab_length_squared <= 1e-12:
        return math.hypot(
            px - ax,
            py - ay,
        )

    projection = (
        ap_x * ab_x
        + ap_y * ab_y
    ) / ab_length_squared

    projection = max(
        0.0,
        min(
            1.0,
            projection,
        ),
    )

    closest_x = (
        ax
        + projection * ab_x
    )

    closest_y = (
        ay
        + projection * ab_y
    )

    return math.hypot(
        px - closest_x,
        py - closest_y,
    )


def distance_to_trajectory(
    x,
    y,
    waypoints,
):

    if len(waypoints) < 2:
        return 0.0

    distances = []

    for index in range(
        len(waypoints) - 1
    ):

        ax, ay = waypoints[index]
        bx, by = waypoints[index + 1]

        distances.append(
            point_to_segment_distance(
                x,
                y,
                ax,
                ay,
                bx,
                by,
            )
        )

    return float(
        min(distances)
    )


# =========================================================
# SUMMARY HELPERS
# =========================================================

def mean_std(values):

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return (
            float("nan"),
            float("nan"),
        )

    return (
        float(np.mean(array)),
        float(np.std(array)),
    )


def percentage(
    count,
    total,
):

    if total == 0:
        return 0.0

    return (
        100.0
        * float(count)
        / float(total)
    )


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        RESULTS_DIRECTORY,
        exist_ok=True,
    )

    if not os.path.isfile(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            "PPO V2.1 model not found: "
            f"{MODEL_PATH}"
        )

    rclpy.init()

    robot = ReachyGazeboInterface(
        reset_service="/reset_world"
    )

    executor = (
        SingleThreadedExecutor()
    )

    executor.add_node(
        robot
    )

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )

    spin_thread.start()

    env = None

    try:

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

        env = ReachyGazeboEnv(
            robot=robot,
            control_duration=(
                CONTROL_DURATION
            ),
            max_steps=MAX_STEPS,
        )

        print(
            "Observation space:",
            env.observation_space.shape,
        )

        if (
            env.observation_space.shape
            != (10,)
        ):
            raise RuntimeError(
                "PPO V2 no-human evaluator "
                "expected observation shape "
                f"(10,), got "
                f"{env.observation_space.shape}."
            )

        print(
            "\nLoading PPO V2.1 model..."
        )

        print(
            MODEL_PATH
        )

        model = PPO.load(
            MODEL_PATH,
            device="cpu",
        )

        print(
            "PPO V2.1 model loaded successfully."
        )

        episode_results = []

        print(
            "\n========================================"
        )

        print(
            "PPO V2.1 NO-HUMAN 2M SEED42 "
            "DETERMINISTIC EVALUATION"
        )

        print(
            "========================================"
        )

        print(
            f"Episodes: {NUM_EPISODES}"
        )

        print(
            f"Maximum steps: {MAX_STEPS}"
        )

        print(
            "Policy: deterministic"
        )

        # =================================================
        # EPISODE LOOP
        # =================================================

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            print(
                "\n----------------------------------------"
            )

            print(
                f"EVALUATION EPISODE "
                f"{episode}/{NUM_EPISODES}"
            )

            print(
                "----------------------------------------"
            )

            reset_success = False

            for reset_retry in range(
                1,
                6,
            ):

                try:

                    (
                        observation,
                        reset_info,
                    ) = env.reset()

                    reset_success = True

                    break

                except RuntimeError as error:

                    error_text = str(
                        error
                    )

                    retryable = (
                        "stable upright and stationary reset"
                        in error_text
                        or
                        "Timed out waiting for fresh odometry after Gazebo reset"
                        in error_text
                    )

                    if not retryable:
                        raise

                    print(
                        f"Evaluation reset cycle "
                        f"{reset_retry}/5 failed; "
                        "retrying..."
                    )

            if not reset_success:
                raise RuntimeError(
                    "Evaluation could not obtain "
                    "a valid reset after "
                    "5 reset cycles."
                )

            total_reward = 0.0

            steps = 0

            terminated = False
            truncated = False

            final_info = (
                reset_info
            )

            path_errors = []

            smoothness_values = []

            previous_physical_action = (
                np.zeros(
                    3,
                    dtype=np.float32,
                )
            )

            episode_start_time = (
                time.monotonic()
            )

            # =============================================
            # STEP LOOP
            # =============================================

            while not (
                terminated
                or truncated
            ):

                action, _ = (
                    model.predict(
                        observation,
                        deterministic=True,
                    )
                )

                action = np.asarray(
                    action,
                    dtype=np.float32,
                )

                action = np.clip(
                    action,
                    -1.0,
                    1.0,
                )

                physical_action = np.array(
                    [
                        0.30
                        * float(
                            action[0]
                        ),

                        0.30
                        * float(
                            action[1]
                        ),

                        0.60
                        * float(
                            action[2]
                        ),
                    ],
                    dtype=np.float32,
                )

                action_change = (
                    physical_action
                    - previous_physical_action
                )

                smoothness_value = float(
                    np.sum(
                        np.square(
                            action_change
                        )
                    )
                )

                smoothness_values.append(
                    smoothness_value
                )

                previous_physical_action = (
                    physical_action.copy()
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                total_reward += float(
                    reward
                )

                steps += 1

                final_info = info

                state = (
                    robot.get_state()
                )

                if state is not None:

                    path_error = (
                        distance_to_trajectory(
                            state.x,
                            state.y,
                            env.world_waypoints,
                        )
                    )

                    path_errors.append(
                        path_error
                    )

            # =============================================
            # END EPISODE
            # =============================================

            elapsed_wall_time = (
                time.monotonic()
                - episode_start_time
            )

            final_state = (
                robot.get_state()
            )

            (
                final_goal_x,
                final_goal_y,
            ) = env.world_waypoints[-1]

            if final_state is not None:

                final_trajectory_error = (
                    math.hypot(
                        final_goal_x
                        - final_state.x,

                        final_goal_y
                        - final_state.y,
                    )
                )

            else:

                final_trajectory_error = (
                    float("nan")
                )

            if path_errors:

                mean_path_error = float(
                    np.mean(
                        path_errors
                    )
                )

            else:

                mean_path_error = (
                    float("nan")
                )

            if smoothness_values:

                mean_smoothness = float(
                    np.mean(
                        smoothness_values
                    )
                )

            else:

                mean_smoothness = (
                    float("nan")
                )

            termination_reason = str(
                final_info.get(
                    "termination_reason",
                    "unknown",
                )
            )

            success = bool(
                final_info.get(
                    "success",
                    False,
                )
                or
                termination_reason == "goal"
            )

            fell_over = bool(
                final_info.get(
                    "fell_over",
                    False,
                )
            )

            timeout = bool(
                termination_reason
                == "timeout"
            )

            final_waypoint = int(
                final_info.get(
                    "waypoint",
                    -1,
                )
            )

            final_waypoint_distance = float(
                final_info.get(
                    "distance_to_goal",
                    float("nan"),
                )
            )

            final_progress_ratio = float(
                final_info.get(
                    "path_progress_ratio",
                    float("nan"),
                )
            )

            final_progress_percent = float(
                final_info.get(
                    "path_progress_percent",
                    (
                        100.0
                        * final_progress_ratio
                    ),
                )
            )

            simulated_control_time = (
                steps
                * CONTROL_DURATION
            )

            result = {

                "episode":
                    episode,

                "success":
                    int(success),

                "termination_reason":
                    termination_reason,

                "fall":
                    int(fell_over),

                "timeout":
                    int(timeout),

                "steps":
                    steps,

                "simulated_control_time_s":
                    simulated_control_time,

                "wall_time_s":
                    elapsed_wall_time,

                "total_reward":
                    total_reward,

                "final_trajectory_error_m":
                    final_trajectory_error,

                "mean_path_error_m":
                    mean_path_error,

                "smoothness":
                    mean_smoothness,

                "final_waypoint":
                    final_waypoint,

                "final_waypoint_distance_m":
                    final_waypoint_distance,

                "final_progress_ratio":
                    final_progress_ratio,

                "final_progress_percent":
                    final_progress_percent,
            }

            episode_results.append(
                result
            )

            print(
                "\nEpisode result:"
            )

            print(
                f"  termination: "
                f"{termination_reason}"
            )

            print(
                f"  success: "
                f"{success}"
            )

            print(
                f"  fall: "
                f"{fell_over}"
            )

            print(
                f"  timeout: "
                f"{timeout}"
            )

            print(
                f"  steps: "
                f"{steps}"
            )

            print(
                f"  reward: "
                f"{total_reward:.3f}"
            )

            print(
                f"  final trajectory error: "
                f"{final_trajectory_error:.3f} m"
            )

            print(
                f"  mean path error: "
                f"{mean_path_error:.3f} m"
            )

            print(
                f"  smoothness: "
                f"{mean_smoothness:.6f}"
            )

            print(
                f"  final progress: "
                f"{final_progress_percent:.1f}%"
            )

        # =================================================
        # SAVE EPISODE CSV
        # =================================================

        with open(
            EPISODE_CSV_PATH,
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    episode_results[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                episode_results
            )

        # =================================================
        # SUMMARY
        # =================================================

        successes = sum(
            row["success"]
            for row in episode_results
        )

        falls = sum(
            row["fall"]
            for row in episode_results
        )

        timeouts = sum(
            row["timeout"]
            for row in episode_results
        )

        reward_mean, reward_std = (
            mean_std(
                [
                    row["total_reward"]
                    for row
                    in episode_results
                ]
            )
        )

        steps_mean, steps_std = (
            mean_std(
                [
                    row["steps"]
                    for row
                    in episode_results
                ]
            )
        )

        final_error_mean, final_error_std = (
            mean_std(
                [
                    row[
                        "final_trajectory_error_m"
                    ]
                    for row
                    in episode_results
                ]
            )
        )

        path_error_mean, path_error_std = (
            mean_std(
                [
                    row["mean_path_error_m"]
                    for row
                    in episode_results
                ]
            )
        )

        smooth_mean, smooth_std = (
            mean_std(
                [
                    row["smoothness"]
                    for row
                    in episode_results
                ]
            )
        )

        progress_mean, progress_std = (
            mean_std(
                [
                    row[
                        "final_progress_ratio"
                    ]
                    for row
                    in episode_results
                ]
            )
        )

        summary_rows = [
            (
                "episodes",
                NUM_EPISODES,
            ),
            (
                "goals",
                successes,
            ),
            (
                "success_rate_percent",
                percentage(
                    successes,
                    NUM_EPISODES,
                ),
            ),
            (
                "falls",
                falls,
            ),
            (
                "fall_rate_percent",
                percentage(
                    falls,
                    NUM_EPISODES,
                ),
            ),
            (
                "timeouts",
                timeouts,
            ),
            (
                "timeout_rate_percent",
                percentage(
                    timeouts,
                    NUM_EPISODES,
                ),
            ),
            (
                "reward_mean",
                reward_mean,
            ),
            (
                "reward_std",
                reward_std,
            ),
            (
                "steps_mean",
                steps_mean,
            ),
            (
                "steps_std",
                steps_std,
            ),
            (
                "final_trajectory_error_mean_m",
                final_error_mean,
            ),
            (
                "final_trajectory_error_std_m",
                final_error_std,
            ),
            (
                "mean_path_error_mean_m",
                path_error_mean,
            ),
            (
                "mean_path_error_std_m",
                path_error_std,
            ),
            (
                "smoothness_mean",
                smooth_mean,
            ),
            (
                "smoothness_std",
                smooth_std,
            ),
            (
                "final_progress_ratio_mean",
                progress_mean,
            ),
            (
                "final_progress_ratio_std",
                progress_std,
            ),
            (
                "final_progress_percent_mean",
                100.0
                * progress_mean,
            ),
        ]

        with open(
            SUMMARY_CSV_PATH,
            "w",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "metric",
                    "value",
                ]
            )

            writer.writerows(
                summary_rows
            )

        # =================================================
        # PRINT SUMMARY
        # =================================================

        print(
            "\n========================================"
        )

        print(
            "PPO V2.1 NO-HUMAN 2M SEED42 "
            "EVALUATION SUMMARY"
        )

        print(
            "========================================"
        )

        print(
            f"Episodes: "
            f"{NUM_EPISODES}"
        )

        print(
            f"Goals: "
            f"{successes}/{NUM_EPISODES} "
            f"({percentage(successes, NUM_EPISODES):.1f}%)"
        )

        print(
            f"Falls: "
            f"{falls}/{NUM_EPISODES} "
            f"({percentage(falls, NUM_EPISODES):.1f}%)"
        )

        print(
            f"Timeouts: "
            f"{timeouts}/{NUM_EPISODES} "
            f"({percentage(timeouts, NUM_EPISODES):.1f}%)"
        )

        print(
            "\nReward:"
        )

        print(
            f"  {reward_mean:.3f} "
            f"+/- {reward_std:.3f}"
        )

        print(
            "\nSteps:"
        )

        print(
            f"  {steps_mean:.2f} "
            f"+/- {steps_std:.2f}"
        )

        print(
            "\nFinal trajectory error:"
        )

        print(
            f"  {final_error_mean:.3f} "
            f"+/- {final_error_std:.3f} m"
        )

        print(
            "\nMean path error:"
        )

        print(
            f"  {path_error_mean:.3f} "
            f"+/- {path_error_std:.3f} m"
        )

        print(
            "\nSmoothness:"
        )

        print(
            f"  {smooth_mean:.6f} "
            f"+/- {smooth_std:.6f}"
        )

        print(
            "\nFinal path progress:"
        )

        print(
            f"  {100.0 * progress_mean:.1f}% "
            f"+/- {100.0 * progress_std:.1f}%"
        )

        print(
            "\nEpisode results saved to:"
        )

        print(
            EPISODE_CSV_PATH
        )

        print(
            "\nSummary saved to:"
        )

        print(
            SUMMARY_CSV_PATH
        )

    except KeyboardInterrupt:

        print(
            "\nPPO V2.1 evaluation interrupted."
        )

    except Exception as error:

        print(
            f"\nPPO V2 EVALUATION ERROR: "
            f"{error}"
        )

        raise

    finally:

        print(
            "\nStopping evaluation safely..."
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
            "PPO V2.1 evaluation process closed."
        )


if __name__ == "__main__":
    main()
