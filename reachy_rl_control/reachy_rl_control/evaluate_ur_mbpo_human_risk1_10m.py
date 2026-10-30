#!/usr/bin/env python3

import csv
import math
import os
import threading
import time

import numpy as np
import rclpy

from rclpy.executors import SingleThreadedExecutor

from stable_baselines3 import SAC

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)

from reachy_rl_control.environment_human_10m import (
    ReachyHumanGazeboEnv,
)


# =========================================================
# CONFIGURATION
# =========================================================

NUM_EPISODES = 20

MAX_STEPS = 1250

CONTROL_DURATION = 0.1

MODEL_PATH = (
    "/opt/reachy_ws/src/reachy_rl_control/"
    "models/ur_mbpo_human_risk1_10k.zip"
)

RESULTS_DIRECTORY = (
    "/opt/reachy_ws/src/results"
)

EPISODE_CSV_PATH = os.path.join(
    RESULTS_DIRECTORY,
    "ur_mbpo_human_risk1_10m_20episodes.csv",
)

SUMMARY_CSV_PATH = os.path.join(
    RESULTS_DIRECTORY,
    "ur_mbpo_human_risk1_10m_summary.csv",
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
    """
    Euclidean distance between point P and line segment AB.
    """

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
    """
    Minimum distance from a robot position to the
    complete piecewise-linear trajectory.
    """

    if len(waypoints) < 2:

        return 0.0

    distances = []

    for index in range(
        len(waypoints) - 1
    ):

        ax, ay = (
            waypoints[index]
        )

        bx, by = (
            waypoints[index + 1]
        )

        distance = (
            point_to_segment_distance(
                x,
                y,
                ax,
                ay,
                bx,
                by,
            )
        )

        distances.append(
            distance
        )

    return float(
        min(distances)
    )


# =========================================================
# SUMMARY HELPERS
# =========================================================

def mean_std(values):
    """
    Return population mean and standard deviation.
    """

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:

        return (
            float("nan"),
            float("nan"),
        )

    return (
        float(
            np.mean(array)
        ),
        float(
            np.std(array)
        ),
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
            "PPO model not found: "
            f"{MODEL_PATH}"
        )

    # -----------------------------------------------------
    # ROS INITIALISATION
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
        # WAIT FOR REACHY ODOMETRY
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
        # CREATE HUMAN-AWARE ENVIRONMENT
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

        # -------------------------------------------------
        # LOAD UR-MBPO MODEL
        # -------------------------------------------------

        print(
            "\nLoading UR-MBPO-human 10k policy..."
        )

        print(
            MODEL_PATH
        )

        model = SAC.load(
            MODEL_PATH,
            device="cpu",
        )

        print(
            "UR-MBPO policy loaded successfully."
        )

        # -------------------------------------------------
        # STORAGE
        # -------------------------------------------------

        episode_results = []

        # -------------------------------------------------
        # EVALUATION
        # -------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "UR-MBPO HUMAN RISK-1 10M DETERMINISTIC EVALUATION"
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

        print(
            "Human reference frame: reachy"
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

            for reset_retry in range(1, 6):

                try:

                    observation, reset_info = env.reset()

                    reset_success = True

                    break

                except RuntimeError as error:

                    if (
                        "stable upright and stationary reset"
                        not in str(error)
                    ):
                        raise

                    print(
                        f"Evaluation reset cycle "
                        f"{reset_retry}/5 failed; retrying..."
                    )

            if not reset_success:

                raise RuntimeError(
                    "Evaluation could not obtain a valid "
                    "stable reset after 5 reset cycles."
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

                # -----------------------------------------
                # Physical action used for smoothness
                # -----------------------------------------

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

                # -----------------------------------------
                # ENVIRONMENT STEP
                # -----------------------------------------

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

                # -----------------------------------------
                # TRAJECTORY ERROR
                # -----------------------------------------

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
            # END OF EPISODE
            # =============================================

            elapsed_wall_time = (
                time.monotonic()
                - episode_start_time
            )

            final_state = (
                robot.get_state()
            )

            final_goal_x, final_goal_y = (
                env.world_waypoints[-1]
            )

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
            )

            collision = bool(
                final_info.get(
                    "collision",
                    False,
                )
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

            near_miss = bool(
                final_info.get(
                    "near_miss_occurred",
                    False,
                )
            )

            minimum_lidar_clearance = float(
                final_info.get(
                    "episode_min_lidar_distance",
                    float("nan"),
                )
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

                "collision":
                    int(collision),

                "fall":
                    int(fell_over),

                "timeout":
                    int(timeout),

                "near_miss":
                    int(near_miss),

                "steps":
                    steps,

                "simulated_control_time_s":
                    simulated_control_time,

                "wall_time_s":
                    elapsed_wall_time,

                "total_reward":
                    total_reward,

                "minimum_lidar_clearance_m":
                    minimum_lidar_clearance,

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
            }

            episode_results.append(
                result
            )

            # ---------------------------------------------
            # PRINT EPISODE RESULT
            # ---------------------------------------------

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
                f"  collision: "
                f"{collision}"
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
                f"  near miss: "
                f"{near_miss}"
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
                "  minimum LiDAR clearance: "
                f"{minimum_lidar_clearance:.3f} m"
            )

            print(
                "  final trajectory error: "
                f"{final_trajectory_error:.3f} m"
            )

            print(
                "  mean path error: "
                f"{mean_path_error:.3f} m"
            )

            print(
                "  smoothness: "
                f"{mean_smoothness:.6f}"
            )

        # =================================================
        # SAVE EPISODE CSV
        # =================================================

        fieldnames = [
            "episode",
            "success",
            "termination_reason",
            "collision",
            "fall",
            "timeout",
            "near_miss",
            "steps",
            "simulated_control_time_s",
            "wall_time_s",
            "total_reward",
            "minimum_lidar_clearance_m",
            "final_trajectory_error_m",
            "mean_path_error_m",
            "smoothness",
            "final_waypoint",
            "final_waypoint_distance_m",
        ]

        with open(
            EPISODE_CSV_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                episode_results
            )

        # =================================================
        # SUMMARY
        # =================================================

        successes = sum(
            result["success"]
            for result in episode_results
        )

        collisions = sum(
            result["collision"]
            for result in episode_results
        )

        falls = sum(
            result["fall"]
            for result in episode_results
        )

        timeouts = sum(
            result["timeout"]
            for result in episode_results
        )

        near_misses = sum(
            result["near_miss"]
            for result in episode_results
        )

        rewards = [
            result["total_reward"]
            for result in episode_results
        ]

        steps_list = [
            result["steps"]
            for result in episode_results
        ]

        clearances = [
            result[
                "minimum_lidar_clearance_m"
            ]
            for result in episode_results
        ]

        final_errors = [
            result[
                "final_trajectory_error_m"
            ]
            for result in episode_results
        ]

        path_errors_summary = [
            result[
                "mean_path_error_m"
            ]
            for result in episode_results
        ]

        smoothness_summary = [
            result[
                "smoothness"
            ]
            for result in episode_results
        ]

        (
            mean_reward,
            std_reward,
        ) = mean_std(
            rewards
        )

        (
            mean_steps,
            std_steps,
        ) = mean_std(
            steps_list
        )

        (
            mean_clearance,
            std_clearance,
        ) = mean_std(
            clearances
        )

        (
            mean_final_error,
            std_final_error,
        ) = mean_std(
            final_errors
        )

        (
            mean_path_error,
            std_path_error,
        ) = mean_std(
            path_errors_summary
        )

        (
            mean_smoothness,
            std_smoothness,
        ) = mean_std(
            smoothness_summary
        )

        summary = {

            "algorithm":
                "UR-MBPO",

            "model":
                "ur_mbpo_human_10k",

            "episodes":
                NUM_EPISODES,

            "successes":
                successes,

            "success_rate_percent":
                percentage(
                    successes,
                    NUM_EPISODES,
                ),

            "collisions":
                collisions,

            "collision_rate_percent":
                percentage(
                    collisions,
                    NUM_EPISODES,
                ),

            "falls":
                falls,

            "fall_rate_percent":
                percentage(
                    falls,
                    NUM_EPISODES,
                ),

            "timeouts":
                timeouts,

            "timeout_rate_percent":
                percentage(
                    timeouts,
                    NUM_EPISODES,
                ),

            "near_misses":
                near_misses,

            "near_miss_rate_percent":
                percentage(
                    near_misses,
                    NUM_EPISODES,
                ),

            "mean_reward":
                mean_reward,

            "std_reward":
                std_reward,

            "mean_steps":
                mean_steps,

            "std_steps":
                std_steps,

            "mean_minimum_lidar_clearance_m":
                mean_clearance,

            "std_minimum_lidar_clearance_m":
                std_clearance,

            "mean_final_trajectory_error_m":
                mean_final_error,

            "std_final_trajectory_error_m":
                std_final_error,

            "mean_path_error_m":
                mean_path_error,

            "std_path_error_m":
                std_path_error,

            "mean_smoothness":
                mean_smoothness,

            "std_smoothness":
                std_smoothness,
        }

        # =================================================
        # SAVE SUMMARY CSV
        # =================================================

        with open(
            SUMMARY_CSV_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow(
                [
                    "metric",
                    "value",
                ]
            )

            for (
                metric,
                value,
            ) in summary.items():

                writer.writerow(
                    [
                        metric,
                        value,
                    ]
                )

        # =================================================
        # PRINT SUMMARY
        # =================================================

        print(
            "\n========================================"
        )

        print(
            "UR-MBPO HUMAN RISK-1 10M EVALUATION SUMMARY"
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
            f"("
            f"{percentage(successes, NUM_EPISODES):.1f}%"
            f")"
        )

        print(
            f"Collisions: "
            f"{collisions}/{NUM_EPISODES} "
            f"("
            f"{percentage(collisions, NUM_EPISODES):.1f}%"
            f")"
        )

        print(
            f"Falls: "
            f"{falls}/{NUM_EPISODES} "
            f"("
            f"{percentage(falls, NUM_EPISODES):.1f}%"
            f")"
        )

        print(
            f"Timeouts: "
            f"{timeouts}/{NUM_EPISODES} "
            f"("
            f"{percentage(timeouts, NUM_EPISODES):.1f}%"
            f")"
        )

        print(
            f"Near misses: "
            f"{near_misses}/{NUM_EPISODES} "
            f"("
            f"{percentage(near_misses, NUM_EPISODES):.1f}%"
            f")"
        )

        print(
            "\nReward:"
        )

        print(
            f"  {mean_reward:.3f} "
            f"+/- "
            f"{std_reward:.3f}"
        )

        print(
            "\nSteps:"
        )

        print(
            f"  {mean_steps:.2f} "
            f"+/- "
            f"{std_steps:.2f}"
        )

        print(
            "\nMinimum LiDAR clearance:"
        )

        print(
            f"  {mean_clearance:.3f} "
            f"+/- "
            f"{std_clearance:.3f} m"
        )

        print(
            "\nFinal trajectory error:"
        )

        print(
            f"  {mean_final_error:.3f} "
            f"+/- "
            f"{std_final_error:.3f} m"
        )

        print(
            "\nMean path error:"
        )

        print(
            f"  {mean_path_error:.3f} "
            f"+/- "
            f"{std_path_error:.3f} m"
        )

        print(
            "\nSmoothness:"
        )

        print(
            f"  {mean_smoothness:.6f} "
            f"+/- "
            f"{std_smoothness:.6f}"
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
            "\nEvaluation interrupted."
        )

    finally:

        print(
            "\nStopping evaluation safely..."
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

        if robot_spin_thread.is_alive():

            robot_spin_thread.join(
                timeout=1.0
            )

        print(
            "UR-MBPO evaluation process closed."
        )


if __name__ == "__main__":
    main()
