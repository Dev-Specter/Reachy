#!/usr/bin/env python3

import csv
import math
import os
import threading

import numpy as np
import rclpy
from rclpy.executors import (
    SingleThreadedExecutor,
)

from stable_baselines3 import PPO

from reachy_rl_control.environment import (
    ReachyGazeboEnv,
)
from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)


# ---------------------------------------------------------
# EVALUATION SETTINGS
# ---------------------------------------------------------

NUM_EPISODES = 20

CONTROL_DURATION = 0.1

MAX_STEPS = 250


MODEL_PATH = (
    "/opt/reachy_ws/src/"
    "reachy_rl_control/models/"
    "ppo_reachy_10k.zip"
)


# Save this separately from the earlier evaluation so the
# old results are preserved.
RESULTS_PATH = (
    "/opt/reachy_ws/src/"
    "reachy_rl_control/results/"
    "ppo_10k_20episodes_safety.csv"
)


# ---------------------------------------------------------
# PATH ERROR
# ---------------------------------------------------------

def lateral_error(
    x,
    y,
    start_x,
    start_y,
    start_yaw,
):
    """
    Calculate perpendicular distance from Reachy's
    position to the straight reference path.

    This metric is appropriate for the current 2 m
    straight-path experiment.
    """

    dx = (
        x
        - start_x
    )

    dy = (
        y
        - start_y
    )

    return abs(
        -math.sin(start_yaw) * dx
        + math.cos(start_yaw) * dy
    )


# ---------------------------------------------------------
# FINAL DESTINATION ERROR
# ---------------------------------------------------------

def final_goal_error(
    x,
    y,
    final_goal_x,
    final_goal_y,
):
    """
    Distance from the robot to the actual final destination,
    regardless of which waypoint the robot reached.
    """

    return math.hypot(
        final_goal_x - x,
        final_goal_y - y,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main(args=None):

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

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )

    spin_thread.start()

    env = None

    try:

        # -------------------------------------------------
        # WAIT FOR GAZEBO
        # -------------------------------------------------

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

        # -------------------------------------------------
        # CHECK MODEL
        # -------------------------------------------------

        if not os.path.exists(
            MODEL_PATH
        ):
            raise FileNotFoundError(
                f"Model not found: "
                f"{MODEL_PATH}"
            )

        # -------------------------------------------------
        # RESULTS DIRECTORY
        # -------------------------------------------------

        os.makedirs(
            os.path.dirname(
                RESULTS_PATH
            ),
            exist_ok=True,
        )

        # -------------------------------------------------
        # LOAD PPO
        # -------------------------------------------------

        print(
            "\nLoading PPO model..."
        )

        print(
            MODEL_PATH
        )

        model = PPO.load(
            MODEL_PATH
        )

        print(
            "PPO model loaded."
        )

        # -------------------------------------------------
        # CREATE ENVIRONMENT
        # -------------------------------------------------

        env = ReachyGazeboEnv(
            robot=robot,
            control_duration=(
                CONTROL_DURATION
            ),
            max_steps=MAX_STEPS,
        )

        results = []

        print(
            "\nStarting "
            f"{NUM_EPISODES}-episode "
            "PPO evaluation..."
        )

        # =================================================
        # EPISODE LOOP
        # =================================================

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            observation, info = (
                env.reset()
            )

            # ---------------------------------------------
            # STARTING POSE
            # ---------------------------------------------

            start_x = float(
                observation[0]
            )

            start_y = float(
                observation[1]
            )

            start_yaw = float(
                observation[2]
            )

            # Actual final destination in world coordinates.
            final_goal_x = float(
                env.world_waypoints[-1][0]
            )

            final_goal_y = float(
                env.world_waypoints[-1][1]
            )

            total_reward = 0.0

            cross_track_errors = []

            action_changes = []

            previous_physical_action = (
                np.zeros(
                    3,
                    dtype=np.float32,
                )
            )

            terminated = False
            truncated = False

            fell_over = False

            termination_reason = (
                "running"
            )

            # =============================================
            # STEP LOOP
            # =============================================

            for step in range(
                1,
                MAX_STEPS + 1,
            ):

                # -----------------------------------------
                # PPO ACTION
                # -----------------------------------------

                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                # Convert normalized policy output back into
                # actual Reachy physical commands for the
                # evaluation smoothness metric.
                physical_action = (
                    np.array(
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
                )

                # -----------------------------------------
                # SMOOTHNESS METRIC
                # -----------------------------------------

                action_change = (
                    physical_action
                    - previous_physical_action
                )

                action_changes.append(
                    float(
                        np.sum(
                            np.square(
                                action_change
                            )
                        )
                    )
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

                total_reward += (
                    reward
                )

                # -----------------------------------------
                # FALL INFORMATION
                # -----------------------------------------

                if info.get(
                    "fell_over",
                    False,
                ):
                    fell_over = True

                termination_reason = (
                    info.get(
                        "termination_reason",
                        "unknown",
                    )
                )

                # -----------------------------------------
                # POSITION
                # -----------------------------------------

                x = float(
                    observation[0]
                )

                y = float(
                    observation[1]
                )

                # -----------------------------------------
                # PATH TRACKING ERROR
                # -----------------------------------------

                cross_track_errors.append(
                    lateral_error(
                        x,
                        y,
                        start_x,
                        start_y,
                        start_yaw,
                    )
                )

                # -----------------------------------------
                # EPISODE FINISHED
                # -----------------------------------------

                if (
                    terminated
                    or truncated
                ):
                    break

            # =================================================
            # EPISODE RESULT
            # =================================================

            # A successful episode must explicitly finish
            # because Reachy reached the final goal.
            success = (
                termination_reason
                == "goal"
            )

            timeout = (
                termination_reason
                == "timeout"
            )

            # Final robot position.
            final_x = float(
                observation[0]
            )

            final_y = float(
                observation[1]
            )

            true_final_goal_error = (
                final_goal_error(
                    final_x,
                    final_y,
                    final_goal_x,
                    final_goal_y,
                )
            )

            mean_path_error = float(
                np.mean(
                    cross_track_errors
                )
            )

            max_path_error = float(
                np.max(
                    cross_track_errors
                )
            )

            mean_smoothness = float(
                np.mean(
                    action_changes
                )
            )

            nominal_duration = (
                step
                * CONTROL_DURATION
            )

            result = {
                "episode":
                    episode,

                "success":
                    int(success),

                "fell_over":
                    int(fell_over),

                "timeout":
                    int(timeout),

                "termination_reason":
                    termination_reason,

                "steps":
                    step,

                "duration_seconds":
                    nominal_duration,

                "total_reward":
                    total_reward,

                "final_waypoint":
                    info["waypoint"],

                # Distance to whatever waypoint was active
                # when the episode ended.
                "distance_to_current_waypoint_m":
                    info["distance_to_goal"],

                # Distance to the actual final 2 m goal.
                "final_goal_error_m":
                    true_final_goal_error,

                "mean_path_error_m":
                    mean_path_error,

                "max_path_error_m":
                    max_path_error,

                "mean_squared_action_change":
                    mean_smoothness,

                "final_roll_rad":
                    info.get(
                        "roll",
                        0.0,
                    ),

                "final_pitch_rad":
                    info.get(
                        "pitch",
                        0.0,
                    ),
            }

            results.append(
                result
            )

            # -------------------------------------------------
            # EPISODE PRINT
            # -------------------------------------------------

            print(
                f"Episode "
                f"{episode:02d}/"
                f"{NUM_EPISODES} | "

                f"Success: "
                f"{success} | "

                f"Fall: "
                f"{fell_over} | "

                f"Reason: "
                f"{termination_reason} | "

                f"Steps: "
                f"{step:3d} | "

                f"Reward: "
                f"{total_reward:7.3f} | "

                f"Final error: "
                f"{true_final_goal_error:.3f} m | "

                f"Path error: "
                f"{mean_path_error:.3f} m"
            )

        # =================================================
        # SAVE CSV
        # =================================================

        fieldnames = list(
            results[0].keys()
        )

        with open(
            RESULTS_PATH,
            "w",
            newline="",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                results
            )

        # =================================================
        # OVERALL METRICS
        # =================================================

        successes = [
            row["success"]
            for row in results
        ]

        falls = [
            row["fell_over"]
            for row in results
        ]

        timeouts = [
            row["timeout"]
            for row in results
        ]

        rewards = [
            row["total_reward"]
            for row in results
        ]

        steps_list = [
            row["steps"]
            for row in results
        ]

        final_errors = [
            row["final_goal_error_m"]
            for row in results
        ]

        path_errors = [
            row["mean_path_error_m"]
            for row in results
        ]

        smoothness_values = [
            row[
                "mean_squared_action_change"
            ]
            for row in results
        ]

        success_rate = (
            100.0
            * sum(successes)
            / NUM_EPISODES
        )

        fall_rate = (
            100.0
            * sum(falls)
            / NUM_EPISODES
        )

        timeout_rate = (
            100.0
            * sum(timeouts)
            / NUM_EPISODES
        )

        # =================================================
        # OVERALL SUMMARY
        # =================================================

        print(
            "\n"
            "================================"
        )

        print(
            "PPO 10K - SAFETY EVALUATION"
        )

        print(
            "================================"
        )

        print(
            f"Success rate: "
            f"{success_rate:.1f}% "
            f"({sum(successes)}/"
            f"{NUM_EPISODES})"
        )

        print(
            f"Fall rate: "
            f"{fall_rate:.1f}% "
            f"({sum(falls)}/"
            f"{NUM_EPISODES})"
        )

        print(
            f"Timeout rate: "
            f"{timeout_rate:.1f}% "
            f"({sum(timeouts)}/"
            f"{NUM_EPISODES})"
        )

        print(
            f"Mean reward: "
            f"{np.mean(rewards):.3f} "
            f"+/- "
            f"{np.std(rewards):.3f}"
        )

        print(
            f"Mean steps: "
            f"{np.mean(steps_list):.1f} "
            f"+/- "
            f"{np.std(steps_list):.1f}"
        )

        print(
            f"Mean final goal error: "
            f"{np.mean(final_errors):.4f} m"
        )

        print(
            f"Mean path error: "
            f"{np.mean(path_errors):.4f} m"
        )

        print(
            f"Mean squared action change: "
            f"{np.mean(smoothness_values):.6f}"
        )

        # =================================================
        # SUCCESSFUL EPISODES ONLY
        # =================================================

        successful_results = [
            row
            for row in results
            if row["success"] == 1
        ]

        if successful_results:

            success_steps = [
                row["steps"]
                for row
                in successful_results
            ]

            success_rewards = [
                row["total_reward"]
                for row
                in successful_results
            ]

            success_goal_errors = [
                row["final_goal_error_m"]
                for row
                in successful_results
            ]

            success_path_errors = [
                row["mean_path_error_m"]
                for row
                in successful_results
            ]

            success_smoothness = [
                row[
                    "mean_squared_action_change"
                ]
                for row
                in successful_results
            ]

            print(
                "\n"
                "SUCCESSFUL EPISODES ONLY"
            )

            print(
                "================================"
            )

            print(
                f"Successful episodes: "
                f"{len(successful_results)}"
            )

            print(
                f"Mean steps: "
                f"{np.mean(success_steps):.2f} "
                f"+/- "
                f"{np.std(success_steps):.2f}"
            )

            print(
                f"Mean nominal time: "
                f"{np.mean(success_steps) * CONTROL_DURATION:.2f} s"
            )

            print(
                f"Mean reward: "
                f"{np.mean(success_rewards):.3f} "
                f"+/- "
                f"{np.std(success_rewards):.3f}"
            )

            print(
                f"Mean final goal error: "
                f"{np.mean(success_goal_errors):.4f} m"
            )

            print(
                f"Mean path error: "
                f"{np.mean(success_path_errors):.4f} m"
            )

            print(
                f"Mean squared action change: "
                f"{np.mean(success_smoothness):.6f}"
            )

        print(
            f"\nResults saved to:\n"
            f"{RESULTS_PATH}"
        )

    except KeyboardInterrupt:

        print(
            "\nEvaluation interrupted."
        )

    except Exception as error:

        print(
            f"\nPPO EVALUATION ERROR: "
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
            "Evaluation closed."
        )


if __name__ == "__main__":
    main()
