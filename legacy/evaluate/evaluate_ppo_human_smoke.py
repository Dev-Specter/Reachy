#!/usr/bin/env python3

import threading

import numpy as np
import rclpy

from rclpy.executors import (
    ExternalShutdownException,
    SingleThreadedExecutor,
)

from stable_baselines3 import PPO

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)

from reachy_rl_control.environment_human import (
    ReachyHumanGazeboEnv,
)


MODEL_PATH = (
    "/opt/reachy_ws/src/reachy_rl_control/models/"
    "ppo_human_smoke.zip"
)

NUM_EPISODES = 5
MAX_STEPS = 250


def spin_robot_executor(executor):

    try:
        executor.spin()

    except ExternalShutdownException:
        pass


def main():

    rclpy.init()

    robot = ReachyGazeboInterface()

    robot_executor = SingleThreadedExecutor()

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

        env = ReachyHumanGazeboEnv(
            robot=robot,
            control_duration=0.1,
            max_steps=MAX_STEPS,
        )

        print(
            "\nLoading PPO human smoke model..."
        )

        print(
            MODEL_PATH
        )

        model = PPO.load(
            MODEL_PATH,
            env=env,
            device="cpu",
        )

        print(
            "Model loaded."
        )

        results = []

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            print(
                "\n=================================="
            )

            print(
                f"EVALUATION EPISODE {episode}"
            )

            print(
                "=================================="
            )

            observation, info = (
                env.reset()
            )

            total_reward = 0.0

            terminated = False
            truncated = False

            final_info = info

            for step_number in range(
                1,
                MAX_STEPS + 1,
            ):

                action, _ = model.predict(
                    observation,
                    deterministic=True,
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

                total_reward += reward

                final_info = info

                if (
                    step_number == 1
                    or step_number % 25 == 0
                    or terminated
                    or truncated
                ):

                    print(
                        f"Step {step_number:03d} | "
                        f"action="
                        f"[{action[0]:+.3f}, "
                        f"{action[1]:+.3f}, "
                        f"{action[2]:+.3f}] | "
                        f"reward={reward:+.3f} | "
                        f"waypoint="
                        f"{info['waypoint']} | "
                        f"goal_dist="
                        f"{info['distance_to_goal']:.3f} m | "
                        f"human_dist="
                        f"{info['min_lidar_distance']:.3f} m"
                    )

                if (
                    terminated
                    or truncated
                ):
                    break

            reason = final_info.get(
                "termination_reason",
                "unknown",
            )

            print(
                "\nEpisode result:"
            )

            print(
                "Total reward:",
                f"{total_reward:.3f}",
            )

            print(
                "Steps:",
                step_number,
            )

            print(
                "Termination reason:",
                reason,
            )

            print(
                "Success:",
                final_info.get(
                    "success",
                    False,
                ),
            )

            print(
                "Collision:",
                final_info.get(
                    "collision",
                    False,
                ),
            )

            print(
                "Fall:",
                final_info.get(
                    "fell_over",
                    False,
                ),
            )

            print(
                "Near miss:",
                final_info.get(
                    "near_miss_occurred",
                    False,
                ),
            )

            print(
                "Minimum human distance:",
                f"{final_info.get('episode_min_lidar_distance', 0.0):.3f} m",
            )

            print(
                "Final waypoint:",
                final_info.get(
                    "waypoint",
                    -1,
                ),
            )

            print(
                "Final waypoint distance:",
                f"{final_info.get('distance_to_goal', 0.0):.3f} m",
            )

            results.append(
                {
                    "reason": reason,
                    "reward": total_reward,
                    "steps": step_number,
                    "min_distance":
                        final_info.get(
                            "episode_min_lidar_distance",
                            0.0,
                        ),
                    "near_miss":
                        final_info.get(
                            "near_miss_occurred",
                            False,
                        ),
                }
            )

        # ---------------------------------------------
        # SUMMARY
        # ---------------------------------------------

        goals = sum(
            result["reason"] == "goal"
            for result in results
        )

        collisions = sum(
            result["reason"] == "collision"
            for result in results
        )

        falls = sum(
            result["reason"] == "fall"
            for result in results
        )

        timeouts = sum(
            result["reason"] == "timeout"
            for result in results
        )

        near_misses = sum(
            result["near_miss"]
            for result in results
        )

        mean_reward = np.mean(
            [
                result["reward"]
                for result in results
            ]
        )

        mean_min_distance = np.mean(
            [
                result["min_distance"]
                for result in results
            ]
        )

        print(
            "\n=================================="
        )

        print(
            "PPO HUMAN SMOKE EVALUATION SUMMARY"
        )

        print(
            "=================================="
        )

        print(
            "Episodes:",
            NUM_EPISODES,
        )

        print(
            "Goals:",
            goals,
        )

        print(
            "Collisions:",
            collisions,
        )

        print(
            "Falls:",
            falls,
        )

        print(
            "Timeouts:",
            timeouts,
        )

        print(
            "Near misses:",
            near_misses,
        )

        print(
            "Mean reward:",
            f"{mean_reward:.3f}",
        )

        print(
            "Mean minimum human distance:",
            f"{mean_min_distance:.3f} m",
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

        if robot_thread.is_alive():

            robot_thread.join(
                timeout=1.0
            )

        print(
            "Evaluation process closed."
        )


if __name__ == "__main__":
    main()
