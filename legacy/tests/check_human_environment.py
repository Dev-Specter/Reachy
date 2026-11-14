#!/usr/bin/env python3

import threading

import numpy as np

import rclpy
from rclpy.executors import (
    ExternalShutdownException,
    SingleThreadedExecutor,
)

from stable_baselines3.common.env_checker import (
    check_env,
)

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
)

from reachy_rl_control.environment_human import (
    ReachyHumanGazeboEnv,
)


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

    robot_spin_thread = threading.Thread(
        target=spin_robot_executor,
        args=(robot_executor,),
        daemon=True,
    )

    robot_spin_thread.start()

    env = None

    try:

        # ---------------------------------------------
        # WAIT FOR REACHY ODOMETRY
        # ---------------------------------------------

        print(
            "\nWaiting for Reachy /odom..."
        )

        if not robot.wait_for_odometry(
            timeout=10.0
        ):

            raise RuntimeError(
                "No /odom received. "
                "Is Reachy Gazebo running?"
            )

        print(
            "Reachy odometry received."
        )

        # ---------------------------------------------
        # CREATE HUMAN-AWARE ENVIRONMENT
        # ---------------------------------------------

        print(
            "\nCreating human-aware environment..."
        )

        env = ReachyHumanGazeboEnv(
            robot=robot,
            control_duration=0.1,
            max_steps=300,
        )

        print(
            "Human-aware environment created."
        )

        # ---------------------------------------------
        # MANUAL RESET TEST
        # ---------------------------------------------

        print(
            "\nRunning reset test..."
        )

        observation, info = env.reset()

        print(
            "\nReset successful."
        )

        print(
            "Observation shape:",
            observation.shape,
        )

        print(
            "Observation dtype:",
            observation.dtype,
        )

        print(
            "\nRobot/goal observation:"
        )

        print(
            np.round(
                observation[:8],
                3,
            )
        )

        print(
            "\nNormalized LiDAR observation:"
        )

        print(
            np.round(
                observation[8:],
                3,
            )
        )

        print(
            "\nInitial minimum LiDAR distance:",
            f"{info['min_lidar_distance']:.3f} m",
        )

        print(
            "Human spawn:",
            f"x={info['human_spawn_x']:.3f}, "
            f"y={info['human_spawn_y']:.3f}",
        )

        print(
            "Human velocity:",
            f"vx={info['human_velocity_x']:.3f}, "
            f"vy={info['human_velocity_y']:.3f}",
        )

        # ---------------------------------------------
        # VERIFY OBSERVATION
        # ---------------------------------------------

        assert observation.shape == (
            20,
        ), (
            "Expected observation shape "
            "(20,), got "
            f"{observation.shape}"
        )

        assert observation.dtype == (
            np.float32
        ), (
            "Observation must be float32."
        )

        assert env.observation_space.contains(
            observation
        ), (
            "Observation is outside "
            "observation_space."
        )

        print(
            "\n20-value observation test PASSED."
        )

        # ---------------------------------------------
        # ONE ZERO-ACTION STEP
        # ---------------------------------------------

        print(
            "\nRunning one controlled zero-action step..."
        )

        action = np.zeros(
            3,
            dtype=np.float32,
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

        print(
            "\nStep successful."
        )

        print(
            "Reward:",
            reward,
        )

        print(
            "Terminated:",
            terminated,
        )

        print(
            "Truncated:",
            truncated,
        )

        print(
            "Termination reason:",
            info[
                "termination_reason"
            ],
        )

        print(
            "Current minimum LiDAR distance:",
            f"{info['min_lidar_distance']:.3f} m",
        )

        print(
            "Collision:",
            info[
                "collision"
            ],
        )

        print(
            "Near human:",
            info[
                "near_human"
            ],
        )

        print(
            "Fell over:",
            info[
                "fell_over"
            ],
        )

        # ---------------------------------------------
        # STABLE BASELINES ENV CHECKER
        # ---------------------------------------------

        print(
            "\nRunning Stable-Baselines3 "
            "environment checker..."
        )

        print(
            "The checker may reset Reachy "
            "and apply random actions."
        )

        check_env(
            env,
            warn=True,
        )

        print(
            "\n=================================="
        )

        print(
            "HUMAN ENVIRONMENT CHECKER PASSED"
        )

        print(
            "=================================="
        )

    except KeyboardInterrupt:

        print(
            "\nEnvironment test interrupted."
        )

    finally:

        print(
            "\nStopping test safely..."
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
            "Test closed."
        )


if __name__ == "__main__":
    main()
