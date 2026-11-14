#!/usr/bin/env python3

import math
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from reachy_rl_control.reachy_gazebo_interface import (
    ReachyGazeboInterface,
    angle_difference,
)


def main(args=None):
    # ---------------------------------------------------------
    # Start ROS 2
    # ---------------------------------------------------------
    rclpy.init(args=args)

    robot = ReachyGazeboInterface(
        reset_service="/reset_world"
    )

    # ---------------------------------------------------------
    # Run ROS executor in background
    # ---------------------------------------------------------
    executor = SingleThreadedExecutor()
    executor.add_node(robot)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )

    spin_thread.start()

    try:
        # -----------------------------------------------------
        # Wait for odometry
        # -----------------------------------------------------
        print("\nWaiting for Reachy odometry...")

        if not robot.wait_for_odometry(timeout=5.0):
            raise RuntimeError(
                "No /odom received. Is Gazebo running?"
            )

        print("Odometry received.")

        # -----------------------------------------------------
        # Establish reference position
        # -----------------------------------------------------
        print("\nResetting Gazebo to get reference position...")

        reference = robot.reset_gazebo(
            timeout=5.0,
            settle_time=0.5,
        )

        print("\nReference state:")
        print(reference)

        # -----------------------------------------------------
        # Number of reset tests
        # -----------------------------------------------------
        number_of_tests = 10

        position_errors = []
        yaw_errors = []

        # -----------------------------------------------------
        # Reset reliability test
        # -----------------------------------------------------
        for episode in range(number_of_tests):

            print(
                f"\n========================================"
            )
            print(
                f"RESET TEST {episode + 1}/{number_of_tests}"
            )
            print(
                f"========================================"
            )

            # -------------------------------------------------
            # Move Reachy forward
            # -------------------------------------------------
            print("\nMoving Reachy forward...")

            robot.command_for(
                vx=0.15,
                vy=0.0,
                omega=0.0,
                duration=1.0,
                frequency=20.0,
            )

            time.sleep(0.2)

            moved_state = robot.get_state()

            if moved_state is None:
                raise RuntimeError(
                    "No odometry available after movement."
                )

            print("\nState before reset:")
            print(moved_state)

            # -------------------------------------------------
            # Reset Gazebo
            # -------------------------------------------------
            print("\nResetting Gazebo...")

            reset_state = robot.reset_gazebo(
                timeout=5.0,
                settle_time=0.5,
            )

            print("\nState after reset:")
            print(reset_state)

            # -------------------------------------------------
            # Calculate position reset error
            # -------------------------------------------------
            dx = reset_state.x - reference.x
            dy = reset_state.y - reference.y

            position_error = math.hypot(
                dx,
                dy,
            )

            # -------------------------------------------------
            # Calculate yaw reset error
            # -------------------------------------------------
            yaw_error = abs(
                angle_difference(
                    reset_state.yaw,
                    reference.yaw,
                )
            )

            position_errors.append(
                position_error
            )

            yaw_errors.append(
                yaw_error
            )

            # -------------------------------------------------
            # Print results
            # -------------------------------------------------
            print("\nReset error:")

            print(
                f"  dx             = {dx:.4f} m"
            )

            print(
                f"  dy             = {dy:.4f} m"
            )

            print(
                f"  position error = "
                f"{position_error:.4f} m"
            )

            print(
                f"  yaw error      = "
                f"{yaw_error:.4f} rad"
            )

            print(
                f"  yaw error      = "
                f"{math.degrees(yaw_error):.2f} degrees"
            )

        # -----------------------------------------------------
        # Final summary
        # -----------------------------------------------------
        print(
            "\n========================================"
        )
        print(
            "RESET TEST SUMMARY"
        )
        print(
            "========================================"
        )

        average_position_error = (
            sum(position_errors)
            / len(position_errors)
        )

        maximum_position_error = max(
            position_errors
        )

        average_yaw_error = (
            sum(yaw_errors)
            / len(yaw_errors)
        )

        maximum_yaw_error = max(
            yaw_errors
        )

        print(
            f"\nTests completed: {number_of_tests}"
        )

        print(
            f"Average position error: "
            f"{average_position_error:.4f} m"
        )

        print(
            f"Maximum position error: "
            f"{maximum_position_error:.4f} m"
        )

        print(
            f"Average yaw error: "
            f"{math.degrees(average_yaw_error):.2f} degrees"
        )

        print(
            f"Maximum yaw error: "
            f"{math.degrees(maximum_yaw_error):.2f} degrees"
        )

        print(
            "\nAll reset tests completed successfully."
        )

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    except Exception as error:
        print(
            f"\nERROR: {error}"
        )

    finally:
        # -----------------------------------------------------
        # Always stop Reachy safely
        # -----------------------------------------------------
        print("\nStopping Reachy...")

        robot.stop_for(
            duration=1.0,
            frequency=20.0,
        )

        # -----------------------------------------------------
        # Shutdown ROS
        # -----------------------------------------------------
        executor.shutdown()

        robot.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(
            timeout=1.0
        )

        print("Test closed safely.")


if __name__ == "__main__":
    main()