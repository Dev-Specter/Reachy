#!/usr/bin/env python3

import math
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from reachy_rl_control.reachy_gazebo_interface import ReachyGazeboInterface
from reachy_rl_control.trajectory import STRAIGHT_PATH, WAYPOINT_TOLERANCE


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def main(args=None):
    rclpy.init(args=args)

    robot = ReachyGazeboInterface(
        reset_service="/reset_world"
    )

    executor = SingleThreadedExecutor()
    executor.add_node(robot)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )
    spin_thread.start()

    try:
        print("\nWaiting for /odom...")

        if not robot.wait_for_odometry(timeout=5.0):
            raise RuntimeError(
                "No /odom received. Is Gazebo running?"
            )

        print("Odometry received.")

        print("\nResetting Gazebo...")

        start_state = robot.reset_gazebo(
            timeout=5.0,
            settle_time=0.5,
        )

        print(
            f"Start position: "
            f"x={start_state.x:.3f}, "
            f"y={start_state.y:.3f}, "
            f"yaw={start_state.yaw:.3f}"
        )

        start_x = start_state.x
        start_y = start_state.y
        start_yaw = start_state.yaw

        # Convert trajectory coordinates into Gazebo coordinates
        waypoints = []

        for path_x, path_y in STRAIGHT_PATH:
            world_x = (
                start_x
                + math.cos(start_yaw) * path_x
                - math.sin(start_yaw) * path_y
            )

            world_y = (
                start_y
                + math.sin(start_yaw) * path_x
                + math.cos(start_yaw) * path_y
            )

            waypoints.append(
                (world_x, world_y)
            )

        print("\nStarting 2 m trajectory...\n")

        # Skip waypoint 0 because it is the starting position
        for waypoint_number, (target_x, target_y) in enumerate(
            waypoints[1:],
            start=1,
        ):
            print(
                f"Moving to waypoint {waypoint_number}: "
                f"x={target_x:.3f}, "
                f"y={target_y:.3f}"
            )

            waypoint_start_time = time.time()

            while rclpy.ok():

                state = robot.get_state()

                if state is None:
                    time.sleep(0.05)
                    continue

                dx = target_x - state.x
                dy = target_y - state.y

                distance = math.hypot(dx, dy)

                if distance <= WAYPOINT_TOLERANCE:
                    print(
                        f"Waypoint {waypoint_number} reached. "
                        f"Distance error = {distance:.3f} m"
                    )
                    break

                # Convert world-frame position error
                # into Reachy's body frame
                cos_yaw = math.cos(state.yaw)
                sin_yaw = math.sin(state.yaw)

                error_forward = (
                    cos_yaw * dx
                    + sin_yaw * dy
                )

                error_sideways = (
                    -sin_yaw * dx
                    + cos_yaw * dy
                )

                # Simple proportional controller
                vx = 0.8 * error_forward
                vy = 0.8 * error_sideways

                # Maintain original orientation
                yaw_error = math.atan2(
                    math.sin(start_yaw - state.yaw),
                    math.cos(start_yaw - state.yaw),
                )

                omega = 1.0 * yaw_error

                # Safe velocity limits
                vx = clamp(vx, -0.18, 0.18)
                vy = clamp(vy, -0.18, 0.18)
                omega = clamp(omega, -0.40, 0.40)

                robot.send_velocity(
                    vx=vx,
                    vy=vy,
                    omega=omega,
                )

                # Prevent infinite loop
                if time.time() - waypoint_start_time > 20.0:
                    raise RuntimeError(
                        f"Timeout while moving to "
                        f"waypoint {waypoint_number}"
                    )

                time.sleep(0.05)

        robot.stop_for(
            duration=1.0,
            frequency=20.0,
        )

        final_state = robot.get_state()

        final_target_x, final_target_y = waypoints[-1]

        final_error = math.hypot(
            final_target_x - final_state.x,
            final_target_y - final_state.y,
        )

        print("\nStraight trajectory completed.")
        print(
            f"Final position: "
            f"x={final_state.x:.3f}, "
            f"y={final_state.y:.3f}"
        )
        print(
            f"Final goal error: "
            f"{final_error:.3f} m"
        )

    except KeyboardInterrupt:
        print("\nTrajectory test interrupted.")

    except Exception as error:
        print(f"\nERROR: {error}")

    finally:
        print("\nStopping Reachy safely...")

        robot.stop_for(
            duration=1.0,
            frequency=20.0,
        )

        executor.shutdown()
        robot.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(timeout=1.0)

        print("Trajectory test finished.")


if __name__ == "__main__":
    main()
