#!/usr/bin/env bash
echo "sim_down: stopping..."
pkill -f 'lib/controller_manager/spawner'
pkill -f 'lib/gazebo_ros/spawn_entity.py'
pkill -f 'lib/robot_state_publisher/robot_state_publisher'
pkill -x gzclient; pkill -x gzserver
sleep 3
pkill -9 -x gzserver; pkill -9 -x gzclient
pkill -9 -f '_ros2_daemon'
echo "sim down"
