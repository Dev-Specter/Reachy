# Reachy
MSc in AI; Reachy Mobile Base using RL


1. Project architecture and permanent paths
HOST project root: ~/reachy_rl_dissertation
Git repository root: ~/reachy_rl_dissertation/src
ROS package on HOST: ~/reachy_rl_dissertation/src/reachy_rl_control
Models on HOST: ~/reachy_rl_dissertation/models
Git-tracked results: ~/reachy_rl_dissertation/src/results
Docker container: reachy_2023
Docker image: pollenrobotics/reachy_2023:1.1
Docker ROS workspace: /opt/reachy_ws
Docker ROS package source: /opt/reachy_ws/src/reachy_rl_control
ROS distribution: ROS 2 Humble; Python 3.10
Git remote: origin https://github.com/Dev-Specter/Reachy.git
Primary branch: main

2. HOST vs DOCKER rule

The HOST is used for editing source files, Git, results, and permanent model storage. ROS 2 / Gazebo code is built and executed inside Docker because the HOST does not provide the required ROS environment.

# HOST example
cd ~/reachy_rl_dissertation/src/reachy_rl_control

# DOCKER shell
docker exec -it reachy_2023 bash

3. Launching Reachy 2023 in Gazebo

CONFIRMED:
 
Gazebo GUI and Reachy simulation were working in the Docker environment.

DOCKER:
source /opt/ros/humble/setup.bash
source /opt/reachy_ws/install/setup.bash
ros2 launch reachy_bringup reachy.launch.py gazebo:=true start_sdk_server:=true
