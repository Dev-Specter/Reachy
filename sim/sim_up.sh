#!/usr/bin/env bash
# usage: sim_up.sh <world_file> [x] [y] [yaw] [gui]
echo "sim_up: starting"
USER_GMP="$GAZEBO_MODEL_PATH"
WORLD=$(realpath "${1:?world file}"); X=${2:-0}; Y=${3:-0}; YAW=${4:-0}; GUI=${5:-}
ROOT=$(realpath "$(dirname "$WORLD")/..")
URDF=/opt/reachy_ws/rl_sim/reachy_sim.urdf
LOG=/tmp/sim_logs; mkdir -p "$LOG"
[ -f "$WORLD" ] || { echo "world not found: $WORLD"; exit 1; }

source /opt/ros/humble/setup.bash; source /opt/reachy_ws/install/setup.bash
export DISPLAY=${DISPLAY:-:1}
export GAZEBO_MODEL_PATH="$ROOT/models:$ROOT/fuel_models${USER_GMP:+:$USER_GMP}"
echo "world root: $ROOT"
echo "GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH"

wait_for() { for i in $(seq "$2"); do eval "$1" >/dev/null 2>&1 && return 0; sleep 1; done; echo "TIMEOUT waiting for: $3"; exit 1; }

/opt/reachy_ws/rl_sim/sim_down.sh

echo "[1/5] gzserver: $WORLD"
cd "$ROOT"
setsid nohup gzserver --verbose -s libgazebo_ros_init.so -s libgazebo_ros_factory.so \
  "$WORLD" > "$LOG/gzserver.log" 2>&1 &
wait_for "ros2 service list | grep -q /set_entity_state" 90 "gazebo services"

echo "[2/5] robot_state_publisher"
setsid nohup ros2 run robot_state_publisher robot_state_publisher "$URDF" \
  --ros-args -p use_sim_time:=true > "$LOG/rsp.log" 2>&1 &
wait_for "timeout 5 ros2 param get /robot_state_publisher robot_description" 30 "RSP"

echo "[3/5] spawn reachy at ($X, $Y, yaw $YAW)"
ros2 run gazebo_ros spawn_entity.py -entity reachy -file "$URDF" \
  -x "$X" -y "$Y" -z 0.02 -Y "$YAW" > "$LOG/spawn.log" 2>&1 || { echo "spawn failed, see $LOG/spawn.log"; exit 1; }
wait_for "timeout 5 ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers '{}'" 30 "controller_manager"

echo "[4/5] controllers"
for c in joint_state_broadcaster neck_forward_position_controller \
         r_arm_forward_position_controller l_arm_forward_position_controller \
         antenna_forward_position_controller gripper_forward_position_controller; do
  ros2 run controller_manager spawner "$c" -c /controller_manager --controller-manager-timeout 20 \
    >> "$LOG/spawner.log" 2>&1 || { echo "FAILED: $c (see $LOG/spawner.log)"; exit 1; }
done

echo "[5/5] checks"
ros2 control list_controllers
echo "pose: $(gz model -m reachy -p)"
timeout 5 ros2 topic echo /reachy/body_contact --once | grep states
[ -n "$GUI" ] && { setsid nohup gzclient > "$LOG/gzclient.log" 2>&1 & }
echo "READY"
