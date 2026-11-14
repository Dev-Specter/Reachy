# Simulation setup: Reachy 2023 · ROS 2 Humble · Gazebo Classic 11

## Worlds (installed in Docker under /opt/reachy_ws/external/)
| World | Source | Branch | Commit |
|---|---|---|---|
| Hospital | aws-robotics/aws-robomaker-hospital-world (+43 Fuel models via setup.sh) | ros2 | 7161eb8448f5cba6a469da7a79ae5d660b0b7f58 |
| Bookstore (Retail) | aws-robotics/aws-robomaker-bookstore-world | ros2 | f5a2aaacd5418f9ecfa30f79e788856417f17c13 |

RL world variants (`*_rl.world`) are generated with `make_rl_world.py`: same 1 ms ODE step,
`real_time_update_rate=0` (free-running, ~3.3x real time headless), plus the `gazebo_ros_state`
plugin for `/set_entity_state` resets.

## Robot URDF
- `reachy_xacro_generated.urdf`: Reachy xacro output (use_fake_hardware:=true, use_gazebo:=true, depth_camera:=true)
- `build_sim_urdf.py`: produces `reachy_sim.urdf` from it:
  1. removes `zuuu_ros2_control` (fake_components cannot load inside gazebo_ros2_control and blocked
     controller_manager start-up; base wheels are fixed joints, motion comes from `planar_move`)
  2. body contact sensor -> `/reachy/body_contact` (50 Hz): base, mast, torso, camera, lidar (wheels excluded)
  3. arm + head contact sensors -> `/reachy/arm_contact` (10 Hz, 17 links; contacts are batched per update)
- `compare_urdf.py`: checks two URDFs are functionally equivalent.

## Bringup (inside Docker)
```bash
rl_sim/sim_up.sh <world>/worlds/hospital_rl.world 0.37 1.07 0.0  > /tmp/sim_up.log 2>&1   # Hospital
rl_sim/sim_up.sh <world>/worlds/bookstore_rl.world 0 0 0          > /tmp/sim_up.log 2>&1   # Bookstore
rl_sim/sim_down.sh
```
Order matters: gzserver -> robot_state_publisher -> spawn -> controllers
(joint_state_broadcaster + neck/arm/antenna/gripper forward position controllers).

## Design decisions / known limitations
- Hospital spawn at (0,0) overlaps furniture -> use (0.37, 1.07).
- LiDAR: 360 beams, plane at 0.25 m, mounted 0.184 m ahead of base centre.
  Clearance = |(0.184 + r cos a, r sin a)| - 0.245. Beams 174-186 deg masked (mast self-hit at 0.29 m).
  Near-collision threshold 0.10 m (front beams go below range_min at ~0.06 m clearance).
- LiDAR cannot see overhangs (tabletops, seats): contact sensors are the ground truth for collisions.
- Subscribers to `/reachy/arm_contact` need queue depth >= 50 (17 sensors publish in bursts).
- Any body or arm contact terminates the episode: arm joints are rigidly position-held, so
  post-contact dynamics are non-physical.
- Small internal base creep (~0.2 mm per simulated second); odom equals true world pose; resets teleport.
- RL step() must be timed by simulation time (/clock), not wall time.
- Action space: (v, omega), small reverse speed only.
