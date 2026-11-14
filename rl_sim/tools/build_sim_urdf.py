#!/usr/bin/env python3
"""Xacro-generated Reachy URDF (use_gazebo:=true) -> RL simulation URDF.
1. remove zuuu_ros2_control (fake_components; base is driven by gazebo planar_move)
2. LiDAR update rate 5 Hz -> 10 Hz (one fresh scan per 0.1 s RL step)
3. add body contact sensor  -> /reachy/body_contact (50 Hz)
4. add arm+head contact sensors -> /reachy/arm_contact (10 Hz, 17 links)
usage: build_sim_urdf.py <in.urdf> <out.urdf>"""
import re, sys

LIDAR_RATE_HZ = 10

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

# 1. mobile-base ros2_control block
s, n = re.subn(r'\s*<ros2_control name="zuuu_ros2_control".*?</ros2_control>', '', s, flags=re.S)
assert n == 1 and 'fake_components' not in s, "unexpected ros2_control layout"

# 2. LiDAR update rate (only inside the <sensor name="lidar" type="ray"> block)
s, n = re.subn(
    r'(<sensor name="lidar" type="ray">(?:(?!</sensor>).)*?<update_rate>)[^<]*(</update_rate>)',
    rf'\g<1>{LIDAR_RATE_HZ}\g<2>', s, flags=re.S)
assert n == 1, f"expected one lidar update_rate, found {n}"


def sensor(ref, name, cols, rate, topic):
    c = '\n'.join(f'        <collision>{x}</collision>' for x in cols)
    return f'''
  <gazebo reference="{ref}">
    <sensor name="{name}_contact" type="contact">
      <always_on>true</always_on>
      <update_rate>{rate}</update_rate>
      <contact>
{c}
      </contact>
      <plugin name="{name}_bumper" filename="libgazebo_ros_bumper.so">
        <ros><namespace>/reachy</namespace><remapping>bumper_states:={topic}</remapping></ros>
        <frame_name>{ref}</frame_name>
      </plugin>
    </sensor>
  </gazebo>'''


# 3-4. contact sensors
L = 'base_footprint_fixed_joint_lump__'
blocks = sensor('base_footprint', 'body',
                [L + x for x in ('base_link_collision', 'bar_link_collision_1', 'torso_collision_2',
                                 'camera_link_collision_6', 'lidar_link_collision_10')], 50, 'body_contact')
links = [f'{sd}_{k}' for sd in ('r', 'l') for k in ('shoulder', 'shoulder_x', 'upper_arm', 'forearm',
                                                     'wrist', 'wrist2hand', 'gripper_thumb', 'gripper_finger')]
for l in links:
    blocks += sensor(l, l, [f'{l}_collision'], 10, 'arm_contact')
blocks += sensor('head_z', 'head_z', ['head_z_fixed_joint_lump__head_collision'], 10, 'arm_contact')

i = s.rfind('</robot>')
open(dst, 'w').write(s[:i] + blocks + '\n' + s[i:])
print(f"wrote {dst}: zuuu block removed, lidar {LIDAR_RATE_HZ} Hz, "
      f"1 body + {len(links) + 1} arm/head contact sensors")
