# Simulation setup: Reachy 2023 · ROS 2 Humble · Gazebo Classic 11

## Worlds

The AWS RoboMaker environments are installed inside Docker under:

```text
/opt/reachy_ws/external/
```

| World | Source | Branch | Commit |
|---|---|---|---|
| Hospital | `aws-robotics/aws-robomaker-hospital-world` (+43 Fuel models via setup script) | `ros2` | `7161eb8448f5cba6a469da7a79ae5d660b0b7f58` |
| Bookstore | `aws-robotics/aws-robomaker-bookstore-world` | `ros2` | `f5a2aaacd5418f9ecfa30f79e788856417f17c13` |

The Bookstore revision was verified directly from the final Docker environment.

The Hospital revision is the pinned revision recorded for the final simulation setup; the current installed Hospital directory does not retain its `.git` metadata.

RL world variants such as `*_rl.world` are generated with `tools/make_rl_world.py`.

The modified worlds use:

- 1 ms ODE simulation step
- `real_time_update_rate=0`
- Gazebo ROS state plugin for entity-state resets

## Directory structure

```text
rl_sim/
├── README.md
├── sim_up.sh
├── sim_down.sh
├── pedestrian_proxy_controller.py
├── maps/
│   ├── bookstore/
│   ├── hospital/
│   │   ├── static/
│   │   └── dynamic/
│   │       ├── train/
│   │       └── eval/
│   └── tests/
├── urdf/
│   ├── reachy_sim.urdf
│   ├── reachy_sim_bodyonly.urdf
│   ├── reachy_sim_nocontact.urdf
│   └── reachy_xacro_generated.urdf
├── tools/
│   ├── build_sim_urdf.py
│   ├── compare_urdf.py
│   └── make_rl_world.py
└── tests/
    ├── contact_drive_test.py
    └── wall_test.py
```

## Robot URDF

`urdf/reachy_xacro_generated.urdf` is the Reachy xacro-generated model used as the source simulation description.

`tools/build_sim_urdf.py` produces:

```text
urdf/reachy_sim.urdf
```

The final simulation startup script uses:

```text
/opt/reachy_ws/rl_sim/urdf/reachy_sim.urdf
```

## Bring-up

Run inside the Docker container.

### Hospital

```bash
/opt/reachy_ws/rl_sim/sim_up.sh \
  /opt/reachy_ws/external/aws-robomaker-hospital-world/worlds/hospital_rl.world \
  0.37 1.07 0.0
```

### Dynamic Hospital

```bash
/opt/reachy_ws/rl_sim/sim_up.sh \
  /opt/reachy_ws/rl_sim/worlds/hospital_dynamic_rl.world \
  0.37 1.07 0.0
```

### Bookstore

The final Bookstore runs used the spawn pose:

```text
x   = -2.8
y   = -0.2
yaw = -0.813669
```

Bring-up command:

```bash
/opt/reachy_ws/rl_sim/sim_up.sh \
  /opt/reachy_ws/external/aws-robomaker-bookstore-world/worlds/bookstore_rl.world \
  -2.8 -0.2 -0.813669
```

## Shutdown

```bash
/opt/reachy_ws/rl_sim/sim_down.sh
```

## Reachy container shell note

The `pollenrobotics/reachy_2023:1.1` image uses a wrapper for `/bin/bash`.

The simulation startup script explicitly invokes `/bin/bashroot` when calling `sim_down.sh` so startup works reliably inside this container.

Do not replace the container's `/bin/bash` wrapper globally.

## Static task sets

Hospital:

```text
maps/hospital/static/dev.json
maps/hospital/static/eval.json
maps/hospital/static/free.json
maps/hospital/static/free.txt
```

Bookstore:

```text
maps/bookstore/eval.json
maps/bookstore/free.json
maps/bookstore/free.txt
```

## Dynamic task sets

Training pools:

```text
maps/hospital/dynamic/train/complex_general.json
maps/hospital/dynamic/train/complex_interaction.json
maps/hospital/dynamic/train/straight_general.json
maps/hospital/dynamic/train/straight_interaction.json
maps/hospital/dynamic/train/turning_general.json
maps/hospital/dynamic/train/turning_interaction.json
```

Evaluation pools:

```text
maps/hospital/dynamic/eval/complex_general.json
maps/hospital/dynamic/eval/complex_interaction.json
maps/hospital/dynamic/eval/straight_general.json
maps/hospital/dynamic/eval/straight_interaction.json
maps/hospital/dynamic/eval/turning_general.json
maps/hospital/dynamic/eval/turning_interaction.json
```

A local smoke-test task set may also exist at:

```text
maps/tests/hospital_dynamic_smoke.json
```

That smoke file is ignored by Git and is not part of the final benchmark.

## Dynamic pedestrian control

The pedestrian proxy controller is:

```text
pedestrian_proxy_controller.py
```

It supports the dynamic navigation experiments and controlled pedestrian-motion evaluation.

The dissertation includes both seen and held-out pedestrian-motion evaluations.

## Utilities

### Generate RL world variants

```bash
python3 tools/make_rl_world.py
```

### Build simulation URDF

```bash
python3 tools/build_sim_urdf.py
```

### Compare URDF variants

```bash
python3 tools/compare_urdf.py
```

## Simulation checks

Simulation-specific checks are stored under:

```text
tests/
```

These include contact and wall-interaction tests.

## Final runtime validation

The final repository layout was validated by:

1. copying the reorganized `rl_sim/` tree into the Reachy Docker container,
2. rebuilding `reachy_rl_control`,
3. launching `hospital_dynamic_rl.world`,
4. loading the nested dynamic-training task pool from `maps/hospital/dynamic/train/`,
5. running a one-episode SAC dynamic navigation smoke evaluation.

The smoke evaluation completed successfully through the full navigation loop.
