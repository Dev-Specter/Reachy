# Risk-Aware Model-Based Reinforcement Learning for Dynamic Navigation of the Reachy Mobile Robot

This repository contains the final dissertation implementation and evaluation artifacts for reinforcement-learning-based navigation of the Reachy mobile robot in ROS 2 and Gazebo.

The project compares model-free and model-based reinforcement learning for static and dynamic indoor navigation, with particular emphasis on collision risk and pedestrian interaction.

## Methods

The final navigation stack supports:

- PPO
- Soft Actor-Critic (SAC)
- Model-Based Policy Optimisation (MBPO)
- Uncertainty/Risk-Aware MBPO (UR-MBPO)

The primary dissertation comparisons focus on SAC, MBPO and UR-MBPO, with PPO retained as an additional baseline in the matched static benchmark.

## System

The project uses:

- ROS 2 Humble
- Gazebo Classic
- Reachy mobile base simulation
- 2D LiDAR-based observations
- continuous linear and angular velocity control
- AWS RoboMaker Hospital and Bookstore environments
- dynamic pedestrian proxy evaluation

The final policy observation contains robot motion state, goal-relative features, previous actions and LiDAR sectors. Actions are continuous mobile-base velocity commands.

## Repository structure

```text
.
├── reachy_rl_control/
│   └── reachy_rl_control/
│       ├── environment_nav.py
│       ├── evaluate_nav.py
│       ├── make_eval_set.py
│       ├── mbpo_dynamics.py
│       ├── nav_interface.py
│       ├── nav_mbpo.py
│       ├── nav_tasks.py
│       ├── nav_wrappers.py
│       ├── probe_free_space.py
│       ├── train_nav.py
│       └── ur_mbpo_dynamics.py
│
├── rl_sim/
│   ├── maps/
│   ├── tests/
│   ├── tools/
│   ├── urdf/
│   ├── pedestrian_proxy_controller.py
│   ├── sim_up.sh
│   └── sim_down.sh
│
├── results/
│   ├── static_hospital/
│   ├── dynamic_hospital/
│   ├── transfer/
│   ├── tables/
│   └── manifests/
│
├── docs/
│   └── media/
│
├── legacy/
│   ├── environments/
│   ├── train/
│   ├── evaluate/
│   ├── experiments/
│   ├── helpers/
│   ├── tests/
│   └── results/
│
└── REPRODUCIBILITY.md
```

Historical development scripts and exploratory results are retained under `legacy/` for provenance but are not part of the final dissertation implementation.

## Final static Hospital benchmark

The main static benchmark uses:

- 50,000 real environment training steps
- three training seeds: 42, 43 and 44
- 50 fixed Hospital evaluation tasks per seed
- PPO, SAC, MBPO and UR-MBPO

Final aggregate results are stored in:

```text
results/tables/hospital_static_matched50_summary.csv
```

The final static benchmark showed:

| Method | Success | Collision | Mean return |
|---|---:|---:|---:|
| PPO | 76.0% | 22.7% | +50.26 |
| SAC | 92.0% | 2.0% | +60.94 |
| MBPO | 85.3% | 3.3% | +58.17 |
| UR-MBPO | 91.3% | 6.7% | +59.14 |

SAC achieved the strongest overall static result in the final matched benchmark. MBPO and UR-MBPO remained competitive, but UR-MBPO did not outperform SAC on collision rate.

## Dynamic Hospital benchmark

The dynamic benchmark uses policies trained with moving pedestrian scenarios and evaluates held-out diagonal pedestrian motion.

The final held-out evaluation includes:

- straight, turning and complex robot task geometries
- general and interaction exposure groups
- pedestrian phase offsets of 0, 2, 4 and 6 seconds
- SAC, MBPO and UR-MBPO
- seed 42

Aggregate held-out results:

| Method | Goal | Human collision | Static collision | Timeout |
|---|---:|---:|---:|---:|
| SAC | 52.9% | 10.8% | 27.1% | 9.2% |
| MBPO | 59.6% | 11.7% | 23.8% | 5.0% |
| UR-MBPO | 55.4% | 14.6% | 23.3% | 6.7% |

MBPO achieved the strongest overall result on the final held-out dynamic benchmark.

The risk-aware method should be interpreted as an empirical reward-shaping variant rather than a formal safety method. The project does not claim deployment-ready safety guarantees.

## Transfer experiments

The repository also contains single-seed transfer experiments:

- Hospital-trained policies evaluated in Bookstore
- Bookstore-trained policies evaluated in Hospital
- Bookstore-trained policies evaluated in Bookstore

These are stored under:

```text
results/transfer/
```

Bookstore navigation was substantially harder than Hospital navigation, including for policies trained directly in Bookstore, so these transfer results should be interpreted descriptively rather than as strong generalisation evidence.

## Running the simulation

Inside the Docker workspace:

```bash
cd /opt/reachy_ws
source /opt/ros/humble/setup.bash
source /opt/reachy_ws/install/setup.bash
export PYTHONPATH=/opt/reachy_ws/src/reachy_rl_control:$PYTHONPATH
```

Hospital:

```bash
/opt/reachy_ws/rl_sim/sim_up.sh \
  /opt/reachy_ws/external/aws-robomaker-hospital-world/worlds/hospital_rl.world \
  0.37 1.07 0.0
```

Bookstore:

```bash
/opt/reachy_ws/rl_sim/sim_up.sh \
  /opt/reachy_ws/external/aws-robomaker-bookstore-world/worlds/bookstore_rl.world \
  -2.8 -0.2 -0.813669
```

Shutdown:

```bash
/opt/reachy_ws/rl_sim/sim_down.sh
```

## Training and evaluation entry points

The final ROS 2 package exposes:

```text
train_nav
evaluate_nav
make_eval_set
probe_free_space
```

Examples:

```bash
ros2 run reachy_rl_control train_nav --help
```

```bash
ros2 run reachy_rl_control evaluate_nav --help
```

## Task sets

Static Hospital task sets:

```text
rl_sim/maps/hospital/static/
```

Dynamic Hospital task sets:

```text
rl_sim/maps/hospital/dynamic/train/
rl_sim/maps/hospital/dynamic/eval/
```

Bookstore task sets:

```text
rl_sim/maps/bookstore/
```

## Results

Final dissertation result artifacts are stored under:

```text
results/
```

This includes:

- per-episode evaluation CSVs
- summary JSON files
- compact training monitor logs
- final summary tables
- frozen experiment manifests

Large model checkpoints and replay buffers are intentionally excluded from Git.

## Figures and screenshots

Dissertation media is stored under:

```text
docs/media/
```

Current figures include:

- Hospital static learning success
- Hospital static learning collision rate
- Hospital/Bookstore clearance and exposure analysis

Setup screenshots are also included for reproducibility documentation.

## Reproducibility

See:

```text
REPRODUCIBILITY.md
```

for the final training budgets, seeds, benchmark definitions and repository conventions.

## Legacy code

Earlier experiments, tuning scripts, ablations and historical result files are retained under:

```text
legacy/
```

These files are preserved for project history and reproducibility but are not maintained as part of the final navigation stack.

## Disclaimer

All experiments were conducted in simulation.

The safety-related metrics in this project are empirical evaluation metrics only. UR-MBPO uses risk-sensitive reward shaping and synthetic model rollouts, but does not provide a formal safety guarantee.
