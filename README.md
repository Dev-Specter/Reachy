# Risk-Aware Model-Based Reinforcement Learning for Dynamic Navigation of the Reachy Mobile Robot

This repository contains the final code, simulation assets, task sets, evaluation outputs, and reproducibility metadata for the dissertation:

**Risk-Aware Model-Based Reinforcement Learning for Dynamic Navigation of the Reachy Mobile Robot**

The project studies reinforcement-learning-based mobile navigation for Reachy in ROS 2 and Gazebo, with particular emphasis on model-based reinforcement learning and navigation around moving pedestrians.

## Methods

The final navigation stack supports:

- PPO
- SAC
- MBPO
- UR-MBPO

The dissertation evaluates PPO, SAC, MBPO and UR-MBPO in the matched static benchmark and transfer experiments, while the dynamic-navigation experiments focus on SAC, MBPO and UR-MBPO.

UR-MBPO extends the model-based approach with uncertainty-aware synthetic rollout handling and a risk penalty applied during synthetic model rollouts. It should not be interpreted as providing a formal safety guarantee.

## System

The experiments use:

- Reachy 2023 mobile robot simulation
- ROS 2 Humble
- Gazebo Classic 11
- LiDAR-based navigation observations
- continuous mobile-base velocity actions
- AWS RoboMaker Hospital world
- AWS RoboMaker Bookstore world
- static and dynamic pedestrian navigation tasks

The final experiments were run in simulation.

## Repository structure

```text
.
├── docs/
│   └── media/
├── legacy/
├── reachy_rl_control/
├── results/
├── rl_sim/
├── requirements.txt
├── REPRODUCIBILITY.md
└── README.md
```

### `reachy_rl_control/`

Active ROS 2 Python package containing the final dissertation navigation stack.

Important modules include:

```text
environment_nav.py
evaluate_nav.py
make_eval_set.py
mbpo_dynamics.py
nav_interface.py
nav_mbpo.py
nav_tasks.py
nav_wrappers.py
probe_free_space.py
train_nav.py
ur_mbpo_dynamics.py
```

Final ROS 2 console entry points:

```text
train_nav
evaluate_nav
make_eval_set
probe_free_space
```

### `rl_sim/`

Simulation support files, including:

- Reachy simulation URDF
- simulation startup and shutdown scripts
- static and dynamic task sets
- pedestrian proxy controller
- world-generation utilities
- simulation checks

See `rl_sim/README.md` for world revisions and simulation details.

### `results/`

Contains the final dissertation evaluation artifacts and compact training metadata.

See `results/README.md` for the mapping between dissertation tables/figures and repository files.

### `legacy/`

Contains historical experiments, superseded scripts, older evaluation outputs, and development code retained for provenance.

The contents of `legacy/` are not part of the final active navigation stack.

## Final matched Hospital benchmark

The final static benchmark uses 50 fixed Hospital evaluation tasks, 50,000 training steps, and seeds 42–44.

| Algorithm | Success | Collision | Mean return |
|---|---:|---:|---:|
| PPO | 76.0% | 22.7% | +50.26 |
| SAC | 41.3% | 9.3% | +27.42 |
| MBPO | 65.3% | 6.0% | +50.60 |
| UR-MBPO | 64.0% | 7.3% | +48.37 |

PPO achieved the highest success rate on the corrected final matched static benchmark, but also had the highest collision rate.

MBPO and UR-MBPO achieved similar success, substantially fewer collisions than PPO, and more timeout-dominated failures.

SAC underperformed the other three methods on success and showed the highest timeout rate.

## Dynamic-navigation evaluation

Dynamic-trained SAC, MBPO and UR-MBPO policies were evaluated on held-out diagonal pedestrian motion.

Seed 42 final results:

| Algorithm | Goal | Human collision | Static collision | Timeout |
|---|---:|---:|---:|---:|
| SAC | 52.9% | 10.8% | 27.1% | 9.2% |
| MBPO | 59.6% | 11.7% | 23.8% | 5.0% |
| UR-MBPO | 55.4% | 14.6% | 23.3% | 6.7% |

MBPO produced the strongest overall held-out dynamic result in these experiments.

UR-MBPO did not uniformly improve pedestrian safety, so the results should be interpreted as an uncertainty-aware model-based navigation study rather than a formal safety result.

## Seen pedestrian crossing

Static-trained SAC, MBPO and UR-MBPO policies were evaluated against a seen pedestrian-crossing pattern.

Raw outputs are stored under:

```text
results/dynamic_hospital/seen_crossing/
```

The aggregate dissertation table is:

```text
results/tables/table_7_1_seen_crossing.csv
```

## Unseen pedestrian motion

Static-trained policies were evaluated against four unseen pedestrian-motion variants:

- slow
- fast
- reverse
- stop-go

Raw outputs are stored under:

```text
results/dynamic_hospital/unseen_motion/
```

The aggregate dissertation table is:

```text
results/tables/table_7_2_unseen_motion.csv
```

These results show that no method is uniformly safest across all pedestrian-motion conditions.

## Transfer experiments

The repository includes:

- static Hospital-trained policies evaluated in Bookstore
- dynamic Hospital-trained policies evaluated in Bookstore
- Bookstore-trained policies evaluated in Hospital
- Bookstore-trained policies evaluated in Bookstore

Relevant directories:

```text
results/transfer/hospital_to_bookstore/
results/transfer/dynamic_hospital_to_bookstore/
results/transfer/bookstore_to_hospital/
results/transfer/bookstore_in_domain/
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

### Hospital

```bash
/opt/reachy_ws/rl_sim/sim_up.sh \
  /opt/reachy_ws/external/aws-robomaker-hospital-world/worlds/hospital_rl.world \
  0.37 1.07 0.0
```

### Dynamic Hospital

```bash
cp /opt/reachy_ws/rl_sim/worlds/hospital_dynamic_rl.world \
   /opt/reachy_ws/external/aws-robomaker-hospital-world/worlds/

/opt/reachy_ws/rl_sim/sim_up.sh \
  /opt/reachy_ws/external/aws-robomaker-hospital-world/worlds/hospital_dynamic_rl.world \
  0.37 1.07 0.0
```

### Bookstore

```bash
/opt/reachy_ws/rl_sim/sim_up.sh \
  /opt/reachy_ws/external/aws-robomaker-bookstore-world/worlds/bookstore_rl.world \
  -2.8 -0.2 -0.813669
```

### Shutdown

```bash
/opt/reachy_ws/rl_sim/sim_down.sh
```

## Training

The final training entry point is:

```bash
ros2 run reachy_rl_control train_nav --help
```

Available algorithms include:

```text
ppo
sac
mbpo
ur_mbpo
```

## Evaluation

The final evaluator is:

```bash
ros2 run reachy_rl_control evaluate_nav --help
```

Supported worlds include:

```text
hospital
bookstore
hospital_dynamic_train
hospital_dynamic_eval
```

A fixed task set can also be supplied using:

```text
--eval-set
```

## Task sets

### Hospital static

```text
rl_sim/maps/hospital/static/dev.json
rl_sim/maps/hospital/static/eval.json
rl_sim/maps/hospital/static/free.json
```

### Hospital dynamic training

```text
rl_sim/maps/hospital/dynamic/train/
```

Contains balanced task pools for:

```text
straight_general
straight_interaction
turning_general
turning_interaction
complex_general
complex_interaction
```

### Hospital dynamic evaluation

```text
rl_sim/maps/hospital/dynamic/eval/
```

Uses the same geometry/exposure groups with held-out pedestrian motion used by the final dynamic evaluation.

### Bookstore

```text
rl_sim/maps/bookstore/
```

Contains:

```text
eval.json
free.json
free.txt
```

## Reproducibility environment

The final experiments were run using the Docker image:

```text
pollenrobotics/reachy_2023:1.1
```

Python dependencies from the final container environment are recorded in:

```text
requirements.txt
```

The pinned AWS RoboMaker Hospital and Bookstore world revisions used by the simulation setup are documented in:

```text
rl_sim/README.md
```

Additional experiment protocol and repository reproducibility information is provided in:

```text
REPRODUCIBILITY.md
```

## Results

Final result artifacts are organized under:

```text
results/static_hospital/
results/dynamic_hospital/
results/bookstore_in_domain/
results/transfer/
results/tables/
results/manifests/
```

The frozen final-run manifest is:

```text
results/manifests/frozen_50k_manifest.txt
```

## Media

Dissertation figures and supporting media are stored under:

```text
docs/media/
```

Current figures include:

```text
docs/media/figures/hospital_50k_learning_success.png
docs/media/figures/hospital_50k_learning_collision.png
docs/media/figures/clearance_exposure_hospital_bookstore.png
```

Setup screenshots are stored under:

```text
docs/media/screenshots/setup/
```

## Trained models

Final trained model archives are published in the GitHub Release:

https://github.com/Dev-Specter/Reachy/releases/tag/models-50k-final

The release contains:

- static Hospital SAC, MBPO and UR-MBPO models for seeds 42–44
- dynamic Hospital SAC, MBPO and UR-MBPO models for seed 42
- `SHA256SUMS.txt` containing checksums for all released model archives
- `frozen_50k_manifest.txt` containing the frozen static-benchmark hashes

The static Hospital model hashes can be checked against:

```text
results/manifests/frozen_50k_manifest.txt
```

Large trained-model files are intentionally excluded from the Git repository itself.

## Notes on interpretation

The project is a simulation-based research study.

The reported collision, clearance, success and uncertainty metrics apply to the specific Gazebo environments, robot model, task sets and pedestrian proxies used here.

UR-MBPO uses a synthetic-rollout risk penalty and uncertainty-aware model-based learning. It does not provide a formal safety guarantee.

No claim of real-world deployment safety is made.
