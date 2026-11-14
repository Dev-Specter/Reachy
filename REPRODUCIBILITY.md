# Reproducibility

## Environment

- ROS 2 Humble
- Gazebo Classic 11
- Docker container: `reachy_2023`
- Workspace inside container: `/opt/reachy_ws`

## Main source package

`reachy_rl_control/reachy_rl_control/`

Primary entry points:

- `train_nav.py`
- `evaluate_nav.py`
- `make_eval_set.py`
- `probe_free_space.py`

## Simulation

Simulation assets are under:

`rl_sim/`

Key files:

- `rl_sim/sim_up.sh`
- `rl_sim/sim_down.sh`
- `rl_sim/pedestrian_proxy_controller.py`
- `rl_sim/urdf/`
- `rl_sim/maps/`

## Static Hospital benchmark

Training budget:

- 50,000 real environment steps

Algorithms:

- PPO
- SAC
- MBPO
- UR-MBPO

Training seeds:

- 42
- 43
- 44

Evaluation:

- 50 fixed Hospital tasks per seed
- 150 episodes per algorithm

Final outputs:

`results/static_hospital/`

## Dynamic Hospital benchmark

Algorithms:

- SAC
- MBPO
- UR-MBPO

Training seed:

- 42

Training:

- 50,000 real environment steps
- balanced dynamic-human task pools

Evaluation:

- held-out diagonal pedestrian motion
- straight, turning and complex robot geometry groups
- general and interaction exposure groups
- pedestrian phase offsets 0, 2, 4 and 6

Final outputs:

`results/dynamic_hospital/`

## Transfer evaluation

Final transfer outputs:

`results/transfer/`

Contains:

- Hospital-trained -> Bookstore
- Bookstore-trained -> Hospital
- Bookstore-trained -> Bookstore in-domain

These transfer experiments use seed 42 only and should be interpreted descriptively.

## Historical experiments

Earlier exploratory training scripts, evaluation scripts and results are retained under:

`legacy/`

They are not part of the final dissertation benchmark.

## Trained models

Model checkpoints are intentionally not stored in Git.

The Git repository contains compact configuration, training-monitor and evaluation data required to reproduce and inspect the dissertation results.
## Trained model release

Final trained models used for the dissertation are published at:

https://github.com/Dev-Specter/Reachy/releases/tag/models-50k-final

The release includes model archives, `SHA256SUMS.txt`, and the frozen 50k benchmark manifest. The static Hospital model hashes can be verified against `results/manifests/frozen_50k_manifest.txt` in this repository.
