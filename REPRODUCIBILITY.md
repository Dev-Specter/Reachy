# Reproducibility

## Environment

- ROS 2 Humble
- Gazebo Classic 11
- Docker image: `pollenrobotics/reachy_2023:1.1`
- Docker container: `reachy_2023`
- Workspace inside container: `/opt/reachy_ws`

Python dependencies from the final container environment are recorded in:

`requirements.txt`

## Main source package

`reachy_rl_control/reachy_rl_control/`

Primary entry points:

- `train_nav.py`
- `evaluate_nav.py`
- `make_eval_set.py`
- `probe_free_space.py`

The active navigation stack supports:

- PPO
- SAC
- MBPO
- UR-MBPO

## Simulation

Simulation assets are under:

`rl_sim/`

Key files:

- `rl_sim/sim_up.sh`
- `rl_sim/sim_down.sh`
- `rl_sim/pedestrian_proxy_controller.py`
- `rl_sim/urdf/`
- `rl_sim/maps/`

World revisions, spawn poses, task-set layout, and simulation setup details are documented in:

`rl_sim/README.md`

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
- 150 episodes per algorithm across the three seeds

Final outputs:

`results/static_hospital/`

The corresponding aggregate summary is:

`results/tables/hospital_static_matched50_summary.csv`

## Hospital development benchmark

The retained development benchmark uses:

- 20 Hospital tasks
- seed 42
- 50,000 training steps

Algorithms:

- SAC
- MBPO
- UR-MBPO

Final development summaries are stored under:

`results/static_hospital/dev/`

These development results are separate from the final matched 50-task Hospital benchmark.

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

Training task categories include:

- straight general
- straight interaction
- turning general
- turning interaction
- complex general
- complex interaction

Final dynamic training metadata is stored under:

`results/dynamic_hospital/train/`

### Held-out dynamic evaluation

The final dynamic evaluation uses held-out diagonal pedestrian motion across:

- straight geometry
- turning geometry
- complex geometry
- general exposure
- interaction exposure
- pedestrian phase offsets 0, 2, 4 and 6

Final outputs are stored under:

`results/dynamic_hospital/eval/`

### Seen pedestrian crossing

The seen pedestrian-crossing evaluation covers:

- SAC
- MBPO
- UR-MBPO
- seeds 42–44
- phase offsets 0, 2, 4 and 6
- 5 episodes per seed/phase combination

Raw outputs are stored under:

`results/dynamic_hospital/seen_crossing/`

The aggregate dissertation table is:

`results/tables/table_7_1_seen_crossing.csv`

### Unseen pedestrian motion

The unseen-motion evaluation covers:

- slow
- fast
- reverse
- stop-go

It uses:

- SAC
- MBPO
- UR-MBPO
- seeds 42–44
- phase offsets 0, 2, 4 and 6
- 5 episodes per seed/phase combination

Raw outputs are stored under:

`results/dynamic_hospital/unseen_motion/`

The aggregate dissertation table is:

`results/tables/table_7_2_unseen_motion.csv`

## Transfer evaluation

Final transfer outputs are stored under:

`results/transfer/`

They include:

- static Hospital-trained → Bookstore
- dynamic Hospital-trained → Bookstore
- Bookstore-trained → Hospital
- Bookstore-trained → Bookstore in-domain

Relevant directories are:

`results/transfer/hospital_to_bookstore/`

`results/transfer/dynamic_hospital_to_bookstore/`

`results/transfer/bookstore_to_hospital/`

`results/transfer/bookstore_in_domain/`

These transfer experiments use seed 42 only and should be interpreted descriptively rather than as multi-seed statistical comparisons.

## Bookstore training

Final Bookstore in-domain training metadata is stored under:

`results/bookstore_in_domain/train/`

Subdirectories are provided for:

- PPO
- SAC
- MBPO
- UR-MBPO

Each retained training directory contains compact metadata such as:

- `config.json`
- `summary.json`
- `train.monitor.csv`

## Frozen benchmark manifest

The frozen 50k benchmark manifest is:

`results/manifests/frozen_50k_manifest.txt`

It records the final static benchmark model references and hashes used for reproducibility.

## Trained models

Model checkpoints are intentionally not stored directly in Git.

Final trained models used for the dissertation are published at:

https://github.com/Dev-Specter/Reachy/releases/tag/models-50k-final

The release contains:

- static Hospital SAC, MBPO and UR-MBPO models for seeds 42–44
- dynamic Hospital SAC, MBPO and UR-MBPO models for seed 42
- `SHA256SUMS.txt` for all released model archives
- `frozen_50k_manifest.txt` for the frozen static-benchmark hashes

The static Hospital model hashes can be verified against:

`results/manifests/frozen_50k_manifest.txt`

Large model archives remain outside the Git repository itself.

## Historical experiments

Earlier exploratory training scripts, evaluation scripts, development runs, superseded result files, and legacy simulation artifacts are retained under:

`legacy/`

They are not part of the final active dissertation benchmark.

## Interpretation

The project is a simulation-based research study.

UR-MBPO applies uncertainty-aware model-based learning and a risk penalty during synthetic model rollouts.

The method does not provide a formal safety guarantee.

Reported success, collision, clearance, return, and uncertainty results apply to the specific simulation environments, robot model, task distributions, and pedestrian proxies used in this repository.

No claim of real-world deployment safety is made.
