# Final Dissertation Results

This directory contains the final evaluation outputs used in the dissertation.

Historical and exploratory runs are preserved under `legacy/results/`.

## Structure

### `static_hospital/`

Final matched static Hospital benchmark.

- 50 fixed evaluation tasks
- 50,000 real environment training steps
- Seeds 42, 43 and 44
- PPO, SAC, MBPO and UR-MBPO
- `train/` contains compact training metadata and monitor logs
- `eval/` contains per-episode evaluation CSVs and summary JSON files

### `dynamic_hospital/`

Final dynamically trained Hospital benchmark.

- Seed 42
- SAC, MBPO and UR-MBPO
- Balanced dynamic-human training task pools
- Held-out diagonal pedestrian evaluation
- Straight, turning and complex geometry groups
- General and interaction exposure groups
- Four pedestrian phase offsets

### `transfer/`

Cross-environment evaluation.

- `hospital_to_bookstore/` — Hospital-trained seed-42 policies evaluated on Bookstore
- `bookstore_to_hospital/` — Bookstore-trained seed-42 policies evaluated on Hospital
- `bookstore_in_domain/` — Bookstore-trained seed-42 policies evaluated on Bookstore

These transfer experiments are single-seed evaluations and should be interpreted descriptively.

### `tables/`

Final dissertation summary tables:

- `hospital_static_matched50_summary.csv`
- `compute_cost_summary.csv`
- `failure_taxonomy_summary.csv`

### `manifests/`

Frozen experiment metadata used to identify the final benchmark runs.

## Excluded artifacts

Trained model archives, replay buffers, checkpoints and large raw run directories are not stored in Git.

Historical development experiments, tuning runs, smoke tests and earlier human-navigation experiments are retained under `legacy/`.
