# Legacy experimental code

This directory contains historical reinforcement-learning experiments and
earlier Reachy navigation implementations retained for reproducibility and
project history.

These files are **not part of the final dissertation navigation stack**.

The final implementation is located in:

- `reachy_rl_control/reachy_rl_control/`
- `rl_sim/`

## Contents

- `environments/` — earlier environment variants
- `train/` — historical one-off training scripts
- `evaluate/` — historical evaluation scripts
- `helpers/` — earlier interfaces, wrappers, and utilities
- `experiments/` — experimental UR-MBPO variants and ablations
- `tests/` — tests associated with the earlier implementation
- `root/` — historical files previously stored at repository root

The final navigation experiments use the consolidated `train_nav.py` and
`evaluate_nav.py` entry points.
