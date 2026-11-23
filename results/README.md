# Final dissertation results

This directory contains the compact final experiment artifacts used to support the dissertation.

Large trained-model files and intermediate checkpoints are intentionally excluded from Git.

Historical and superseded result files are stored under:

```text
../legacy/results/
```

## Dissertation result mapping

| Dissertation item | Repository artifact |
|---|---|
| Table 6.1, Figure 6.1 — Hospital development benchmark | `static_hospital/dev/`; figure `../docs/media/figures/dev_vs_final.png` |
| Table 6.2, Figure 6.2 — Final matched Hospital benchmark | `static_hospital/eval/`, `tables/hospital_static_matched50_summary.csv`; figure `../docs/media/figures/final_static.png` |
| Table 6.3 — Pooled bootstrap 95% intervals | `tables/hospital_static_bootstrap95.csv` |
| Table 6.4 — Exact McNemar tests with Holm correction | `tables/hospital_static_mcnemar.csv` |
| Tables 6.2–6.4 regeneration | `../analysis/static_stats.py` (summary and McNemar reproduce exactly; bootstrap is Monte Carlo) |
| Figure 6.3 — Static learning curves | `../docs/media/figures/hospital_50k_learning_success.png`, `../docs/media/figures/hospital_50k_learning_collision.png` |
| Table 6.5 — UR-MBPO component ablation | `ablation/` (filter-only and penalties-only); full UR-MBPO and MBPO from `static_hospital/eval/*_seed42.csv` |
| Table 7.1 — Seen pedestrian crossing | `dynamic_hospital/seen_crossing/` and `tables/table_7_1_seen_crossing.csv` |
| Table 7.2 — Unseen pedestrian motion | `dynamic_hospital/unseen_motion/` and `tables/table_7_2_unseen_motion.csv` |
| Tables 7.3–7.6, Figure 7.2 — Dynamic training and held-out diagonal evaluation | `dynamic_hospital/train/`, `dynamic_hospital/eval/`; figure `../docs/media/figures/diagonal.png` |
| Table 8.1 — Static Hospital → Bookstore | `transfer/hospital_to_bookstore/` |
| Table 8.2, Figure 8.2 — Bookstore in-domain | `transfer/bookstore_in_domain/`, `bookstore_in_domain/train/`; figure `../docs/media/figures/bookstore_indomain.png` |
| Table 8.3, Figure 8.3 — Bookstore → Hospital and bidirectional transfer | `transfer/bookstore_to_hospital/`; figure `../docs/media/figures/transfer.png` |
| Section 8.4 — Dynamic Hospital → Static Hospital / Bookstore | `transfer/dynamic_hospital_to_static_hospital/`, `transfer/dynamic_hospital_to_bookstore/` |
| Table 9.1 — Failure taxonomy | `tables/failure_taxonomy_summary.csv` |
| Figure 9.1, Table 9.2 — Seed-42 trajectories and timeout classification | `trajectories/` (per-step `*_traj.csv`); plotted with `../analysis/plot_trajectories.py` → `../docs/media/figures/hospital_trajectories.png` |
| Figure 9.2 — Hospital/Bookstore clearance exposure | `../docs/media/figures/clearance_exposure_hospital_bookstore.png` |
| Table 9.3 — Compute cost | `tables/compute_cost_summary.csv` |
| Figure 4.3 — Frozen Hospital task sets | `../rl_sim/maps/hospital/static/{eval,dev,free}.*`; plotted with `../analysis/plot_task_map.py` → `../docs/media/figures/hospital_task_map.png` |
| Table E.1 — Evaluation audit | `tables/evaluation_audit.csv`, produced by `../analysis/audit_results.py` |
| Figures 7.1, 8.1, 9.3, F.1–F.6 — Screenshots and video stills | `../docs/media/screenshots/` (videos are submitted separately, not stored in Git) |
| Appendix F recording setup | `../rl_sim/tools/goal_marker.py`, `../rl_sim/tools/one_task.py`, `../rl_sim/tools/reachy_nav.rviz` |

## Structure

```text
results/
├── bookstore_in_domain/
│   └── train/
├── dynamic_hospital/
│   ├── eval/
│   ├── train/
│   ├── seen_crossing/
│   └── unseen_motion/
├── manifests/
├── static_hospital/
│   ├── dev/
│   ├── eval/
│   └── train/
├── tables/
└── transfer/
    ├── bookstore_in_domain/
    ├── bookstore_to_hospital/
    ├── dynamic_hospital_to_bookstore/
    └── hospital_to_bookstore/
```

## Static Hospital benchmark

### Development benchmark

```text
static_hospital/dev/
```

Contains the 20-task Hospital development-set artifacts used in the dissertation development benchmark.

The retained 50k seed-42 development artifacts are:

```text
dev_hospital_sac_seed42_50k.csv
dev_hospital_sac_seed42_50k_summary.json
dev_hospital_mbpo_seed42_50k.csv
dev_hospital_mbpo_seed42_50k_summary.json
dev_hospital_ur_mbpo_seed42_50k.csv
dev_hospital_ur_mbpo_seed42_50k_summary.json
```

These are 20-task development-set results for:

- SAC
- MBPO
- UR-MBPO

A directly comparable final PPO development-set run was not retained, so PPO is not included in Table 6.1.

These development results should not be confused with the final matched 50-task Hospital benchmark.

### Final matched evaluation

```text
static_hospital/eval/
```

Contains the final 50-task matched Hospital evaluation for:

- PPO
- SAC
- MBPO
- UR-MBPO

Seeds:

```text
42
43
44
```

The corresponding aggregate table is:

```text
tables/hospital_static_matched50_summary.csv
```

A post-hoc evaluation audit found that the original SAC, MBPO and UR-MBPO static evaluations had been run with the Bookstore world loaded instead of Hospital (every recorded contact involved a Bookstore object), and that, because many Hospital start poses were invalid in that geometry, the evaluator silently advanced past frozen tasks that failed the reset-validity check. The affected evaluations were re-run with explicit task forcing and a recorded `task_index`, ensuring that tasks 0--49 from the frozen Hospital evaluation set are executed exactly once and in order.

The corrected evaluation files are stored in `static_hospital/eval/`. The original affected files are retained for provenance in `static_hospital/eval_superseded/`, and the strict re-evaluation copies are retained in `static_hospital/reeval_strict/`.

PPO was not affected by the reset-skipping issue, so its original evaluation files remain unchanged.

### Final static training metadata

```text
static_hospital/train/
```

Contains compact final training metadata such as:

- `config.json`
- `summary.json`
- `train.monitor.csv`

Model checkpoints are not included directly in Git.

## Dynamic Hospital

### Dynamic training metadata

```text
dynamic_hospital/train/
```

Contains final seed-42 dynamic training metadata for:

- SAC
- MBPO
- UR-MBPO

### Held-out diagonal evaluation

```text
dynamic_hospital/eval/
```

Contains the final held-out diagonal pedestrian-motion evaluation.

The evaluation covers:

- straight geometry
- turning geometry
- complex geometry
- interaction exposure
- general exposure
- pedestrian phases 0, 2, 4 and 6

Algorithms:

- SAC
- MBPO
- UR-MBPO

### Seen pedestrian crossing

```text
dynamic_hospital/seen_crossing/
```

Contains raw CSV and summary JSON files for the seen pedestrian-crossing evaluation.

The evaluation covers:

- SAC
- MBPO
- UR-MBPO
- seeds 42–44
- phases 0, 2, 4 and 6
- 5 episodes per seed/phase combination

The aggregate dissertation table is:

```text
tables/table_7_1_seen_crossing.csv
```

### Unseen pedestrian motion

```text
dynamic_hospital/unseen_motion/
```

Contains raw evaluation data for the four unseen pedestrian-motion patterns:

- slow
- fast
- reverse
- stop-go

The evaluation covers:

- SAC
- MBPO
- UR-MBPO
- seeds 42–44
- phases 0, 2, 4 and 6
- 5 episodes per seed/phase combination

The aggregate dissertation table is:

```text
tables/table_7_2_unseen_motion.csv
```

## Bookstore training

Final Bookstore in-domain training metadata is stored under:

```text
bookstore_in_domain/train/
```

Subdirectories:

```text
ppo_seed42/
sac_seed42/
mbpo_seed42/
ur_mbpo_seed42/
```

Each contains:

```text
config.json
summary.json
train.monitor.csv
```

The full model files remain outside Git.

## Transfer evaluations

### Static Hospital → Bookstore

```text
transfer/hospital_to_bookstore/
```

Contains seed-42 static Hospital-trained policies evaluated in Bookstore for:

- PPO
- SAC
- MBPO
- UR-MBPO

### Dynamic Hospital → Bookstore

```text
transfer/dynamic_hospital_to_bookstore/
```

Contains final dynamic-trained Hospital policies evaluated zero-shot in Bookstore.

Algorithms:

- SAC
- MBPO
- UR-MBPO

This evaluation is also used for the dissertation's clearance-shift discussion.

### Bookstore → Hospital

```text
transfer/bookstore_to_hospital/
```

Contains Bookstore-trained seed-42 policies evaluated in Hospital.

### Bookstore in-domain evaluation

```text
transfer/bookstore_in_domain/
```

Contains Bookstore-trained seed-42 policies evaluated in Bookstore.

Transfer experiments are single-seed descriptive evaluations and should not be interpreted as multi-seed statistical comparisons.

## Tables

```text
tables/
```

Contains compact dissertation-level summaries, including:

```text
compute_cost_summary.csv
failure_taxonomy_summary.csv
hospital_static_matched50_summary.csv
table_7_1_seen_crossing.csv
table_7_2_unseen_motion.csv
```

## Manifest

```text
manifests/frozen_50k_manifest.txt
```

Records the frozen final 50k experiment/model references used by the dissertation.

The trained models themselves are excluded from Git and are published separately in the final model release:

https://github.com/Dev-Specter/Reachy/releases/tag/models-50k-final

The release also contains checksums for the published model archives.

## Final benchmark interpretation

The final matched Hospital benchmark shows SAC as the strongest overall static result.

MBPO performs strongly but exhibits seed sensitivity.

UR-MBPO achieves competitive success but does not uniformly reduce collision rates.

The pedestrian-motion evaluations likewise do not show a uniformly safest method across all motion types.

The risk-aware term in UR-MBPO is applied to synthetic model rollouts and is not a formal safety guarantee.

## Excluded artifacts

The final results directory intentionally excludes:

- trained model archives
- intermediate checkpoints
- temporary smoke-test outputs
- partial training runs
- superseded 10k development experiments
- obsolete safety-filter experiments
- other historical outputs

Those files remain outside the final result set or under `legacy/` where appropriate.
