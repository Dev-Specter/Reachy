# Results

`final/` holds one folder per training run (`config.json`, `summary.json`, `train.monitor.csv`)
and the evaluation outputs (`*.csv` per task, `*_summary.json`). `summaries/` holds the tables
quoted in the dissertation. Models are in the GitHub Release `v1.0-dissertation`.

## Dissertation item -> source

| Dissertation | Source file(s) | Produced by |
|---|---|---|
| Table 6.1 development benchmark | `final/eval/...dev...` | `analysis/...` |
| Table 6.2 final matched 50-task benchmark | `summaries/hospital_static_matched50_summary.csv` | `analysis/...` |
| Table 6.3 bootstrap intervals | | |
| Table 6.4 paired McNemar tests | | |
| Fig. learning curves | `final/hospital_*_50k_*/train.monitor.csv` | |
| Tables 7.1-7.5 dynamic pedestrian | | |
| Tables 8.1-8.3 cross-environment | | |
| Table 9.1 failure taxonomy | `summaries/failure_taxonomy_summary.csv` | |
| Table 9.3 compute cost | `summaries/compute_cost_summary.csv` | |

## Run naming

`<world>_<algo>_seed<seed>_<steps>k_<tag>`, e.g. `hospital_sac_seed42_50k_finaldev50k_lr2e4`.
Evaluation outputs: `eval/<set>_<world>_<algo>_seed<seed>`.
