# Superseded static Hospital evaluations

These SAC, MBPO and UR-MBPO evaluation files were produced by the original evaluator before strict task forcing was added.

A post-hoc audit found that some frozen evaluation tasks could fail the reset-validity check. When this happened, the original evaluator silently advanced to the next task. As a result, the nominal 50-episode evaluations did not execute all 50 frozen tasks exactly once and instead repeated some accepted tasks.

The affected evaluations were re-run with a strict evaluator that explicitly forces each frozen task by index and records `task_index` in every CSV row. The corrected results are stored in:

`results/static_hospital/eval/`

The files in this directory are retained only for provenance and should not be used for reported benchmark results.
