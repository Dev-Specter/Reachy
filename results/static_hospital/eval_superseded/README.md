Superseded static Hospital evaluations

These files contain the original static Hospital evaluation outputs for SAC, MBPO and UR-MBPO and are retained for auditability only. They must not be used as reported benchmark results.

A post-hoc audit found that these evaluations were run with the Bookstore world loaded instead of the Hospital world: every collision recorded in these files involves a Bookstore object (for example aws_robomaker_retail_ChairB_01). Because many Hospital start poses were invalid in that geometry, the original evaluator also skipped 16 of the 50 frozen tasks at reset and repeated other accepted tasks, so each nominal 50-episode run covered only about 34 unique tasks.

The evaluator was then modified to force each indexed frozen task, so that an invalid reset stops the evaluation with an error, and to record task_index for every episode. All nine affected evaluations were re-run in the Hospital world on the complete frozen 50-task set. The corrected results are in:

results/static_hospital/eval/
results/static_hospital/reeval_strict/ (identical copies)

All other evaluations (PPO, the development set, the dynamic experiments, the UR-MBPO ablation and all transfer tests) were checked in the same way and are consistent with their intended worlds and task sets.

The superseded outputs in this directory are preserved solely to document the audit trail and to explain the difference between the original and corrected reported results.
