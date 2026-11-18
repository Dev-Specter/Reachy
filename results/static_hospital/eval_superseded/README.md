# Superseded static Hospital evaluations

These files contain the original static Hospital evaluation outputs for SAC, MBPO and UR-MBPO and are retained for auditability only. They must not be used as the final reported benchmark results.

A post-hoc audit identified that the original evaluator could silently advance past a frozen evaluation task when that task failed the environment's reset-validity check. As a result, some nominal 50-episode evaluations did not execute the complete frozen 50-task benchmark and could instead repeat other accepted tasks.

The evaluator was subsequently modified to force each indexed frozen task explicitly and to record `task_index` for every episode. All affected SAC, MBPO and UR-MBPO static Hospital evaluations were then re-run on the complete frozen 50-task set.

The corrected results are stored in:

- `results/static_hospital/eval/`
- `results/static_hospital/reeval_strict/`

The superseded outputs in this directory are preserved solely to document the audit trail and explain the difference between the original and corrected reported results.

PPO was not affected by this issue because its original evaluation executed the frozen task set correctly.
