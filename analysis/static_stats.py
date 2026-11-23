"""Regenerate the static Hospital benchmark statistics (Tables 6.2-6.4).

Reads the per-episode evaluation CSVs in results/static_hospital/eval/
(three seeds x 50 frozen tasks per algorithm) and writes:

  results/tables/hospital_static_matched50_summary.csv  per-seed rates and mean/SD (Table 6.2)
  results/tables/hospital_static_bootstrap95.csv        pooled episode bootstrap 95% intervals (Table 6.3)
  results/tables/hospital_static_mcnemar.csv            exact McNemar tests with Holm correction (Table 6.4)

By default the files are written to results/tables/regenerated/ so that the archived
tables are not overwritten; compare them with `diff`. The summary and McNemar tables
reproduce the archived files exactly. The bootstrap is a Monte Carlo procedure, so its
interval bounds can differ from the archived ones by a few tenths of a percentage point.

Usage (from the repository root):
    python3 analysis/static_stats.py [--eval-dir results/static_hospital/eval] [--out-dir results/tables/regenerated]
"""
import argparse
import csv
import itertools
import os
from math import comb

import numpy as np

ALGOS = [("PPO", "ppo"), ("SAC", "sac"), ("MBPO", "mbpo"), ("UR-MBPO", "ur_mbpo")]
SEEDS = [42, 43, 44]


def load(eval_dir, prefix, seed):
    rows = list(csv.DictReader(open(os.path.join(eval_dir, f"{prefix}_seed{seed}.csv"))))
    if len(rows) != 50:
        raise SystemExit(f"{prefix}_seed{seed}: expected 50 episodes, found {len(rows)}")
    # PPO files predate the task_index column; their episodes are tasks 0-49 in order.
    for k, r in enumerate(rows):
        r["task"] = int(r["task_index"]) if r.get("task_index") not in (None, "") else k
    return sorted(rows, key=lambda r: r["task"])


def num(v, nd=3):
    v = round(float(v), nd)
    return str(int(v)) if v == int(v) else repr(v)


def pfmt(p):
    s = "%.6f" % p
    return s if s != "0.000000" else "%.6e" % p


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/static_hospital/eval")
    ap.add_argument("--out-dir", default="results/tables/regenerated")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    data = {name: {s: load(a.eval_dir, pre, s) for s in SEEDS} for name, pre in ALGOS}

    # ---- Table 6.2: per-seed rates ----
    with open(os.path.join(a.out_dir, "hospital_static_matched50_summary.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["algorithm", "seed", "success", "collision", "timeout", "return",
                    "success_sd", "collision_sd", "timeout_sd", "return_sd"])
        for name, _ in ALGOS:
            per = []
            for s in SEEDS:
                rows = data[name][s]
                vals = [100 * np.mean([int(r[k]) for r in rows]) for k in ("success", "collision", "timeout")]
                vals.append(np.mean([float(r["return"]) for r in rows]))
                per.append(vals)
                w.writerow([name, s] + [num(v) for v in vals] + ["", "", "", ""])
            per = np.array(per)
            w.writerow([name, "mean"] + [num(v) for v in per.mean(0)] + [num(v) for v in per.std(0, ddof=1)])

    # ---- Table 6.3: pooled episode bootstrap ----
    rng = np.random.default_rng(a.seed)
    with open(os.path.join(a.out_dir, "hospital_static_bootstrap95.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["algorithm", "success_95ci", "collision_95ci", "timeout_95ci", "return_95ci"])
        for name, _ in ALGOS:
            rows = [r for s in SEEDS for r in data[name][s]]
            m = np.array([[int(r["success"]), int(r["collision"]), int(r["timeout"]), float(r["return"])] for r in rows])
            idx = rng.integers(0, len(m), size=(a.boot, len(m)))
            means = m[idx].mean(1)
            lo, hi = np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)
            cells = ["%.1f-%.1f%%" % (100 * lo[j], 100 * hi[j]) for j in range(3)]
            cells.append("%+.2f to %+.2f" % (lo[3], hi[3]))
            w.writerow([name] + cells)

    # ---- Table 6.4: exact McNemar with Holm correction ----
    out = []
    for metric in ("success", "collision", "timeout"):
        fam = []
        for (n1, _), (n2, _) in itertools.combinations(ALGOS, 2):
            x = [int(r[metric]) for s in SEEDS for r in data[n1][s]]
            y = [int(r[metric]) for s in SEEDS for r in data[n2][s]]
            b = sum(1 for u, v in zip(x, y) if u and not v)
            c = sum(1 for u, v in zip(x, y) if v and not u)
            fam.append([metric, f"{n1} vs {n2}", b, c, exact_mcnemar(b, c)])
        for row, hp in zip(fam, holm([r[4] for r in fam])):
            out.append(row + [hp])
    with open(os.path.join(a.out_dir, "hospital_static_mcnemar.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "comparison", "a_only", "b_only", "p", "holm_p"])
        for r in out:
            w.writerow(r[:4] + [pfmt(r[4]), pfmt(r[5])])
    print("written to", a.out_dir)


if __name__ == "__main__":
    main()
