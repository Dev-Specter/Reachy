#!/usr/bin/env python3
"""Plot evaluation trajectories on the probed free-space map.

Each --run is LABEL=OUT_PREFIX, where OUT_PREFIX is the --out value given to
evaluate_nav (the script reads OUT_PREFIX.csv and OUT_PREFIX_traj.csv).
All runs must have been evaluated on the same frozen task set.

Rows are tasks, columns are runs. By default the script picks the tasks whose
outcomes differ most between runs; use --task-index to choose them yourself.

Example (repo root):
  python3 analysis/plot_trajectories.py \
     --map rl_sim/maps/hospital/static/free.txt \
     --tasks rl_sim/maps/hospital/static/eval.json \
     --run PPO=traj/ppo_seed42 --run SAC=traj/sac_seed42 \
     --run MBPO=traj/mbpo_seed42 --run UR-MBPO=traj/ur_mbpo_seed42 \
     --auto 3 --out docs/media/figures/hospital_trajectories.png
"""
import argparse, csv, json, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COL = {"goal": "#2F5D8A", "collision": "#C4553B", "timeout": "#8A8A8A", "fall": "#7B3FA0"}


def read_map(path):
    lines = open(path).read().splitlines()
    m = re.search(r"grid\s+([\d.]+)\s*m.*x from\s+(-?[\d.]+).*y from\s+(-?[\d.]+)", lines[0])
    res, x0, y0 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    rows = lines[1:]
    w = max(len(r) for r in rows)
    code = {" ": 0, "#": 1, ":": 2, ".": 3, "S": 3}
    g = np.array([[code.get(ch, 0) for ch in r.ljust(w)] for r in rows])
    extent = [x0 - res / 2, x0 + (w - 0.5) * res, y0 - (len(rows) - 0.5) * res, y0 + res / 2]
    return g, extent


def read_run(prefix):
    eps = {}
    for r in csv.DictReader(open(prefix + ".csv")):
        ti = r.get("task_index", "")
        key = int(ti) if ti != "" else int(r["episode"]) - 1
        eps[key] = r["termination_reason"].replace("human_collision", "collision")
    traj = defaultdict(list)
    for r in csv.DictReader(open(prefix + "_traj.csv")):
        ti = r["task_index"]
        key = int(ti) if ti != "" else int(r["episode"]) - 1
        traj[key].append((float(r["x"]), float(r["y"])))
    return eps, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--tasks", help="frozen task JSON, used to mark the goal of each task")
    ap.add_argument("--run", action="append", required=True, help="LABEL=OUT_PREFIX")
    ap.add_argument("--task-index", type=int, nargs="*")
    ap.add_argument("--auto", type=int, default=3, help="number of tasks to pick automatically")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    grid, extent = read_map(a.map)
    goals = {}
    if a.tasks:
        T = json.load(open(a.tasks)); T = T["tasks"] if isinstance(T, dict) else T
        goals = {i: (t["goal_x"], t["goal_y"]) for i, t in enumerate(T)}
    runs = []
    for spec in a.run:
        label, prefix = spec.split("=", 1)
        runs.append((label, *read_run(prefix)))

    if a.task_index:
        tasks = a.task_index
    else:
        common = set.intersection(*[set(e) & set(t) for _, e, t in runs])
        score = lambda k: (len({e[k] for _, e, _ in runs}), -k)
        tasks = sorted(common, key=score, reverse=True)[: a.auto]

    cmap = ListedColormap(["#EDEDED", "#4A4A4A", "#D9D2C3", "#FFFFFF"])
    nr, nc = len(tasks), len(runs)
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.1 * nr), squeeze=False)
    for i, k in enumerate(tasks):
        pts = np.vstack([np.array(t[k]) for _, _, t in runs if k in t] + ([np.array([goals[k]])] if k in goals else []))
        lo, hi = pts.min(0) - a.margin, pts.max(0) + a.margin
        span = max(hi - lo); c = (lo + hi) / 2
        for j, (label, eps, traj) in enumerate(runs):
            ax = axes[i][j]
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, extent=extent, origin="upper", interpolation="nearest")
            if k in goals:
                ax.plot(*goals[k], "*", ms=12, mfc="#E3B23C", mec="k", zorder=5)
            if k in traj:
                p = np.array(traj[k]); out = eps.get(k, "?")
                ax.plot(p[:, 0], p[:, 1], color=COL.get(out, "k"), lw=1.8)
                ax.plot(*p[0], "o", ms=6, mfc="white", mec="k")
                ax.plot(*p[-1], "x" if out == "collision" else ".", ms=8, color=COL.get(out, "k"))
                ax.set_title(f"{label}: {out}", fontsize=9, color=COL.get(out, "k"))
            ax.set_xlim(c[0] - span / 2, c[0] + span / 2); ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"task {k}", fontsize=9)
    handles = [plt.Line2D([], [], color=v, lw=2, label=k) for k, v in COL.items() if k != "fall"]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="start"))
    if goals:
        handles.append(plt.Line2D([], [], marker="*", ls="", ms=11, mfc="#E3B23C", mec="k", label="goal position"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(a.out, dpi=200)
    print("tasks plotted:", tasks, "->", a.out)


if __name__ == "__main__":
    main()
