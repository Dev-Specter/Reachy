"""Plot the frozen static task sets on the probed free-space map.

Each task is drawn as an arrow from its start to its goal. Held-out
evaluation tasks and development tasks are shown in different colours,
and selected task indices can be labelled.

Example:
    python3 analysis/plot_task_map.py \
        --map rl_sim/maps/hospital/static/free.txt \
        --eval rl_sim/maps/hospital/static/eval.json \
        --dev rl_sim/maps/hospital/static/dev.json \
        --label 7 15 17 28 \
        --out docs/media/figures/hospital_task_map.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from plot_trajectories import read_map

EVAL_COL = "#2F5D8A"
DEV_COL = "#C98A1B"


def load_tasks(path):
    data = json.load(open(path))
    return data["tasks"] if isinstance(data, dict) else data


def draw(ax, tasks, colour, lw, alpha, labels=()):
    for i, t in enumerate(tasks):
        sx, sy, gx, gy = t["start_x"], t["start_y"], t["goal_x"], t["goal_y"]
        ax.annotate("", xy=(gx, gy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw,
                                    alpha=alpha, shrinkA=0, shrinkB=0,
                                    mutation_scale=9))
        ax.plot(sx, sy, "o", ms=3.2, mfc="white", mec=colour, mew=1.0, alpha=alpha)
        if i in labels:
            ax.annotate(str(i), xy=(sx, sy), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, fontweight="bold", color=colour,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--eval", required=True, help="held-out evaluation task JSON")
    ap.add_argument("--dev", help="development task JSON (optional)")
    ap.add_argument("--label", type=int, nargs="*", default=[], help="evaluation task indices to label")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    grid, extent = read_map(a.map)
    ev = load_tasks(a.eval)
    dv = load_tasks(a.dev) if a.dev else []

    pts = np.array([(t[k + "_x"], t[k + "_y"]) for t in ev + dv for k in ("start", "goal")])
    lo, hi = pts.min(0) - a.margin, pts.max(0) + a.margin

    cmap = ListedColormap(["#EDEDED", "#4A4A4A", "#D9D2C3", "#FFFFFF"])
    w, h = hi - lo
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * h / w + 0.3))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, extent=extent, origin="upper", interpolation="nearest")
    if dv:
        draw(ax, dv, DEV_COL, lw=1.0, alpha=0.9)
    draw(ax, ev, EVAL_COL, lw=1.2, alpha=0.95, labels=set(a.label))

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=9); ax.set_ylabel("y (m)", fontsize=9)
    ax.tick_params(labelsize=8)

    handles = [plt.Line2D([], [], color=EVAL_COL, lw=1.5, marker=">", ms=5,
                          label=f"held-out evaluation tasks ({len(ev)})")]
    if dv:
        handles.append(plt.Line2D([], [], color=DEV_COL, lw=1.5, marker=">", ms=5,
                                  label=f"development tasks ({len(dv)})"))
    handles.append(plt.Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec="k", label="start"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
