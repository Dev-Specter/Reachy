#!/usr/bin/env python3
"""Audit every archived evaluation CSV in results/.

For each evaluation file it checks:
  1. episode count matches the expected count for that experiment;
  2. task order: every episode used the intended frozen task, in order
     (matched by reference path length + straight-line distance);
  3. world consistency: every recorded collision object belongs to the
     world the evaluation claims to have used.

Usage (repo root):  python3 analysis/audit_results.py [--out results/tables]
Writes evaluation_audit.csv and prints a summary table.
"""
import argparse, csv, glob, json, math, os, re
from collections import Counter

MAPS = "rl_sim/maps"

# folder glob -> (intended world, task file or None, expected episodes per file)
GROUPS = [
    ("static_hospital/dev/*.csv",                        "Hospital",  f"{MAPS}/hospital/static/dev.json", 20),
    ("static_hospital/eval/*.csv",                       "Hospital",  f"{MAPS}/hospital/static/eval.json", 50),
    ("static_hospital/eval_superseded/*.csv",            "Hospital",  f"{MAPS}/hospital/static/eval.json", 50),
    ("ablation/*.csv",                                   "Hospital",  f"{MAPS}/hospital/static/eval.json", 50),
    ("dynamic_hospital/seen_crossing/*.csv",             "Hospital",  None, None),
    ("dynamic_hospital/unseen_motion/*.csv",             "Hospital",  None, None),
    ("dynamic_hospital/eval/*.csv",                      "Hospital",  "GROUP", 10),
    ("transfer/hospital_to_bookstore/*.csv",             "Bookstore", f"{MAPS}/bookstore/eval.json", 50),
    ("transfer/bookstore_in_domain/*.csv",               "Bookstore", f"{MAPS}/bookstore/eval.json", 50),
    ("transfer/dynamic_hospital_to_bookstore/*.csv",     "Bookstore", f"{MAPS}/bookstore/eval.json", 50),
    ("transfer/bookstore_to_hospital/*.csv",             "Hospital",  f"{MAPS}/hospital/static/eval.json", 50),
    ("transfer/dynamic_hospital_to_static_hospital/*.csv","Hospital", f"{MAPS}/hospital/static/eval.json", 50),
]

HOSPITAL_HINTS = ("hospital", "patient", "potatochip", "wheelchair", "nurse", "doctor",
                  "visitor", "kitchensink", "mopcart", "scrubs", "chair_", "vendingmachine",
                  "bed", "cabinet", "curtain", "elevator", "trolley")
BOOKSTORE_HINTS = ("retail", "bookshelf", "bookstore")


def world_of(obj):
    o = obj.lower()
    if "pedestrian" in o:
        return "Pedestrian"
    if any(h in o for h in BOOKSTORE_HINTS):
        return "Bookstore"
    if any(h in o for h in HOSPITAL_HINTS):
        return "Hospital"
    return "Unknown"


def load_tasks(path):
    d = json.load(open(path))
    return d["tasks"] if isinstance(d, dict) else d


def task_key(ref, straight):
    return (round(float(ref), 3), round(float(straight), 2))


def keys_for(tasks):
    return [task_key(t["reference_path_length"],
                     math.hypot(t["goal_x"] - t["start_x"], t["goal_y"] - t["start_y"]))
            for t in tasks]


def group_file(csv_path):
    b = os.path.basename(csv_path)
    m = re.match(r"(?:ur_mbpo|mbpo|sac|ppo)_(\w+?_(?:general|interaction))_", b)
    return f"{MAPS}/hospital/dynamic/eval/{m.group(1)}.json" if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()

    rows = []
    for pattern, world, taskfile, n_expected in GROUPS:
        for f in sorted(glob.glob(os.path.join(a.results, pattern))):
            data = list(csv.DictReader(open(f)))
            if not data or "contact_other" not in data[0]:
                continue
            n = len(data)
            # task order
            tf = group_file(f) if taskfile == "GROUP" else taskfile
            if tf:
                exp = keys_for(load_tasks(tf))
                got = [task_key(r["reference_path_length"], r["straight_line_distance"]) for r in data]
                on_task = sum(g == exp[i % len(exp)] for i, g in enumerate(got))
                order = f"{on_task}/{n}"
            else:
                order = "n/a (pool-sampled)"
            # world consistency
            worlds = Counter(world_of(r["contact_other"]) for r in data if r["contact_other"])
            wrong = sum(v for k, v in worlds.items() if k not in (world, "Pedestrian", "Unknown"))
            ok_n = (n_expected is None) or (n == n_expected)
            ok_order = (not tf) or on_task == n
            if ok_n and ok_order and wrong == 0:
                status = "PASS"
            elif ok_n and wrong == 0 and tf and len(set(exp) - set(got)) <= 1 and set(got) <= set(exp):
                # one task rejected at reset (e.g. start pose occupied) and the next task used instead
                status = "NOTE: 1 task skipped at reset"
            else:
                status = "FAIL"
            rows.append({
                "file": os.path.relpath(f, a.results), "intended_world": world,
                "episodes": n, "on_intended_task": order,
                "contacts_by_world": "; ".join(f"{k}={v}" for k, v in sorted(worlds.items())) or "none",
                "contacts_in_wrong_world": wrong, "status": status,
            })

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "evaluation_audit.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # folder summary
    summ = {}
    for r in rows:
        k = os.path.dirname(r["file"])
        s = summ.setdefault(k, {"files": 0, "pass": 0, "episodes": 0, "wrong": 0, "world": r["intended_world"]})
        s["files"] += 1; s["pass"] += r["status"] != "FAIL"
        s["episodes"] += r["episodes"]; s["wrong"] += r["contacts_in_wrong_world"]
    print(f"{'folder':45s} {'world':9s} files ok episodes wrong-world-contacts")
    for k, s in summ.items():
        print(f"{k:45s} {s['world']:9s} {s['files']:5d} {s['pass']:4d} {s['episodes']:8d} {s['wrong']:6d}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
