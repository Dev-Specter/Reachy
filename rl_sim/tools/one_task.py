#!/usr/bin/env python3
"""Write a one-task evaluation set so evaluate_nav replays exactly one frozen task.

  python3 one_task.py /opt/reachy_ws/rl_sim/maps/hospital/static/eval.json 17 /tmp/task17.json
"""
import json, sys
src, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(src))
tasks = d["tasks"] if isinstance(d, dict) else d
sel = dict(d) if isinstance(d, dict) else {}
sel["tasks"] = [tasks[idx]]
sel.setdefault("meta", {})
if isinstance(sel["meta"], dict):
    sel["meta"] = dict(sel["meta"], source=src, source_index=idx)
json.dump(sel, open(out, "w"), indent=2)
print(f"wrote {out}: task {idx} of {src}")
