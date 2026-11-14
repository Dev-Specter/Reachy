#!/usr/bin/env python3

import csv
import math
from pathlib import Path


CSV_PATH = Path(
    "/opt/reachy_ws/src/results/"
    "ur_mbpo_human_v2_risk1_2m_seed42_20episodes.csv"
)

SUCCESS_THRESHOLD_PERCENT = 0.0
PROGRESS_THRESHOLD = 0.60


if not CSV_PATH.is_file():
    raise FileNotFoundError(
        f"Evaluation CSV not found: {CSV_PATH}"
    )


with CSV_PATH.open(
    newline="",
) as file:

    rows = list(
        csv.DictReader(file)
    )


if not rows:
    raise RuntimeError(
        "Evaluation CSV contains no episodes."
    )


successes = sum(
    int(float(row["success"]))
    for row in rows
)


progress_values = []

for row in rows:

    value = float(
        row["final_progress_ratio"]
    )

    if math.isfinite(value):
        progress_values.append(value)


if not progress_values:
    raise RuntimeError(
        "No valid progress-ratio values found."
    )


success_percent = (
    100.0
    * successes
    / len(rows)
)

mean_progress_ratio = (
    sum(progress_values)
    / len(progress_values)
)


run_zero_shot = (
    success_percent
    > SUCCESS_THRESHOLD_PERCENT
    or
    mean_progress_ratio
    >= PROGRESS_THRESHOLD
)


print(
    "\n======================================"
)
print(
    "V2 2M PREDECLARED DECISION RULE"
)
print(
    "======================================"
)

print(
    f"Episodes: {len(rows)}"
)

print(
    f"Successes: {successes}"
)

print(
    f"Success rate: "
    f"{success_percent:.1f}%"
)

print(
    f"Mean final progress ratio: "
    f"{mean_progress_ratio:.4f}"
)

print(
    f"Mean final progress: "
    f"{100.0 * mean_progress_ratio:.1f}%"
)

print(
    "\nRule:"
)

print(
    "success > 0% OR "
    "mean progress ratio >= 0.60"
)

print()

if run_zero_shot:

    print(
        "DECISION: ZERO-SHOT GENERALISATION"
    )

    print(
        "Next evaluations:"
    )

    print(
        "  3.5 m straight"
    )

    print(
        "  5 m straight"
    )

    print(
        "  L-shape"
    )

    print(
        "  curve"
    )

else:

    print(
        "DECISION: V2.1 RETRAINING"
    )

    print(
        "Current V2 does not meet the "
        "predeclared threshold."
    )

    print(
        "Next experiment:"
    )

    print(
        "  V2.1"
    )

    print(
        "  2 m human-aware"
    )

    print(
        "  UR-MBPO risk=1"
    )

    print(
        "  10k steps"
    )

    print(
        "  seed=42"
    )
