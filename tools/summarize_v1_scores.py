"""Official v1 score CSV -> mean PDMS and sub-scores (x100).

    python tools/summarize_v1_scores.py <csv> [<csv2> ...]
"""

from __future__ import annotations

import sys

import pandas as pd

COLS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
        "time_to_collision_within_bound", "comfort", "score"]

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    df = df[~df["token"].isin(["average", "average_all_frames"])]
    print(f"\n== {path}\n   n={len(df)}")
    for c in COLS:
        print(f"   {c:32s} {df[c].mean() * 100:8.2f}")
