"""GT trajectories (8,3) [x, y, heading] per token + speeds.json for the referee rollout.

    python tools/gen_gt_trajectories.py --gear <gear>/navtrain --out <dir> [--episodes list.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cosmos_framework.data.vfm.action.datasets.navsim_lerobot_dataset import ego_relative_waypoints  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gear", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default=None, help="episode index list (default: all episodes)")
    args = ap.parse_args()

    gear, out = Path(args.gear), Path(args.out)
    (out / "gt_npys").mkdir(parents=True, exist_ok=True)
    info = json.load(open(gear / "meta/info.json"))
    recs = {json.loads(l)["episode_index"]: json.loads(l)["navsim_token"] for l in open(gear / "meta/episodes.jsonl")}
    eps = [int(x) for x in open(args.episodes).read().split()] if args.episodes else sorted(recs)

    speeds: dict[str, float] = {}
    for i, idx in enumerate(eps):
        token = recs[idx]
        t = pq.read_table(str(gear / info["data_path"].format(episode_chunk=idx // int(info.get("chunks_size", 1000)),
                                                              episode_index=idx)))
        action = np.asarray(t["action"].to_pylist(), dtype=np.float64)  # (9,4) absolute world
        state = np.asarray(t["observation.state"].to_pylist(), dtype=np.float32)
        wp = ego_relative_waypoints(action)[1:]  # (8,4) ego-relative future
        gt = np.stack([wp[:, 0], wp[:, 1], np.arctan2(wp[:, 3], wp[:, 2])], axis=1).astype(np.float32)
        np.save(out / "gt_npys" / f"{token}.npy", gt)
        speeds[token] = float(state[0, 0])
        if (i + 1) % 5000 == 0:
            print(f"{i + 1}/{len(eps)}")
    json.dump(speeds, open(out / "speeds.json", "w"))
    print(f"done: {len(speeds)} tokens -> {out}")


if __name__ == "__main__":
    main()
