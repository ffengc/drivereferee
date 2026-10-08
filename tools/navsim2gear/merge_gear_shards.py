"""Merge GEAR shards (parallel conversions over disjoint log groups) into one dataset.

Rewrites the episode_index / task_index columns inside every parquet, unifies the task
table and recomputes stats over all episodes.

    python merge_gear_shards.py --out <gear>/navtrain --shards '<gear>/_navtrain_shard_*' [--move]
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_navsim_to_gear import CAMERA_KEY, CHUNK_SIZE, ego_relative_waypoints, write_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", nargs="+", required=True, help="shard GEAR dirs (globs supported)")
    ap.add_argument("--move", action="store_true", help="move mp4s instead of copying")
    args = ap.parse_args()

    shard_dirs = sorted({d for s in args.shards for d in glob.glob(s)})
    if not shard_dirs:
        print("no shards matched!")
        return
    out = Path(args.out)
    print(f"merging {len(shard_dirs)} shards -> {out}")

    task_set: dict[str, int] = {}
    ep_records, state_arrays, action_arrays = [], [], []
    has_video_any = False
    g = 0
    for sd in shard_dirs:
        sd = Path(sd)
        ep_file = sd / "meta/episodes.jsonl"
        if not ep_file.exists():
            print(f"  ! {sd.name}: no episodes.jsonl, skip")
            continue
        eps = [json.loads(line) for line in open(ep_file)]
        eps.sort(key=lambda r: r["episode_index"])
        has_video = (sd / "videos").exists()
        has_video_any = has_video_any or has_video
        for ep in eps:
            old = ep["episode_index"]
            cmd = ep["tasks"][0]
            if cmd not in task_set:
                task_set[cmd] = len(task_set)
            tidx = task_set[cmd]
            oc, nc = old // CHUNK_SIZE, g // CHUNK_SIZE
            df = pd.read_parquet(sd / f"data/chunk-{oc:03d}/episode_{old:06d}.parquet")
            n = len(df)
            df["episode_index"] = np.full(n, g, dtype=np.int64)
            df["task_index"] = np.full(n, tidx, dtype=np.int64)
            df["annotation.language.cot"] = np.full(n, tidx, dtype=np.int64)
            dst_pq = out / f"data/chunk-{nc:03d}/episode_{g:06d}.parquet"
            dst_pq.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_pq)
            if has_video:
                src_mp4 = sd / f"videos/chunk-{oc:03d}/observation.images.{CAMERA_KEY}/episode_{old:06d}.mp4"
                dst_mp4 = out / f"videos/chunk-{nc:03d}/observation.images.{CAMERA_KEY}/episode_{g:06d}.mp4"
                dst_mp4.parent.mkdir(parents=True, exist_ok=True)
                (shutil.move if args.move else shutil.copy2)(str(src_mp4), str(dst_mp4))
            state_arrays.append(np.stack(df["observation.state"].values))
            action_arrays.append(ego_relative_waypoints(np.stack(df["action"].values)))
            ep_records.append({"episode_index": g, "tasks": [cmd], "length": n, "navsim_token": ep.get("navsim_token")})
            g += 1
        print(f"  {sd.name}: +{len(eps)} -> total {g}")

    if g == 0:
        print("no episodes merged!")
        return
    task_records = [{"task_index": i, "task": t} for t, i in sorted(task_set.items(), key=lambda kv: kv[1])]
    write_meta(out, ep_records, task_records, np.concatenate(state_arrays, 0), np.concatenate(action_arrays, 0),
               has_video_any)
    print(f"=== MERGED === {g} episodes, {len(task_records)} tasks -> {out}")


if __name__ == "__main__":
    main()
