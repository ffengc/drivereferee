"""History-frame ego poses for the map generator (GEAR only stores current + future poses).

Reproduces the converter's frame selection (cur_idx = max(3, H); missing history is padded
with the earliest frame). Output npz: tokens (N,), hist_poses (N,H,3) [x, y, heading] in the
frame-0 rear-axle frame, padded (N,) count of padded frames.

    python tools/gen_history_poses.py --gear <gear>/navtest --logs <dataset>/navsim_logs/test --out <out.npz>
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from pyquaternion import Quaternion

CUR_IDX = 3
N_FUTURE = 8


def _frame_pose(frame: dict[str, Any]) -> np.ndarray:
    t = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([t[0], t[1], yaw], dtype=np.float64)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _to_frame0(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    d = poses[:, :2] - origin[None, :2]
    c, s = math.cos(float(origin[2])), math.sin(float(origin[2]))
    xy = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=-1)
    return np.column_stack([xy, _wrap(poses[:, 2] - origin[2])]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gear", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True, help="navsim_logs/<split> pkl dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--verify", type=int, default=200, help="scenes whose current+future poses are checked against GEAR")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    H = args.history
    cur_idx = max(CUR_IDX, H)
    recs = [json.loads(l) for l in (args.gear / "meta/episodes.jsonl").open() if l.strip()]
    want = {str(r["navsim_token"]): int(r["episode_index"]) for r in recs}
    info = json.load(open(args.gear / "meta/info.json"))
    chunks = int(info.get("chunks_size", 1000))
    print(f"{args.gear.name}: {len(want)} tokens, cur_idx={cur_idx}, H={H}")

    tokens: list[str] = []
    hist_out: list[np.ndarray] = []
    padded_out: list[int] = []
    worst = 0.0
    n_verified = 0

    logs = sorted(args.logs.glob("*.pkl"))
    for li, lp in enumerate(logs):
        with lp.open("rb") as fh:
            frames = pickle.load(fh)
        by_tok = {str(f.get("token")): i for i, f in enumerate(frames)}
        for tok in [t for t in by_tok if t in want]:
            i0 = by_tok[tok]
            lo = max(i0 - H, 0)
            hidx = list(range(lo, i0))
            pad = H - len(hidx)
            while len(hidx) < H:
                hidx.insert(0, hidx[0] if hidx else i0)
            origin = _frame_pose(frames[i0])
            tokens.append(tok)
            hist_out.append(_to_frame0(np.stack([_frame_pose(frames[j]) for j in hidx]), origin))
            padded_out.append(pad)
            if n_verified < args.verify and i0 + N_FUTURE < len(frames):
                ep = want[tok]
                pfile = args.gear / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
                if pfile.exists():
                    act = np.array(pq.read_table(pfile).column("action").to_pylist(), dtype=np.float64)
                    raw = np.stack([_frame_pose(frames[j]) for j in range(i0, i0 + N_FUTURE + 1)])
                    # GEAR stores float32 world coordinates: compare after the same rounding.
                    d_xy = np.abs(raw[:, :2].astype(np.float32).astype(np.float64) - act[:, :2]).max()
                    d_ang = np.abs(_wrap(raw[:, 2] - np.arctan2(act[:, 3], act[:, 2]))).max()
                    worst = max(worst, float(max(d_xy, d_ang)))
                    n_verified += 1
        if len(tokens) == len(want):
            break
        if (li + 1) % 20 == 0:
            print(f"  {li + 1}/{len(logs)} logs, {len(tokens)}/{len(want)} tokens")

    missing = sorted(set(want) - set(tokens))
    assert not missing, f"{len(missing)} tokens not found in raw logs, e.g. {missing[:5]}"
    if n_verified:
        print(f"verify: {n_verified} scenes, max deviation {worst:.2e} (tol {args.tol})")
        assert worst < args.tol, "pose convention differs from the converter"

    order = np.argsort(tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, tokens=np.array(tokens)[order], hist_poses=np.stack(hist_out)[order].astype(np.float32),
                        padded=np.array(padded_out, dtype=np.int8)[order])
    print(f"wrote {args.out}: {len(tokens)} tokens, {int((np.array(padded_out) > 0).sum())} padded")


if __name__ == "__main__":
    main()
