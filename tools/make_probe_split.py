"""Train / calibration split for the map generator, assigned by source log (no leakage).

    python tools/make_probe_split.py --occ <occ>/navtrain --occ-test <occ>/navtest --out <splits> --train 60000 --calib 2000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _meta(path: str) -> tuple[str, str] | None:
    try:
        with np.load(path, allow_pickle=True) as z:
            m = json.loads(str(z["metadata_json"]))
        return str(m["navsim_token"]), Path(str(m["source_log"])).name
    except Exception:
        return None


def scan(occ_dir: Path, workers: int = 16) -> dict[str, str]:
    files = [str(p) for p in sorted(occ_dir.glob("*.npz"))]
    print(f"scanning {occ_dir.name}: {len(files)} npz")
    out: dict[str, str] = {}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_meta, files, chunksize=256):
            if r:
                out[r[0]] = r[1]
    print(f"  {len(out)} tokens, {len(set(out.values()))} logs")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ", required=True)
    ap.add_argument("--occ-test", default=None, help="assert zero log overlap with this split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--calib", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    tok2log = scan(Path(args.occ), args.workers)
    by_log: dict[str, list[str]] = defaultdict(list)
    for tok, log in tok2log.items():
        by_log[log].append(tok)
    logs = sorted(by_log)
    random.Random(args.seed).shuffle(logs)

    train, calib, train_logs, calib_logs = [], [], [], []
    for log in logs:
        if len(train) < args.train:
            train += sorted(by_log[log])
            train_logs.append(log)
        elif len(calib) < args.calib:
            calib += sorted(by_log[log])
            calib_logs.append(log)
        else:
            break
    assert not (set(train_logs) & set(calib_logs))

    if args.occ_test:
        test_logs = set(scan(Path(args.occ_test), args.workers).values())
        overlap = test_logs & (set(train_logs) | set(calib_logs))
        assert not overlap, f"test split shares logs with train/calib: {sorted(overlap)[:5]}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, toks, lgs in (("train", train, train_logs), ("calib", calib, calib_logs)):
        p = out / f"probe_{name}.txt"
        p.write_text("\n".join(toks) + "\n")
        print(f"{name}: {len(toks)} tokens / {len(lgs)} logs -> {p}")
    (out / "probe_meta.json").write_text(json.dumps({
        "seed": args.seed, "split_unit": "source_log", "train_tokens": len(train), "calib_tokens": len(calib),
        "train_logs": sorted(train_logs), "calib_logs": sorted(calib_logs)}, indent=1))


if __name__ == "__main__":
    main()
