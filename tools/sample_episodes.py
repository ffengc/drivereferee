"""Sample N episode indices from a GEAR split (self-sampling scene pool).

    python tools/sample_episodes.py --gear <gear>/navtrain --n 15000 --seed 42 --out episodes_15k.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gear", required=True)
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    total = sum(1 for l in open(Path(args.gear) / "meta/episodes.jsonl") if l.strip())
    idx = np.sort(np.random.default_rng(args.seed).choice(total, size=args.n, replace=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(str(i) for i in idx) + "\n")
    print(f"{args.n} of {total} episodes -> {args.out}")


if __name__ == "__main__":
    main()
