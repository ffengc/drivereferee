"""Paired per-scene score difference between two NavSim score CSVs with a bootstrap CI.

    python tools/paired_bootstrap.py --base <base.csv> --test <test.csv> [--column score] [--n-boot 10000]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def load(path: str, column: str) -> pd.Series:
    df = pd.read_csv(path)
    df = df[~df["token"].isin(["average", "average_all_frames"])]
    df = df[df["token"].astype(str).str.len() == 16]
    return df.set_index("token")[column].astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--column", default="score")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = pd.concat([load(args.base, args.column), load(args.test, args.column)], axis=1, join="inner")
    d = (m.iloc[:, 1] - m.iloc[:, 0]).values
    rng = np.random.default_rng(args.seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5]) * 100
    print(f"n={len(d)}  base={m.iloc[:, 0].mean() * 100:.3f}  test={m.iloc[:, 1].mean() * 100:.3f}  "
          f"delta={d.mean() * 100:+.3f}  CI95=[{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
