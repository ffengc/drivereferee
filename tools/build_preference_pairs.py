"""Build winner/loser pairs from K self-sampled candidates and their referee scores.

A scene yields a pair only if it has both passing and violating candidates:
    winner = passing candidate with the largest margin,
    loser  = violating candidate closest to the GT trajectory (mean L2 over waypoints).

    python tools/build_preference_pairs.py --selfplay-root <root with seed*/test/*.npy and referee_seed*.csv> \\
        --gt-dir <gt_npys> --out-root <out> [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_referee(root: Path, seeds: list[int]) -> dict[tuple[str, int], dict]:
    out = {}
    for s in seeds:
        for r in csv.DictReader(open(root / f"referee_seed{s}.csv")):
            out[(r["token"], s)] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfplay-root", required=True)
    ap.add_argument("--gt-dir", required=True, help="GT (8,3) npy dir from gen_gt_trajectories.py")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    root, out, gt_dir = Path(args.selfplay_root), Path(args.out_root), Path(args.gt_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref = load_referee(root, args.seeds)
    tokens = sorted({t for (t, _) in ref})
    print(f"tokens={len(tokens)} candidates={len(ref)}")

    def cand_path(tok: str, seed: int) -> Path:
        return root / f"seed{seed}" / "test" / f"{tok}.npy"

    pairs, n_scenes, n_fail = [], 0, 0
    for tok in tokens:
        cands = []
        for s in args.seeds:
            r = ref.get((tok, s))
            if r is None:
                continue
            margin = min(float(r["dac_margin"] or 9e9), float(r["nc_margin"] or 9e9))
            fail = r["ref_dac_fail"] == "1" or r["ref_nc_fail"] == "1"
            cands.append((s, fail, margin, r))
        if len(cands) < 2:
            continue
        n_scenes += 1
        passes = [c for c in cands if not c[1]]
        fails = [c for c in cands if c[1]]
        n_fail += int(bool(fails))
        if not (passes and fails):
            continue
        w = max(passes, key=lambda c: c[2])
        gt_p = gt_dir / f"{tok}.npy"
        if gt_p.exists():
            gt = np.load(gt_p)[:, :2]

            def _dist(c):
                p = cand_path(tok, c[0])
                return float(np.linalg.norm(np.load(p)[:, :2] - gt, axis=1).mean()) if p.exists() else 9e9

            l = min(fails, key=_dist)
        else:
            l = min(fails, key=lambda c: c[2])
        pairs.append({"token": tok, "win_seed": w[0], "lose_seed": l[0],
                      "win_margin": round(w[2], 3), "lose_margin": round(l[2], 3),
                      "lose_dac": l[3]["ref_dac_fail"], "lose_nc": l[3]["ref_nc_fail"]})

    summary = {"scenes": n_scenes, "scenes_with_fail": n_fail, "pairs": len(pairs),
               "pair_yield": round(len(pairs) / max(n_scenes, 1), 4)}
    with open(out / "pairs.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in pairs)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
