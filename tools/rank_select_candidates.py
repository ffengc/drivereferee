"""Gated selection among K candidates with the referee on predicted maps.

Rank key: violating rollout steps (asc), then margin (desc), then deviation from the default
candidate (asc). The default (first dir) is replaced only if the best candidate improves by
at least --delta-steps violating steps, or, at equal steps, by --delta-margin meters.

    LD_LIBRARY_PATH= python tools/rank_select_candidates.py --cand-dirs <seed0>/test <seed1>/test \\
        --occ <predicted maps> --speeds <speeds.json> --out <selected>/test --report <csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import referee as RF  # noqa: E402


def score(traj: np.ndarray, sp: float, occ: dict, sim) -> tuple[int, float]:
    """-> (violating steps, margin); margin = min over the two gates."""
    r = RF.judge_one(traj, sp, occ, sim)
    viol = int(r["dac_viol"]) + int(r["nc_viol"])
    m = [float(r[k]) for k in ("dac_margin", "nc_margin") if r[k] != ""]
    return viol, (min(m) if m else float("inf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dirs", nargs="+", required=True, help="candidate dirs; the first is the default output")
    ap.add_argument("--occ", required=True, help="predicted occupancy dir (<token>.npz)")
    ap.add_argument("--speeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta-steps", type=int, default=1)
    ap.add_argument("--delta-margin", type=float, default=0.4)
    ap.add_argument("--report", default=None, help="per-token CSV")
    args = ap.parse_args()

    sim = RF.PDMSimulator(RF._SAMPLING)
    speeds = json.load(open(args.speeds))
    dirs = [Path(d) for d in args.cand_dirs]
    occ_root = Path(args.occ)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = dirs[0]
    toks = sorted(p.stem for p in base.glob("*.npy"))
    print(f"{len(toks)} default tokens, {len(dirs)} candidate dirs", flush=True)

    rows, n_repl, n_multi = [], 0, 0
    for i, t in enumerate(toks):
        d0 = np.load(base / f"{t}.npy")
        cands = [(0, d0)]
        for k, d in enumerate(dirs[1:], start=1):
            p = d / f"{t}.npy"
            if p.exists():
                cands.append((k, np.load(p)))
        if len(cands) == 1 or not (occ_root / f"{t}.npz").exists() or t not in speeds:
            np.save(out / f"{t}.npy", d0)  # no candidate or no map: keep the default
            rows.append({"token": t, "n_cand": len(cands), "picked": 0, "replaced": 0,
                         "v0": "", "vbest": "", "m0": "", "mbest": ""})
            continue
        n_multi += 1
        occ = dict(np.load(occ_root / f"{t}.npz", allow_pickle=True))
        sp = float(speeds[t])
        scored = []
        for k, tr in cands:
            v, m = score(tr, sp, occ, sim)
            dev = float(np.abs(tr - d0).mean())
            scored.append((v, -m, dev, k, tr))
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        v0, m0 = score(d0, sp, occ, sim)
        vb, mb, kb, trb = scored[0][0], -scored[0][1], scored[0][3], scored[0][4]
        better = (v0 - vb >= args.delta_steps) or (vb == v0 and mb - m0 >= args.delta_margin)
        pick = trb if (kb != 0 and better) else d0
        n_repl += int(kb != 0 and better)
        np.save(out / f"{t}.npy", pick)
        rows.append({"token": t, "n_cand": len(cands), "picked": kb if better else 0,
                     "replaced": int(kb != 0 and better),
                     "v0": v0, "vbest": vb, "m0": round(m0, 3), "mbest": round(mb, 3)})
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(toks)}  replaced {n_repl}", flush=True)

    print(f"done: {len(toks)} tokens -> {out}; multi-candidate scenes {n_multi}, replaced {n_repl}")
    if args.report:
        with open(args.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
