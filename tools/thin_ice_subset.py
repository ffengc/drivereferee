"""Thin-ice subset: scenes whose referee margin on the predicted map is <= threshold or that
already violate a gate. Writes the GEAR episode indices (space separated).

    python tools/thin_ice_subset.py --margins <referee.csv> --gear <gear>/navtest --out subset.txt
"""

from __future__ import annotations

import argparse
import csv
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margins", required=True, help="referee.py output on the default candidates")
    ap.add_argument("--gear", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.margins)))
    tok2ep = {json.loads(l)["navsim_token"]: json.loads(l)["episode_index"]
              for l in open(args.gear + "/meta/episodes.jsonl") if l.strip()}

    def m(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", "None", None) else 99.0

    thin = [r["token"] for r in rows
            if min(m(r, "dac_margin"), m(r, "nc_margin")) <= args.threshold
            or r.get("ref_dac_fail") == "1" or r.get("ref_nc_fail") == "1"]
    eps = [str(tok2ep[t]) for t in thin if t in tok2ep]
    open(args.out, "w").write(" ".join(eps))
    print(f"thin-ice subset: {len(eps)} / {len(rows)} scenes ({len(eps) / max(len(rows), 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
