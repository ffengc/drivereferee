"""npy dir -> official NavSim v1 submission pickle.

    LD_LIBRARY_PATH= python tools/build_v1_submission.py --npy-dir <dir with <token>.npy> --out <submission.pkl>
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from navsim.common.dataclasses import Trajectory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = {}
    for p in sorted(Path(args.npy_dir).glob("*.npy")):
        poses = np.load(p).astype(np.float32)
        assert poses.shape == (8, 3), f"{p.name}: shape {poses.shape} != (8,3)"
        preds[p.stem] = Trajectory(poses=poses)
    assert preds, f"empty npy dir: {args.npy_dir}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"predictions": [preds]}, f)
    print(f"submission: {len(preds)} tokens -> {out}")


if __name__ == "__main__":
    main()
