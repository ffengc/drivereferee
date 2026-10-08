"""Predict occupancy maps with a trained map generator; output keys mirror the occupancy GT
so referee.py can read them unchanged (drivable / vehicle bool, *_prob float16, vis_union).

    torchrun --nproc_per_node=6 tools/dump_map_generator.py --ckpt <mapgen>/ckpt.pt --tower <tower dir> \\
        --gear <gear>/navtest --hist <history_poses/navtest.npz> --out <maps_navtest> [--tokens list.txt]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from export_vision_tower import load_frozen_vision_tower  # noqa: E402
from map_generator_model import MapGenerator, build_sample_grid  # noqa: E402
from train_map_generator import MapGenDataset, attach_hooks, collate, tower_tokens  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tower", required=True)
    ap.add_argument("--gear", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=None, help="token list file (default: whole split)")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--processor", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank, world = 0, 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    is_main = rank == 0
    dev = "cuda"
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    if args.tokens:
        toks = Path(args.tokens).read_text().split()
    else:
        toks = sorted(json.loads(l)["navsim_token"] for l in (Path(args.gear) / "meta/episodes.jsonl").open() if l.strip())
    have = {p.stem for p in out.glob("*.npz")} if out.exists() else set()
    todo = [t for t in toks if t not in have]
    shard = todo[rank::world]
    if is_main:
        print(f"{len(toks)} tokens, {len(have)} done, {len(todo)} to go; world={world}, this rank {len(shard)}")
    if not shard:
        if ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(args.processor)
    ds = MapGenDataset(shard, proc, gear=args.gear, occ="", hist=args.hist, with_gt=False)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, collate_fn=collate)

    tower = load_frozen_vision_tower(args.tower, device=dev)
    store: dict = {}
    attach_hooks(tower, store)
    model = MapGenerator().to(dev).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])

    meta_base = {"grid": {"size": 128, "resolution": 0.4, "x_back": 6.4, "y_half": 25.6,
                          "coordinate_frame": "frame0_rear_axle"},
                 "source": "map_generator", "ckpt": str(args.ckpt), "ckpt_step": int(ck.get("step", -1)),
                 "threshold": args.thr}
    n, t0 = 0, time.time()
    with torch.no_grad():
        for b in dl:
            px = b["pixel_values"].to(dev, non_blocking=True)
            gthw = b["grid_thw"].to(dev)
            B = b["hist"].shape[0]
            tok = tower_tokens(tower, px, gthw, store, B).float()
            grid, valid, dep = build_sample_grid(b["hist"].to(dev))
            o = model(tok, grid, valid, dep)
            dp = torch.sigmoid(o["drv_logit"].float()).cpu().numpy()
            vp = torch.sigmoid(o["veh_logit"].float()).cpu().numpy()
            vu = o["vis_union"].cpu().numpy()
            for i, t in enumerate(b["token"]):
                np.savez_compressed(
                    out / f"{t}.npz",
                    drivable=(dp[i] > args.thr), vehicle=(vp[i] > args.thr),
                    drv_prob=dp[i].astype(np.float16), veh_prob=vp[i].astype(np.float16),
                    vis_union=vu[i],
                    metadata_json=np.asarray(json.dumps({**meta_base, "navsim_token": t})))
            n += B
            if is_main and n % (args.bs * 20) == 0:
                r = n / max(time.time() - t0, 1e-6)
                print(f"  rank0 {n}/{len(shard)}  {r:.1f} scenes/s")
    if ddp:
        dist.barrier()
    if is_main:
        print(f"done: {len(list(out.glob('*.npz')))} maps -> {out} (threshold {args.thr})")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
