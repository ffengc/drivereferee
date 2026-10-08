"""Train the map generator on frozen vision-tower tokens (features extracted online).

Targets: occupancy GT drivable (128,128) + vehicle (9,128,128); TSDF targets from signed EDT.
Loss: BCE (drivable), focal + soft-Dice (vehicle), truncated L1 (TSDF), multi-scale aux.

    torchrun --nproc_per_node=6 tools/train_map_generator.py --gear <gear>/navtrain --occ <occ>/navtrain \\
        --hist <history_poses/navtrain.npz> --splits <splits> --tower <tower dir> --out <out> --steps 10000 --bs 10
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
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent))
from export_vision_tower import load_frozen_vision_tower  # noqa: E402
from map_generator_model import N_FUT, RES, TOK_H, TOK_W, MapGenerator, build_sample_grid, n_params  # noqa: E402

TSDF_TRUNC = 3.0


def signed_tsdf(mask: np.ndarray) -> np.ndarray:
    inside = distance_transform_edt(mask) * RES
    outside = distance_transform_edt(~mask) * RES
    return np.clip(inside - outside, -TSDF_TRUNC, TSDF_TRUNC).astype(np.float32)


class MapGenDataset(Dataset):
    def __init__(self, tokens: list[str], processor, gear: str, occ: str, hist: str, with_gt: bool = True):
        self.tokens = tokens
        self.proc = processor
        self.gear, self.occ, self.with_gt = gear, occ, with_gt
        self.info = json.load(open(Path(gear) / "meta/info.json"))
        self.chunks = int(self.info.get("chunks_size", 1000))
        self.ep = {json.loads(l)["navsim_token"]: json.loads(l)["episode_index"]
                   for l in (Path(gear) / "meta/episodes.jsonl").open() if l.strip()}
        h = np.load(hist)
        self.hist = {t: p for t, p in zip(h["tokens"], h["hist_poses"])}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i: int):
        import av as pyav

        tok = self.tokens[i]
        ep = self.ep[tok]
        mp4 = Path(self.gear) / self.info["video_path"].format(
            episode_chunk=ep // self.chunks, episode_index=ep, video_key="observation.images.front_wide")
        with pyav.open(str(mp4)) as c:
            frames = [f.to_ndarray(format="rgb24") for j, f in enumerate(c.decode(video=0)) if j <= 4]
        assert len(frames) == 5, f"{tok}: got {len(frames)} frames (expected 5)"
        enc = self.proc(images=frames, return_tensors="pt")
        item = {
            "pixel_values": enc["pixel_values"].to(torch.bfloat16),
            "grid_thw": enc["image_grid_thw"],
            "hist": torch.from_numpy(self.hist[tok].copy()),
            "token": tok,
        }
        if not self.with_gt:
            return item
        z = np.load(Path(self.occ) / f"{tok}.npz", allow_pickle=True)
        drv = z["drivable"].astype(bool)
        veh = z["vehicle"].astype(bool)
        item.update({
            "drv": torch.from_numpy(drv),
            "veh": torch.from_numpy(veh),
            "drv_tsdf": torch.from_numpy(signed_tsdf(drv)),
            "veh_tsdf": torch.from_numpy(np.stack([signed_tsdf(veh[t]) for t in range(N_FUT)])),
        })
        return item


def collate(batch: list[dict]) -> dict:
    out = {"token": [b["token"] for b in batch]}
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch])
    out["grid_thw"] = torch.cat([b["grid_thw"] for b in batch])
    for k in ("hist", "drv", "veh", "drv_tsdf", "veh_tsdf"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def attach_hooks(tower, store: dict) -> None:
    """Capture the pre-merger tokens and the deepstack intermediate layers."""
    tower.merger.register_forward_pre_hook(lambda _m, inp: store.__setitem__("f", inp[0]))
    for i in tower.config.deepstack_visual_indexes:
        tower.blocks[i].register_forward_hook(
            lambda _m, _i, o, k=i: store.__setitem__(f"l{k}", o[0] if isinstance(o, tuple) else o))


@torch.no_grad()
def tower_tokens(tower, px, grid, store: dict, B: int) -> torch.Tensor:
    """-> (B, 5, HW, d_vit*4): deepstack layers + pre-merger tokens."""
    store.clear()
    tower(px, grid)
    keys = [f"l{i}" for i in tower.config.deepstack_visual_indexes] + ["f"]
    return torch.cat([store[k] for k in keys], dim=-1).view(B, 5, TOK_H * TOK_W, -1)


def focal_bce(logit, target, gamma: float = 2.0, alpha: float = 0.75):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    at = alpha * target + (1 - alpha) * (1 - target)
    return (at * ((1 - pt) ** gamma) * ce).mean()


def soft_dice(logit, target, eps: float = 1.0):
    p = torch.sigmoid(logit).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def losses(out: dict, b: dict, dev: str):
    drv = b["drv"].to(dev).float()
    veh = b["veh"].to(dev).float()
    l_drv = F.binary_cross_entropy_with_logits(out["drv_logit"].float(), drv)
    l_veh = focal_bce(out["veh_logit"].float(), veh) + 0.5 * soft_dice(out["veh_logit"].float(), veh)
    l_td = F.l1_loss(out["drv_tsdf"].float().clamp(-TSDF_TRUNC, TSDF_TRUNC), b["drv_tsdf"].to(dev))
    l_tv = F.l1_loss(out["veh_tsdf"].float().clamp(-TSDF_TRUNC, TSDF_TRUNC), b["veh_tsdf"].to(dev))
    aux = 0.0
    for name, w in (("aux2", 0.4), ("aux3", 0.2)):
        s = out[name].shape[-1]
        d_s = F.adaptive_avg_pool2d(drv[:, None], s)[:, 0]
        v_s = F.adaptive_avg_pool2d(veh, s)
        aux = aux + w * (F.binary_cross_entropy_with_logits(out[name][:, 0].float(), d_s)
                         + focal_bce(out[name][:, 1:].float(), v_s))
    tot = l_drv + l_veh + 0.5 * l_td + 0.5 * l_tv + aux
    d = {"drv": l_drv, "veh": l_veh, "tsdf_d": l_td, "tsdf_v": l_tv, "aux": aux}
    return tot, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in d.items()}


@torch.no_grad()
def iou_stats(out: dict, b: dict, dev: str) -> dict:
    r = {}
    for key, logit, gt in (("drv", out["drv_logit"], b["drv"]), ("veh", out["veh_logit"], b["veh"])):
        p = torch.sigmoid(logit.float()) > 0.5
        g = gt.to(dev).bool()
        r[f"iou_{key}"] = (p & g).sum().item() / max((p | g).sum().item(), 1)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gear", required=True)
    ap.add_argument("--occ", required=True)
    ap.add_argument("--hist", required=True, help="history poses npz")
    ap.add_argument("--splits", required=True, help="dir with probe_train.txt / probe_calib.txt")
    ap.add_argument("--tower", required=True, help="exported vision tower dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=10, help="per-GPU batch size")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=400)
    ap.add_argument("--processor", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank, world = 0, 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = "cuda"
    is_main = rank == 0
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(args.processor)

    train_toks = Path(f"{args.splits}/probe_train.txt").read_text().split()
    calib_toks = Path(f"{args.splits}/probe_calib.txt").read_text().split()
    if is_main:
        print(f"train {len(train_toks)} / calib {len(calib_toks)}")

    ds = MapGenDataset(train_toks, proc, args.gear, args.occ, args.hist)
    samp = DistributedSampler(ds, shuffle=True, drop_last=True) if ddp else None
    dl = DataLoader(ds, batch_size=args.bs, shuffle=samp is None, sampler=samp, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    # Calib tokens are grouped by log: sample across logs for evaluation.
    ev_toks = list(np.random.RandomState(0).permutation(np.array(calib_toks))[: args.eval_n])
    ds_ev = MapGenDataset(ev_toks, proc, args.gear, args.occ, args.hist)
    dl_ev = DataLoader(ds_ev, batch_size=args.bs, shuffle=False, num_workers=max(2, args.workers // 2),
                       collate_fn=collate)

    tower = load_frozen_vision_tower(args.tower, device=dev)
    store: dict = {}
    attach_hooks(tower, store)
    assert sum(1 for p in tower.parameters() if p.requires_grad) == 0
    model = MapGenerator().to(dev)
    if is_main:
        print(f"map generator params {n_params(model)} | world_size={world} global batch={args.bs * world}")
    core = model
    if ddp:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    log = open(out / "train.log", "a") if is_main else None
    step, t0, epoch = 0, time.time(), 0

    def run(batch):
        px = batch["pixel_values"].to(dev, non_blocking=True)
        gthw = batch["grid_thw"].to(dev)
        B = batch["hist"].shape[0]
        tok = tower_tokens(tower, px, gthw, store, B).float()
        grid, valid, dep = build_sample_grid(batch["hist"].to(dev))
        return model(tok, grid, valid, dep)

    while step < args.steps:
        if samp is not None:
            samp.set_epoch(epoch)
        epoch += 1
        for b in dl:
            if step >= args.steps:
                break
            o = run(b)
            loss, parts = losses(o, b, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if is_main and (step % 20 == 0 or step == 1):
                st = iou_stats(o, b, dev)
                msg = (f"[{step}/{args.steps}] loss {float(loss):.4f} "
                       + " ".join(f"{k} {v:.4f}" for k, v in parts.items())
                       + f" | train IoU drv {st['iou_drv']:.3f} veh {st['iou_veh']:.3f}"
                       + f" | {(time.time() - t0) / step:.2f}s/it")
                print(msg)
                log.write(msg + "\n")
                log.flush()
            if is_main and (step % args.eval_every == 0 or step == args.steps):
                model.eval()
                accs = {"iou_drv": [], "iou_veh": []}
                with torch.no_grad():
                    for be in dl_ev:
                        for k, v in iou_stats(run(be), be, dev).items():
                            accs[k].append(v)
                msg = (f"* [{step}] eval IoU drv {np.mean(accs['iou_drv']):.4f} "
                       f"veh {np.mean(accs['iou_veh']):.4f} (n={len(ds_ev)})")
                print(msg)
                log.write(msg + "\n")
                log.flush()
                torch.save({"model": core.state_dict(), "step": step, "args": vars(args)}, out / "ckpt.pt")
                model.train()
    if is_main:
        print(f"done, weights -> {out / 'ckpt.pt'}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
