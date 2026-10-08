"""Map generator: frozen vision-tower tokens -> BEV drivable area + future vehicle occupancy.

Lift = deterministic projection + bilinear sampling weighted by a learned per-token depth
distribution; 5 frames (4 history + current) are pose-synchronized into the current frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from probe_geometry import DIST, K_ORIG, OUT_H, OUT_W, R_VALID, SENSOR2LIDAR_R, SENSOR2LIDAR_T, SX, SY, bev_grid_points  # noqa: E402

SIZE, RES, X_BACK = 128, 0.4, 6.4
HEIGHTS = (0.25, 1.0, 1.75, 2.5)  # sampling heights (m)
N_FUT = 9  # current + 8 future frames
TOK_H, TOK_W = 30, 52  # pre-merger token grid for 832x480
D_VIT = 1152
DEPTH_MIN, DEPTH_MAX, N_DEPTH = 1.0, 49.0, 48

_PTS_CACHE: np.ndarray | None = None


def _bev_points_all_heights() -> np.ndarray:
    """(N_z*SIZE*SIZE, 3) ego points, flattened as (height, row, col)."""
    global _PTS_CACHE
    if _PTS_CACHE is None:
        base = bev_grid_points(SIZE, RES, X_BACK, z=0.0)
        _PTS_CACHE = np.concatenate(
            [np.column_stack([base[:, 0], base[:, 1], np.full(len(base), z)]) for z in HEIGHTS]
        ).astype("float32")
    return _PTS_CACHE


def _project_batch_depth(pts: torch.Tensor):
    """(B,P,3) ego -> (B,P,2) pixels, (B,P) validity, (B,P) camera depth. Torch port of ego_to_pixel."""
    dev, dt = pts.device, pts.dtype
    R = torch.as_tensor(np.asarray(SENSOR2LIDAR_R), dtype=dt, device=dev)
    T = torch.as_tensor(np.asarray(SENSOR2LIDAR_T), dtype=dt, device=dev)
    cam = (pts - T) @ R
    z = cam[..., 2]
    ok = z > 1e-3
    zz = torch.where(ok, z, torch.ones_like(z))
    x, y = cam[..., 0] / zz, cam[..., 1] / zz
    k1, k2, p1, p2, k3 = [float(v) for v in DIST]
    r2 = x * x + y * y
    ok = ok & (r2 <= R_VALID**2)
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = (float(K_ORIG[0, 0]) * xd + float(K_ORIG[0, 2])) * SX
    v = (float(K_ORIG[1, 1]) * yd + float(K_ORIG[1, 2])) * SY
    ok = ok & (u >= 0) & (u < OUT_W) & (v >= 0) & (v < OUT_H)
    return torch.stack([u, v], -1), ok, z


def build_sample_grid(hist_poses: torch.Tensor):
    """hist_poses (B,4,3) history poses in the current ego frame -> grid (B,5,P,2), valid (B,5,P), depth (B,5,P)."""
    B = hist_poses.shape[0]
    pts = torch.as_tensor(_bev_points_all_heights(), dtype=torch.float32, device=hist_poses.device)
    P = pts.shape[0]
    grids, valids, depths = [], [], []
    for k in range(5):
        if k < 4:
            x0, y0, th = hist_poses[:, k, 0], hist_poses[:, k, 1], hist_poses[:, k, 2]
            c, s = torch.cos(th), torch.sin(th)
            dx = pts[None, :, 0] - x0[:, None]
            dy = pts[None, :, 1] - y0[:, None]
            pk = torch.stack([c[:, None] * dx + s[:, None] * dy, -s[:, None] * dx + c[:, None] * dy,
                              pts[None, :, 2].expand(B, P)], dim=-1)
        else:
            pk = pts[None].expand(B, P, 3)
        uv, ok, z = _project_batch_depth(pk)
        grids.append(torch.stack([uv[..., 0] / OUT_W * 2 - 1, uv[..., 1] / OUT_H * 2 - 1], -1))
        valids.append(ok)
        depths.append(z)
    return torch.stack(grids, 1), torch.stack(valids, 1), torch.stack(depths, 1)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.f(x)


class MapGenerator(nn.Module):
    """Multi-layer tokens -> conv neck -> depth-weighted lift -> BEV U-Net -> drivable / vehicle heads."""

    def __init__(self, d_vit: int = D_VIT, n_layers: int = 4, d_neck: int = 256, d_feat: int = 96, width: int = 192):
        super().__init__()
        self.d_feat = d_feat
        self.proj = nn.Conv2d(d_vit * n_layers, d_neck, 1)
        self.neck = nn.Sequential(ConvBlock(d_neck, d_neck), ConvBlock(d_neck, d_neck))
        self.to_feat = nn.Conv2d(d_neck, d_feat, 1)
        self.depth_head = nn.Conv2d(d_neck, N_DEPTH, 1)
        cin = 5 * len(HEIGHTS) * d_feat + 5 * len(HEIGHTS) + 5 + 1
        self.fuse = nn.Sequential(nn.Conv2d(cin, width, 1, bias=False), nn.GroupNorm(8, width), nn.SiLU())
        self.enc1, self.enc2, self.enc3 = (ConvBlock(width, width), ConvBlock(width, 2 * width),
                                           ConvBlock(2 * width, 4 * width))
        self.dec2 = ConvBlock(4 * width + 2 * width, 2 * width)
        self.dec1 = ConvBlock(2 * width + width, width)
        self.head_drv = nn.Conv2d(width, 2, 1)  # [occ logit, tsdf]
        self.head_veh = nn.Conv2d(width, 2 * N_FUT, 1)  # [occ logit x9, tsdf x9]
        self.aux2 = nn.Conv2d(2 * width, 1 + N_FUT, 1)
        self.aux3 = nn.Conv2d(4 * width, 1 + N_FUT, 1)

    def forward(self, tokens: torch.Tensor, grid: torch.Tensor, valid: torch.Tensor, depth: torch.Tensor) -> dict:
        """tokens (B,5,HW,d_vit*n_layers), grid (B,5,P,2), valid (B,5,P), depth (B,5,P) m."""
        B = tokens.shape[0]
        Nz = len(HEIGHTS)
        x = tokens.view(B * 5, TOK_H, TOK_W, -1).permute(0, 3, 1, 2)
        x = self.neck(self.proj(x))
        f = self.to_feat(x)
        dprob = self.depth_head(x).softmax(1)  # (B*5, N_DEPTH, 30, 52)

        g2 = grid.reshape(B * 5, -1, 1, 2)
        s = F.grid_sample(f, g2, mode="bilinear", padding_mode="zeros", align_corners=False)
        s = s.view(B, 5, self.d_feat, Nz, SIZE, SIZE)

        # Depth weighting: trilinear lookup of the depth distribution at (depth, v, u).
        dn = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 2 - 1
        g3 = torch.stack([grid[..., 0], grid[..., 1], dn], -1).reshape(B * 5, -1, 1, 1, 3)
        w = F.grid_sample(dprob.unsqueeze(1), g3, mode="bilinear", padding_mode="zeros", align_corners=False)
        w = w.view(B, 5, 1, Nz, SIZE, SIZE)
        vmask = valid.view(B, 5, 1, Nz, SIZE, SIZE).to(s.dtype)
        s = s * w * vmask * N_DEPTH

        vis = valid.view(B, 5, Nz, SIZE, SIZE).any(2).to(s.dtype)
        cat = torch.cat([s.reshape(B, 5 * self.d_feat * Nz, SIZE, SIZE),
                         (w * vmask).view(B, 5 * Nz, SIZE, SIZE) * N_DEPTH,
                         vis, vis.any(1, keepdim=True).to(s.dtype)], 1)
        h = self.fuse(cat)
        e1 = self.enc1(h)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        drv, veh = self.head_drv(d1), self.head_veh(d1)
        return {"drv_logit": drv[:, 0], "drv_tsdf": drv[:, 1],
                "veh_logit": veh[:, :N_FUT], "veh_tsdf": veh[:, N_FUT:],
                "aux2": self.aux2(d2), "aux3": self.aux3(e3),
                "vis_union": vis.any(1)}


def n_params(m: nn.Module) -> str:
    return f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M"


if __name__ == "__main__":
    m = MapGenerator()
    print("params:", n_params(m))
    hp = torch.zeros(2, 4, 3)
    hp[:, :, 0] = torch.tensor([-6.0, -4.0, -2.0, -1.0])
    grid, valid, dep = build_sample_grid(hp)
    out = m(torch.randn(2, 5, TOK_H * TOK_W, D_VIT * 4), grid, valid, dep)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
