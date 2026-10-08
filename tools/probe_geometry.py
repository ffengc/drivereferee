"""Camera geometry: ego (rear-axle) 3D points -> 832x480 front-camera pixels.

ego (x fwd / y left / z up) -> camera (R^T (p - t)) -> normalized plane -> radial/tangential
distortion -> original intrinsics (1920x1120) -> anisotropic resize to 832x480.
"""

from __future__ import annotations

import numpy as np

K_ORIG = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.356, 0.173, -0.002, 0.0, -0.052])  # [k1, k2, p1, p2, k3]
SENSOR2LIDAR_T = np.array([1.6701, -0.0259, 1.5226])
SENSOR2LIDAR_R = np.array([
    [0.003096, -0.025271, 0.999676],
    [-0.999957, -0.008835, 0.002873],
    [0.008759, -0.999642, -0.025297],
])
ORIG_W, ORIG_H = 1920, 1120
OUT_W, OUT_H = 832, 480
SX, SY = OUT_W / ORIG_W, OUT_H / ORIG_H
R_VALID = 1.27  # distortion polynomial is monotonic only for r <= R_VALID


def ego_to_pixel(pts_ego: np.ndarray, apply_distortion: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) ego points -> (N,2) pixels and (N,) validity mask."""
    p = np.asarray(pts_ego, dtype=np.float64).reshape(-1, 3)
    cam = (p - SENSOR2LIDAR_T[None, :]) @ SENSOR2LIDAR_R
    z = cam[:, 2]
    ok = z > 1e-3
    zz = np.where(ok, z, 1.0)
    x, y = cam[:, 0] / zz, cam[:, 1] / zz
    if apply_distortion:
        k1, k2, p1, p2, k3 = DIST
        r2 = x * x + y * y
        ok &= r2 <= R_VALID * R_VALID
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = xd, yd
    u = (K_ORIG[0, 0] * x + K_ORIG[0, 2]) * SX
    v = (K_ORIG[1, 1] * y + K_ORIG[1, 2]) * SY
    ok &= (u >= 0) & (u < OUT_W) & (v >= 0) & (v < OUT_H)
    return np.stack([u, v], axis=-1), ok


def bev_grid_points(size: int = 128, res: float = 0.4, x_back: float = 6.4, z: float = 0.0) -> np.ndarray:
    """Cell centers (size*size, 3) of the occupancy grid, ego frame, (row, col) order."""
    xs = -x_back + (np.arange(size) + 0.5) * res
    ys = -(size * res) / 2 + (np.arange(size) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=-1)


if __name__ == "__main__":
    print(f"{ORIG_W}x{ORIG_H} -> {OUT_W}x{OUT_H}; sx={SX:.5f} sy={SY:.5f}")
