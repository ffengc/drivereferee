"""Analytic referee: roll out each candidate trajectory and check it against an occupancy map.

For every (8,3) candidate: LQR + bicycle rollout with the NavSim devkit simulator (41 poses
@ 10 Hz), then per pose the four vehicle corners are looked up in the drivable grid (DAC)
and the time-aligned vehicle grid (NC). Outputs violation counts and EDT margins (m).

Runs in the NavSim devkit environment (CPU):
    LD_LIBRARY_PATH= python tools/referee.py score --pred-dirs <seed0>/test <seed1>/test ... \\
        --occ-root <occ dir> --speeds <speeds.json> --out <referee.csv> [--source-names seed0 seed1 ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from scipy.ndimage import distance_transform_edt

# Occupancy grid (same as gen_occ_gt_navsim.py / dump_map_generator.py).
X_BACK, X_FRONT, Y_HALF, SIZE = 6.4, 44.8, 25.6, 128
RES = (X_FRONT + X_BACK) / SIZE  # 0.4 m
HALF_W, FRONT, BACK = 0.95, 3.5, 1.1  # vehicle half width / front / rear overhang from the rear axle
_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
OCC_FPS, OCC_T = 2.0, 9


def make_ego_state(speed: float) -> EgoState:
    params = get_pacifica_parameters()
    footprint = CarFootprint.build_from_rear_axle(StateSE2(0.0, 0.0, 0.0), params)
    dyn = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=params.rear_axle_to_center,
        rear_axle_velocity_2d=StateVector2D(max(speed, 0.0), 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
    )
    return EgoState(footprint, dyn, tire_steering_angle=0.0, is_in_auto_mode=True, time_point=TimePoint(0))


def rollout(traj83: np.ndarray, speed: float, sim: PDMSimulator) -> np.ndarray:
    ego = make_ego_state(speed)
    interp = transform_trajectory(Trajectory(traj83.astype(np.float32)), ego)
    states = get_trajectory_as_array(interp, _SAMPLING, ego.time_point)
    simulated = sim.simulate_proposals(states[None, ...], ego)[0]
    return simulated[:, [StateIndex.X, StateIndex.Y, StateIndex.HEADING]]  # (41,3)


def corners(x: float, y: float, h: float) -> np.ndarray:
    c, s = np.cos(h), np.sin(h)
    fwd, left = np.array([c, s]), np.array([-s, c])
    p = np.array([x, y])
    return np.stack([p + FRONT * fwd + HALF_W * left, p + FRONT * fwd - HALF_W * left,
                     p - BACK * fwd - HALF_W * left, p - BACK * fwd + HALF_W * left])


def to_cell(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N,2) meters -> (row, col, inside_window)."""
    row = np.floor((pts[:, 0] + X_BACK) / RES).astype(int)
    col = np.floor((pts[:, 1] + Y_HALF) / RES).astype(int)
    inside = (row >= 0) & (row < SIZE) & (col >= 0) & (col < SIZE)
    return row, col, inside


def judge_one(traj83: np.ndarray, speed: float, occ: dict, sim: PDMSimulator) -> dict:
    sim_path = rollout(traj83, speed, sim)
    drv = occ["drivable"].astype(bool)
    edt_drv = distance_transform_edt(drv) * RES  # distance to non-drivable (m)
    veh = occ["vehicle"].astype(bool)  # (9,H,W)
    edt_veh = [distance_transform_edt(~veh[t]) * RES for t in range(veh.shape[0])]

    dac_viol, nc_viol = 0, 0
    dac_margin, nc_margin = np.inf, np.inf
    max_dev = 0.0
    for i, (x, y, h) in enumerate(sim_path):
        pts = np.vstack([corners(x, y, h), [[x, y]]])
        row, col, inside = to_cell(pts)
        if inside.any():
            m = edt_drv[row[inside], col[inside]]
            dac_margin = min(dac_margin, float(m.min()))
            if (m <= 0).any():
                dac_viol += 1
            f = min(int(round(i * 0.1 * OCC_FPS)), OCC_T - 1)
            mv = edt_veh[f][row[inside], col[inside]]
            nc_margin = min(nc_margin, float(mv.min()))
            if (mv <= 0).any():
                nc_viol += 1
        if i % 5 == 0 and i > 0 and i // 5 - 1 < len(traj83):
            ix, iy, _ = traj83[i // 5 - 1]
            max_dev = max(max_dev, float(np.hypot(x - ix, y - iy)))
    return {
        "dac_viol": dac_viol, "nc_viol": nc_viol,
        "dac_margin": round(dac_margin, 3) if np.isfinite(dac_margin) else "",
        "nc_margin": round(nc_margin, 3) if np.isfinite(nc_margin) else "",
        "track_dev": round(max_dev, 3),
        "ref_dac_fail": int(dac_viol >= 1), "ref_nc_fail": int(nc_viol >= 1),
    }


def cmd_score(args) -> None:
    speeds = json.loads(Path(args.speeds).read_text())
    tokens = {l.strip() for l in open(args.tokens) if l.strip()} if args.tokens else None
    sim = PDMSimulator(_SAMPLING)
    rows = []
    for pred_dir in args.pred_dirs:
        pred_dir = Path(pred_dir)
        name = args.source_names.pop(0) if args.source_names else pred_dir.parent.name
        done = 0
        for p in sorted(pred_dir.glob("*.npy")):
            tok = p.stem
            if tokens is not None and tok not in tokens:
                continue
            occ_path = Path(args.occ_root) / f"{tok}.npz"
            if not occ_path.exists():
                continue
            r = judge_one(np.load(p), float(speeds.get(tok, 0.0)), np.load(occ_path), sim)
            r.update({"source": name, "token": tok})
            rows.append(r)
            done += 1
        print(f"[score] {name}: {done} candidates")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[score] total {len(rows)} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--pred-dirs", nargs="+", required=True)
    s.add_argument("--source-names", nargs="*", default=[])
    s.add_argument("--occ-root", required=True)
    s.add_argument("--speeds", required=True)
    s.add_argument("--tokens", default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
