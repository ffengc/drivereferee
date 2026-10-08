"""Rasterize NavSim ground-truth occupancy for the referee: drivable area + vehicle boxes.

Per scene one npz in the frame-0 rear-axle frame (rows = x forward, cols = y left):
    drivable      uint8 [128, 128]        vehicle   uint8 [9, 128, 128] (t = 0..4 s @ 2 Hz)
    ego_poses     float32 [9, 3]          metadata_json (token, source_log, grid)
Grid: 0.4 m cells, x in [-6.4, 44.8) m, y in [-25.6, 25.6) m.

    python gen_occ_gt_navsim.py --gear-root <gear>/navtrain --logs-dir <dataset>/navsim_logs/trainval \\
        --maps-root <dataset>/maps --out-dir <occ>/navtrain [--shard i/n]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyquaternion import Quaternion
from shapely.ops import unary_union

X_BACK, X_FRONT, Y_HALF = 6.4, 44.8, 25.6
SIZE = 128
SUPER = 4
QUERY_RADIUS_M = 100.0
FUTURE_FRAMES = 8
VEHICLE_CLASSES = {"vehicle"}


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _frame_pose(frame: dict[str, Any]) -> np.ndarray:
    t = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([t[0], t[1], yaw], dtype=np.float64)


def _global_to_local(points_xy: np.ndarray, origin: np.ndarray) -> np.ndarray:
    delta = np.asarray(points_xy, dtype=np.float64) - origin[None, :2]
    c, s = math.cos(float(origin[2])), math.sin(float(origin[2]))
    return np.stack([c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]], axis=-1)


def _local_to_pixel(local_xy: np.ndarray, super_scale: int = 1) -> np.ndarray:
    res = (X_FRONT + X_BACK) / (SIZE * super_scale)
    local = np.asarray(local_xy, dtype=np.float64)
    return np.stack([(local[:, 1] + Y_HALF) / res, (local[:, 0] + X_BACK) / res], axis=-1)


def _iter_polygons(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def _rasterize_geometry(geometry, origin: np.ndarray) -> np.ndarray:
    high = SIZE * SUPER
    canvas = np.zeros((high, high), dtype=np.uint8)
    for polygon in _iter_polygons(geometry):
        ext = np.round(_local_to_pixel(_global_to_local(np.asarray(polygon.exterior.coords)[:, :2], origin), SUPER))
        cv2.fillPoly(canvas, [ext.astype(np.int32)], 1)
        for interior in polygon.interiors:
            hole = np.round(_local_to_pixel(_global_to_local(np.asarray(interior.coords)[:, :2], origin), SUPER))
            cv2.fillPoly(canvas, [hole.astype(np.int32)], 0)
    pooled = canvas.reshape(SIZE, SUPER, SIZE, SUPER).mean(axis=(1, 3))
    return (pooled >= 0.5).astype(np.uint8)


def _box_corners_frame0(box: np.ndarray, ego_pose: np.ndarray, origin: np.ndarray) -> np.ndarray:
    c, s = math.cos(float(ego_pose[2])), math.sin(float(ego_pose[2]))
    center_global = np.array([ego_pose[0] + c * box[0] - s * box[1], ego_pose[1] + s * box[0] + c * box[1]])
    center = _global_to_local(center_global[None], origin)[0]
    heading = float(_wrap_angle(ego_pose[2] + box[6] - origin[2]))
    fwd = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-math.sin(heading), math.cos(heading)])
    hl, hw = float(box[3]) / 2.0, float(box[4]) / 2.0
    return np.stack([center + hl * fwd + hw * left, center + hl * fwd - hw * left,
                     center - hl * fwd - hw * left, center - hl * fwd + hw * left])


def _rasterize_vehicles(frames: list[dict[str, Any]], origin: np.ndarray) -> np.ndarray:
    high = SIZE * SUPER
    out = np.zeros((len(frames), SIZE, SIZE), dtype=np.uint8)
    for t, frame in enumerate(frames):
        canvas = np.zeros((high, high), dtype=np.uint8)
        ego_pose = _frame_pose(frame)
        boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float64)
        names = [str(n) for n in frame["anns"]["gt_names"]]
        for box, name in zip(boxes, names):
            if name not in VEHICLE_CLASSES:
                continue
            px = np.round(_local_to_pixel(_box_corners_frame0(box, ego_pose, origin), SUPER)).astype(np.int32)
            cv2.fillPoly(canvas, [px], 1)
        pooled = canvas.reshape(SIZE, SUPER, SIZE, SUPER).mean(axis=(1, 3))
        out[t] = (pooled >= 0.25).astype(np.uint8)
    return out


@lru_cache(maxsize=8)
def _map_api(maps_root: str, city: str):
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    return get_maps_api(maps_root, "nuplan-maps-v1.0", city)


def _drivable(frame0: dict[str, Any], origin: np.ndarray, maps_root: Path) -> np.ndarray:
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.abstract_map import SemanticMapLayer

    map_api = _map_api(str(maps_root), str(frame0["map_location"]))
    # Query around the vehicle center like the official metric cache does.
    d = float(get_pacifica_parameters().rear_axle_to_center)
    center = origin[:2] + d * np.array([math.cos(float(origin[2])), math.sin(float(origin[2]))])
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA]
    nearby = map_api.get_proximal_map_objects(Point2D(float(center[0]), float(center[1])), QUERY_RADIUS_M, layers)
    union = unary_union([obj.polygon for layer in layers for obj in nearby[layer]])
    return _rasterize_geometry(union, origin)


def _relative_ego_poses(frames: list[dict[str, Any]], origin: np.ndarray) -> np.ndarray:
    poses = np.stack([_frame_pose(f) for f in frames])
    xy = _global_to_local(poses[:, :2], origin)
    return np.column_stack([xy, _wrap_angle(poses[:, 2] - origin[2])]).astype(np.float32)


def _find_raw_scenes(logs_dir: Path, wanted: set[str]) -> dict[str, tuple[Path, list[dict[str, Any]]]]:
    found: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    expected = FUTURE_FRAMES + 1
    for log_path in sorted(logs_dir.glob("*.pkl")):
        with log_path.open("rb") as fh:
            frames = pickle.load(fh)
        for i, frame in enumerate(frames):
            token = str(frame.get("token"))
            if token not in wanted or token in found:
                continue
            scene = frames[i : i + expected]
            if len(scene) == expected:
                found[token] = (log_path, scene)
        if len(found) == len(wanted):
            break
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gear-root", type=Path, required=True)
    ap.add_argument("--logs-dir", type=Path, required=True)
    ap.add_argument("--maps-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokens-file", type=Path, default=None, help="subset of tokens (default: all)")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th token starting at i")
    args = ap.parse_args()

    records = [json.loads(l) for l in (args.gear_root / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    tokens = [str(r["navsim_token"]) for r in records]
    if args.tokens_file:
        keep = {t.strip() for t in args.tokens_file.read_text().split()}
        tokens = [t for t in tokens if t in keep]
    i, n = (int(x) for x in args.shard.split("/"))
    tokens = tokens[i::n]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokens = [t for t in tokens if not (args.out_dir / f"{t}.npz").exists()]
    print(f"shard {args.shard}: {len(tokens)} tokens to generate")

    scenes = _find_raw_scenes(args.logs_dir, set(tokens))
    missing = sorted(set(tokens) - set(scenes))
    if missing:
        print(f"{len(missing)} tokens without a complete raw window (skipped): {missing[:5]}")

    n_ok = n_bad = 0
    for token in tokens:
        if token not in scenes:
            continue
        source_log, frames = scenes[token]
        origin = _frame_pose(frames[0])
        try:
            drivable = _drivable(frames[0], origin, args.maps_root)
        except Exception as exc:  # rare bad scenes: skip, do not abort the shard
            print(f"[skip] {token}: {exc}")
            n_bad += 1
            continue
        vehicle = _rasterize_vehicles(frames, origin)
        ego_poses = _relative_ego_poses(frames, origin)
        metadata = {
            "navsim_token": token,
            "source_log": str(source_log),
            "coordinate_frame": "frame0_rear_axle",
            "size": SIZE,
            "resolution": (X_FRONT + X_BACK) / SIZE,
            "x_back": X_BACK,
            "x_front": X_FRONT,
            "y_half": Y_HALF,
        }
        np.savez_compressed(args.out_dir / f"{token}.npz", drivable=drivable, vehicle=vehicle, ego_poses=ego_poses,
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        n_ok += 1
        if n_ok % 1000 == 0:
            print(f"  {n_ok} done")
    print(f"=== DONE === {n_ok} scenes written, {n_bad} skipped -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
