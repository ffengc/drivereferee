"""Convert NavSim (OpenScene) scenes into the GEAR / LeRobot v2.0 format.

One navsim token -> one episode: H history frames + current + 8 future frames of the front
camera (mp4 @ 2 Hz), a (9, 2) [speed, yaw] state and a (9, 4) absolute world pose
[x, y, cos(yaw), sin(yaw)] for current + future. Poses are made ego-relative at load time.

Run inside the NavSim devkit environment:
    OPENSCENE_DATA_ROOT=<dataset> NUPLAN_MAPS_ROOT=<dataset>/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \\
    PYTHONPATH=<navsim_devkit_root> python convert_navsim_to_gear.py --split trainval \\
        --scene-filter-yaml <devkit>/.../scene_filter/navtrain.yaml --make-video --history-frames 4 --out <gear>/navtrain
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

EMBODIMENT_TAG = "navsim"
TARGET_FPS = 2
CHUNK_SIZE = 1000
CAMERA_KEY = "front_wide"
OUT_W, OUT_H = 832, 480
CUR_IDX = 3  # index of the current frame in a devkit scene
N_FUTURE = 8
EP_LEN = 1 + N_FUTURE
COMMAND_TEXT = ["turn left", "keep straight", "turn right", "unknown"]


def ego_relative_waypoints(action_abs: np.ndarray) -> np.ndarray:
    """(N,4) absolute world [x,y,cos,sin] -> ego-relative to frame 0 (used for stats only)."""
    x, y, c, s = action_abs[:, 0], action_abs[:, 1], action_abs[:, 2], action_abs[:, 3]
    x0, y0, c0, s0 = x[0], y[0], c[0], s[0]
    dx, dy = x - x0, y - y0
    xe = c0 * dx + s0 * dy
    ye = -s0 * dx + c0 * dy
    cd = c * c0 + s * s0
    sd = s * c0 - c * s0
    return np.stack([xe, ye, cd, sd], axis=1).astype(np.float32)


def command_to_text(cmd: np.ndarray) -> str:
    cmd = np.asarray(cmd).reshape(-1)
    if cmd.sum() == 0:
        return COMMAND_TEXT[-1]
    return COMMAND_TEXT[int(np.argmax(cmd))]


def extract_token(scene, history_frames: int = 0) -> dict | None:
    """State/action/command for one scene; the video window is H history + current + 8 future."""
    frames = scene.frames
    cur_idx = max(CUR_IDX, history_frames)
    if len(frames) < cur_idx + N_FUTURE + 1:
        return None
    sel = frames[cur_idx : cur_idx + EP_LEN]

    poses = np.array([fr.ego_status.ego_pose for fr in sel], dtype=np.float64)  # (9,3)
    yaw = poses[:, 2]
    action = np.stack([poses[:, 0], poses[:, 1], np.cos(yaw), np.sin(yaw)], axis=1).astype(np.float32)
    vel = np.array([fr.ego_status.ego_velocity for fr in sel], dtype=np.float64)
    speed = np.linalg.norm(vel, axis=1)
    state = np.stack([speed, yaw], axis=1).astype(np.float32)
    cmd_text = command_to_text(sel[0].ego_status.driving_command)

    if history_frames:
        hist = list(frames[max(cur_idx - history_frames, 0) : cur_idx])
        while len(hist) < history_frames:  # log start: repeat the earliest frame
            hist.insert(0, hist[0] if hist else frames[cur_idx])
        vid_frames = hist + list(sel)
    else:
        vid_frames = sel
    return {"state": state, "action": action, "command": cmd_text, "video_frames": vid_frames}


def encode_front_video(sel_frames, out_path: Path) -> bool:
    """Front-camera frames -> h264 mp4 at OUT_W x OUT_H. False if any image is missing."""
    import cv2
    import imageio

    imgs = []
    for fr in sel_frames:
        im = fr.cameras.cam_f0.image
        if im is None:
            return False
        imgs.append(cv2.resize(im, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), imgs, fps=TARGET_FPS, codec="libx264")
    return True


def build_episode_parquet(state, action, ep_idx, task_idx, fps) -> pd.DataFrame:
    n = len(state)
    next_done = np.zeros(n, dtype=bool)
    next_done[-1] = True
    is_first = np.zeros(n, dtype=bool)
    is_first[0] = True
    return pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(action),
            "timestamp": (np.arange(n) / fps).astype(np.float64),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep_idx, dtype=np.int64),
            "task_index": np.full(n, task_idx, dtype=np.int64),
            "annotation.language.cot": np.full(n, task_idx, dtype=np.int64),
            "next.reward": np.zeros(n, dtype=np.float64),
            "next.done": next_done,
            "is_terminal": next_done.copy(),
            "is_first": is_first,
            "discount": np.ones(n, dtype=np.float64),
        }
    )


def _stats_dict(data: np.ndarray) -> dict:
    return {
        "mean": data.mean(0).tolist(),
        "std": data.std(0).tolist(),
        "min": data.min(0).tolist(),
        "max": data.max(0).tolist(),
        "q01": np.quantile(data, 0.01, 0).tolist(),
        "q99": np.quantile(data, 0.99, 0).tolist(),
    }


def write_meta(out: Path, ep_records, task_records, state_all, action_all, has_video) -> None:
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    n_ep = len(ep_records)
    total_frames = sum(r["length"] for r in ep_records)
    info = {
        "codebase_version": "v2.0",
        "robot_type": EMBODIMENT_TAG,
        "total_episodes": n_ep,
        "total_frames": total_frames,
        "total_tasks": len(task_records),
        "total_videos": n_ep if has_video else 0,
        "total_chunks": (n_ep + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": TARGET_FPS,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **(
                {
                    f"observation.images.{CAMERA_KEY}": {
                        "dtype": "video",
                        "shape": [OUT_H, OUT_W, 3],
                        "names": ["height", "width", "channel"],
                        "video_info": {
                            "video.fps": TARGET_FPS,
                            "video.codec": "h264",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "has_audio": False,
                        },
                    }
                }
                if has_video
                else {}
            ),
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["speed", "yaw"]},
            "action": {"dtype": "float32", "shape": [4], "names": ["x", "y", "cos_theta", "sin_theta"]},
            "timestamp": {"dtype": "float64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "next.reward": {"dtype": "float64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "is_terminal": {"dtype": "bool", "shape": [1]},
            "is_first": {"dtype": "bool", "shape": [1]},
            "discount": {"dtype": "float64", "shape": [1]},
            "annotation.language.cot": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    def _field(key, start, end):
        return {"original_key": key, "start": start, "end": end, "rotation_type": None,
                "absolute": True, "dtype": "float32", "range": None}

    modality = {
        "state": {"speed": _field("observation.state", 0, 1), "yaw": _field("observation.state", 1, 2)},
        "action": {
            "x": _field("action", 0, 1),
            "y": _field("action", 1, 2),
            "cos_theta": _field("action", 2, 3),
            "sin_theta": _field("action", 3, 4),
        },
        "video": {CAMERA_KEY: {"original_key": f"observation.images.{CAMERA_KEY}"}},
        "annotation": {"language.cot": {"original_key": "annotation.language.cot"}},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))
    (meta / "embodiment.json").write_text(
        json.dumps({"robot_type": EMBODIMENT_TAG, "embodiment_tag": EMBODIMENT_TAG}, indent=4)
    )
    with open(meta / "tasks.jsonl", "w") as f:
        for t in task_records:
            f.write(json.dumps(t) + "\n")
    with open(meta / "episodes.jsonl", "w") as f:
        for r in ep_records:
            f.write(json.dumps(r) + "\n")
    stats = {
        "observation.state": _stats_dict(state_all.astype(np.float64)),
        "action": _stats_dict(action_all.astype(np.float64)),
    }
    (meta / "stats.json").write_text(json.dumps(stats, indent=4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="trainval", help="navsim data split dir (trainval / test / mini)")
    ap.add_argument("--out", required=True, help="GEAR output dir")
    ap.add_argument("--make-video", action="store_true", help="encode cam_f0 mp4 (needs camera blobs)")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N tokens")
    ap.add_argument("--history-frames", type=int, default=0, help="history frames prepended to the video")
    ap.add_argument("--scene-filter-yaml", default=None, help="official navtrain/navtest scene_filter yaml")
    args = ap.parse_args()

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    out = Path(args.out)

    sensor_cfg = (
        SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False,
                     cam_r2=False, cam_b0=False, lidar_pc=False)
        if args.make_video
        else SensorConfig.build_no_sensors()
    )
    if args.scene_filter_yaml:
        import yaml

        fcfg = yaml.safe_load(open(args.scene_filter_yaml))
        scene_filter = SceneFilter(
            num_history_frames=max(fcfg["num_history_frames"], args.history_frames + 1),
            num_future_frames=fcfg["num_future_frames"],
            frame_interval=fcfg.get("frame_interval", 1),
            has_route=fcfg.get("has_route", False),
            max_scenes=fcfg.get("max_scenes"),
            log_names=fcfg.get("log_names"),
            tokens=fcfg.get("tokens"),
        )
    else:
        scene_filter = SceneFilter(num_history_frames=max(4, args.history_frames + 1), num_future_frames=10,
                                   has_route=False)
    loader = SceneLoader(
        data_path=data_root / "navsim_logs" / args.split,
        original_sensor_path=data_root / "sensor_blobs" / args.split,
        scene_filter=scene_filter,
        sensor_config=sensor_cfg,
    )
    tokens = loader.tokens
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"converting {len(tokens)} tokens (split={args.split}, make_video={args.make_video})")

    task_set: dict[str, int] = {}
    ep_records, state_arrays, action_arrays = [], [], []
    errors = 0
    ep_idx = 0
    for tok in tokens:
        try:
            scene = loader.get_scene_from_token(tok)
            rec = extract_token(scene, history_frames=args.history_frames)
            if rec is None:
                continue
            chunk_idx = ep_idx // CHUNK_SIZE
            if args.make_video:
                vpath = out / f"videos/chunk-{chunk_idx:03d}/observation.images.{CAMERA_KEY}/episode_{ep_idx:06d}.mp4"
                if not encode_front_video(rec["video_frames"], vpath):
                    continue
            cmd = rec["command"]
            if cmd not in task_set:
                task_set[cmd] = len(task_set)
            task_idx = task_set[cmd]
            df = build_episode_parquet(rec["state"], rec["action"], ep_idx, task_idx, TARGET_FPS)
            ppath = out / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
            ppath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ppath)
            ep_records.append({"episode_index": ep_idx, "tasks": [cmd], "length": len(rec["state"]),
                               "navsim_token": tok})
            state_arrays.append(rec["state"])
            action_arrays.append(ego_relative_waypoints(rec["action"]))
            ep_idx += 1
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ! {tok}: {type(e).__name__}: {e}")

    if not ep_records:
        print("no episodes written!")
        return
    task_records = [{"task_index": i, "task": t} for t, i in sorted(task_set.items(), key=lambda kv: kv[1])]
    write_meta(out, ep_records, task_records, np.concatenate(state_arrays, 0), np.concatenate(action_arrays, 0),
               args.make_video)
    print(f"=== DONE === {len(ep_records)} episodes -> {out} (errors={errors})")


if __name__ == "__main__":
    main()
