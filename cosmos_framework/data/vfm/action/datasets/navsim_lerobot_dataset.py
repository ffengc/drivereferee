# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Modified by Fengcheng Yu, 2026.

"""NavSim GEAR dataset for Cosmos3 action-policy training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Joint, build_action_spec
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id

_VIDEO_KEY = "observation.images.front_wide"
_CAPTION_TEMPLATE = "You are an autonomous vehicle planning system. Driving command: {command}."
_NORMALIZER_PATH = Path(__file__).parent / "stats/navsim_lerobot_stats.json"


def ego_relative_waypoints(action_abs: np.ndarray) -> np.ndarray:
    """(N,4) absolute world poses ``[x, y, cos, sin]`` -> poses relative to frame 0.

    x = forward, y = left, heading kept as [cos, sin] of the yaw delta. Frame 0 maps
    to ``[0, 0, 1, 0]``. Must match the transform used by the converter and the scorer.
    """
    x, y, c, s = action_abs[:, 0], action_abs[:, 1], action_abs[:, 2], action_abs[:, 3]
    x0, y0, c0, s0 = x[0], y[0], c[0], s[0]
    dx, dy = x - x0, y - y0
    xe = c0 * dx + s0 * dy
    ye = -s0 * dx + c0 * dy
    cd = c * c0 + s * s0
    sd = s * c0 - c * s0
    return np.stack([xe, ye, cd, sd], axis=1)


class NavSimLeRobotDataset(ActionBaseDataset):
    """NavSim GEAR dataset (LeRobot v2.0 layout: ``meta/episodes.jsonl`` plus one
    parquet and one mp4 per episode).

    One sample = one scene: ``num_history_frames`` history frames + current frame +
    8 future frames (2 Hz) and the 8 future ego-relative waypoints
    ``[x, y, cos(dyaw), sin(dyaw)]`` in the frame-0 rear-axle frame. ``use_state=True``
    prepends ``[speed, yaw, 0, 0]`` as a conditioning row -> ``(chunk+1, 4)`` action.
    The whole action tensor is quantile-normalized with the waypoint stats.
    """

    def __init__(
        self,
        root: str,
        fps: float = 2.0,
        chunk_length: int = 8,
        mode: str = "policy",
        pose_convention: str = "backward_framewise",
        tolerance_s: float = 0.02,
        viewpoint: str = "ego_view",
        use_state: bool = True,
        action_normalization: str | None = "quantile",
        num_history_frames: int = 0,
    ) -> None:
        # ActionBaseDataset.__init__ assumes the LeRobot v3.0 meta layout; GEAR is v2.0,
        # so replicate the field setup and reuse only the sample-building machinery.
        Dataset.__init__(self)
        self._fps = float(fps)
        self._dt = 1.0 / self._fps
        self._chunk_length = int(chunk_length)
        self._num_history_frames = int(num_history_frames)
        self._sample_stride = 1
        self._mode = mode
        self._pose_convention = pose_convention
        self._tolerance_s = float(tolerance_s)
        self._viewpoint = viewpoint
        self._domain_name = "navsim"
        self._domain_id = get_domain_id("navsim")
        self._action_normalization = action_normalization
        self._norm_stats = None
        self._use_state = bool(use_state)
        # token -> npy path; when set, the future waypoints are read from that file
        # (eval-format (8,3) [x, y, heading]) instead of the parquet. Used by the pair dataset.
        self._winner_paths: dict[str, Path] = {}

        self._root = Path(root)
        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        if float(self._info.get("fps", fps)) != self._fps:
            raise ValueError(f"fps mismatch: dataset meta says {self._info.get('fps')}, got fps={fps}")
        self._chunks_size = int(self._info.get("chunks_size", 1000))

        self._episodes: dict[int, dict[str, Any]] = {}
        with (self._root / "meta" / "episodes.jsonl").open() as f:
            for line in f:
                rec = json.loads(line)
                self._episodes[int(rec["episode_index"])] = rec
        self._tasks: dict[int, str] = {}
        with (self._root / "meta" / "tasks.jsonl").open() as f:
            for line in f:
                rec = json.loads(line)
                self._tasks[int(rec["task_index"])] = str(rec["task"])

    @property
    def action_dim(self) -> int:
        return 4

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Joint(n=4, label="waypoint"))

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    def _compute_idle_frames(self, action: torch.Tensor) -> int:
        # Waypoints are cumulative poses, so the pose-delta idle heuristic does not apply.
        return 0

    def _decode_video(self, video_path: Path) -> torch.Tensor:
        """Decode the whole episode mp4 with PyAV -> (T, C, H, W) float in [0, 1]."""
        frames: list[np.ndarray] = []
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        expected = self._chunk_length + 1 + self._num_history_frames
        if len(frames) != expected:
            raise ValueError(f"{video_path}: expected {expected} frames, got {len(frames)}")
        stacked = np.stack(frames)  # (T, H, W, C) uint8
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0

    def navsim_token(self, idx: int) -> str:
        """NavSim scene token for episode ``idx``."""
        return str(self._episodes[int(idx)]["navsim_token"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = int(idx)
        chunk_idx = idx // self._chunks_size
        parquet_path = self._root / self._info["data_path"].format(episode_chunk=chunk_idx, episode_index=idx)
        table = pq.read_table(parquet_path)

        action_abs = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if action_abs.shape != (self._chunk_length + 1, 4):
            raise ValueError(f"episode {idx}: expected action ({self._chunk_length + 1}, 4), got {action_abs.shape}")

        rel = ego_relative_waypoints(action_abs)
        if not np.allclose(rel[0], [0.0, 0.0, 1.0, 0.0], atol=1e-3):
            raise ValueError(f"episode {idx}: frame-0 ego invariant broken: {rel[0]}")
        traj = rel[1:].astype(np.float32)  # (chunk, 4) future waypoints

        # Optional label replacement by a sampled trajectory (heading -> cos/sin).
        if self._winner_paths:
            wp = self._winner_paths.get(self.navsim_token(idx))
            if wp is not None:
                w = np.load(wp)
                if w.shape != (self._chunk_length, 3):
                    raise ValueError(f"winner {wp}: expected ({self._chunk_length}, 3), got {w.shape}")
                traj = np.stack([w[:, 0], w[:, 1], np.cos(w[:, 2]), np.sin(w[:, 2])], axis=1).astype(np.float32)

        if self._use_state:
            state_row = np.array([[state[0, 0], state[0, 1], 0.0, 0.0]], dtype=np.float32)
            action = np.concatenate([state_row, traj], axis=0)  # (chunk + 1, 4)
        else:
            action = traj

        video_path = self._root / self._info["video_path"].format(
            episode_chunk=chunk_idx, episode_index=idx, video_key=_VIDEO_KEY
        )
        video = self._decode_video(video_path)  # (T, C, H, W) in [0, 1]

        command = self._tasks.get(int(table["task_index"][0].as_py()), "unknown")
        ai_caption = _CAPTION_TEMPLATE.format(command=command)

        return self._build_result(
            mode=self._choose_mode(),
            video=video,
            action=torch.from_numpy(action).float(),
            ai_caption=ai_caption,
        )

    def __len__(self) -> int:
        return len(self._episodes)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """One block per episode (each sample is a whole scene)."""
        return [(i, 1) for i in range(len(self._episodes))]
