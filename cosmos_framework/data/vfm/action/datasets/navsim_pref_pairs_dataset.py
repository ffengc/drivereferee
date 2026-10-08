# SPDX-License-Identifier: Apache-2.0
# Written by Fengcheng Yu, 2026.

"""Preference-pair dataset for referee distillation.

Each item is a normal NavSim sample whose action rows are the winner trajectory,
plus a ``loser_action`` tensor holding the loser trajectory with identical
processing (ego-relative -> [x, y, cos, sin] -> state row -> normalization).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cosmos_framework.data.vfm.action.datasets.navsim_lerobot_dataset import NavSimLeRobotDataset


class NavSimPrefPairsDataset(NavSimLeRobotDataset):
    def __init__(
        self,
        root: str,
        pairs_manifest: str,
        selfplay_root: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(root=root, **kwargs)
        self._selfplay_root = Path(selfplay_root)
        self._pairs = [json.loads(line) for line in open(pairs_manifest)]
        if not self._pairs:
            raise ValueError(f"empty pairs manifest: {pairs_manifest}")
        self._tok2ep: dict[str, int] = {}
        for line in open(Path(root) / "meta" / "episodes.jsonl"):
            r = json.loads(line)
            self._tok2ep[r["navsim_token"]] = int(r["episode_index"])
        missing = [p["token"] for p in self._pairs if p["token"] not in self._tok2ep]
        if missing:
            raise ValueError(f"{len(missing)} pair tokens missing from episodes.jsonl: {missing[:3]}")

    def _cand_path(self, token: str, seed: int) -> Path:
        return self._selfplay_root / f"seed{seed}" / "test" / f"{token}.npy"

    def __len__(self) -> int:
        return len(self._pairs)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return [(i, 1) for i in range(len(self._pairs))]

    def __getitem__(self, i: int) -> dict[str, Any]:
        p = self._pairs[int(i) % len(self._pairs)]
        ep = self._tok2ep[p["token"]]
        # Winner and loser go through the same parent pipeline.
        self._winner_paths = {p["token"]: self._cand_path(p["token"], int(p["win_seed"]))}
        sample = super().__getitem__(ep)
        self._winner_paths = {p["token"]: self._cand_path(p["token"], int(p["lose_seed"]))}
        loser_sample = super().__getitem__(ep)
        self._winner_paths = {}
        sample["loser_action"] = loser_sample["action"]  # (chunk+1, 4), same normalization
        return sample
