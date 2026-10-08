# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Modified by Fengcheng Yu, 2026.

"""Merge a slim NavSim checkpoint (trained tensors only) into a base DCP -> full DCP.

Every ``net.*`` tensor of the base is kept; keys present in the chosen slim tree
(``net_ema.*`` by default) replace the base tensor. CPU-only, single process.

    CUDA_VISIBLE_DEVICES= python -m cosmos_framework.scripts.merge_navsim_slim_dcp \\
        --base-path <base DCP> --slim-path <run>/checkpoints/iter_XXXXXXXXX -o <out DCP>
"""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

import math
import pickle
import shutil
from pathlib import Path
from typing import Annotated, Literal

import pydantic
import torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.inference.common.args import ResolvedPath
from cosmos_framework.utils import log


class Args(pydantic.BaseModel):
    base_path: ResolvedPath
    """Base DCP checkpoint dir (contains model/.metadata)."""
    slim_path: ResolvedPath
    """Slim training checkpoint dir (the iter_XXXXXXXXX directory containing model/)."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output dir for the merged full DCP checkpoint."""
    tree: Literal["net_ema", "net"] = "net_ema"
    """Which slim tree to merge: EMA weights (inference default) or the regular net."""


def _load_dcp_flat(model_dir: Path, keys: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Load a (subset of a) DCP model component into plain CPU tensors."""
    with (model_dir / ".metadata").open("rb") as f:
        metadata = pickle.load(f)
    state_dict: dict[str, torch.Tensor] = {}
    for key, tensor_md in metadata.state_dict_metadata.items():
        if not isinstance(tensor_md, TensorStorageMetadata):
            continue
        if keys is not None and key not in keys:
            continue
        state_dict[key] = torch.empty(tensor_md.size, dtype=tensor_md.properties.dtype)
    if keys is not None:
        missing = sorted(set(keys) - set(state_dict))
        if missing:
            raise KeyError(f"{len(missing)} requested keys absent from {model_dir}: {missing[:5]} ...")
    dcp.load(state_dict=state_dict, storage_reader=FileSystemReader(model_dir))
    return state_dict


def _list_dcp_keys(model_dir: Path) -> list[str]:
    with (model_dir / ".metadata").open("rb") as f:
        metadata = pickle.load(f)
    return [k for k, v in metadata.state_dict_metadata.items() if isinstance(v, TensorStorageMetadata)]


def merge_navsim_slim_dcp(args: Args) -> None:
    base_model_dir = args.base_path / "model"
    slim_model_dir = args.slim_path / "model"
    for d in (base_model_dir, slim_model_dir):
        if not (d / ".metadata").exists():
            raise FileNotFoundError(f"Not a DCP model dir (no .metadata): {d}")
    if args.output_path.exists() and any(args.output_path.iterdir()):
        raise FileExistsError(f"Output dir exists and is non-empty: {args.output_path}")

    prefix = f"{args.tree}."
    slim_keys_all = _list_dcp_keys(slim_model_dir)
    tree_keys = [k for k in slim_keys_all if k.startswith(prefix)]
    if not tree_keys:
        raise KeyError(
            f"Slim checkpoint has no '{prefix}*' keys (trees present: {sorted({k.split('.', 1)[0] for k in slim_keys_all})})"
        )
    log.info(f"Slim checkpoint: {len(slim_keys_all)} keys total, {len(tree_keys)} in tree '{args.tree}'")

    log.info(f"Loading slim tree from {slim_model_dir} ...")
    slim = {k.removeprefix(prefix): v for k, v in _load_dcp_flat(slim_model_dir, keys=tree_keys).items()}

    log.info(f"Loading full base from {base_model_dir} (CPU) ...")
    merged = _load_dcp_flat(base_model_dir)

    n_replaced = 0
    for key, value in slim.items():
        base_key = f"net.{key}"
        if base_key not in merged:
            raise KeyError(f"Slim key '{base_key}' not found in the base checkpoint")
        if tuple(value.shape) != tuple(merged[base_key].shape):
            raise ValueError(f"{base_key}: slim shape {tuple(value.shape)} != base shape {tuple(merged[base_key].shape)}")
        merged[base_key] = value.to(merged[base_key].dtype)
        n_replaced += 1
    log.info(f"Merged: replaced {n_replaced} tensors; output keys = {len(merged)}")

    log.info(f"Saving merged checkpoint to {args.output_path} ...")
    out_model_dir = args.output_path / "model"
    model_size = sum(t.numel() * t.element_size() for t in merged.values())
    thread_count = math.ceil(model_size / (5 * 1024**3))
    dcp.save(
        state_dict=merged,
        storage_writer=FileSystemWriter(out_model_dir, thread_count=thread_count),
        planner=CustomSavePlanner(),
    )
    shutil.copy(args.base_path / "checkpoint.json", args.output_path / "checkpoint.json")
    shutil.copy(base_model_dir / "config.json", out_model_dir / "config.json")
    log.info(f"Done. Merged full checkpoint ({model_size / 1024**3:.1f} GiB) at {args.output_path}")


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    merge_navsim_slim_dcp(args)


if __name__ == "__main__":
    main()
