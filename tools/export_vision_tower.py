"""Export the frozen Qwen3-VL vision tower from a Cosmos3-Nano DCP and load it standalone.

    python tools/export_vision_tower.py --dcp <base DCP>/model --config <Cosmos3-Nano config.json> --out <tower dir>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

PREFIX = "net.language_model.visual."


def _has_vc(d: dict) -> bool:
    return "vision_config" in d or any(_has_vc(v) for v in d.values() if isinstance(v, dict))


def export(dcp: str, config: str, out_dir: str) -> None:
    from safetensors.torch import save_file
    from torch.distributed.checkpoint import FileSystemReader, load

    reader = FileSystemReader(dcp)
    md = reader.read_metadata()
    keys = [k for k in md.state_dict_metadata if k.startswith(PREFIX)]
    assert keys, f"no {PREFIX}* keys in {dcp}"
    sd = {k: torch.empty(tuple(md.state_dict_metadata[k].size), dtype=md.state_dict_metadata[k].properties.dtype)
          for k in keys}
    load(sd, storage_reader=reader)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stripped = {k[len(PREFIX):]: v.contiguous() for k, v in sd.items()}
    save_file(stripped, str(out / "vision_tower.safetensors"))

    vc = json.load(open(sorted(glob.glob(config))[0]))
    while isinstance(vc, dict) and "vision_config" not in vc:
        vc = next((v for v in vc.values() if isinstance(v, dict) and _has_vc(v)), None)
        assert vc is not None, "vision_config not found in config"
    (out / "vision_config.json").write_text(json.dumps(vc["vision_config"], indent=1))
    n_par = sum(v.numel() for v in stripped.values())
    print(f"exported {n_par / 1e6:.0f} M params -> {out}")


def load_frozen_vision_tower(tower_dir: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load the exported tower as an eval, no-grad module."""
    from safetensors.torch import load_file
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    d = Path(tower_dir)
    cfg = Qwen3VLVisionConfig(**json.load(open(d / "vision_config.json")))
    model = Qwen3VLVisionModel._from_config(cfg)
    missing, unexpected = model.load_state_dict(load_file(str(d / "vision_tower.safetensors")), strict=False)
    assert not missing, f"missing weights: {missing[:5]}"
    assert not unexpected, f"unexpected weights: {unexpected[:5]}"
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", required=True, help="DCP model dir (contains .metadata)")
    ap.add_argument("--config", required=True, help="Cosmos3-Nano config.json (glob allowed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export(args.dcp, args.config, args.out)


if __name__ == "__main__":
    main()
