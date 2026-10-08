"""Policy inference: merged DCP checkpoint -> one (8,3) [x, y, heading] npy per scene.

Observation = the first H+1 frames of the episode mp4 (history + current); future frames are
zeros. The state row [speed, yaw, 0, 0] is normalized like in training. Sampling uses the
framework policy defaults (30 steps, guidance 1.0, shift 10).

    CUDA_VISIBLE_DEVICES=0 python tools/infer_navsim_dump.py --checkpoint-path <merged DCP> \\
        --gear-root <gear>/navtest --out-dir <out> --split test --seed 0 --history-frames 4
"""

import argparse
import json
import sys
from pathlib import Path

import av as pyav
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cosmos_framework.inference.common.init import init_script  # noqa: E402

init_script(env={})

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG  # noqa: E402
from cosmos_framework.data.vfm.action.action_normalization import denormalize_action, normalize_action  # noqa: E402
from cosmos_framework.data.vfm.action.datasets.navsim_lerobot_dataset import NavSimLeRobotDataset  # noqa: E402
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id  # noqa: E402
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline  # noqa: E402
from cosmos_framework.inference.args import OmniSetupOverrides  # noqa: E402
from cosmos_framework.inference.inference import OmniInference  # noqa: E402
from cosmos_framework.scripts.action_policy_server_robolab import _build_data_batch_from_sample  # noqa: E402
from cosmos_framework.utils import log  # noqa: E402

_CHUNK = 8
_CAPTION = "You are an autonomous vehicle planning system. Driving command: {command}."


def _first_frames(mp4: Path, n: int) -> list[torch.Tensor]:
    """First n frames of an mp4 -> list of (3,H,W) uint8 tensors."""
    out: list[torch.Tensor] = []
    with pyav.open(str(mp4)) as container:
        for frame in container.decode(video=0):
            out.append(torch.from_numpy(frame.to_ndarray(format="rgb24").copy()).permute(2, 0, 1))
            if len(out) >= n:
                return out
    raise ValueError(f"{mp4}: wanted {n} frames, got {len(out)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-path", required=True, help="merged full DCP dir")
    p.add_argument("--gear-root", required=True, help="GEAR split dir")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="test", help="npy files go to <out-dir>/<split>/")
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="episode indices (default: all)")
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--shift", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", default="480")
    p.add_argument("--history-frames", type=int, default=4)
    args = p.parse_args()

    gear = Path(args.gear_root)
    dump_dir = Path(args.out_dir) / args.split
    dump_dir.mkdir(parents=True, exist_ok=True)

    ds = NavSimLeRobotDataset(root=str(gear), num_history_frames=args.history_frames)
    stats = ds.load_action_stats()
    episodes = args.episodes if args.episodes else list(range(len(ds)))
    info = json.loads((gear / "meta" / "info.json").read_text())

    ckpt_dir = Path(args.checkpoint_path)
    config_file = ckpt_dir / "model" / "config.json"
    if not config_file.is_file():
        config_file = ckpt_dir / "config.json"
    overrides = OmniSetupOverrides.model_validate(
        {
            "checkpoint_path": args.checkpoint_path,
            "config_file": str(config_file),
            "output_dir": str(Path(args.out_dir) / "_inference_workdir"),
            "guardrails": False,
            "parallelism_preset": "latency",
            "use_ema_weights": False,  # merged checkpoints carry the EMA weights in net.*
        }
    )
    pipe = OmniInference.create(overrides.build_setup())
    model = pipe.model
    model.eval()

    transform = ActionTransformPipeline(
        tokenizer_config=NANO_MODEL_CONFIG["vlm_config"]["tokenizer"],
        cfg_dropout_rate=0.0,
        max_action_dim=int(getattr(model.config, "max_action_dim", 64)),
        num_history_video_frames=int(args.history_frames),
    )

    import pyarrow.parquet as pq

    results = {}
    H = int(args.history_frames)
    for idx in episodes:
        token = ds.navsim_token(idx)
        chunk_idx = idx // int(info.get("chunks_size", 1000))
        table = pq.read_table(gear / info["data_path"].format(episode_chunk=chunk_idx, episode_index=idx))
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)  # (9,2) [speed,yaw]
        command = ds._tasks.get(int(table["task_index"][0].as_py()), "unknown")
        mp4 = gear / info["video_path"].format(episode_chunk=chunk_idx, episode_index=idx,
                                               video_key="observation.images.front_wide")

        obs_frames = _first_frames(mp4, H + 1)
        obs = obs_frames[-1]
        video = torch.zeros((3, H + _CHUNK + 1, obs.shape[1], obs.shape[2]), dtype=torch.uint8)
        for k, fr in enumerate(obs_frames):
            video[:, k] = fr

        action = torch.zeros((_CHUNK + 1, 4), dtype=torch.float32)
        state_row = torch.tensor([state[0, 0], state[0, 1], 0.0, 0.0], dtype=torch.float32)
        action[0] = normalize_action(state_row, ds.action_normalization, stats)

        sample = {
            "ai_caption": _CAPTION.format(command=command),
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(2, dtype=torch.long),
            "mode": "policy",
            "domain_id": torch.tensor(get_domain_id("navsim"), dtype=torch.long),
            "viewpoint": "ego_view",
        }
        sample = transform(sample, args.resolution)
        data_batch = _build_data_batch_from_sample(sample)

        with torch.inference_mode():
            samples = model.generate_samples_from_batch(
                data_batch,
                guidance=args.guidance,
                seed=[args.seed],
                num_steps=args.num_steps,
                shift=args.shift,
            )

        pred = samples["action"][0][:, :4].detach().float().cpu()[1:]  # drop the state row -> (8,4)
        pred = denormalize_action(pred, ds.action_normalization, stats)
        heading = torch.atan2(pred[:, 3], pred[:, 2])
        traj = torch.stack([pred[:, 0], pred[:, 1], heading], dim=1).numpy().astype(np.float32)
        assert traj.shape == (8, 3)
        np.save(dump_dir / f"{token}.npy", traj)
        results[token] = traj
        log.info(f"[{idx}] {token}: x_fwd {traj[:, 0].min():.2f}..{traj[:, 0].max():.2f} m, "
                 f"|y|max {abs(traj[:, 1]).max():.2f} m")

    (Path(args.out_dir) / f"tokens_{args.split}.txt").write_text("\n".join(results) + "\n")
    log.info(f"dumped {len(results)} tokens -> {dump_dir}")


if __name__ == "__main__":
    main()
