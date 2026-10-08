# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Modified by Fengcheng Yu, 2026.

"""NavSim action-policy recipes on Cosmos3-Nano (full fine-tuning of the generation experts).

``action_policy_navsim_h4_nano``      base policy: 4 history + current + 8 future frames.
``action_policy_navsim_pref_h4_nano`` referee distillation on winner/loser pairs.
"""

import copy
import os
from typing import Any

import torch
from hydra.core.config_store import ConfigStore

from cosmos_framework.checkpoint.dcp import DistributedCheckpointer
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
    get_action_navsim_pref_pairs_dataset,
    get_action_navsim_sft_dataset,
)
from cosmos_framework.data.vfm.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

# Trained parameter set (same as the DROID full-FT recipe) plus the fresh NavSim action heads.
TRAINED_KEYS = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]


class NavSimSlimCheckpointCallback(Callback):
    """Save only the trained tensors (``optimizer.keys_to_select``) instead of the full model.

    Slim checkpoints are turned back into full ones with ``scripts/merge_navsim_slim_dcp.py``.
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self._enabled = enabled

    @staticmethod
    def _nbytes(value: Any) -> int:
        return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0

    def on_save_checkpoint(self, model: Any, state_dict: dict[str, Any]) -> None:
        if not self._enabled:
            return
        keys_to_select = list(self.config.optimizer.keys_to_select or [])
        if not keys_to_select:
            log.warning("NavSimSlimCheckpointCallback: optimizer.keys_to_select is empty; saving full model.")
            return
        model_sd = state_dict.get("model")
        if not model_sd:
            return
        kept = {k: v for k, v in model_sd.items() if any(sel in k for sel in keys_to_select)}
        if not kept:
            raise RuntimeError(f"NavSimSlimCheckpointCallback: keys_to_select={keys_to_select} matched 0 model keys")
        before = sum(self._nbytes(v) for v in model_sd.values())
        after = sum(self._nbytes(v) for v in kept.values())
        state_dict["model"] = kept
        log.info(
            f"NavSimSlimCheckpointCallback: kept {len(kept)}/{len(model_sd)} model keys, "
            f"{before / 1024**3:.1f} GiB -> {after / 1024**3:.2f} GiB"
        )


class NavSimSlimResumeCheckpointer(DistributedCheckpointer):
    """Resume from a slim checkpoint when env ``RESUME_SLIM_DIR`` points at an ``iter_*`` dir.

    Three stages: base warm-start (stock), overlay of the slim model tensors, then
    optimizer / scheduler / trainer state from the slim directory.
    """

    def load(self, model, optimizer=None, scheduler=None, grad_scaler=None) -> int:
        slim_dir = os.environ.get("RESUME_SLIM_DIR", "").strip()
        if not slim_dir:
            return super().load(model, optimizer, scheduler, grad_scaler)

        assert self._read_latest_checkpoint_file() is None, (
            "Job dir already has a resumable checkpoint; unset RESUME_SLIM_DIR or use a fresh job.name."
        )
        assert os.path.isdir(os.path.join(slim_dir, "model")), f"not a checkpoint dir: {slim_dir}"

        import torch.distributed as dist
        import torch.distributed.checkpoint as dcp_mod

        from cosmos_framework.checkpoint.dcp import CustomLoadPlanner, ModelWrapper
        from cosmos_framework.utils.vfm.rand_state import get_rand_state_dict, set_rand_state_dict

        # Stage 1: frozen backbone from the base checkpoint.
        super().load(model, optimizer, scheduler, grad_scaler)
        log.critical(f"[SlimResume] stage 1 done (base warm-start); overlaying {slim_dir}")

        # Stage 2: trained tensors (net.* and net_ema.*).
        dist.barrier()
        reader = self.get_storage_reader(os.path.join(slim_dir, "model"))
        slim_keys = set(reader.read_metadata().state_dict_metadata.keys())
        wrapper = ModelWrapper(model)
        overlay = {k: v for k, v in wrapper.state_dict().items() if k in slim_keys}
        missing = slim_keys - set(overlay.keys())
        assert not missing, f"slim checkpoint keys absent from model: {sorted(missing)[:5]} ..."
        dcp_mod.load(overlay, storage_reader=reader, planner=CustomLoadPlanner())
        wrapper.load_state_dict(overlay)
        log.critical(f"[SlimResume] stage 2 done: overlaid {len(overlay)} trained tensors")

        # Stage 3: optimizer / scheduler / trainer state.
        dist.barrier()
        _sd = optimizer.state_dict()
        dcp_mod.load(
            _sd, storage_reader=self.get_storage_reader(os.path.join(slim_dir, "optim")), planner=CustomLoadPlanner()
        )
        optimizer.load_state_dict(_sd)

        _sd = scheduler.state_dict()
        dcp_mod.load(
            _sd,
            storage_reader=self.get_storage_reader(os.path.join(slim_dir, "scheduler")),
            planner=CustomLoadPlanner(),
        )
        scheduler.load_state_dict(_sd)

        trainer_reader = self.get_storage_reader(os.path.join(slim_dir, "trainer"))
        rng_key = f"rng_state_{dist.get_rank()}"
        current_rng = get_rand_state_dict()
        _sd = {"grad_scaler": grad_scaler.state_dict(), "iteration": 0}
        trainer_md = trainer_reader.read_metadata().state_dict_metadata
        if any(k == rng_key or k.startswith(f"{rng_key}.") for k in trainer_md.keys()):
            _sd[rng_key] = current_rng
        dcp_mod.load(_sd, storage_reader=trainer_reader, planner=CustomLoadPlanner())
        grad_scaler.load_state_dict(_sd["grad_scaler"])
        set_rand_state_dict(_sd.get(rng_key, current_rng))
        iteration = int(_sd["iteration"])
        dist.barrier()
        log.critical(f"[SlimResume] stage 3 done: resuming at iteration {iteration}")
        return iteration


action_policy_navsim_h4_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_navsim_h4_nano",
            wandb_mode="disabled",
        ),
        model=L(OmniMoTModel)(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True, max_action_dim=64
            _recursive_=False,
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=list(TRAINED_KEYS),
            lr=2.0e-04,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[48000],
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=50,
            max_iter=48000,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                navsim_slim_checkpoint=L(NavSimSlimCheckpointCallback)(enabled=True),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            type=L(NavSimSlimResumeCheckpointer)(),
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # EMA warm-starts from net; action heads init fresh for the NavSim domain.
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=8000,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_navsim",
            max_samples_per_batch=4,  # per rank
            max_sequence_length=None,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    navsim=dict(
                        ratio=1,
                        dataset=L(get_action_navsim_sft_dataset)(
                            root="${oc.env:NAVSIM_GEAR_ROOT}",
                            fps=2.0,
                            chunk_length=8,  # 8 future waypoints (+ state row)
                            mode="policy",
                            use_state=True,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            action_normalization="quantile",
                            viewpoint="ego_view",
                            resolution="480",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            num_history_frames=4,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

_model_cfg = action_policy_navsim_h4_nano["model"]["config"]
_model_cfg["tokenizer"]["encode_exact_durations"] = [13]  # 4 history + current + 8 future frames
_model_cfg["max_num_tokens_after_packing"] = -1
_model_cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
_model_cfg["lora_enabled"] = False


# Referee distillation: pairs-only stream, every sample carries a loser trajectory.
action_policy_navsim_pref_h4_nano = copy.deepcopy(action_policy_navsim_h4_nano)
action_policy_navsim_pref_h4_nano["job"]["name"] = "action_policy_navsim_pref_h4_nano"
action_policy_navsim_pref_h4_nano["model"]["config"]["preference_beta"] = 10.0
action_policy_navsim_pref_h4_nano["optimizer"]["lr"] = 2.0e-05
action_policy_navsim_pref_h4_nano["dataloader_train"]["max_samples_per_batch"] = 2
action_policy_navsim_pref_h4_nano["dataloader_train"]["dataloader"]["datasets"] = dict(
    pairs=dict(
        ratio=1,
        dataset=L(get_action_navsim_pref_pairs_dataset)(
            root="${oc.env:NAVSIM_GEAR_ROOT}",
            pairs_manifest="${oc.env:PAIRS_MANIFEST}",
            selfplay_root="${oc.env:SELFPLAY_ROOT}",
            fps=2.0,
            chunk_length=8,
            mode="policy",
            use_state=True,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            action_normalization="quantile",
            viewpoint="ego_view",
            resolution="480",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            num_history_frames=4,
        ),
    ),
)
# Start from a merged base checkpoint: load the trained action heads, only skip EMA.
action_policy_navsim_pref_h4_nano["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]


for _item in [action_policy_navsim_h4_nano, action_policy_navsim_pref_h4_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
