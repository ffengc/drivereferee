#!/usr/bin/env bash
# Base policy: 48k steps from Cosmos3-Nano, slim-resume to 64k, warm-start to 80k (8 GPUs).
# Usage: bash scripts/02_train_base.sh {48k|64k|merge64k|80k|merge80k}
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
export NAVSIM_GEAR_ROOT=$GEAR_ROOT/navtrain
export TOML_FILE=examples/toml/sft_config/action_policy_navsim_h4.toml
FULLFT="model.config.parallelism.data_parallel_shard_degree=$NPROC_PER_NODE"

case "$1" in
48k)
    export BASE_CHECKPOINT_PATH=$BASE_DCP LOG_FILENAME=navsim_h4_base.log
    export EXTRA_TAIL_OVERRIDES="job.name=navsim_h4_base trainer.max_iter=48000 scheduler.cycle_lengths=[48000] checkpoint.save_iter=8000 $FULLFT"
    bash examples/launch_sft_action_policy_navsim.sh
    ;;
64k)
    # Continue the 48k schedule to 64k with the same LambdaLinear shape (LR declared over 64k).
    export BASE_CHECKPOINT_PATH=$BASE_DCP LOG_FILENAME=navsim_h4_ext64k.log
    export RESUME_SLIM_DIR=$CKPT_DIR/navsim_h4_base/checkpoints/iter_000048000
    export EXTRA_TAIL_OVERRIDES="job.name=navsim_h4_ext64k trainer.max_iter=64000 scheduler.cycle_lengths=[64000] checkpoint.save_iter=8000 model.config.activation_checkpointing.mode=none $FULLFT"
    bash examples/launch_sft_action_policy_navsim.sh
    ;;
merge64k)
    CUDA_VISIBLE_DEVICES= python -m cosmos_framework.scripts.merge_navsim_slim_dcp \
        --base-path "$BASE_DCP" --slim-path "$CKPT_DIR/navsim_h4_ext64k/checkpoints/iter_000064000" \
        -o "$WORK/ckpt/h4_64k_merged" --tree net_ema
    ;;
80k)
    # Warm start from the merged 64k weights; this schedule equals the 64k-80k segment of an 80k schedule.
    export BASE_CHECKPOINT_PATH=$WORK/ckpt/h4_64k_merged LOG_FILENAME=navsim_h4_ext80k.log
    export EXTRA_TAIL_OVERRIDES="job.name=navsim_h4_ext80k trainer.max_iter=16000 scheduler.cycle_lengths=[16000] scheduler.f_max=[0.08] scheduler.warm_up_steps=[1000] checkpoint.save_iter=8000 model.config.activation_checkpointing.mode=selective $FULLFT"
    bash examples/launch_sft_action_policy_navsim.sh
    ;;
merge80k)
    CUDA_VISIBLE_DEVICES= python -m cosmos_framework.scripts.merge_navsim_slim_dcp \
        --base-path "$WORK/ckpt/h4_64k_merged" --slim-path "$CKPT_DIR/navsim_h4_ext80k/checkpoints/iter_000016000" \
        -o "$WORK/ckpt/h4_80k_merged" --tree net_ema
    ;;
*) echo "usage: $0 {48k|64k|merge64k|80k|merge80k}"; exit 1 ;;
esac
