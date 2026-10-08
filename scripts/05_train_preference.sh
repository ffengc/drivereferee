#!/usr/bin/env bash
# Referee distillation on the winner/loser pairs (8 GPUs, ~30 exposures per pair), then merge.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
NP=$(wc -l < "$WORK/pairs/pairs.jsonl")
STEPS=$(( 30 * NP / (2 * NPROC_PER_NODE) / 10 * 10 ))   # 2 pairs per rank per step
echo "$NP pairs -> $STEPS steps"

export NAVSIM_GEAR_ROOT=$GEAR_ROOT/navtrain
export BASE_CHECKPOINT_PATH=$WORK/ckpt/h4_80k_merged
export PAIRS_MANIFEST=$WORK/pairs/pairs.jsonl SELFPLAY_ROOT=$WORK/selfplay
export TOML_FILE=examples/toml/sft_config/action_policy_navsim_pref_h4.toml LOG_FILENAME=navsim_h4_pref.log
export EXTRA_TAIL_OVERRIDES="job.name=navsim_h4_pref trainer.max_iter=$STEPS checkpoint.save_iter=$STEPS scheduler.cycle_lengths=[$STEPS] model.config.parallelism.data_parallel_shard_degree=$NPROC_PER_NODE"
bash examples/launch_sft_action_policy_navsim.sh

CUDA_VISIBLE_DEVICES= python -m cosmos_framework.scripts.merge_navsim_slim_dcp \
    --base-path "$WORK/ckpt/h4_80k_merged" --slim-path "$CKPT_DIR/navsim_h4_pref/checkpoints/iter_$(printf %09d "$STEPS")" \
    -o "$WORK/ckpt/distilled_merged" --tree net_ema
