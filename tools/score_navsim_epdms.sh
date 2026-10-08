#!/usr/bin/env bash
# EPDMS (NavSim v2 one-stage) scoring of a directory of <token>.npy trajectories.
# Env: NAVSIM_DEVKIT_ROOT, NAVSIM_PYTHON, OPENSCENE_DATA_ROOT, NAVSIM_EXP_ROOT
# Usage: bash tools/score_navsim_epdms.sh <pred_dir> <split> <train_test_split> <metric_cache_dir> [experiment_name]
#   pred_dir must contain <split>/<token>.npy
set -euo pipefail
: "${NAVSIM_DEVKIT_ROOT:?}"; : "${NAVSIM_PYTHON:?}"; : "${OPENSCENE_DATA_ROOT:?}"; : "${NAVSIM_EXP_ROOT:?}"

PRED_DIR=$1; SPLIT=$2; TTS=$3; CACHE=$4; EXP_NAME=${5:-pdm_score}
export LD_LIBRARY_PATH=
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
mkdir -p "$NAVSIM_EXP_ROOT"

cd "$NAVSIM_DEVKIT_ROOT"
exec "$NAVSIM_PYTHON" navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split="$TTS" \
    agent=human_agent \
    worker=sequential \
    metric_cache_path="$CACHE" \
    pred_dir="$PRED_DIR" \
    split="$SPLIT" \
    experiment_name="$EXP_NAME"
