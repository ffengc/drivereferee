#!/usr/bin/env bash
# Paths used by every script. Edit for your machine, then `source scripts/env.sh`.

export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# NavSim data and devkits
export OPENSCENE_DATA_ROOT=/data/navsim              # navsim_logs/, sensor_blobs/, maps/
export NAVSIM_DEVKIT_ROOT=/data/navsim-devkit         # NavSim v2 devkit (conversion, referee, EPDMS)
export NAVSIM_DEVKIT_V1_ROOT=/data/navsim-devkit-v1   # NavSim v1.1 devkit (official PDMS)
export NAVSIM_PYTHON=/envs/navsim/bin/python          # python of the v2 devkit env
export NAVSIM_V1_PYTHON=/envs/navsim-v1/bin/python    # python of the v1.1 devkit env

# Converted data, checkpoints, outputs
export GEAR_ROOT=/data/gear                           # navtrain/, navtest/ (scripts/01)
export BASE_DCP=/ckpt/Cosmos3-Nano                    # convert_model_to_dcp output
export WAN_VAE_PATH=/ckpt/wan22_vae/Wan2.2_VAE.pth
export HF_HOME=/ckpt/hf_cache
export WORK=/work/drivereferee                        # everything this pipeline writes

export OUTPUT_ROOT=$WORK/train
export IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_ROOT
export CKPT_DIR=$OUTPUT_ROOT/cosmos3_action/action_sft
export NAVSIM_EXP_ROOT=$WORK/navsim_exp
export METRIC_CACHE_V2=$WORK/metric_cache_navtest
export METRIC_CACHE_V1=$WORK/metric_cache_navtest_v1

export GPUS=${GPUS:-0,1,2,3,4,5,6,7}                  # GPUs for sharded inference
export NPROC_PER_NODE=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK" "$OUTPUT_ROOT" "$NAVSIM_EXP_ROOT"
