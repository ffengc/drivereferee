#!/usr/bin/env bash
# Map generator (predicted occupancy for the deployed referee): tower export, history poses,
# log-level split, training (6 GPUs), and maps for navtest.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
MG=$WORK/mapgen
mkdir -p "$MG"

[ -f "$MG/tower/vision_tower.safetensors" ] || python tools/export_vision_tower.py \
    --dcp "$BASE_DCP/model" --config "$HF_HOME/hub/models--nvidia--Cosmos3-Nano/snapshots/*/config.json" --out "$MG/tower"

[ -f "$MG/hist_navtrain.npz" ] || python tools/gen_history_poses.py --gear "$GEAR_ROOT/navtrain" \
    --logs "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" --out "$MG/hist_navtrain.npz" --history 4
[ -f "$MG/hist_navtest.npz" ] || python tools/gen_history_poses.py --gear "$GEAR_ROOT/navtest" \
    --logs "$OPENSCENE_DATA_ROOT/navsim_logs/test" --out "$MG/hist_navtest.npz" --history 4

[ -f "$MG/splits/probe_train.txt" ] || python tools/make_probe_split.py --occ "$WORK/occ_gt/navtrain" \
    --occ-test "$WORK/occ_gt/navtest" --out "$MG/splits" --train 60000 --calib 2000

MG_GPUS=${MG_GPUS:-0,1,2,3,4,5}
N=$(echo "$MG_GPUS" | tr ',' '\n' | wc -l)
[ -f "$MG/model/ckpt.pt" ] || CUDA_VISIBLE_DEVICES=$MG_GPUS torchrun --nproc_per_node="$N" tools/train_map_generator.py \
    --gear "$GEAR_ROOT/navtrain" --occ "$WORK/occ_gt/navtrain" --hist "$MG/hist_navtrain.npz" \
    --splits "$MG/splits" --tower "$MG/tower" --out "$MG/model" --steps 10000 --bs 10 --lr 2e-3

CUDA_VISIBLE_DEVICES=$MG_GPUS torchrun --nproc_per_node="$N" tools/dump_map_generator.py \
    --ckpt "$MG/model/ckpt.pt" --tower "$MG/tower" --gear "$GEAR_ROOT/navtest" --hist "$MG/hist_navtest.npz" \
    --out "$MG/maps_navtest" --thr 0.5
