#!/usr/bin/env bash
# Deployment: default trajectory on navtest, referee margins on predicted maps, second candidate
# on thin-ice scenes, gated selection (K=2).
# Usage: bash scripts/07_gated_inference.sh [merged DCP] [name]
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
MODEL=${1:-$WORK/ckpt/distilled_merged}
NAME=${2:-distilled}
G=$WORK/gated/$NAME
MAPS=$WORK/mapgen/maps_navtest
SPEEDS=$WORK/gt/navtest/speeds.json
mkdir -p "$G"

bash scripts/_dump_sharded.sh "$MODEL" "$GEAR_ROOT/navtest" "$G/seed0" 0

export LD_LIBRARY_PATH= NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 PYTHONPATH=$NAVSIM_DEVKIT_ROOT
[ -s "$G/margins.csv" ] || $NAVSIM_PYTHON tools/referee.py score --pred-dirs "$G/seed0/test" --source-names seed0 \
    --occ-root "$MAPS" --speeds "$SPEEDS" --out "$G/margins.csv"
python tools/thin_ice_subset.py --margins "$G/margins.csv" --gear "$GEAR_ROOT/navtest" --out "$G/thin_ice.txt" --threshold 0.4

bash scripts/_dump_sharded.sh "$MODEL" "$GEAR_ROOT/navtest" "$G/seed1" 1 "$G/thin_ice.txt"

$NAVSIM_PYTHON tools/rank_select_candidates.py --cand-dirs "$G/seed0/test" "$G/seed1/test" \
    --occ "$MAPS" --speeds "$SPEEDS" --out "$G/selected/test" --delta-steps 1 --delta-margin 0.4 --report "$G/select.csv"
echo "selected trajectories: $G/selected/test"
