#!/usr/bin/env bash
# Self-sampling on 15k navtrain scenes (K=5 seeds), referee grading on GT maps, pair building.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
MODEL=$WORK/ckpt/h4_80k_merged
SP=$WORK/selfplay
mkdir -p "$SP"

[ -s "$SP/episodes_15k.txt" ] || python tools/sample_episodes.py --gear "$GEAR_ROOT/navtrain" --n 15000 --seed 42 --out "$SP/episodes_15k.txt"

for S in 0 1 2 3 4; do
    bash scripts/_dump_sharded.sh "$MODEL" "$GEAR_ROOT/navtrain" "$SP/seed$S" "$S" "$SP/episodes_15k.txt"
    [ -s "$SP/referee_seed$S.csv" ] && continue
    ( export LD_LIBRARY_PATH= NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 PYTHONPATH=$NAVSIM_DEVKIT_ROOT
      $NAVSIM_PYTHON tools/referee.py score --pred-dirs "$SP/seed$S/test" --source-names "seed$S" \
          --occ-root "$WORK/occ_gt/navtrain" --speeds "$WORK/gt/navtrain/speeds.json" --out "$SP/referee_seed$S.csv" )
done

python tools/build_preference_pairs.py --selfplay-root "$SP" --gt-dir "$WORK/gt/navtrain/gt_npys" --out-root "$WORK/pairs"
