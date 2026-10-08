#!/usr/bin/env bash
# Referee assets: occupancy GT (drivable + vehicles) and GT trajectories / speeds for both splits.
# Runs in the NavSim devkit env (CPU). Occupancy generation is sharded over $NSHARD processes.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
export LD_LIBRARY_PATH= NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT
NSHARD=${NSHARD:-16}

for KIND in navtrain navtest; do
    LOGS=$([ "$KIND" = navtest ] && echo test || echo trainval)
    mkdir -p "$WORK/occ_gt/$KIND" "$WORK/logs"
    for i in $(seq 0 $((NSHARD-1))); do
        $NAVSIM_PYTHON tools/gen_occ_gt_navsim.py --gear-root "$GEAR_ROOT/$KIND" \
            --logs-dir "$OPENSCENE_DATA_ROOT/navsim_logs/$LOGS" --maps-root "$OPENSCENE_DATA_ROOT/maps" \
            --out-dir "$WORK/occ_gt/$KIND" --shard "$i/$NSHARD" > "$WORK/logs/occ_${KIND}_$i.log" 2>&1 &
    done
    wait
    echo "$KIND: $(ls "$WORK/occ_gt/$KIND" | wc -l) occupancy files"
done

# GT trajectories + initial speeds (training env, uses the dataset class).
python tools/gen_gt_trajectories.py --gear "$GEAR_ROOT/navtest" --out "$WORK/gt/navtest"
python tools/gen_gt_trajectories.py --gear "$GEAR_ROOT/navtrain" --out "$WORK/gt/navtrain"
