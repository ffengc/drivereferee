#!/usr/bin/env bash
# Score a directory of navtest trajectories: official NavSim v1.1 PDMS and EPDMS (v2 devkit).
# Usage: bash scripts/08_evaluate.sh <dir with test/<token>.npy> <name> [base name for paired delta]
# First run (once): bash scripts/08_evaluate.sh --build-caches
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
export LD_LIBRARY_PATH= NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
V1=$WORK/navsim_v1_exp
mkdir -p "$V1/submissions"

if [ "${1:-}" = "--build-caches" ]; then
    (cd "$NAVSIM_DEVKIT_V1_ROOT" && $NAVSIM_V1_PYTHON navsim/planning/script/run_metric_caching.py \
        train_test_split=navtest cache.cache_path="$METRIC_CACHE_V1" \
        worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=32)
    (cd "$NAVSIM_DEVKIT_ROOT" && $NAVSIM_PYTHON navsim/planning/script/run_metric_caching.py \
        train_test_split=navtest cache.cache_path="$METRIC_CACHE_V2")
    exit 0
fi

DIR=$1; NAME=$2; BASE=${3:-}

# Official v1.1 PDMS via submission pickle.
$NAVSIM_V1_PYTHON tools/build_v1_submission.py --npy-dir "$DIR/test" --out "$V1/submissions/$NAME.pkl"
(cd "$NAVSIM_DEVKIT_V1_ROOT" && $NAVSIM_V1_PYTHON navsim/planning/script/run_pdm_score_from_submission.py \
    train_test_split=navtest metric_cache_path="$METRIC_CACHE_V1" \
    submission_file_path="$V1/submissions/$NAME.pkl" output_dir="$V1/${NAME}_v1")
CSV=$(ls -t "$V1/${NAME}_v1"/*.csv | head -1)
$NAVSIM_V1_PYTHON tools/summarize_v1_scores.py "$CSV"
if [ -n "$BASE" ]; then
    $NAVSIM_V1_PYTHON tools/paired_bootstrap.py --base "$(ls -t "$V1/${BASE}_v1"/*.csv | head -1)" --test "$CSV"
fi

# EPDMS (NavSim v2, one-stage).
bash tools/score_navsim_epdms.sh "$DIR" test navtest "$METRIC_CACHE_V2" "${NAME}_epdms" 2>&1 | grep -E "Final average score" || true
