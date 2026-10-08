#!/bin/bash
# Full navtrain / navtest -> one GEAR dataset: split the official scene filter by log into K
# shards, convert in parallel, merge with global re-indexing.
#
# Required env: OPENSCENE_DATA_ROOT, NAVSIM_DEVKIT_ROOT, GEAR_OUT_ROOT
# Optional env: NAVSIM_PYTHON (default: python), HIST_FRAMES (default: 4)
# Usage: bash convert_navsim_full.sh [K=48] [navtrain|navtest]
set -eu
: "${OPENSCENE_DATA_ROOT:?}"
: "${NAVSIM_DEVKIT_ROOT:?}"
: "${GEAR_OUT_ROOT:?}"

K=${1:-48}
KIND=${2:-navtrain}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${NAVSIM_PYTHON:-python}
V=$GEAR_OUT_ROOT
NTYAML=$NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/train_test_split/scene_filter/${KIND}.yaml
DATASPLIT=$([ "$KIND" = navtest ] && echo test || echo trainval)
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT

if [ -s "$V/$KIND/meta/episodes.jsonl" ]; then
  echo "$KIND already converted ($(grep -c '' "$V/$KIND/meta/episodes.jsonl") episodes), skip."
  exit 0
fi

SHARD_DIR=$V/_${KIND}_shards
rm -rf "$SHARD_DIR"; mkdir -p "$SHARD_DIR"
echo "splitting into $K shard yamls (KIND=$KIND, data_split=$DATASPLIT)"
$PY - "$NTYAML" "$K" "$SHARD_DIR" <<'PY'
import sys, yaml
nt = yaml.safe_load(open(sys.argv[1])); K = int(sys.argv[2]); out = sys.argv[3]
logs = nt['log_names']
base = {k: nt[k] for k in ('_target_', '_convert_', 'num_history_frames', 'num_future_frames',
                           'frame_interval', 'has_route', 'max_scenes', 'tokens')}
n = 0
for k in range(K):
    grp = logs[k::K]
    if not grp:
        continue
    d = dict(base); d['log_names'] = grp
    yaml.safe_dump(d, open(f'{out}/shard_{k:03d}.yaml', 'w')); n += 1
print(f'{n} shards over {len(logs)} logs')
PY

echo "launching parallel converters..."
for y in "$SHARD_DIR"/shard_*.yaml; do
  k=$(basename "$y" .yaml | sed 's/shard_//')
  $PY "$HERE/convert_navsim_to_gear.py" --split "$DATASPLIT" --scene-filter-yaml "$y" \
      --make-video --history-frames "${HIST_FRAMES:-4}" --out "$V/_${KIND}_shard_$k" > "$SHARD_DIR/conv_$k.log" 2>&1 &
done
wait
echo "conversion done, merging..."
$PY "$HERE/merge_gear_shards.py" --out "$V/$KIND" --shards "$V/_${KIND}_shard_*" --move
echo "=== $KIND GEAR done -> $V/$KIND ($(grep -c '' "$V/$KIND/meta/episodes.jsonl") episodes) ==="
