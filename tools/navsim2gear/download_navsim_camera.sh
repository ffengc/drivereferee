#!/bin/bash
# Download OpenScene camera shards [START, END) for <trainval|test>, unpack and merge them into
# $OPENSCENE_DATA_ROOT/sensor_blobs/<KIND>. Camera only (no lidar), streaming, resumable.
# Usage: OPENSCENE_DATA_ROOT=<dataset> bash download_navsim_camera.sh <trainval|test> <START> <END>
set -u
: "${OPENSCENE_DATA_ROOT:?}"

KIND=$1; START=$2; END=$3
DATA=$OPENSCENE_DATA_ROOT
STAGE=$DATA/_staging_$KIND/w_${START}_${END}
DEST=$DATA/sensor_blobs/$KIND
DONE=$DATA/_staging_$KIND/done
mkdir -p "$STAGE" "$DEST" "$DONE"
BASE=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_${KIND}_camera

for s in $(seq "$START" $((END-1))); do
  if [ -f "$DONE/cam_$s.done" ]; then echo "split $s already done, skipping"; continue; fi
  f="$STAGE/cam_$s.tgz"
  echo "split $s: downloading..."
  if ! wget -q "$BASE/openscene_sensor_${KIND}_camera_${s}.tgz" -O "$f"; then
    echo "split $s download failed, skipping (re-run to retry)"; rm -f "$f"; continue
  fi
  echo "split $s: extracting..."
  if ! tar xzf "$f" -C "$STAGE"; then
    echo "split $s extract failed, skipping"; rm -f "$f"; rm -rf "$STAGE/openscene-v1.1"; continue
  fi
  rsync -a "$STAGE/openscene-v1.1/sensor_blobs/$KIND/" "$DEST/"
  rm -rf "$f" "$STAGE/openscene-v1.1"
  touch "$DONE/cam_$s.done"
  echo "split $s: ok"
done
echo "worker [$START,$END) finished."
