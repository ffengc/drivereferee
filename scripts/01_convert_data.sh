#!/usr/bin/env bash
# NavSim -> GEAR (832x480 front camera, 4 history + current + 8 future frames).
# Before this: download navsim_logs/ and maps/ per the NavSim devkit docs, and the camera blobs:
#   OPENSCENE_DATA_ROOT=... bash tools/navsim2gear/download_navsim_camera.sh trainval 0 200
#   OPENSCENE_DATA_ROOT=... bash tools/navsim2gear/download_navsim_camera.sh test 0 32
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export GEAR_OUT_ROOT=$GEAR_ROOT HIST_FRAMES=4 LD_LIBRARY_PATH=
bash tools/navsim2gear/convert_navsim_full.sh 48 navtrain
bash tools/navsim2gear/convert_navsim_full.sh 16 navtest
