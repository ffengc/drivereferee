#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Modified by Fengcheng Yu, 2026.

# Launch NavSim action-policy training via cosmos_framework.scripts.train.
#
# Env vars:
#   TOML_FILE             recipe TOML (default: the h4 base recipe)
#   NAVSIM_GEAR_ROOT      NavSim GEAR split root (.../gear/navtrain)
#   BASE_CHECKPOINT_PATH  DCP checkpoint to start from
#   WAN_VAE_PATH          Wan2.2 VAE .pth
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_navsim_h4.toml}"
: "${NAVSIM_GEAR_ROOT:?set NAVSIM_GEAR_ROOT to the GEAR split root}"
: "${BASE_CHECKPOINT_PATH:?set BASE_CHECKPOINT_PATH to the base DCP dir}"
export NAVSIM_GEAR_ROOT
DATASET_PATH="$NAVSIM_GEAR_ROOT"

EXTRA_DATASET_CHECK='[[ -f "$NAVSIM_GEAR_ROOT/meta/info.json" ]] || { echo "ERROR: missing $NAVSIM_GEAR_ROOT/meta/info.json" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
