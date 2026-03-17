#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# One-factor ablation:
# - keep feature freezing on
# - remove the positive non-degrade improvement margin
export TASK_NAME_BASE="${TASK_NAME_BASE:-[2024-nswelecLOAD-tinynews-margin0]}"
export DELTA_FREEZE_FEATURE_MODULES="${DELTA_FREEZE_FEATURE_MODULES:-1}"
export DELTA_NON_DEGRADE_LAMBDA="${DELTA_NON_DEGRADE_LAMBDA:-1.0}"
export DELTA_NON_DEGRADE_MARGIN="${DELTA_NON_DEGRADE_MARGIN:-0.0}"

exec bash "$SCRIPT_DIR/nswelecLOAD_2024_tinynews.sh"
