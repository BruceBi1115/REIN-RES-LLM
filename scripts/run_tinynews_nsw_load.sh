#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT_NAME="scripts/run_tinynews_nsw_load.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_tinynews_nsw_load.sh"

DATASET_KEY="nsw_load"
TIME_COL="date"
VALUE_COL="TOTALDEMAND"
ID_COL=""
UNIT="MW"
DESCRIPTION="This dataset records the electricity load demand data in Australia NSW from 2024, collected from National electricity market."
REGION="Australia, NSW"
FREQ_MIN=""

TRAIN_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_trainset.csv"
VAL_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_valset.csv"
TEST_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_testset.csv"
DEFAULT_NEWS_PATH="dataset/watt_free_2024_load.json"

BASE_EPOCHS="40"

BATCH_SIZE="16"
GRAD_ACCS=(
    "4"
)
NEWS_TOPK="3"
BASE_BACKBONES=("mlp")
DELTA_FREEZE_FEATURE_MODULES="1"
TASK_NAME_BASE="[batch16-acc4-load-distilbert-topk3-gatedadd-summarygated]"
PRE_RUN_HOOK=""

source "$SCRIPT_DIR/_run_tinynews_common.sh"
run_tinynews_dataset_main "$@"
