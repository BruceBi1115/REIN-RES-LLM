#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT_NAME="scripts/run_tinynews_nsw_price.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_tinynews_nsw_price.sh"

DATASET_KEY="nsw_price"
TIME_COL="date"
VALUE_COL="RRP"
ID_COL=""
UNIT=""
DESCRIPTION="This dataset records the electricity price data in Australia NSW from 2024, collected from National electricity market."
REGION="Australia, NSW"
FREQ_MIN=""

TRAIN_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_trainset.csv"
VAL_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_valset.csv"
TEST_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_testset.csv"
DEFAULT_NEWS_PATH="dataset/watt_free_2024_load.json"

BATCH_SIZE="8"
NEWS_TOPK="10"
BASE_BACKBONES=("mlp")
DELTA_FREEZE_FEATURE_MODULES="1"
TASK_NAME_BASE="[Price-distilbert-topk10-gatedadd-summarygated]"
PRE_RUN_HOOK=""

source "$SCRIPT_DIR/_run_tinynews_common.sh"
run_tinynews_dataset_main "$@"
