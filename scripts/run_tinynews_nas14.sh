#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT_NAME="scripts/run_tinynews_nas14.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_tinynews_nas14.sh"

DATASET_KEY="nas14"
TIME_COL="date"
VALUE_COL="open"
ID_COL="ticker"
UNIT="USD"
DESCRIPTION="This dataset records 14 tickers' daily opening stock prices on the NASDAQ market from 2022 to 2023, paired with related NASDAQ news articles."
REGION="United States, NASDAQ"
FREQ_MIN="1440"

TRAIN_FILE="dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_trainset.csv"
VAL_FILE="dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_valset.csv"
TEST_FILE="dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_testset.csv"
DEFAULT_NEWS_PATH="dataset/nasdaq_news_22_23.json"

BATCH_SIZE="4"
BASE_BACKBONES=("dlinear")
DELTA_FREEZE_FEATURE_MODULES="0"
TASK_NAME_BASE="[2022_2023-nasdaq14tickeropen-tinynews]"
PRE_RUN_HOOK="prepare_chronological_timeseries_splits"

source "$SCRIPT_DIR/_run_tinynews_common.sh"
run_tinynews_dataset_main "$@"
