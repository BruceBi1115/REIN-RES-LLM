#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT_NAME="scripts/run_nas14_news.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_nas14_news.sh"

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

STAGE="all"
BASE_EPOCHS="40"
DELTA_EPOCHS="30"
BATCH_SIZE="4"
GRAD_ACCS=("1")
LRS=("1e-4")
HORIZONS=("48")
SCHEDULERS=("1")
BASE_BACKBONES=("dlinear")
DAY_FIRST="0"

TASK_NAME_BASE="delta_v3_nas14"
PRE_RUN_HOOK="prepare_chronological_timeseries_splits"

NEWS_API_ENABLE="1"
DELTA_V3_SCHEMA_VARIANT="price"
DELTA_V3_REGIME_BANK_PATH="checkpoints/_shared_refine_cache/v4/regime_bank_nas14.npz"
DELTA_V3_REGIME_BANK_BUILD="0"


if [[ -z "${DELTA_V3_REGIME_BANK_BUILD:-}" ]]; then
  if [[ -f "checkpoints/_shared_refine_cache/v4/regime_bank_nas14__nasdaq_news_22_23.npz" || -f "$DELTA_V3_REGIME_BANK_PATH" ]]; then
    DELTA_V3_REGIME_BANK_BUILD="0"
  else
    DELTA_V3_REGIME_BANK_BUILD="1"
  fi
fi

source "$SCRIPT_DIR/_run_forecast_common.sh"
run_forecast_dataset_main "$@"
