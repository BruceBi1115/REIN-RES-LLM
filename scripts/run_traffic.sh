#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT_NAME="scripts/run_traffic.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_traffic.sh"

DATASET_KEY="traffic_count"
TIME_COL="time"
VALUE_COL="traffic_count"
ID_COL=""
UNIT="count"
DESCRIPTION="This dataset records hourly traffic counts of year 2024 in Australia, NSW, aggregated into a single hourly time series."
REGION="Traffic monitoring network"
FREQ_MIN="60"
DAY_FIRST="0"

TRAIN_FILE="dataset/traffic_count/road_traffic_counts_hourly_permanent3_2024_hourly_trainset.csv"
VAL_FILE="dataset/traffic_count/road_traffic_counts_hourly_permanent3_2024_hourly_valset.csv"
TEST_FILE="dataset/traffic_count/road_traffic_counts_hourly_permanent3_2024_hourly_testset.csv"
DEFAULT_NEWS_PATH="dataset/news_from_sources/reneweconomy_web_stories_2024.json"

# News is intentionally left empty for now, so default to base-only runs.
# When a traffic-news source is ready, set:
#   STAGE="all"
#   DEFAULT_NEWS_PATH="path/to/traffic_news.json"
STAGE="all"
BASE_EPOCHS="100"
DELTA_EPOCHS="50"
BATCH_SIZE="16"
GRAD_ACCS=("4")
LRS=("1e-4")
HORIZONS=( "48")
SCHEDULERS=("1")
BASE_BACKBONES=("mlp")

EARLY_STOP_PATIENCE="5"
NEWS_WINDOW_DAYS="1"

TASK_NAME_BASE="traffic_count"
PRE_RUN_HOOK=""

NEWS_API_ENABLE="1"

DELTA_V3_REFINED_BANK_BUILD="1"
DELTA_V3_ACTIVE_MASS_THRESHOLD="0.7"
DELTA_V3_SCHEMA_VARIANT="traffic"
DELTA_V3_REGIME_BANK_PATH=""

DELTA_V3_TEXT_ENCODER_MODEL_ID="intfloat/e5-small-v2"
DELTA_V3_TEXT_ENCODER_MAX_LENGTH="256"
DELTA_V3_REGIME_TAU_DAYS="5.0"
DELTA_V3_REGIME_EMA_ALPHA="0.5"
DELTA_V3_REGIME_EMA_WINDOW="5"
DELTA_V3_ARCH="patchtst_regime_modulation"
DELTA_V3_HIDDEN_SIZE="128"
DELTA_V3_NUM_LAYERS="2"
DELTA_V3_NUM_HEADS="4"
DELTA_V3_PATCH_LEN="8"
DELTA_V3_PATCH_STRIDE="4"
DELTA_V3_DROPOUT="0.1"
DELTA_V3_USE_BASE_HIDDEN="1"
DELTA_V3_SLOW_WEIGHT="1.0"
DELTA_V3_SHAPE_WEIGHT="1.0"
DELTA_V3_SPIKE_WEIGHT="1.0"
DELTA_V3_SPIKE_GATE_THRESHOLD="0.8"
DELTA_V3_SPIKE_K="3.0"
DELTA_V3_SPIKE_TARGET_PCT="0.10"
DELTA_V3_SPIKE_GATE_LOSS_WEIGHT="0.25"
DELTA_V3_NEWS_BLANK_PROB="0.3"
DELTA_V3_CONSISTENCY_WEIGHT="0.05"
DELTA_V3_COUNTERFACTUAL_WEIGHT="0.1"
DELTA_V3_COUNTERFACTUAL_MARGIN="0.02"
DELTA_V3_SPIKE_BIAS_L2="1e-3"

DELTA_V3_LAMBDA_MIN="0.05"
DELTA_V3_LAMBDA_TS_CAP="0.30"
DELTA_V3_LAMBDA_NEWS_CAP="0.12"
DELTA_V3_LAMBDA_MAX="0.45"
DELTA_V3_SHAPE_GAIN_CAP="0.20"
DELTA_V3_SPIKE_BIAS_CAP="0.75"
DELTA_V3_SELECTION_COUNTERFACTUAL_GAIN_MIN="0.01"
DELTA_V3_SELECTION_LAMBDA_SATURATION_MAX_PCT="0.35"
DELTA_V3_HARD_RESIDUAL_FRAC="0.6"
DELTA_V3_HARD_RESIDUAL_PCT="0.10"
DELTA_V3_PRETRAIN_EPOCHS="12"
DELTA_V3_PRETRAIN_LR="1e-3"
DELTA_V3_PRICE_WINSOR_LOW="0.005"
DELTA_V3_PRICE_WINSOR_HIGH="0.995"
DELTA_V3_GRAD_CLIP="1.0"
DELTA_V3_EVAL_PERMUTATION_SEED="2024"
DELTA_V3_SELECT_METRIC="mae"

source "$SCRIPT_DIR/_run_forecast_common.sh"
run_forecast_dataset_main "$@"
