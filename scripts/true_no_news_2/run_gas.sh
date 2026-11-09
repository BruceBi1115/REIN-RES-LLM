#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUN_SCRIPT_NAME="scripts/true_no_news_2/run_gas.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_gas.sh"

DATASET_KEY="gas_demand"
TIME_COL="date"
VALUE_COL="total_value"
ID_COL=""
UNIT=""
DESCRIPTION="This dataset records hourly gas demand values for the Netherlands in 2025, extracted from ENTSOG."
REGION="Netherlands"
FREQ_MIN="60"
DAY_FIRST="0"

TRAIN_FILE="dataset/gas_demand/entsog_gas_demand_nl_2025_hour_extracted_trainset.csv"
VAL_FILE="dataset/gas_demand/entsog_gas_demand_nl_2025_hour_extracted_valset.csv"
TEST_FILE="dataset/gas_demand/entsog_gas_demand_nl_2025_hour_extracted_testset.csv"
DEFAULT_NEWS_PATH="dataset/gdelt_gas_nl_2025.json"

STAGE="all"
BASE_EPOCHS="40"
DELTA_EPOCHS="30"
BATCH_SIZE="16"
GRAD_ACCS=("4")
LRS=("5e-6")
HORIZONS=("48" "96" "192" "336" "720")
# HORIZONS=( "48")
SCHEDULERS=("1")
BASE_BACKBONES=("dlinear")


DELTA_V3_REFINED_BANK_BUILD="0"
# TRUE NO-NEWS: theta=-1 -> gate always open (lambda_ts trains normally), news knobs always on
DELTA_V3_ACTIVE_MASS_THRESHOLDS=("-1")
NORMALIZATION_MODE="zscore"
DELTA_V3_TEXT_ENCODER_MODEL_ID="intfloat/e5-small-v2"

TASK_NAME_BASE="delta_v3_gas_demand_true_no_news_2"
PRE_RUN_HOOK=""

# DELTA_V3_WARMUP_PCT=0.10
EARLY_STOP_PATIENCE=5

NEWS_API_ENABLE="1"
DELTA_V3_SCHEMA_VARIANT="gas_demand"
# TRUE NO-NEWS: all regime content + relevance mass are 0 (built from the HEAD bank via
# scripts/make_regime_bank_ablation.py --mode blank_news_features)
DELTA_V3_REGIME_BANK_PATH="_shared_refine_cache/v4/ablations/true_nonews/gas_demand_blank_news_features.npz"
# TRUE NO-NEWS 1: all three news knobs pinned to neutral (lambda_news=0, shape_gain=1, spike_bias=0).
#   lambda_news = cap*tanh(.), shape_gain = 1 + cap*tanh(.), spike_bias = cap*tanh(.)  -> cap ~0 = neutral.
#   1e-12 instead of 0 because src/delta_v3/config.py uses `value or <default>`, which turns 0 back into the default.
DELTA_V3_LAMBDA_NEWS_CAP="1e-12"
DELTA_V3_SHAPE_GAIN_CAP="1e-12"
DELTA_V3_SPIKE_BIAS_CAP="1e-12"
# TRUE NO-NEWS 2: additionally pin shape_gain to 0 (removes the shape residual branch entirely).
# L2 pull toward 1 is turned off since shape_gain is constant.
DELTA_V3_SHAPE_GAIN_FIXED="0"
DELTA_V3_SHAPE_GAIN_L2_WEIGHT="0"

if [[ -z "${DELTA_V3_REFINED_BANK_BUILD:-}" ]]; then
  NEWS_PATH="${NEWS_PATH:-$DEFAULT_NEWS_PATH}"
  NEWS_CACHE_TAG="$(basename -- "${NEWS_PATH%.json}")"
  if [[ -f "_shared_refine_cache/v4/regime_bank_${NEWS_CACHE_TAG}.npz" ]]; then
    DELTA_V3_REFINED_BANK_BUILD="0"
  else
    DELTA_V3_REFINED_BANK_BUILD="1"
  fi
fi

cd "$REPO_ROOT"
source "$COMMON_DIR/_run_forecast_common.sh"
run_forecast_dataset_main "$@"
