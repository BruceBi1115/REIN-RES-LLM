#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT_NAME="scripts/run_bitcoin_hourly.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_bitcoin_hourly.sh"

DATASET_KEY="bitcoin_hourly_24"
TIME_COL="DATETIME"
VALUE_COL="OPEN"
ID_COL=""
UNIT="USD"
DESCRIPTION="This dataset records Bitcoin hourly opening prices from 2022 to 2023."
REGION="Global crypto market"
FREQ_MIN="60"
DAY_FIRST="1"

TRAIN_FILE="dataset/bitcoin_hourly_24/bitcoin-hourly-open-2024_trainset.csv"
VAL_FILE="dataset/bitcoin_hourly_24/bitcoin-hourly-open-2024_valset.csv"
TEST_FILE="dataset/bitcoin_hourly_24/bitcoin-hourly-open-2024_testset.csv"
DEFAULT_NEWS_PATH="dataset/news_from_sources/coindesk_archive_2024_new.json"

STAGE="all"
BASE_EPOCHS="40"
DELTA_EPOCHS="30"
BATCH_SIZE="16"
GRAD_ACCS=("4")
LRS=("1e-4")
HORIZONS=("48")
SCHEDULERS=("1")
BASE_BACKBONES=("dlinear")
# NORMALIZATION_MODE="zscore"
EARLY_STOP_PATIENCE="5"
NEWS_WINDOW_DAYS="1"

TASK_NAME_BASE="delta_v3_bitcoin_hourly"
PRE_RUN_HOOK=""

DELTA_V3_REFINED_BANK_BUILD="1"
DELTA_V3_ACTIVE_MASS_THRESHOLDS=("0.7")

NEWS_API_ENABLE="1"
DELTA_V3_SCHEMA_VARIANT="bitcoin"
DELTA_V3_REGIME_BANK_PATH="_shared_refine_cache/v4/regime_bank_$(basename -- "${NEWS_PATH:-$DEFAULT_NEWS_PATH}" .json).npz"

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
DELTA_V3_CONSISTENCY_WEIGHT="0.30"
DELTA_V3_COUNTERFACTUAL_WEIGHT="0.25"
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
SPIKE_CLIP_THRESHOLD="0"

NEWS_PATH="${NEWS_PATH:-$DEFAULT_NEWS_PATH}"
NEWS_CACHE_TAG="$(basename -- "${NEWS_PATH%.json}")"

if [[ -z "${DELTA_V3_REFINED_BANK_BUILD+x}" ]]; then
  if [[ -f "_shared_refine_cache/v4/regime_bank_${NEWS_CACHE_TAG}.npz" ]]; then
    DELTA_V3_REFINED_BANK_BUILD="0"
  else
    DELTA_V3_REFINED_BANK_BUILD="1"
  fi
fi

set_horizon_specific_params() {
  local h="$1"
  if [[ "$h" == "24" || "$h" == "48" ]]; then
    DELTA_V3_LAMBDA_MAX="0.45"
    DELTA_V3_LAMBDA_TS_CAP="0.30"
    DELTA_V3_LAMBDA_NEWS_CAP="0.12"
    DELTA_V3_SPIKE_BIAS_CAP="0.75"
    DELTA_V3_SHAPE_GAIN_CAP="0.30"
  elif [[ "$h" == "96" ]]; then
    DELTA_V3_LAMBDA_MAX="0.20"
    DELTA_V3_LAMBDA_TS_CAP="0.15"
    DELTA_V3_LAMBDA_NEWS_CAP="0.05"
    DELTA_V3_SPIKE_BIAS_CAP="0.40"
    DELTA_V3_SHAPE_GAIN_CAP="0.20"
  else
    DELTA_V3_LAMBDA_MAX="0.12"
    DELTA_V3_LAMBDA_TS_CAP="0.08"
    DELTA_V3_LAMBDA_NEWS_CAP="0.02"
    DELTA_V3_SPIKE_BIAS_CAP="0.30"
    DELTA_V3_SHAPE_GAIN_CAP="0.15"
  fi
}

source "$SCRIPT_DIR/_run_forecast_common.sh"

run_bitcoin_hourly_main() {
  parse_forecast_runner_args "$@"
  maybe_run_in_tmux
  init_common_defaults
  validate_dataset_config

  PYTHON_BIN="${PYTHON_BIN:-}"
  if ! PYTHON_BIN="$(pick_python_bin "$PYTHON_BIN")"; then
    echo "[ERROR] No suitable Python found for this project." >&2
    echo "        Need modules: matplotlib, openai, pandas, torch, transformers" >&2
    echo "        You can run with an explicit interpreter, e.g.:" >&2
    echo "        PYTHON_BIN=/path/to/env/bin/python bash ${RUN_SCRIPT_NAME:-scripts/run_bitcoin_hourly.sh}" >&2
    exit 1
  fi
  echo "[env] Using PYTHON_BIN=$PYTHON_BIN"

  NEWS_PATH="${NEWS_PATH:-$DEFAULT_NEWS_PATH}"
  set_array_default NEWS_CHOICES "$NEWS_PATH"

  if [[ -n "${PRE_RUN_HOOK:-}" ]]; then
    "$PRE_RUN_HOOK"
  fi

  echo "[config] dataset=$DATASET_KEY stage=$STAGE task_base=$TASK_NAME_BASE"
  for news_path in "${NEWS_CHOICES[@]}"; do
    for base_backbone in "${BASE_BACKBONES[@]}"; do
      for lr in "${LRS[@]}"; do
        for scheduler in "${SCHEDULERS[@]}"; do
          for grad_acc in "${GRAD_ACCS[@]}"; do
            for horizon in "${HORIZONS[@]}"; do
              for active_mass_threshold in "${DELTA_V3_ACTIVE_MASS_THRESHOLDS[@]}"; do
                local task_name
                DELTA_V3_ACTIVE_MASS_THRESHOLD="$active_mass_threshold"
                set_horizon_specific_params "$horizon"
                task_name="${TASK_NAME_BASE}__lr${lr}__ga${grad_acc}__sch${scheduler}${TASK_NAME_SUFFIX}"
                build_run_args "$task_name" "$base_backbone" "$horizon" "$lr" "$scheduler" "$grad_acc" "$news_path"
                if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
                  RUN_ARGS+=( "${EXTRA_RUN_ARGS[@]}" )
                fi

                echo "==> Running dataset=$DATASET_KEY backbone=$base_backbone horizon=$horizon lr=$lr grad_acc=$grad_acc active_mass_threshold=$DELTA_V3_ACTIVE_MASS_THRESHOLD stage=$STAGE"
                echo "    horizon_caps: lambda_max=$DELTA_V3_LAMBDA_MAX lambda_ts_cap=$DELTA_V3_LAMBDA_TS_CAP lambda_news_cap=$DELTA_V3_LAMBDA_NEWS_CAP spike_bias_cap=$DELTA_V3_SPIKE_BIAS_CAP shape_gain_cap=$DELTA_V3_SHAPE_GAIN_CAP early_stop_patience=$EARLY_STOP_PATIENCE"
                "$PYTHON_BIN" "$ENTRY" "${RUN_ARGS[@]}"
              done
            done
          done
        done
      done
    done
  done
}

run_bitcoin_hourly_main "$@"
