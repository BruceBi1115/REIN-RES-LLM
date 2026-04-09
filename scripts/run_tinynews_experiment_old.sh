#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tinynews_experiment.sh --dataset <nas14|nsw_load|nsw_price> [--profile <default|relative_text|additive>] [--no-signnet] [-- <extra run.py args>]

Examples:
  bash scripts/run_tinynews_experiment.sh --dataset nas14
  bash scripts/run_tinynews_experiment.sh --dataset nas14 --no-signnet
  bash scripts/run_tinynews_experiment.sh --dataset nsw_load
  bash scripts/run_tinynews_experiment.sh --dataset nsw_price
  bash scripts/run_tinynews_experiment.sh --dataset nsw_price --profile additive

Notes:
  - `--profile additive` is only meaningful for `nsw_price`; it matches the old `_tinynews_2` defaults.
  - `--no-signnet` forces internal DELTA sign mode and appends `__nosignnet` to the task suffix when none is provided.
  - Scripts now default to `DELTA_MULTIMODAL_ARCH=plan_c_mvp`; set `DELTA_MULTIMODAL_ARCH=summary_gated` if you want the older fallback path.
  - Environment overrides from the old scripts are still supported.
EOF
}

DATASET="${DATASET:-}"
PROFILE="${PROFILE:-default}"
NO_SIGNNET=0
EXTRA_RUN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --dataset requires a value." >&2
        usage
        exit 1
      fi
      DATASET="$2"
      shift 2
      ;;
    --profile)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --profile requires a value." >&2
        usage
        exit 1
      fi
      PROFILE="$2"
      shift 2
      ;;
    --no-signnet)
      NO_SIGNNET=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_RUN_ARGS=("$@")
      break
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "[ERROR] You must specify --dataset." >&2
  usage
  exit 1
fi

normalize_dataset() {
  local raw="$1"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    nas14|nas_14ticker|nasdaq14|nasdaq_14ticker|nas)
      printf 'nas14\n'
      ;;
    nsw_load|nswelecload|load|elecload)
      printf 'nsw_load\n'
      ;;
    nsw_price|nswelecprice|price|elecprice)
      printf 'nsw_price\n'
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_profile() {
  local raw="$1"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    default|"")
      printf 'default\n'
      ;;
    relative|relative_text|tinynews)
      printf 'relative_text\n'
      ;;
    additive|tinynews_2|alt)
      printf 'additive\n'
      ;;
    *)
      return 1
      ;;
  esac
}

if ! DATASET_KEY="$(normalize_dataset "$DATASET")"; then
  echo "[ERROR] Unsupported dataset: $DATASET" >&2
  usage
  exit 1
fi

if ! PROFILE_KEY="$(normalize_profile "$PROFILE")"; then
  echo "[ERROR] Unsupported profile: $PROFILE" >&2
  usage
  exit 1
fi

# =======================
# 1) Core run config
# =======================
PYTHON_BIN="${PYTHON_BIN:-}"
ENTRY="run.py"

pick_python_bin() {
  local requested="$1"
  local cand resolved
  local -a candidates=()

  if [[ -n "$requested" ]]; then
    candidates+=("$requested")
  fi
  candidates+=("python" "python3")

  local env_root
  for env_root in "$HOME/miniconda3/envs" "$HOME/anaconda3/envs"; do
    if [[ -d "$env_root" ]]; then
      while IFS= read -r cand; do
        candidates+=("$cand")
      done < <(ls "$env_root"/*/bin/python 2>/dev/null || true)
    fi
  done

  local -A seen=()
  for cand in "${candidates[@]}"; do
    resolved="$(command -v "$cand" 2>/dev/null || true)"
    if [[ -z "$resolved" ]]; then
      continue
    fi
    if [[ -n "${seen[$resolved]:-}" ]]; then
      continue
    fi
    seen[$resolved]=1

    if "$resolved" - <<'PY' >/dev/null 2>&1
import importlib
for m in ["pandas", "torch", "transformers", "peft", "openai"]:
    importlib.import_module(m)
PY
    then
      echo "$resolved"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(pick_python_bin "$PYTHON_BIN")"; then
  echo "[ERROR] No suitable Python found for this project." >&2
  echo "        Need modules: pandas, torch, transformers, peft, openai" >&2
  echo "        You can run with an explicit interpreter, e.g.:" >&2
  echo "        PYTHON_BIN=/path/to/env/bin/python bash scripts/run_tinynews_experiment.sh --dataset nsw_price" >&2
  exit 1
fi
echo "[env] Using PYTHON_BIN=$PYTHON_BIN"

prepare_chronological_timeseries_splits() {
  "$PYTHON_BIN" - <<'PY'
import os
import pandas as pd

raw_file = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine.csv"
source_train = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_trainset.csv"
source_val = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_valset.csv"
source_test = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_testset.csv"
out_train = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_trainset.csv"
out_val = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_valset.csv"
out_test = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_testset.csv"
time_col = "date"

def parse_time_series(series):
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], dayfirst=True, errors="coerce")
    return parsed

for path in (raw_file, source_train, source_val, source_test):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

src_train = pd.read_csv(source_train)
src_val = pd.read_csv(source_val)
src_test = pd.read_csv(source_test)
full_df = pd.read_csv(raw_file)

expected_total = len(src_train) + len(src_val) + len(src_test)
if len(full_df) != expected_total:
    raise ValueError(
        f"NAS_14ticker split source mismatch: full={len(full_df)} split_sum={expected_total}"
    )

parsed = parse_time_series(full_df[time_col])
if parsed.isna().any():
    bad_rows = full_df.loc[parsed.isna(), [time_col]].head(5).to_dict("records")
    raise ValueError(f"Failed to parse {time_col} in raw file. examples={bad_rows}")

full_sorted = full_df.copy()
full_sorted["_parsed_time"] = parsed
full_sorted = full_sorted.sort_values("_parsed_time", kind="mergesort").reset_index(drop=True)
full_sorted = full_sorted.drop(columns=["_parsed_time"])

n_train = len(src_train)
n_val = len(src_val)
train_df = full_sorted.iloc[:n_train].copy()
val_df = full_sorted.iloc[n_train:n_train + n_val].copy()
test_df = full_sorted.iloc[n_train + n_val:].copy()

for path, df in ((out_train, train_df), (out_val, val_df), (out_test, test_df)):
    df.to_csv(path, index=False)

def _span(df):
    ts = parse_time_series(df[time_col])
    return str(ts.min()), str(ts.max()), int(len(df))

tr_min, tr_max, tr_n = _span(train_df)
va_min, va_max, va_n = _span(val_df)
te_min, te_max, te_n = _span(test_df)
print(
    "[DATASET] prepared chronological splits "
    f"train={tr_n}[{tr_min} -> {tr_max}] "
    f"val={va_n}[{va_min} -> {va_max}] "
    f"test={te_n}[{te_min} -> {te_max}]"
)
PY
}

init_common_defaults() {
  NEWS_TEXT_COL="content"
  NEWS_TIME_COL="date"

  DELTA_EPOCHS="${DELTA_EPOCHS:-100}"
  BASE_EPOCHS="${BASE_EPOCHS:-40}"
  NEWS_TOPM="${NEWS_TOPM:-999}"
  NEWS_TOPK="${NEWS_TOPK:-999}"
  GPU_ID="${GPU_ID:-0}"
  DEFAULT_POLICY="${DEFAULT_POLICY:-all}"

  TEMPLATE_POOL_2="configs/deltaWithNews_template.yaml"
  TEMPLATE_POOL="${TEMPLATE_POOL:-$TEMPLATE_POOL_2}"

  RESIDUAL_LOSS="${RESIDUAL_LOSS:-smooth_l1}"
  SELECT_METRIC="${SELECT_METRIC:-mae}"
  STRIDE="${STRIDE:-1}"

  STAGE="${STAGE:-all}"
  PATCH_DROPOUT="${PATCH_DROPOUT:-0}"
  HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"

  DELTA_VAL_MODE="${DELTA_VAL_MODE:-each_epoch}"
  DELTA_CLIP="${DELTA_CLIP:-1.0}"
  EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
  DELTA_SIGN_EPS="${DELTA_SIGN_EPS:-0.03}"

  DELTA_SIGN_TAU="${DELTA_SIGN_TAU:-1.0}"
  DELTA_SIGN_MODE="${DELTA_SIGN_MODE:-signnet_binary}"
  RESIDUAL_ARCH="${RESIDUAL_ARCH:-unified}"
  DELTA_SIGN_EXTERNAL_EPOCHS="${DELTA_SIGN_EXTERNAL_EPOCHS:-80}"
  DELTA_SIGN_EXTERNAL_DROPOUT="${DELTA_SIGN_EXTERNAL_DROPOUT:-0.2}"
  DELTA_SIGN_EXTERNAL_LR="${DELTA_SIGN_EXTERNAL_LR:-3e-4}"
  DELTA_SIGN_EXTERNAL_WEIGHT_DECAY="${DELTA_SIGN_EXTERNAL_WEIGHT_DECAY:-5e-4}"
  DELTA_SIGN_EXTERNAL_GRAD_CLIP="${DELTA_SIGN_EXTERNAL_GRAD_CLIP:-1.0}"
  DELTA_SIGN_EXTERNAL_PATIENCE="${DELTA_SIGN_EXTERNAL_PATIENCE:-5}"
  DELTA_SIGN_EXTERNAL_SELECT_METRIC="${DELTA_SIGN_EXTERNAL_SELECT_METRIC:-balanced_acc}"
  DELTA_SIGN_EXTERNAL_MIN_DELTA="${DELTA_SIGN_EXTERNAL_MIN_DELTA:-1e-4}"
  DELTA_SIGN_EXTERNAL_LR_FACTOR="${DELTA_SIGN_EXTERNAL_LR_FACTOR:-0.5}"
  DELTA_SIGN_EXTERNAL_LR_PATIENCE="${DELTA_SIGN_EXTERNAL_LR_PATIENCE:-1}"
  DELTA_SIGN_EXTERNAL_MIN_LR="${DELTA_SIGN_EXTERNAL_MIN_LR:-1e-5}"
  DELTA_SIGN_EXTERNAL_CALIBRATE_BIAS="${DELTA_SIGN_EXTERNAL_CALIBRATE_BIAS:-1}"
  DELTA_SIGN_EXTERNAL_BIAS_CLIP="${DELTA_SIGN_EXTERNAL_BIAS_CLIP:-1.5}"
  DELTA_SIGN_EXTERNAL_NEWS_DROPOUT="${DELTA_SIGN_EXTERNAL_NEWS_DROPOUT:-0}"
  DELTA_SIGN_EXTERNAL_USE_NEWS_WEIGHTING="${DELTA_SIGN_EXTERNAL_USE_NEWS_WEIGHTING:-0}"
  DELTA_SIGN_EXTERNAL_USE_RESIDUAL_WEIGHTING="${DELTA_SIGN_EXTERNAL_USE_RESIDUAL_WEIGHTING:-0}"
  DELTA_SIGN_EXTERNAL_USE_POS_WEIGHT="${DELTA_SIGN_EXTERNAL_USE_POS_WEIGHT:-1}"
  DELTA_SIGN_EXTERNAL_POS_WEIGHT_FLOOR="${DELTA_SIGN_EXTERNAL_POS_WEIGHT_FLOOR:-0.5}"
  DELTA_SIGN_EXTERNAL_POS_WEIGHT_CLIP="${DELTA_SIGN_EXTERNAL_POS_WEIGHT_CLIP:-3.0}"
  DELTA_SIGN_EXTERNAL_TAU="${DELTA_SIGN_EXTERNAL_TAU:-1.0}"
  DELTA_MAG_TARGET="${DELTA_MAG_TARGET:-log1p}"
  DELTA_MAG_MAX="${DELTA_MAG_MAX:-0.0}"
  DELTA_RESIDUAL_WEIGHT_SCALE="${DELTA_RESIDUAL_WEIGHT_SCALE:-1.0}"

  BASE_HIDDEN_DIM="${BASE_HIDDEN_DIM:-256}"
  BASE_MOVING_AVG="${BASE_MOVING_AVG:-25}"
  BASE_DROPOUT="${BASE_DROPOUT:-0.0}"
  BASE_LOSS="${BASE_LOSS:-smooth_l1}"
  BASE_LR="${BASE_LR:-1e-3}"
  BASE_WEIGHT_DECAY="${BASE_WEIGHT_DECAY:-1e-5}"

  UTILITY_RERANK_ENABLE="${UTILITY_RERANK_ENABLE:-0}"
  UTILITY_KEYWORD_WEIGHT="${UTILITY_KEYWORD_WEIGHT:-0.35}"
  UTILITY_RECENCY_WEIGHT="${UTILITY_RECENCY_WEIGHT:-0.25}"
  UTILITY_RATE_WEIGHT="${UTILITY_RATE_WEIGHT:-0.35}"
  UTILITY_SENTIMENT_WEIGHT="${UTILITY_SENTIMENT_WEIGHT:-0.05}"
  UTILITY_RECENCY_TAU_HOURS="${UTILITY_RECENCY_TAU_HOURS:-6}"
  UTILITY_MMR_ENABLE="${UTILITY_MMR_ENABLE:-1}"
  UTILITY_MMR_LAMBDA="${UTILITY_MMR_LAMBDA:-0.8}"
  UTILITY_DEDUP_THRESHOLD="${UTILITY_DEDUP_THRESHOLD:-0.95}"
  UTILITY_KEEP_TOPK="${UTILITY_KEEP_TOPK:--1}"
  UTILITY_MIN_SCORE="${UTILITY_MIN_SCORE:--1.0}"

  NEWS_API_ENABLE="${NEWS_API_ENABLE:-1}"
  NEWS_API_MODEL="${NEWS_API_MODEL:-gpt-5.1}"
  NEWS_API_KEY_PATH="${NEWS_API_KEY_PATH:-api_key.txt}"
  NEWS_API_BASE_URL="${NEWS_API_BASE_URL:-}"
  NEWS_API_TIMEOUT_SEC="${NEWS_API_TIMEOUT_SEC:-30}"
  NEWS_API_MAX_RETRIES="${NEWS_API_MAX_RETRIES:-2}"
  if [[ "$NEWS_API_ENABLE" == "1" ]]; then
    NEWS_REFINE_MODE="${NEWS_REFINE_MODE:-api}"
    NEWS_STRUCTURED_MODE="${NEWS_STRUCTURED_MODE:-api}"
  else
    NEWS_REFINE_MODE="${NEWS_REFINE_MODE:-local}"
    NEWS_STRUCTURED_MODE="${NEWS_STRUCTURED_MODE:-off}"
  fi
  NEWS_REFINE_CACHE_ENABLE="${NEWS_REFINE_CACHE_ENABLE:-1}"
  NEWS_STRUCTURED_CACHE_ENABLE="${NEWS_STRUCTURED_CACHE_ENABLE:-1}"
  NEWS_DOC_CACHE_PATH="${NEWS_DOC_CACHE_PATH:-}"
  NEWS_REFINE_PREWARM_MAX_BATCHES="${NEWS_REFINE_PREWARM_MAX_BATCHES:--1}"
  NEWS_REFINE_SHOW_PROGRESS="${NEWS_REFINE_SHOW_PROGRESS:-1}"
  NEWS_STRUCTURED_SHOW_PROGRESS="${NEWS_STRUCTURED_SHOW_PROGRESS:-1}"
  DELTA_INCLUDE_STRUCTURED_NEWS="${DELTA_INCLUDE_STRUCTURED_NEWS:-0}"
  DELTA_STRUCTURED_ENABLE="${DELTA_STRUCTURED_ENABLE:-1}"
  DELTA_TEMPORAL_TEXT_SOURCE="${DELTA_TEMPORAL_TEXT_SOURCE:-refined}"
  # DELTA_TEMPORAL_TEXT_DIM="${DELTA_TEMPORAL_TEXT_DIM:-8}"
  # DELTA_TEMPORAL_TEXT_MAX_LEN="${DELTA_TEMPORAL_TEXT_MAX_LEN:-96}"
  DELTA_TEMPORAL_TEXT_FUSE_LAMBDA="${DELTA_TEMPORAL_TEXT_FUSE_LAMBDA:-0.5}"
  DELTA_TEMPORAL_TEXT_FREEZE_ENCODER="${DELTA_TEMPORAL_TEXT_FREEZE_ENCODER:-1}"
  DELTA_MULTIMODAL_ARCH="${DELTA_MULTIMODAL_ARCH:-summary_gated}"  # default to Plan C; set summary_gated to fall back to the older summary/gated path
  DELTA_MULTIMODAL_FUSE_LAMBDA="${DELTA_MULTIMODAL_FUSE_LAMBDA:-1.0}"
  DELTA_MODEL_VARIANT="${DELTA_MODEL_VARIANT:-tiny_news_ts}"
  TINY_NEWS_HIDDEN_SIZE="${TINY_NEWS_HIDDEN_SIZE:-256}"
  DELTA_RELATIVE_DENOM_FLOOR="${DELTA_RELATIVE_DENOM_FLOOR:-0.0}"
  DELTA_RELATIVE_RATIO_CLIP="${DELTA_RELATIVE_RATIO_CLIP:-0.0}"
  DELTA_HEAD_LR_SCALE="${DELTA_HEAD_LR_SCALE:-1.0}"

  RUN_OR_NOT="${RUN_OR_NOT:-1}"
  LOOKBACK_WINDOWS=("1")
  SCHEDULERS=("1")
  GRAD_ACCS=("8")

  ID_COL="${ID_COL:-}"
  FREQ_MIN="${FREQ_MIN:-}"
}

configure_dataset_defaults() {
  case "$DATASET_KEY" in
    nas14)
      if [[ "$PROFILE_KEY" != "default" ]]; then
        echo "[ERROR] dataset=$DATASET_KEY only supports --profile default." >&2
        exit 1
      fi
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
      ALLOW_AUTO_DISCOVER_CACHE=1

      BATCH_SIZE="${BATCH_SIZE:-4}"
      HORIZONS=("48")
      BASE_BACKBONES=("dlinear")
      LRS=("5e-6")
      DELTA_FREEZE_FEATURE_MODULES="${DELTA_FREEZE_FEATURE_MODULES:-0}"
      DELTA_SIGN_EXTERNAL_HIDDEN="${DELTA_SIGN_EXTERNAL_HIDDEN:-192}"
      DELTA_STRUCTURED_FEATURE_DIM="${DELTA_STRUCTURED_FEATURE_DIM:-12}"

      DELTA_TEMPORAL_TEXT_ENABLE="${DELTA_TEMPORAL_TEXT_ENABLE:-1}"
      DELTA_TEMPORAL_TEXT_DIM="${DELTA_TEMPORAL_TEXT_DIM:-64}"
      DELTA_TEMPORAL_TEXT_MAX_LEN="${DELTA_TEMPORAL_TEXT_MAX_LEN:-196}"
      # DELTA_TEMPORAL_TEXT_FREEZE_ENCODER=0
      # DELTA_SIGN_EXTERNAL_USE_NEWS_WEIGHTING=1
      # DELTA_SIGN_EXTERNAL_USE_RESIDUAL_WEIGHTING=1
      DELTA_TEMPORAL_TEXT_FUSE_LAMBDA=1

      TEMPORAL_TEXT_MODEL_ID="${TEMPORAL_TEXT_MODEL_ID:-distilbert-base-uncased}"
      DELTA_TEMPORAL_TEXT_PER_STEP_TOPK="${DELTA_TEMPORAL_TEXT_PER_STEP_TOPK:-10}"
      DELTA_RESIDUAL_MODE="${DELTA_RESIDUAL_MODE:-relative}"
      TASK_NAME_BASE="${TASK_NAME_BASE:-[2022_2023-nasdaq14tickeropen-tinynews]}"
      ;;
    nsw_load)
      if [[ "$PROFILE_KEY" != "default" ]]; then
        echo "[ERROR] dataset=$DATASET_KEY only supports --profile default." >&2
        exit 1
      fi
      TIME_COL="date"
      VALUE_COL="TOTALDEMAND"
      UNIT="MW"
      DESCRIPTION="This dataset records the electricity load demand data in Australia NSW from 2024, collected from National electricity market."
      REGION="Australia, NSW"

      TRAIN_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_trainset.csv"
      VAL_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_valset.csv"
      TEST_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_testset.csv"
      DEFAULT_NEWS_PATH="dataset/watt_free_2024_load.json"
      ALLOW_AUTO_DISCOVER_CACHE=1

      BATCH_SIZE="${BATCH_SIZE:-8}"
      # HORIZONS=("48" "96" "192" "336" "720")
      HORIZONS=("48")
      BASE_BACKBONES=("mlp")
      LRS=("5e-6")
      STRIDE=1

      NEWS_TOPK=10
      delta_text_fuse_mode="gated_add"
      DELTA_MULTIMODAL_ARCH="summary_gated"

      # NEWS_TOPK=3
      DELTA_FREEZE_FEATURE_MODULES="${DELTA_FREEZE_FEATURE_MODULES:-1}"
      DELTA_SIGN_EXTERNAL_HIDDEN="${DELTA_SIGN_EXTERNAL_HIDDEN:-192}"
      DELTA_STRUCTURED_FEATURE_DIM="${DELTA_STRUCTURED_FEATURE_DIM:-12}"

      DELTA_TEMPORAL_TEXT_FUSE_LAMBDA="${DELTA_TEMPORAL_TEXT_FUSE_LAMBDA:-1}"
      #temporal text config
      DELTA_TEMPORAL_TEXT_ENABLE="${DELTA_TEMPORAL_TEXT_ENABLE:-1}"
      DELTA_TEMPORAL_TEXT_DIM="${DELTA_TEMPORAL_TEXT_DIM:-64}"
      DELTA_TEMPORAL_TEXT_MAX_LEN="${DELTA_TEMPORAL_TEXT_MAX_LEN:-196}"
      TEMPORAL_TEXT_MODEL_ID="${TEMPORAL_TEXT_MODEL_ID:-distilbert-base-uncased}"
      DELTA_TEMPORAL_TEXT_PER_STEP_TOPK="${DELTA_TEMPORAL_TEXT_PER_STEP_TOPK:-10}"
      
      DELTA_RESIDUAL_MODE="${DELTA_RESIDUAL_MODE:-relative}"
      TASK_NAME_BASE="${TASK_NAME_BASE:-[LOAD-topk10-gatedadd-summarygated]}"
      ;;
    nsw_price)
      if [[ "$PROFILE_KEY" != "default" ]]; then
        echo "[ERROR] dataset=$DATASET_KEY only supports --profile default." >&2
        exit 1
      fi
      TIME_COL="date"
      VALUE_COL="RRP"
      UNIT=""
      DESCRIPTION="This dataset records the electricity price data in Australia NSW from 2024, collected from National electricity market."
      REGION="Australia, NSW"

      NEWS_TOPK=10
      delta_text_fuse_mode="gated_add"
      DELTA_MULTIMODAL_ARCH="summary_gated"

      TRAIN_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_trainset.csv"
      VAL_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_valset.csv"
      TEST_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_testset.csv"
      DEFAULT_NEWS_PATH="dataset/watt_free_2024_load.json"
      ALLOW_AUTO_DISCOVER_CACHE=1

      BATCH_SIZE="${BATCH_SIZE:-8}"
      
      # HORIZONS=("48" "96" "192" "336" "720")
      HORIZONS=("48")
      BASE_BACKBONES=("mlp")
      LRS=("5e-6")
      DELTA_FREEZE_FEATURE_MODULES="${DELTA_FREEZE_FEATURE_MODULES:-1}"
      DELTA_SIGN_EXTERNAL_HIDDEN="${DELTA_SIGN_EXTERNAL_HIDDEN:-192}"
      DELTA_STRUCTURED_FEATURE_DIM="${DELTA_STRUCTURED_FEATURE_DIM:-12}"

      # DELTA_SIGN_MODE="internal"
      # DELTA_TEMPORAL_TEXT_SOURCE="raw"

      DELTA_TEMPORAL_TEXT_FUSE_LAMBDA="${DELTA_TEMPORAL_TEXT_FUSE_LAMBDA:-1}"
      #temporal text config
      DELTA_TEMPORAL_TEXT_ENABLE="${DELTA_TEMPORAL_TEXT_ENABLE:-1}"
      DELTA_TEMPORAL_TEXT_DIM="${DELTA_TEMPORAL_TEXT_DIM:-64}"
      DELTA_TEMPORAL_TEXT_MAX_LEN="${DELTA_TEMPORAL_TEXT_MAX_LEN:-196}"
      TEMPORAL_TEXT_MODEL_ID="${TEMPORAL_TEXT_MODEL_ID:-distilbert-base-uncased}"
      DELTA_TEMPORAL_TEXT_PER_STEP_TOPK="${DELTA_TEMPORAL_TEXT_PER_STEP_TOPK:-10}"

      DELTA_RESIDUAL_MODE="${DELTA_RESIDUAL_MODE:-relative}"
      TASK_NAME_BASE="${TASK_NAME_BASE:-[Price-distilbert-topk10-gatedadd-summarygated]}"
      ;;
  esac

  NEWS_PATH="${NEWS_PATH:-$DEFAULT_NEWS_PATH}"
  NEWS_CHOICES=("$NEWS_PATH")
}

apply_no_signnet_overrides() {
  if [[ "$NO_SIGNNET" != "1" ]]; then
    return
  fi
  DELTA_SIGN_MODE="internal"
  DELTA_FREEZE_FEATURE_MODULES="${DELTA_FREEZE_FEATURE_MODULES:-0}"
  if [[ -z "${TASK_NAME_SUFFIX:-}" ]]; then
    TASK_NAME_SUFFIX="__nosignnet"
  fi
}

build_common_args() {
  COMMON_ARGS=()
  if [[ -n "$FREQ_MIN" ]]; then
    COMMON_ARGS+=( --freq_min "$FREQ_MIN" )
  fi
  COMMON_ARGS+=(
    --delta_freeze_feature_modules "$DELTA_FREEZE_FEATURE_MODULES"
    --delta_sign_eps "$DELTA_SIGN_EPS"
    --delta_sign_tau "$DELTA_SIGN_TAU"
    --delta_sign_mode "$DELTA_SIGN_MODE"
    --residual_arch "$RESIDUAL_ARCH"
    --delta_sign_external_epochs "$DELTA_SIGN_EXTERNAL_EPOCHS"
    --delta_sign_external_hidden "$DELTA_SIGN_EXTERNAL_HIDDEN"
    --delta_sign_external_dropout "$DELTA_SIGN_EXTERNAL_DROPOUT"
    --delta_sign_external_lr "$DELTA_SIGN_EXTERNAL_LR"
    --delta_sign_external_weight_decay "$DELTA_SIGN_EXTERNAL_WEIGHT_DECAY"
    --delta_sign_external_grad_clip "$DELTA_SIGN_EXTERNAL_GRAD_CLIP"
    --delta_sign_external_patience "$DELTA_SIGN_EXTERNAL_PATIENCE"
    --delta_sign_external_select_metric "$DELTA_SIGN_EXTERNAL_SELECT_METRIC"
    --delta_sign_external_min_delta "$DELTA_SIGN_EXTERNAL_MIN_DELTA"
    --delta_sign_external_lr_factor "$DELTA_SIGN_EXTERNAL_LR_FACTOR"
    --delta_sign_external_lr_patience "$DELTA_SIGN_EXTERNAL_LR_PATIENCE"
    --delta_sign_external_min_lr "$DELTA_SIGN_EXTERNAL_MIN_LR"
    --delta_sign_external_calibrate_bias "$DELTA_SIGN_EXTERNAL_CALIBRATE_BIAS"
    --delta_sign_external_bias_clip "$DELTA_SIGN_EXTERNAL_BIAS_CLIP"
    --delta_sign_external_news_dropout "$DELTA_SIGN_EXTERNAL_NEWS_DROPOUT"
    --delta_sign_external_use_news_weighting "$DELTA_SIGN_EXTERNAL_USE_NEWS_WEIGHTING"
    --delta_sign_external_use_residual_weighting "$DELTA_SIGN_EXTERNAL_USE_RESIDUAL_WEIGHTING"
    --delta_sign_external_use_pos_weight "$DELTA_SIGN_EXTERNAL_USE_POS_WEIGHT"
    --delta_sign_external_pos_weight_floor "$DELTA_SIGN_EXTERNAL_POS_WEIGHT_FLOOR"
    --delta_sign_external_pos_weight_clip "$DELTA_SIGN_EXTERNAL_POS_WEIGHT_CLIP"
    --delta_sign_external_tau "$DELTA_SIGN_EXTERNAL_TAU"
    --delta_mag_target "$DELTA_MAG_TARGET"
    --delta_mag_max "$DELTA_MAG_MAX"
    --delta_residual_weight_scale "$DELTA_RESIDUAL_WEIGHT_SCALE"
    --delta_residual_mode "$DELTA_RESIDUAL_MODE"
    --delta_relative_denom_floor "$DELTA_RELATIVE_DENOM_FLOOR"
    --delta_relative_ratio_clip "$DELTA_RELATIVE_RATIO_CLIP"
    --delta_head_lr_scale "$DELTA_HEAD_LR_SCALE"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --time_col "$TIME_COL"
    --value_col "$VALUE_COL"
  )
  if [[ -n "$ID_COL" ]]; then
    COMMON_ARGS+=( --id_col "$ID_COL" )
  fi
  COMMON_ARGS+=(
    --unit "$UNIT"
    --description "$DESCRIPTION"
    --region "$REGION"
    --train_file "$TRAIN_FILE"
    --val_file "$VAL_FILE"
    --test_file "$TEST_FILE"
    --news_text_col "$NEWS_TEXT_COL"
    --news_time_col "$NEWS_TIME_COL"
    --news_topM "$NEWS_TOPM"
    --news_topK "$NEWS_TOPK"
    --batch_size "$BATCH_SIZE"
    --gpu "$GPU_ID"
    --select_metric "$SELECT_METRIC"
    --default_policy "$DEFAULT_POLICY"
    --utility_rerank_enable "$UTILITY_RERANK_ENABLE"
    --utility_keyword_weight "$UTILITY_KEYWORD_WEIGHT"
    --utility_recency_weight "$UTILITY_RECENCY_WEIGHT"
    --utility_rate_weight "$UTILITY_RATE_WEIGHT"
    --utility_sentiment_weight "$UTILITY_SENTIMENT_WEIGHT"
    --utility_recency_tau_hours "$UTILITY_RECENCY_TAU_HOURS"
    --utility_mmr_enable "$UTILITY_MMR_ENABLE"
    --utility_mmr_lambda "$UTILITY_MMR_LAMBDA"
    --utility_dedup_threshold "$UTILITY_DEDUP_THRESHOLD"
    --utility_keep_topk "$UTILITY_KEEP_TOPK"
    --utility_min_score "$UTILITY_MIN_SCORE"
    --news_refine_mode "$NEWS_REFINE_MODE"
    --news_refine_cache_enable "$NEWS_REFINE_CACHE_ENABLE"
    --news_structured_cache_enable "$NEWS_STRUCTURED_CACHE_ENABLE"
    --news_refine_show_progress "$NEWS_REFINE_SHOW_PROGRESS"
    --news_structured_show_progress "$NEWS_STRUCTURED_SHOW_PROGRESS"
    --news_api_model "$NEWS_API_MODEL"
    --news_api_key_path "$NEWS_API_KEY_PATH"
    --news_api_base_url "$NEWS_API_BASE_URL"
    --news_api_timeout_sec "$NEWS_API_TIMEOUT_SEC"
    --news_api_max_retries "$NEWS_API_MAX_RETRIES"
    --delta_include_structured_news "$DELTA_INCLUDE_STRUCTURED_NEWS"
    --news_structured_mode "$NEWS_STRUCTURED_MODE"
    --delta_structured_enable "$DELTA_STRUCTURED_ENABLE"
    --delta_structured_feature_dim "$DELTA_STRUCTURED_FEATURE_DIM"
    --delta_temporal_text_enable "$DELTA_TEMPORAL_TEXT_ENABLE"
    --temporal_text_model_id "$TEMPORAL_TEXT_MODEL_ID"
    --delta_temporal_text_source "$DELTA_TEMPORAL_TEXT_SOURCE"
    --delta_temporal_text_dim "$DELTA_TEMPORAL_TEXT_DIM"
    --delta_temporal_text_max_len "$DELTA_TEMPORAL_TEXT_MAX_LEN"
    --delta_temporal_text_per_step_topk "$DELTA_TEMPORAL_TEXT_PER_STEP_TOPK"
    --delta_temporal_text_fuse_lambda "$DELTA_TEMPORAL_TEXT_FUSE_LAMBDA"
    --delta_temporal_text_freeze_encoder "$DELTA_TEMPORAL_TEXT_FREEZE_ENCODER"
    --delta_multimodal_arch "$DELTA_MULTIMODAL_ARCH"
    --delta_multimodal_fuse_lambda "$DELTA_MULTIMODAL_FUSE_LAMBDA"
    --delta_model_variant "$DELTA_MODEL_VARIANT"
    --tiny_news_hidden_size "$TINY_NEWS_HIDDEN_SIZE"
    --residual_loss "$RESIDUAL_LOSS"
    --stage "$STAGE"
    --delta_val_mode "$DELTA_VAL_MODE"
    --delta_clip "$DELTA_CLIP"
  )
}

normalize_cache_stem() {
  local raw="$1"
  local stem
  stem="$(basename "$raw")"
  stem="${stem%.*}"
  stem="$(printf '%s' "$stem" | sed -E 's/[^0-9A-Za-z]+/_/g; s/^_+//; s/_+$//' | tr '[:upper:]' '[:lower:]')"
  if [[ -z "$stem" ]]; then
    stem="news"
  fi
  printf '%s\n' "$stem"
}

verify_news_doc_cache_file() {
  local news_path="$1"
  local cache_path="$2"
  local verify_output
  if [[ ! -f "$cache_path" ]]; then
    return 1
  fi
  if verify_output="$("$PYTHON_BIN" scripts/verify_refined_news_cache.py \
    --news "$news_path" \
    --cache "$cache_path" \
    --allow-missing 2>&1)"; then
    if [[ -n "$verify_output" ]]; then
      printf '%s\n' "$verify_output" >&2
    fi
    return 0
  fi
  printf '%s\n' "$verify_output" >&2
  return 1
}

resolve_news_doc_cache_write_path() {
  local news_path="$1"
  if [[ -n "$NEWS_DOC_CACHE_PATH" ]]; then
    printf '%s\n' "$NEWS_DOC_CACHE_PATH"
    return
  fi
  local stem
  stem="$(normalize_cache_stem "$news_path")"
  printf 'checkpoints/_shared_refine_cache/news_doc_cache_%s.json\n' "$stem"
}

prepare_news_cache_mode() {
  local news_path="$1"
  local stem doc_cache_path legacy_refine_path legacy_structured_path
  stem="$(normalize_cache_stem "$news_path")"
  doc_cache_path="$(resolve_news_doc_cache_write_path "$news_path")"
  legacy_refine_path="checkpoints/_shared_refine_cache/refine_news_cache_${stem}.json"
  legacy_structured_path="checkpoints/_shared_refine_cache/structured_news_cache_${stem}.json"

  CURRENT_NEWS_DOC_CACHE_PATH="$doc_cache_path"
  CURRENT_NEWS_REFINE_CACHE_READ_PATH=""
  CURRENT_NEWS_STRUCTURED_CACHE_READ_PATH=""
  CURRENT_NEWS_REFINE_PREWARM=1
  CURRENT_NEWS_STRUCTURED_PREWARM=1
  CURRENT_NEWS_DOC_CACHE_EXPLICIT=0
  CURRENT_NEWS_CACHE_MODE="build_mode"

  if [[ -n "$NEWS_DOC_CACHE_PATH" ]]; then
    CURRENT_NEWS_REFINE_CACHE_READ_PATH="$doc_cache_path"
    CURRENT_NEWS_STRUCTURED_CACHE_READ_PATH="$doc_cache_path"
    CURRENT_NEWS_REFINE_PREWARM=0
    CURRENT_NEWS_STRUCTURED_PREWARM=0
    CURRENT_NEWS_DOC_CACHE_EXPLICIT=1
    CURRENT_NEWS_CACHE_MODE="read_only"
    echo "[NEWS_CACHE] mode=read_only cache_path=$doc_cache_path source=explicit"
    if [[ ! -f "$doc_cache_path" ]]; then
      echo "[NEWS_CACHE][ERROR] explicit cache path does not exist: $doc_cache_path" >&2
      return 1
    fi
    if ! verify_news_doc_cache_file "$news_path" "$doc_cache_path"; then
      echo "[NEWS_CACHE][ERROR] explicit cache path is incompatible with current framework identity rules: $doc_cache_path" >&2
      echo "[NEWS_CACHE][ERROR] Repair the cache or point NEWS_DOC_CACHE_PATH to a validated cache file." >&2
      return 1
    fi
    return 0
  fi

  if [[ "$ALLOW_AUTO_DISCOVER_CACHE" == "1" && -f "$doc_cache_path" ]]; then
    if verify_news_doc_cache_file "$news_path" "$doc_cache_path"; then
      CURRENT_NEWS_REFINE_CACHE_READ_PATH="$doc_cache_path"
      CURRENT_NEWS_STRUCTURED_CACHE_READ_PATH="$doc_cache_path"
      CURRENT_NEWS_REFINE_PREWARM=0
      CURRENT_NEWS_STRUCTURED_PREWARM=0
      CURRENT_NEWS_CACHE_MODE="read_only"
      echo "[NEWS_CACHE] mode=read_only cache_path=$doc_cache_path source=auto_discovered"
      return 0
    fi

    echo "[NEWS_CACHE][ERROR] auto-discovered cache is incompatible with current framework identity rules: $doc_cache_path" >&2
    echo "[NEWS_CACHE][ERROR] Repair or delete this cache file before rerunning." >&2
    return 1
  fi

  mkdir -p "$(dirname "$doc_cache_path")"
  if [[ -f "$legacy_refine_path" ]]; then
    rm -f "$legacy_refine_path"
  fi
  if [[ -f "$legacy_structured_path" ]]; then
    rm -f "$legacy_structured_path"
  fi

  echo "[NEWS_CACHE] mode=build_mode news_path=$news_path"
  echo "[NEWS_CACHE] target_doc_cache=$doc_cache_path"
  echo "[NEWS_CACHE] cleared legacy_refine_cache=$legacy_refine_path"
  echo "[NEWS_CACHE] cleared legacy_structured_cache=$legacy_structured_path"
}

init_common_defaults
configure_dataset_defaults
apply_no_signnet_overrides

TASK_NAME_SUFFIX="${TASK_NAME_SUFFIX:-}"
TASK_NAME="${TASK_NAME_BASE}${TASK_NAME_SUFFIX}"
build_common_args

if [[ "$DELTA_TEMPORAL_TEXT_ENABLE" == "1" ]]; then
  TEMPORAL_TEXT_NAME_TAG="DELTA_TEMPORAL_TEXT_on_${DELTA_TEMPORAL_TEXT_SOURCE}"
else
  TEMPORAL_TEXT_NAME_TAG="DELTA_TEMPORAL_TEXT_off"
fi

if [[ "$DATASET_KEY" == "nas14" ]]; then
  prepare_chronological_timeseries_splits
fi

echo "[config] dataset=$DATASET_KEY profile=$PROFILE_KEY no_signnet=$NO_SIGNNET residual_arch=$RESIDUAL_ARCH task_name=$TASK_NAME temporal_text_tag=$TEMPORAL_TEXT_NAME_TAG"

# =======================
# 4) Run combinations
# =======================
for news_path in "${NEWS_CHOICES[@]}"; do
  for lookback_window in "${LOOKBACK_WINDOWS[@]}"; do
    for base_backbone in "${BASE_BACKBONES[@]}"; do
      for lr in "${LRS[@]}"; do
        for sch in "${SCHEDULERS[@]}"; do
          for grad_acc in "${GRAD_ACCS[@]}"; do
            for horizon in "${HORIZONS[@]}"; do
              if [[ "$RUN_OR_NOT" != "1" ]]; then
                continue
              fi
              run_task="${TASK_NAME}_${base_backbone}_h${horizon}"
              display_task="${run_task}_all_s${STRIDE}_h${horizon}_news_${news_path}_${TEMPORAL_TEXT_NAME_TAG}_RESIDUAL_ARCH_${RESIDUAL_ARCH}_DELTA_RESIDUAL_MODE_${DELTA_RESIDUAL_MODE}"
              if [[ "$RESIDUAL_ARCH" == "current" ]]; then
                display_task="${display_task}_DELTA_SIGN_MODE_${DELTA_SIGN_MODE}"
              fi

              prepare_news_cache_mode "$news_path" || exit 1
              args=( --taskName "$run_task" "${COMMON_ARGS[@]}" )

              args+=( --news_api_enable "$NEWS_API_ENABLE" )
              args+=( --news_doc_cache_path "$CURRENT_NEWS_DOC_CACHE_PATH" )
              args+=( --news_doc_cache_explicit "$CURRENT_NEWS_DOC_CACHE_EXPLICIT" )
              args+=( --news_refine_cache_path "$CURRENT_NEWS_DOC_CACHE_PATH" )
              args+=( --news_refine_cache_read_path "$CURRENT_NEWS_REFINE_CACHE_READ_PATH" )
              args+=( --news_structured_cache_path "$CURRENT_NEWS_DOC_CACHE_PATH" )
              args+=( --news_structured_cache_read_path "$CURRENT_NEWS_STRUCTURED_CACHE_READ_PATH" )
              args+=( --news_refine_prewarm "$CURRENT_NEWS_REFINE_PREWARM" )
              args+=( --news_refine_prewarm_max_batches "$NEWS_REFINE_PREWARM_MAX_BATCHES" )
              args+=( --news_structured_prewarm "$CURRENT_NEWS_STRUCTURED_PREWARM" )
              args+=( --news_path "$news_path" )
              if [[ -n "$TEMPLATE_POOL" ]]; then
                args+=( --template_pool "$TEMPLATE_POOL" )
              fi

              args+=( --news_window_days "$lookback_window" )
              args+=( --head_mlp )
              args+=( --patch_dropout "$PATCH_DROPOUT" )
              args+=( --head_dropout "$HEAD_DROPOUT" )
              args+=( --stride "$STRIDE" )
              args+=( --horizon "$horizon" )
              args+=( --base_epochs "$BASE_EPOCHS" )
              args+=( --delta_epochs "$DELTA_EPOCHS" )
              args+=( --base_backbone "$base_backbone" )
              args+=( --base_hidden_dim "$BASE_HIDDEN_DIM" )
              args+=( --base_moving_avg "$BASE_MOVING_AVG" )
              args+=( --base_dropout "$BASE_DROPOUT" )
              args+=( --base_loss "$BASE_LOSS" )
              args+=( --base_lr "$BASE_LR" )
              args+=( --base_weight_decay "$BASE_WEIGHT_DECAY" )
              args+=( --grad_accum "$grad_acc" )
              args+=( --lr "$lr" )
              args+=( --scheduler "$sch" )
              if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
                args+=( "${EXTRA_RUN_ARGS[@]}" )
              fi

              echo "==> Running: ${display_task} (dataset=${DATASET_KEY}, profile=${PROFILE_KEY}, base_backbone=${base_backbone}, horizon=${horizon})"
              "$PYTHON_BIN" "$ENTRY" "${args[@]}"

              # Optional cleanup for disk space after each run:
              # rm -rf "checkpoints/${run_task}/best_base_${run_task}"
              # rm -rf "checkpoints/${run_task}/best_delta_${run_task}"
            done
          done
        done
      done
    done
  done
done
