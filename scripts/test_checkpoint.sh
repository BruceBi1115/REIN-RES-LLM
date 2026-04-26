#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/_run_forecast_common.sh"

PYTHON_BIN="${PYTHON_BIN:-}"
STAGE="${STAGE:-all}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-}"
DELTA_CHECKPOINT_PATH="${DELTA_CHECKPOINT_PATH:-}"

DATASET_KEY="${DATASET_KEY:-nsw_price}"
TRAIN_FILE="${TRAIN_FILE:-dataset/2024NSWelecPRICE/2024NSWelecPRICE_trainset.csv}"
VAL_FILE="${VAL_FILE:-dataset/2024NSWelecPRICE/2024NSWelecPRICE_valset.csv}"
TEST_FILE="${TEST_FILE:-dataset/2024NSWelecPRICE/2024NSWelecPRICE_testset.csv}"
TIME_COL="${TIME_COL:-date}"
VALUE_COL="${VALUE_COL:-RRP}"
DESCRIPTION="${DESCRIPTION:-This dataset records the electricity price data in Australia NSW from 2024, collected from the National Electricity Market.}"
REGION="${REGION:-Australia, NSW}"
UNIT="${UNIT:-}"
FREQ_MIN="${FREQ_MIN:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TASK_NAME="${TASK_NAME:-checkpoint_test}"
SAVE_DIR="${SAVE_DIR:-./checkpoints}"
GPU_ID="${GPU_ID:-0}"
DAY_FIRST="${DAY_FIRST:-1}"

cd "$REPO_ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rein-res-llm-matplotlib}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
mkdir -p "$MPLCONFIGDIR"

if ! PYTHON_BIN="$(pick_python_bin "$PYTHON_BIN")"; then
  echo "[ERROR] No suitable Python found for this project." >&2
  echo "        Need modules: matplotlib, openai, pandas, torch, transformers" >&2
  echo "        Or run with PYTHON_BIN=/path/to/env/bin/python bash scripts/test_checkpoint.sh" >&2
  exit 1
fi

args=(
  --stage "$STAGE"
  --taskName "$TASK_NAME"
  --save_dir "$SAVE_DIR"
  --gpu "$GPU_ID"
  --dataset_key "$DATASET_KEY"
  --train_file "$TRAIN_FILE"
  --val_file "$VAL_FILE"
  --test_file "$TEST_FILE"
  --time_col "$TIME_COL"
  --value_col "$VALUE_COL"
  --unit "$UNIT"
  --description "$DESCRIPTION"
  --region "$REGION"
  --freq_min "$FREQ_MIN"
  --batch_size "$BATCH_SIZE"
)

if [[ "$DAY_FIRST" == "1" ]]; then
  args+=(--dayFirst)
fi
if [[ -n "$CHECKPOINT_DIR" ]]; then
  args+=(--checkpoint_dir "$CHECKPOINT_DIR")
fi
if [[ -n "$BASE_CHECKPOINT_PATH" ]]; then
  args+=(--base_checkpoint_path "$BASE_CHECKPOINT_PATH")
fi
if [[ -n "$DELTA_CHECKPOINT_PATH" ]]; then
  args+=(--delta_checkpoint_path "$DELTA_CHECKPOINT_PATH")
fi

"$PYTHON_BIN" src/test_from_checkpoint.py "${args[@]}" "$@"
