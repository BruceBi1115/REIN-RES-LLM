#!/usr/bin/env bash

run_time_mmd_domain() {
  local time_mmd_script_dir repo_root common_runner converter
  time_mmd_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd -- "$time_mmd_script_dir/../.." && pwd)"
  common_runner="$time_mmd_script_dir/../_run_forecast_common.sh"
  converter="$time_mmd_script_dir/prepare_time_mmd.py"

  : "${TIME_MMD_DOMAIN:?TIME_MMD_DOMAIN is required}"
  : "${TIME_MMD_STEM:?TIME_MMD_STEM is required}"
  : "${TIME_MMD_SLUG:?TIME_MMD_SLUG is required}"
  : "${TIME_MMD_DESCRIPTION:?TIME_MMD_DESCRIPTION is required}"
  : "${TIME_MMD_REGION:?TIME_MMD_REGION is required}"
  : "${TIME_MMD_UNIT:?TIME_MMD_UNIT is required}"
  : "${TIME_MMD_SCHEMA_VARIANT:?TIME_MMD_SCHEMA_VARIANT is required}"

  SCRIPT_DIR="$time_mmd_script_dir"
  REPO_ROOT="$repo_root"
  RUN_SCRIPT_NAME="scripts/Time-MMD-related/run_${TIME_MMD_SLUG}.sh"
  RUN_SCRIPT_PATH="$time_mmd_script_dir/run_${TIME_MMD_SLUG}.sh"

  DATASET_KEY="time_mmd_${TIME_MMD_SLUG}"
  if [[ "$TIME_MMD_SCHEMA_VARIANT" != "$DATASET_KEY" ]]; then
    echo "[ERROR] $TIME_MMD_DOMAIN requires refinement schema $DATASET_KEY, got $TIME_MMD_SCHEMA_VARIANT" >&2
    return 1
  fi
  TIME_COL="date"
  VALUE_COL="OT"
  ID_COL=""
  UNIT="$TIME_MMD_UNIT"
  DESCRIPTION="$TIME_MMD_DESCRIPTION"
  REGION="$TIME_MMD_REGION"
  FREQ_MIN=""
  DAY_FIRST="0"

  local numerical_dir="dataset/Time-MMD-numerical/$TIME_MMD_DOMAIN"
  local numerical_file="$numerical_dir/${TIME_MMD_STEM}.csv"
  TRAIN_FILE="$numerical_dir/${TIME_MMD_STEM}_trainset.csv"
  VAL_FILE="$numerical_dir/${TIME_MMD_STEM}_valset.csv"
  TEST_FILE="$numerical_dir/${TIME_MMD_STEM}_testset.csv"
  DEFAULT_NEWS_PATH="dataset/Time-MMD-text/$TIME_MMD_DOMAIN/${TIME_MMD_STEM}.json"

  STAGE="${STAGE:-all}"
  BASE_EPOCHS="${BASE_EPOCHS:-40}"
  DELTA_EPOCHS="${DELTA_EPOCHS:-30}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  GRAD_ACCS=("${GRAD_ACC:-4}")
  LRS=("${LR:-1e-4}")
  SCHEDULERS=("${SCHEDULER:-1}")
  BASE_BACKBONES=("${BASE_BACKBONE:-dlinear}")
  HORIZONS=( ${TIME_MMD_HORIZONS} )
  HISTORY_LEN="$TIME_MMD_HISTORY_LEN"
  PATCH_LEN="$TIME_MMD_PATCH_LEN"
  PATCH_STRIDE="$TIME_MMD_PATCH_STRIDE"
  NEWS_WINDOW_DAYS="$TIME_MMD_NEWS_WINDOW_DAYS"

  TASK_NAME_BASE="time_mmd_${TIME_MMD_SLUG}"
  EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
  NORMALIZATION_MODE="${NORMALIZATION_MODE:-robust_quantile}"
  NEWS_API_ENABLE="${NEWS_API_ENABLE:-1}"
  DELTA_V3_SCHEMA_VARIANT="$TIME_MMD_SCHEMA_VARIANT"
  DELTA_V3_REFINED_BANK_BUILD="${DELTA_V3_REFINED_BANK_BUILD:-1}"
  DELTA_V3_ACTIVE_MASS_THRESHOLDS=("${DELTA_V3_ACTIVE_MASS_THRESHOLD:-0.2}")
  DELTA_V3_TEXT_ENCODER_MODEL_ID="${DELTA_V3_TEXT_ENCODER_MODEL_ID:-intfloat/e5-small-v2}"
  DELTA_V3_REGIME_TAU_DAYS="$TIME_MMD_REGIME_TAU_DAYS"
  DELTA_V3_REGIME_EMA_ALPHA="${DELTA_V3_REGIME_EMA_ALPHA:-0.5}"
  DELTA_V3_REGIME_EMA_WINDOW="${DELTA_V3_REGIME_EMA_WINDOW:-5}"
  DELTA_V3_REGIME_BANK_PATH=""

  prepare_time_mmd_domain() {
    echo "[Time-MMD][REFINE] domain=$TIME_MMD_DOMAIN schema_variant=$DELTA_V3_SCHEMA_VARIANT"
    local -a required_outputs=(
      "$numerical_file"
      "$TRAIN_FILE"
      "$VAL_FILE"
      "$TEST_FILE"
      "$DEFAULT_NEWS_PATH"
    )
    local -a missing_outputs=()
    local output_path
    for output_path in "${required_outputs[@]}"; do
      if [[ ! -s "$output_path" ]]; then
        missing_outputs+=("$output_path")
      fi
    done

    if [[ "${TIME_MMD_FORCE_PREPARE:-0}" != "1" && ${#missing_outputs[@]} -eq 0 ]]; then
      echo "[Time-MMD][DATA] processed files already exist; skipping source conversion"
      return 0
    fi

    if [[ "${TIME_MMD_FORCE_PREPARE:-0}" == "1" ]]; then
      echo "[Time-MMD][DATA] TIME_MMD_FORCE_PREPARE=1; rebuilding processed files"
    else
      echo "[Time-MMD][DATA] missing processed files; rebuilding from dataset/Time-MMD/$TIME_MMD_DOMAIN"
      printf '  missing: %s\n' "${missing_outputs[@]}"
    fi
    "$PYTHON_BIN" "$converter" --domain "$TIME_MMD_DOMAIN"
  }
  PRE_RUN_HOOK="prepare_time_mmd_domain"

  cd "$repo_root"
  source "$common_runner"
  run_forecast_dataset_main "$@"
}
