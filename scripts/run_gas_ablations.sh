#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT_NAME="scripts/run_gas_ablations.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_gas_ablations.sh"

# Examples:
#   HORIZONS_CSV=48 ABLATIONS_CSV=no_active_threshold bash scripts/run_gas_ablations.sh
#   HORIZONS_CSV=48,96,192,336,720 \
#   ABLATIONS_CSV=no_active_threshold,high_active_threshold,no_slow_aux,no_shape_aux,no_spike_aux \
#   bash scripts/run_gas_ablations.sh
#   HORIZONS_CSV=48,96,192,336,720 ABLATIONS_CSV=no_regime_vec,no_confidence_weighting bash scripts/run_gas_ablations.sh

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
DEFAULT_REGIME_BANK_PATH="_shared_refine_cache/v4/regime_bank_gdelt_gas_nl_2025.npz"
DEFAULT_REFINED_JSONL_PATH="_shared_refine_cache/v4/refined_gdelt_gas_nl_2025.jsonl"
ABLATION_BANK_DIR="_shared_refine_cache/v4/ablations"

STAGE="${STAGE:-all}"
BASE_EPOCHS="${BASE_EPOCHS:-40}"
DELTA_EPOCHS="${DELTA_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCS=("4")
LRS=("5e-6")
HORIZONS=("48" "96" "192" "336" "720")
SCHEDULERS=("1")
BASE_BACKBONES=("dlinear")

DELTA_V3_REFINED_BANK_BUILD="0"
DELTA_V3_ACTIVE_MASS_THRESHOLDS=("0.2")
NORMALIZATION_MODE="zscore"
DELTA_V3_TEXT_ENCODER_MODEL_ID="intfloat/e5-small-v2"

TASK_NAME_BASE="delta_v3_gas_demand_ablation"
PRE_RUN_HOOK=""
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"

NEWS_API_ENABLE="1"
DELTA_V3_SCHEMA_VARIANT="gas_demand"
DELTA_V3_REGIME_BANK_PATH=""

if ! declare -p ABLATIONS >/dev/null 2>&1; then
  ABLATIONS=(
    "no_active_threshold"
    "high_active_threshold"
    "no_slow_aux"
    "no_shape_aux"
    "no_spike_aux"
  )
fi

csv_to_array() {
  local csv="$1"
  local array_name="$2"
  local -n out_ref="$array_name"
  IFS=',' read -r -a out_ref <<< "$csv"
}

if [[ -n "${HORIZONS_CSV:-}" ]]; then
  csv_to_array "$HORIZONS_CSV" HORIZONS
fi
if [[ -n "${ABLATIONS_CSV:-}" ]]; then
  csv_to_array "$ABLATIONS_CSV" ABLATIONS
fi
if [[ -n "${LRS_CSV:-}" ]]; then
  csv_to_array "$LRS_CSV" LRS
fi
if [[ -n "${GRAD_ACCS_CSV:-}" ]]; then
  csv_to_array "$GRAD_ACCS_CSV" GRAD_ACCS
fi

if [[ -z "${DELTA_V3_REFINED_BANK_BUILD:-}" ]]; then
  NEWS_PATH="${NEWS_PATH:-$DEFAULT_NEWS_PATH}"
  NEWS_CACHE_TAG="$(basename -- "${NEWS_PATH%.json}")"
  if [[ -f "_shared_refine_cache/v4/regime_bank_${NEWS_CACHE_TAG}.npz" ]]; then
    DELTA_V3_REFINED_BANK_BUILD="0"
  else
    DELTA_V3_REFINED_BANK_BUILD="1"
  fi
fi

source "$SCRIPT_DIR/_run_forecast_common.sh"

make_bank_if_needed() {
  local mode="$1"
  local source_bank="${DELTA_V3_SOURCE_REGIME_BANK_PATH:-$DEFAULT_REGIME_BANK_PATH}"
  local output_bank="$ABLATION_BANK_DIR/gas_demand_${mode}.npz"

  if [[ ! -f "$source_bank" ]]; then
    echo "[ERROR] Source regime bank does not exist: $source_bank" >&2
    echo "        Run the normal gas script once or set DELTA_V3_SOURCE_REGIME_BANK_PATH." >&2
    exit 1
  fi
  if [[ ! -f "$output_bank" ]]; then
    local refined_jsonl="${DELTA_V3_SOURCE_REFINED_JSONL_PATH:-$DEFAULT_REFINED_JSONL_PATH}"
    local extra_args=()
    if [[ "$mode" == "no_confidence_weighting" ]]; then
      if [[ ! -f "$refined_jsonl" ]]; then
        echo "[ERROR] Refined JSONL does not exist: $refined_jsonl" >&2
        echo "        Set DELTA_V3_SOURCE_REFINED_JSONL_PATH or rebuild the normal refined cache first." >&2
        exit 1
      fi
      extra_args+=( --refined-jsonl "$refined_jsonl" )
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/make_regime_bank_ablation.py" \
      --source "$source_bank" \
      --output "$output_bank" \
      --mode "$mode" \
      "${extra_args[@]}" >&2
  fi
  printf "%s" "$output_bank"
}

reset_ablation_params() {
  DELTA_V3_REGIME_BANK_PATH="${DELTA_V3_SOURCE_REGIME_BANK_PATH:-$DEFAULT_REGIME_BANK_PATH}"
  DELTA_V3_ACTIVE_MASS_THRESHOLD="0.2"
  DELTA_V3_SLOW_WEIGHT="1.0"
  DELTA_V3_SHAPE_WEIGHT="1.0"
  DELTA_V3_SPIKE_WEIGHT="1.0"
  DELTA_V3_SPIKE_GATE_LOSS_WEIGHT="0.25"
  TASK_NAME_SUFFIX=""
}

apply_ablation() {
  local ablation="$1"
  reset_ablation_params
  TASK_NAME_SUFFIX="__${ablation}"

  case "$ablation" in
    no_active_threshold)
      DELTA_V3_ACTIVE_MASS_THRESHOLD="-1.0"
      ;;
    high_active_threshold)
      DELTA_V3_ACTIVE_MASS_THRESHOLD="1.5"
      ;;
    no_slow_aux)
      DELTA_V3_SLOW_WEIGHT="0"
      ;;
    no_shape_aux)
      DELTA_V3_SHAPE_WEIGHT="0"
      ;;
    no_spike_aux)
      DELTA_V3_SPIKE_WEIGHT="0"
      DELTA_V3_SPIKE_GATE_LOSS_WEIGHT="0"
      ;;
    no_regime_vec|no_topic_mass|no_text_emb|no_relevance_mass|no_in_force_doc_count|no_confidence_weighting)
      DELTA_V3_REGIME_BANK_PATH="$(make_bank_if_needed "$ablation")"
      ;;
    *)
      echo "[ERROR] Unknown ablation: $ablation" >&2
      exit 1
      ;;
  esac
}

run_gas_ablations_main() {
  parse_forecast_runner_args "$@"
  maybe_run_in_tmux
  init_common_defaults
  validate_dataset_config

  PYTHON_BIN="${PYTHON_BIN:-}"
  if ! PYTHON_BIN="$(pick_python_bin "$PYTHON_BIN")"; then
    echo "[ERROR] No suitable Python found for this project." >&2
    exit 1
  fi
  echo "[env] Using PYTHON_BIN=$PYTHON_BIN"

  NEWS_PATH="${NEWS_PATH:-$DEFAULT_NEWS_PATH}"
  set_array_default NEWS_CHOICES "$NEWS_PATH"

  echo "[config] dataset=$DATASET_KEY stage=$STAGE ablations=${ABLATIONS[*]}"
  for ablation in "${ABLATIONS[@]}"; do
    apply_ablation "$ablation"
    echo "==> Ablation: $ablation"
    echo "    active_mass_threshold=$DELTA_V3_ACTIVE_MASS_THRESHOLD"

    for news_path in "${NEWS_CHOICES[@]}"; do
      for base_backbone in "${BASE_BACKBONES[@]}"; do
        for lr in "${LRS[@]}"; do
          for scheduler in "${SCHEDULERS[@]}"; do
            for grad_acc in "${GRAD_ACCS[@]}"; do
              for horizon in "${HORIZONS[@]}"; do
                local task_name
                task_name="${TASK_NAME_BASE}__lr${lr}__ga${grad_acc}__sch${scheduler}${TASK_NAME_SUFFIX}"
                build_run_args "$task_name" "$base_backbone" "$horizon" "$lr" "$scheduler" "$grad_acc" "$news_path"
                if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
                  RUN_ARGS+=( "${EXTRA_RUN_ARGS[@]}" )
                fi
                echo "==> Running ablation=$ablation horizon=$horizon backbone=$base_backbone lr=$lr grad_acc=$grad_acc stage=$STAGE"
                "$PYTHON_BIN" "$ENTRY" "${RUN_ARGS[@]}"
              done
            done
          done
        done
      done
    done
  done
}

run_gas_ablations_main "$@"
