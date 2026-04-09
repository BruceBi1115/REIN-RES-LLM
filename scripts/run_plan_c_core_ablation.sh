#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_plan_c_core_ablation.sh --dataset <nas14|nsw_load|nsw_price> [--profile <default|relative_text|additive>] [--ablation <name[,name...]>] [--list] [-- <extra run.py args>]

Examples:
  bash scripts/run_plan_c_core_ablation.sh --dataset nsw_load
  bash scripts/run_plan_c_core_ablation.sh --dataset nsw_price --ablation full_internal,base_only,no_route_regularization
  bash scripts/run_plan_c_core_ablation.sh --dataset nsw_load --ablation all -- --select_metric mae

Ablations:
  full_internal           Plan C baseline with internal sign
  base_only               Run only the BASE stage
  no_temporal_text        Disable the temporal-text branch
  no_structured           Disable structured news vectors
  no_news_features        Disable both temporal-text and structured vectors
  no_cleaned_residual     Disable cleaned-residual supervision
  no_route_regularization Disable Plan C route balancing / abstain regularization
  no_news_dropout         Disable training-time news dropout
  summary_gated           Replace Plan C router/expert path with summary_gated
  all                     Run all of the above

Notes:
  - This wrapper does not modify source code. It only toggles existing env/CLI parameters.
  - It defaults to internal sign (`--no-signnet`), `HORIZONS_OVERRIDE=48`, and `LRS_OVERRIDE=1e-5`.
  - Each ablation appends a task suffix like `__coreabl_no_temporal_text` to avoid collisions.
EOF
}

list_ablations() {
  cat <<'EOF'
full_internal
base_only
no_temporal_text
no_structured
no_news_features
no_cleaned_residual
no_route_regularization
no_news_dropout
summary_gated
all
EOF
}

DATASET="${DATASET:-}"
PROFILE="${PROFILE:-default}"
ABLATION_SPEC="${ABLATION_SPEC:-all}"
EXTRA_RUN_ARGS=()
SHOW_LIST=0

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
    --ablation|--ablations)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --ablation requires a value." >&2
        usage
        exit 1
      fi
      ABLATION_SPEC="$2"
      shift 2
      ;;
    --list|-l)
      SHOW_LIST=1
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

if [[ "$SHOW_LIST" == "1" ]]; then
  list_ablations
  exit 0
fi

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

normalize_ablation() {
  local raw="$1"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | tr '-' '_' )"
  case "$raw" in
    full|full_internal|baseline|plan_c)
      printf 'full_internal\n'
      ;;
    base|base_only)
      printf 'base_only\n'
      ;;
    no_temporal_text|no_text|notext)
      printf 'no_temporal_text\n'
      ;;
    no_structured|nostructured|no_struct)
      printf 'no_structured\n'
      ;;
    no_news_features|nonewsfeatures|no_news_branches)
      printf 'no_news_features\n'
      ;;
    no_cleaned_residual|nocleaned|raw_residual_target)
      printf 'no_cleaned_residual\n'
      ;;
    no_route_regularization|noroutereg|no_route_reg|no_plan_c_reg)
      printf 'no_route_regularization\n'
      ;;
    no_news_dropout|nodropout|no_dropout)
      printf 'no_news_dropout\n'
      ;;
    summary_gated|legacy|legacy_summary_gated)
      printf 'summary_gated\n'
      ;;
    all)
      printf 'all\n'
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

expand_ablations() {
  local spec="$1"
  local token norm
  local -A seen=()
  local -a result=()
  local IFS=','
  read -r -a tokens <<< "$spec"

  if [[ ${#tokens[@]} -eq 0 ]]; then
    tokens=("all")
  fi

  for token in "${tokens[@]}"; do
    token="$(printf '%s' "$token" | xargs)"
    [[ -z "$token" ]] && continue
    if ! norm="$(normalize_ablation "$token")"; then
      echo "[ERROR] Unsupported ablation: $token" >&2
      usage
      exit 1
    fi
    if [[ "$norm" == "all" ]]; then
      for norm in full_internal base_only no_temporal_text no_structured no_news_features no_cleaned_residual no_route_regularization no_news_dropout summary_gated; do
        if [[ -z "${seen[$norm]:-}" ]]; then
          result+=("$norm")
          seen[$norm]=1
        fi
      done
      continue
    fi
    if [[ -z "${seen[$norm]:-}" ]]; then
      result+=("$norm")
      seen[$norm]=1
    fi
  done

  if [[ ${#result[@]} -eq 0 ]]; then
    echo "[ERROR] No ablations selected." >&2
    exit 1
  fi

  printf '%s\n' "${result[@]}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/run_tinynews_experiment.sh"
BASE_TASK_NAME_SUFFIX="${TASK_NAME_SUFFIX:-}"

if [[ ! -f "$MAIN_SCRIPT" ]]; then
  echo "[ERROR] Missing main experiment script: $MAIN_SCRIPT" >&2
  exit 1
fi

mapfile -t ABLATIONS_TO_RUN < <(expand_ablations "$ABLATION_SPEC")

HORIZONS_OVERRIDE="${HORIZONS_OVERRIDE:-48}"
LRS_OVERRIDE="${LRS_OVERRIDE:-1e-5}"

echo "[CORE_ABLATION] dataset=$DATASET_KEY profile=$PROFILE_KEY ablations=$(IFS=,; echo "${ABLATIONS_TO_RUN[*]}")"

run_one_ablation() {
  local ablation="$1"
  local suffix=""
  local label=""
  local -a env_overrides=()
  local -a script_args=("--dataset" "$DATASET_KEY")

  if [[ "$PROFILE_KEY" != "default" ]]; then
    script_args+=("--profile" "$PROFILE_KEY")
  fi

  script_args+=("--no-signnet")

  case "$ablation" in
    full_internal)
      label="Plan C baseline with internal sign"
      suffix="__coreabl_full_internal"
      ;;
    base_only)
      label="BASE stage only"
      suffix="__coreabl_base_only"
      env_overrides+=("STAGE=base")
      ;;
    no_temporal_text)
      label="temporal-text branch disabled"
      suffix="__coreabl_no_temporal_text"
      env_overrides+=("DELTA_TEMPORAL_TEXT_ENABLE=0")
      env_overrides+=("TEMPORAL_TEXT_MODEL_ID=")
      ;;
    no_structured)
      label="structured news vectors disabled"
      suffix="__coreabl_no_structured"
      env_overrides+=("DELTA_STRUCTURED_ENABLE=0")
      env_overrides+=("DELTA_INCLUDE_STRUCTURED_NEWS=0")
      env_overrides+=("NEWS_STRUCTURED_MODE=off")
      ;;
    no_news_features)
      label="temporal-text and structured features disabled"
      suffix="__coreabl_no_news_features"
      env_overrides+=("DELTA_TEMPORAL_TEXT_ENABLE=0")
      env_overrides+=("TEMPORAL_TEXT_MODEL_ID=")
      env_overrides+=("DELTA_STRUCTURED_ENABLE=0")
      env_overrides+=("DELTA_INCLUDE_STRUCTURED_NEWS=0")
      env_overrides+=("NEWS_STRUCTURED_MODE=off")
      ;;
    no_cleaned_residual)
      label="cleaned residual supervision disabled"
      suffix="__coreabl_no_cleaned_residual"
      ;;
    no_route_regularization)
      label="Plan C route regularization disabled"
      suffix="__coreabl_no_route_regularization"
      env_overrides+=("DELTA_ROUTE_BALANCE_LAMBDA=0")
      env_overrides+=("DELTA_ROUTE_ABSTAIN_LAMBDA=0")
      env_overrides+=("DELTA_ROUTE_CONF_FLOOR=0")
      ;;
    no_news_dropout)
      label="training-time news dropout disabled"
      suffix="__coreabl_no_news_dropout"
      env_overrides+=("NEWS_DROPOUT=0")
      env_overrides+=("DELTA_SIGN_EXTERNAL_NEWS_DROPOUT=0")
      ;;
    summary_gated)
      label="legacy summary_gated residual path"
      suffix="__coreabl_summary_gated"
      env_overrides+=("DELTA_MULTIMODAL_ARCH=summary_gated")
      ;;
    *)
      echo "[ERROR] Internal ablation mapping failure: $ablation" >&2
      exit 1
      ;;
  esac

  env_overrides+=("HORIZONS_OVERRIDE=${HORIZONS_OVERRIDE}")
  env_overrides+=("LRS_OVERRIDE=${LRS_OVERRIDE}")
  env_overrides+=("TASK_NAME_SUFFIX=${BASE_TASK_NAME_SUFFIX}${suffix}")

  local -a forwarded_extra_args=("${EXTRA_RUN_ARGS[@]}")
  if [[ "$ablation" == "no_cleaned_residual" ]]; then
    forwarded_extra_args+=("--cleaned_residual_enable" "0")
  fi

  echo "[CORE_ABLATION] start=$ablation desc=\"$label\" suffix=${BASE_TASK_NAME_SUFFIX}${suffix}"

  local -a cmd=("env")
  if [[ ${#env_overrides[@]} -gt 0 ]]; then
    cmd+=("${env_overrides[@]}")
  fi
  cmd+=("bash" "$MAIN_SCRIPT")
  cmd+=("${script_args[@]}")
  if [[ ${#forwarded_extra_args[@]} -gt 0 ]]; then
    cmd+=("--")
    cmd+=("${forwarded_extra_args[@]}")
  fi
  "${cmd[@]}"

  echo "[CORE_ABLATION] done=$ablation"
}

for ablation in "${ABLATIONS_TO_RUN[@]}"; do
  run_one_ablation "$ablation"
done
