#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tinynews_ablation.sh --dataset <nas14|nsw_load|nsw_price> [--profile <default|relative_text|additive>] [--ablation <name[,name...]>] [--list] [-- <extra run.py args>]

Examples:
  bash scripts/run_tinynews_ablation.sh --dataset nsw_price
  bash scripts/run_tinynews_ablation.sh --dataset nsw_price --ablation full,no_signnet,no_temporal_text
  bash scripts/run_tinynews_ablation.sh --dataset nas14 --ablation all
  bash scripts/run_tinynews_ablation.sh --dataset nsw_price --profile additive --ablation base_only
  bash scripts/run_tinynews_ablation.sh --dataset nsw_price --ablation no_structured -- --delta_temporal_text_source raw

Ablations:
  full              Baseline using the dataset/profile defaults from run_tinynews_experiment.sh
  no_signnet        Remove external SignNet; use DELTA internal sign mode instead
  no_temporal_text  Disable the temporal-text branch
  no_structured     Disable structured news vectors for DELTA/SignNet
  no_news_features  Disable both temporal-text and structured news features
  base_only         Run only the BASE stage
  all               Run: full,no_signnet,no_temporal_text,no_structured,no_news_features,base_only

Notes:
  - Each ablation run gets a task suffix like `__abl_no_signnet` to avoid checkpoint/result collisions.
  - Existing environment overrides are preserved; this script only adds ablation-specific overrides.
  - Extra args after `--` are forwarded to every underlying run.py invocation.
EOF
}

list_ablations() {
  cat <<'EOF'
full
no_signnet
no_temporal_text
no_structured
no_news_features
base_only
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
    full|baseline)
      printf 'full\n'
      ;;
    no_signnet|nosignnet|internal_sign)
      printf 'no_signnet\n'
      ;;
    no_temporal_text|no_text|notext|no_news_text)
      printf 'no_temporal_text\n'
      ;;
    no_structured|nostructured|no_struct|no_structured_news)
      printf 'no_structured\n'
      ;;
    no_news_features|nonewsfeatures|no_delta_news|no_news_branches)
      printf 'no_news_features\n'
      ;;
    base_only|base)
      printf 'base_only\n'
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
      for norm in full no_signnet no_temporal_text no_structured no_news_features base_only; do
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

echo "[ABLATION] dataset=$DATASET_KEY profile=$PROFILE_KEY ablations=$(IFS=,; echo "${ABLATIONS_TO_RUN[*]}")"

run_one_ablation() {
  local ablation="$1"
  local suffix=""
  local label=""
  local -a env_overrides=()
  local -a script_args=("--dataset" "$DATASET_KEY")

  if [[ "$PROFILE_KEY" != "default" ]]; then
    script_args+=("--profile" "$PROFILE_KEY")
  fi

  case "$ablation" in
    full)
      label="baseline defaults"
      suffix="__abl_full"
      ;;
    no_signnet)
      label="DELTA internal sign head only"
      suffix="__abl_no_signnet"
      script_args+=("--no-signnet")
      ;;
    no_temporal_text)
      label="temporal-text branch disabled"
      suffix="__abl_no_temporal_text"
      env_overrides+=("DELTA_TEMPORAL_TEXT_ENABLE=0")
      env_overrides+=("TEMPORAL_TEXT_MODEL_ID=")
      ;;
    no_structured)
      label="structured news vectors disabled"
      suffix="__abl_no_structured"
      env_overrides+=("DELTA_STRUCTURED_ENABLE=0")
      env_overrides+=("DELTA_INCLUDE_STRUCTURED_NEWS=0")
      env_overrides+=("NEWS_STRUCTURED_MODE=off")
      ;;
    no_news_features)
      label="temporal-text and structured news features disabled"
      suffix="__abl_no_news_features"
      env_overrides+=("DELTA_TEMPORAL_TEXT_ENABLE=0")
      env_overrides+=("TEMPORAL_TEXT_MODEL_ID=")
      env_overrides+=("DELTA_STRUCTURED_ENABLE=0")
      env_overrides+=("DELTA_INCLUDE_STRUCTURED_NEWS=0")
      env_overrides+=("NEWS_STRUCTURED_MODE=off")
      ;;
    base_only)
      label="BASE stage only"
      suffix="__abl_base_only"
      env_overrides+=("STAGE=base")
      ;;
    *)
      echo "[ERROR] Internal ablation mapping failure: $ablation" >&2
      exit 1
      ;;
  esac

  env_overrides+=("TASK_NAME_SUFFIX=${BASE_TASK_NAME_SUFFIX}${suffix}")

  echo "[ABLATION] start=$ablation desc=\"$label\" suffix=${BASE_TASK_NAME_SUFFIX}${suffix}"

  local -a cmd=("env")
  if [[ ${#env_overrides[@]} -gt 0 ]]; then
    cmd+=("${env_overrides[@]}")
  fi
  cmd+=("bash" "$MAIN_SCRIPT")
  cmd+=("${script_args[@]}")
  if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
    cmd+=("--")
    cmd+=("${EXTRA_RUN_ARGS[@]}")
  fi
  "${cmd[@]}"

  echo "[ABLATION] done=$ablation"
}

for ablation in "${ABLATIONS_TO_RUN[@]}"; do
  run_one_ablation "$ablation"
done
