#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tinynews_experiment.sh --dataset <nas14|nsw_load|nsw_price> [--profile default] [--no-signnet] [-- <extra run.py args>]

Examples:
  bash scripts/run_tinynews_experiment.sh --dataset nas14
  bash scripts/run_tinynews_experiment.sh --dataset nas14 --no-signnet
  bash scripts/run_tinynews_experiment.sh --dataset nsw_load
  bash scripts/run_tinynews_experiment.sh --dataset nsw_price -- --itr 1

Dataset-specific scripts:
  bash scripts/run_tinynews_nas14.sh
  bash scripts/run_tinynews_nsw_load.sh
  bash scripts/run_tinynews_nsw_price.sh
EOF
}

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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET=""
PROFILE="default"
FORWARD_ARGS=()

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
      FORWARD_ARGS+=( --no-signnet )
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      FORWARD_ARGS+=( -- "$@" )
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

PROFILE="$(printf '%s' "$PROFILE" | tr '[:upper:]' '[:lower:]')"
if [[ -n "$PROFILE" && "$PROFILE" != "default" ]]; then
  echo "[ERROR] Only --profile default is supported by the split dataset scripts." >&2
  exit 1
fi

if ! DATASET_KEY="$(normalize_dataset "$DATASET")"; then
  echo "[ERROR] Unsupported dataset: $DATASET" >&2
  usage
  exit 1
fi

case "$DATASET_KEY" in
  nas14)
    TARGET_SCRIPT="$SCRIPT_DIR/run_tinynews_nas14.sh"
    ;;
  nsw_load)
    TARGET_SCRIPT="$SCRIPT_DIR/run_tinynews_nsw_load.sh"
    ;;
  nsw_price)
    TARGET_SCRIPT="$SCRIPT_DIR/run_tinynews_nsw_price.sh"
    ;;
esac

exec bash "$TARGET_SCRIPT" "${FORWARD_ARGS[@]}"
