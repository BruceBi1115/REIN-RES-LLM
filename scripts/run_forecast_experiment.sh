#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_forecast_experiment.sh --dataset <nas14|nsw_load|nsw_price> [dataset-script args...]

Examples:
  bash scripts/run_forecast_experiment.sh --dataset nsw_load
  bash scripts/run_forecast_experiment.sh --dataset nsw_price -- --epochs 5
  bash scripts/run_forecast_experiment.sh --dataset nas14 --session forecast
EOF
}

DATASET=""
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
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      FORWARD_ARGS+=("$@")
      break
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "[ERROR] --dataset is required." >&2
  usage
  exit 1
fi

case "$DATASET" in
  nas14)
    TARGET_SCRIPT="$SCRIPT_DIR/run_nas14.sh"
    ;;
  nsw_load)
    TARGET_SCRIPT="$SCRIPT_DIR/run_nsw_load.sh"
    ;;
  nsw_price)
    TARGET_SCRIPT="$SCRIPT_DIR/run_nsw_price.sh"
    ;;
  *)
    echo "[ERROR] Unsupported dataset: $DATASET" >&2
    usage
    exit 1
    ;;
esac

exec bash "$TARGET_SCRIPT" "${FORWARD_ARGS[@]}"
