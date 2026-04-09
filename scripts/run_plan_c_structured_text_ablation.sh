#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_plan_c_structured_text_ablation.sh --dataset <nas14|nsw_load|nsw_price> [--profile <default|relative_text|additive>] [-- <extra run.py args>]

Examples:
  bash scripts/run_plan_c_structured_text_ablation.sh --dataset nsw_load
  bash scripts/run_plan_c_structured_text_ablation.sh --dataset nsw_price -- --select_metric mae

Notes:
  - This wrapper defaults to internal sign (`--no-signnet`) for cleaner Plan C ablations.
  - It runs these four combinations:
      1. structured on + temporal text on
      2. structured on + temporal text off
      3. structured off + temporal text on
      4. structured off + temporal text off
  - It defaults to `HORIZONS_OVERRIDE=48` and `LRS_OVERRIDE=1e-5`; override via environment if needed.
EOF
}

DATASET="${DATASET:-}"
PROFILE="${PROFILE:-default}"
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

HORIZONS_OVERRIDE="${HORIZONS_OVERRIDE:-48}"
LRS_OVERRIDE="${LRS_OVERRIDE:-1e-5}"

run_one() {
  local structured_enable="$1"
  local temporal_enable="$2"
  local tag="$3"

  echo "==> Ablation ${tag}: structured=${structured_enable} temporal_text=${temporal_enable}"
  DELTA_STRUCTURED_ENABLE="$structured_enable" \
  DELTA_TEMPORAL_TEXT_ENABLE="$temporal_enable" \
  TASK_NAME_SUFFIX="__abl_${tag}" \
  HORIZONS_OVERRIDE="$HORIZONS_OVERRIDE" \
  LRS_OVERRIDE="$LRS_OVERRIDE" \
  bash scripts/run_tinynews_experiment.sh \
    --dataset "$DATASET" \
    --profile "$PROFILE" \
    --no-signnet \
    -- "${EXTRA_RUN_ARGS[@]}"
}

run_one 1 1 "struct_on_text_on"
run_one 1 0 "struct_on_text_off"
run_one 0 1 "struct_off_text_on"
run_one 0 0 "struct_off_text_off"
