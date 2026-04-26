#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints}"
DATASET_FILTER="all"
HORIZON_FILTER="all"
LIST_ONLY="0"
DRY_RUN="0"
STOP_ON_ERROR="0"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_checkpoints.sh [--dataset DATASET] [--horizon H]
                                   [--checkpoint-root DIR] [--list] [--dry-run]
                                   [--stop-on-error] [-- ...extra args for test_from_checkpoint.py]

Examples:
  bash scripts/eval_checkpoints.sh
  bash scripts/eval_checkpoints.sh --dataset nsw_price
  bash scripts/eval_checkpoints.sh --dataset gas --horizon 336
  bash scripts/eval_checkpoints.sh --horizon 48 --list

Supported datasets:
  all, nsw_price, nsw_load, gas_demand, traffic_count, nas14, bitcoin_hourly_24

Notes:
  - The script auto-discovers checkpoint directories under --checkpoint-root.
  - --dataset filters to one dataset; --horizon filters to one horizon.
  - Any arguments after -- are passed through to scripts/test_checkpoint.sh.
EOF
}

normalize_dataset() {
  local raw="${1:-all}"
  case "$raw" in
    ""|all) printf '%s\n' "all" ;;
    nsw_price|price) printf '%s\n' "nsw_price" ;;
    nsw_load|load) printf '%s\n' "nsw_load" ;;
    gas_demand|gas) printf '%s\n' "gas_demand" ;;
    traffic_count|traffic) printf '%s\n' "traffic_count" ;;
    nas14) printf '%s\n' "nas14" ;;
    bitcoin|bitcoin_hourly|bitcoin_hourly_24) printf '%s\n' "bitcoin_hourly_24" ;;
    *)
      echo "[ERROR] Unsupported dataset: $raw" >&2
      return 1
      ;;
  esac
}

dataset_for_checkpoint() {
  local base="$1"
  case "$base" in
    delta_v3_nsw_price_*) printf '%s\n' "nsw_price" ;;
    delta_v3_nsw_load_*) printf '%s\n' "nsw_load" ;;
    delta_v3_gas_demand_*) printf '%s\n' "gas_demand" ;;
    traffic_count_*) printf '%s\n' "traffic_count" ;;
    delta_v3_nas14_*) printf '%s\n' "nas14" ;;
    delta_v3_bitcoin_hourly_*) printf '%s\n' "bitcoin_hourly_24" ;;
    *) return 1 ;;
  esac
}

extract_horizon() {
  local base="$1"
  if [[ "$base" =~ _1_([0-9]+)_ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

task_name_for_dataset() {
  local dataset="$1"
  case "$dataset" in
    nsw_price) printf '%s\n' "delta_v3_nsw_price__lr1e-4__ga4__sch1" ;;
    nsw_load) printf '%s\n' "delta_v3_nsw_load__lr1e-4__ga4__sch1" ;;
    gas_demand) printf '%s\n' "delta_v3_gas_demand__lr5e-6__ga4__sch1" ;;
    traffic_count) printf '%s\n' "traffic_count__lr1e-4__ga4__sch1" ;;
    nas14) printf '%s\n' "delta_v3_nas14__lr1e-4__ga1__sch1" ;;
    bitcoin_hourly_24) printf '%s\n' "delta_v3_bitcoin_hourly__lr2e-5__ga4__sch1" ;;
    *)
      echo "[ERROR] Missing task-name mapping for dataset=$dataset" >&2
      return 1
      ;;
  esac
}

set_dataset_env() {
  local dataset="$1"
  case "$dataset" in
    nsw_price)
      DATASET_KEY="nsw_price"
      TRAIN_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_trainset.csv"
      VAL_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_valset.csv"
      TEST_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_testset.csv"
      TIME_COL="date"
      VALUE_COL="RRP"
      UNIT=""
      DESCRIPTION="This dataset records the electricity price data in Australia NSW from 2024, collected from the National Electricity Market."
      REGION="Australia, NSW"
      FREQ_MIN="30"
      BATCH_SIZE="16"
      DAY_FIRST="1"
      ;;
    nsw_load)
      DATASET_KEY="nsw_load"
      TRAIN_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_trainset.csv"
      VAL_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_valset.csv"
      TEST_FILE="dataset/2024NSWelecLOAD/2024NSWelecLOAD_testset.csv"
      TIME_COL="date"
      VALUE_COL="TOTALDEMAND"
      UNIT="MW"
      DESCRIPTION="This dataset records the electricity load demand data in Australia NSW from 2024, collected from the National Electricity Market."
      REGION="Australia, NSW"
      FREQ_MIN=""
      BATCH_SIZE="16"
      DAY_FIRST="1"
      ;;
    gas_demand)
      DATASET_KEY="gas_demand"
      TRAIN_FILE="dataset/gas_demand/entsog_gas_demand_nl_2025_hour_extracted_trainset.csv"
      VAL_FILE="dataset/gas_demand/entsog_gas_demand_nl_2025_hour_extracted_valset.csv"
      TEST_FILE="dataset/gas_demand/entsog_gas_demand_nl_2025_hour_extracted_testset.csv"
      TIME_COL="date"
      VALUE_COL="total_value"
      UNIT=""
      DESCRIPTION="This dataset records hourly gas demand values for the Netherlands in 2025, extracted from ENTSOG."
      REGION="Netherlands"
      FREQ_MIN="60"
      BATCH_SIZE="16"
      DAY_FIRST="0"
      ;;
    traffic_count)
      DATASET_KEY="traffic_count"
      TRAIN_FILE="dataset/traffic_count/road_traffic_counts_hourly_permanent3_2024_hourly_trainset.csv"
      VAL_FILE="dataset/traffic_count/road_traffic_counts_hourly_permanent3_2024_hourly_valset.csv"
      TEST_FILE="dataset/traffic_count/road_traffic_counts_hourly_permanent3_2024_hourly_testset.csv"
      TIME_COL="time"
      VALUE_COL="traffic_count"
      UNIT="count"
      DESCRIPTION="This dataset records hourly traffic counts of year 2024 in Australia, NSW, aggregated into a single hourly time series."
      REGION="Traffic monitoring network"
      FREQ_MIN="60"
      BATCH_SIZE="16"
      DAY_FIRST="0"
      ;;
    nas14)
      DATASET_KEY="nas14"
      TRAIN_FILE="dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_trainset.csv"
      VAL_FILE="dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_valset.csv"
      TEST_FILE="dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_testset.csv"
      TIME_COL="date"
      VALUE_COL="open"
      UNIT="USD"
      DESCRIPTION="This dataset records 14 tickers' daily opening stock prices on the NASDAQ market from 2022 to 2023, paired with related NASDAQ news articles."
      REGION="United States, NASDAQ"
      FREQ_MIN="1440"
      BATCH_SIZE="4"
      DAY_FIRST="0"
      ;;
    bitcoin_hourly_24)
      DATASET_KEY="bitcoin_hourly_24"
      TRAIN_FILE="dataset/bitcoin_hourly_24/bitcoin-hourly-open-2024_trainset.csv"
      VAL_FILE="dataset/bitcoin_hourly_24/bitcoin-hourly-open-2024_valset.csv"
      TEST_FILE="dataset/bitcoin_hourly_24/bitcoin-hourly-open-2024_testset.csv"
      TIME_COL="DATETIME"
      VALUE_COL="OPEN"
      UNIT="USD"
      DESCRIPTION="This dataset records Bitcoin hourly opening prices from 2022 to 2023."
      REGION="Global crypto market"
      FREQ_MIN="60"
      BATCH_SIZE="16"
      DAY_FIRST="1"
      ;;
    *)
      echo "[ERROR] Missing dataset config for dataset=$dataset" >&2
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      [[ $# -ge 2 ]] || { echo "[ERROR] --dataset requires a value" >&2; exit 1; }
      DATASET_FILTER="$(normalize_dataset "$2")"
      shift 2
      ;;
    --horizon)
      [[ $# -ge 2 ]] || { echo "[ERROR] --horizon requires a value" >&2; exit 1; }
      HORIZON_FILTER="$2"
      shift 2
      ;;
    --checkpoint-root)
      [[ $# -ge 2 ]] || { echo "[ERROR] --checkpoint-root requires a value" >&2; exit 1; }
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --list)
      LIST_ONLY="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --stop-on-error)
      STOP_ON_ERROR="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$HORIZON_FILTER" != "all" && ! "$HORIZON_FILTER" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --horizon must be 'all' or an integer, got: $HORIZON_FILTER" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
  echo "[ERROR] checkpoint root not found: $CHECKPOINT_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"

mapfile -t MATCHES < <(
  while IFS= read -r dir; do
    base="$(basename "$dir")"
    dataset="$(dataset_for_checkpoint "$base" || true)"
    [[ -n "${dataset:-}" ]] || continue
    horizon="$(extract_horizon "$base" || true)"
    [[ -n "${horizon:-}" ]] || continue
    if [[ "$DATASET_FILTER" != "all" && "$dataset" != "$DATASET_FILTER" ]]; then
      continue
    fi
    if [[ "$HORIZON_FILTER" != "all" && "$horizon" != "$HORIZON_FILTER" ]]; then
      continue
    fi
    printf '%s|%s|%s\n' "$dataset" "$horizon" "$dir"
  done < <(find "$CHECKPOINT_ROOT" -maxdepth 1 -mindepth 1 -type d | sort) \
  | sort -t '|' -k1,1 -k2,2n
)

if [[ ${#MATCHES[@]} -eq 0 ]]; then
  echo "[ERROR] No matching checkpoints found under $CHECKPOINT_ROOT" >&2
  exit 1
fi

echo "[INFO] checkpoint_root=$CHECKPOINT_ROOT"
echo "[INFO] dataset_filter=$DATASET_FILTER horizon_filter=$HORIZON_FILTER"
echo "[INFO] matched_checkpoints=${#MATCHES[@]}"

for entry in "${MATCHES[@]}"; do
  IFS='|' read -r dataset horizon dir <<< "$entry"
  echo "  - dataset=$dataset horizon=$horizon checkpoint=$dir"
done

if [[ "$LIST_ONLY" == "1" ]]; then
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[INFO] dry-run only; nothing executed."
  exit 0
fi

ok_count=0
fail_count=0
batch_log="/tmp/eval_checkpoints_$(date '+%Y%m%d_%H%M%S').log"
: > "$batch_log"
echo "[INFO] batch_log=$batch_log"

for idx in "${!MATCHES[@]}"; do
  entry="${MATCHES[$idx]}"
  IFS='|' read -r dataset horizon dir <<< "$entry"
  base="$(basename "$dir")"
  task_name="$(task_name_for_dataset "$dataset")"
  set_dataset_env "$dataset"
  stdout_log="/tmp/${base}.eval.stdout.log"
  train_count=$((idx + 1))

  echo "[RUN][$train_count/${#MATCHES[@]}] dataset=$dataset horizon=$horizon checkpoint=$dir" | tee -a "$batch_log"

  if CHECKPOINT_DIR="$dir" \
    TASK_NAME="$task_name" \
    DATASET_KEY="$DATASET_KEY" \
    TRAIN_FILE="$TRAIN_FILE" \
    VAL_FILE="$VAL_FILE" \
    TEST_FILE="$TEST_FILE" \
    TIME_COL="$TIME_COL" \
    VALUE_COL="$VALUE_COL" \
    UNIT="$UNIT" \
    DESCRIPTION="$DESCRIPTION" \
    REGION="$REGION" \
    FREQ_MIN="$FREQ_MIN" \
    BATCH_SIZE="$BATCH_SIZE" \
    DAY_FIRST="$DAY_FIRST" \
    bash "$SCRIPT_DIR/test_checkpoint.sh" "${EXTRA_ARGS[@]}" >"$stdout_log" 2>&1; then
    ok_count=$((ok_count + 1))
    echo "[OK] dataset=$dataset horizon=$horizon checkpoint=$dir" | tee -a "$batch_log"
    log_path="$dir/$base.log"
    if [[ -f "$log_path" ]]; then
      grep -E '\[TEST\]\[(FINAL|BASE_ONLY|COUNTERFACTUAL)\]' "$log_path" | tail -n 3 | tee -a "$batch_log"
    else
      echo "[WARN] missing checkpoint log: $log_path" | tee -a "$batch_log"
    fi
  else
    code=$?
    fail_count=$((fail_count + 1))
    echo "[FAIL] dataset=$dataset horizon=$horizon checkpoint=$dir exit=$code stdout_log=$stdout_log" | tee -a "$batch_log"
    tail -n 40 "$stdout_log" | tee -a "$batch_log"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      echo "[ABORT] stop-on-error enabled" | tee -a "$batch_log"
      exit "$code"
    fi
  fi
done

echo "[DONE] ok=$ok_count fail=$fail_count batch_log=$batch_log"

if [[ "$fail_count" -gt 0 ]]; then
  exit 1
fi
