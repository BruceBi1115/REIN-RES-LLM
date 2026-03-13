#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/nswelecprice_2024_tinynews_encoder_branches.sh distilbert
#   bash scripts/nswelecprice_2024_tinynews_encoder_branches.sh roberta_base
#   bash scripts/nswelecprice_2024_tinynews_encoder_branches.sh deberta_v3_base
#   bash scripts/nswelecprice_2024_tinynews_encoder_branches.sh all

TARGET="${1:-distilbert}"
PRESETS=(distilbert bert_base roberta_base deberta_v3_base)

run_one() {
  local preset="$1"
  echo "==> Running text-encoder preset: ${preset}"
  TINY_NEWS_PRESET="${preset}" bash scripts/nswelecprice_2024_tinynews.sh
}

if [[ "${TARGET}" == "all" ]]; then
  for p in "${PRESETS[@]}"; do
    run_one "${p}"
  done
else
  run_one "${TARGET}"
fi

