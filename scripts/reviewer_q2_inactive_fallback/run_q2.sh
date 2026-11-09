#!/usr/bin/env bash
# Reviewer Q2: news-inactive subset -- Base / Active / Blank / Perm and |y_REIN - y_base| statistics
# on the paper's 20 REIN checkpoints. No training.
#
# Output:  results/reviewer_q2_inactive_fallback.csv  (one row per run x {news_inactive, news_active})
#
# Env overrides: CKPT_ROOT, BANK_COMMIT, CKPT_COMMIT, GPU_ID, PYTHON_BIN, WORK_DIR
#   (see scripts/reviewer_q1_semantic_cf/_prepare_paper_inputs.sh)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/reviewer_q1_semantic_cf/_prepare_paper_inputs.sh"

OUT_CSV="$REPO_ROOT/results/reviewer_q2_inactive_fallback.csv"
echo "[Q2] ckpt_root=$CKPT_ROOT bank_dir=$BANK_DIR work_dir=$WORK_DIR gpu=$GPU_ID"
cd "$WORK_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" MPLCONFIGDIR="$WORK_DIR/mpl" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" "$SCRIPT_DIR/q2_inactive_fallback.py" \
    --ckpt_root "$CKPT_ROOT" --bank_dir "$BANK_DIR" --out_csv "$OUT_CSV" --save_dir "$WORK_DIR/run_logs"
