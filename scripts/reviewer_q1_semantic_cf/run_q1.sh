#!/usr/bin/env bash
# Reviewer Q1: semantic-only counterfactuals (mass/timestamps/calendar fixed) + multi-seed permutations
# on the paper's 20 REIN checkpoints. No training.
#
# Output:  results/reviewer_q1_semantic_counterfactuals.csv          (one row per run x subset x variant)
#          results/reviewer_q1_semantic_counterfactuals_summary.csv  (aggregated over seeds / shifts)
#
# Env overrides:
#   CKPT_ROOT   dir holding the 20 paper run folders (default: session scratch copy; if missing it is
#               re-extracted from git commit $CKPT_COMMIT)
#   BANK_COMMIT git commit holding the original regime banks (default 9f98750; the current HEAD's
#               load bank is a different, overwritten file)
#   GPU_ID, PYTHON_BIN, WORK_DIR
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/reviewer_q1_semantic_cf/_prepare_paper_inputs.sh"

OUT_CSV="$REPO_ROOT/results/reviewer_q1_semantic_counterfactuals.csv"
echo "[Q1] ckpt_root=$CKPT_ROOT bank_dir=$BANK_DIR work_dir=$WORK_DIR gpu=$GPU_ID"
cd "$WORK_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" MPLCONFIGDIR="$WORK_DIR/mpl" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" "$SCRIPT_DIR/q1_semantic_counterfactuals.py" \
    --ckpt_root "$CKPT_ROOT" --bank_dir "$BANK_DIR" --out_csv "$OUT_CSV" --save_dir "$WORK_DIR/run_logs"
