#!/usr/bin/env bash
# Shared by run_q1.sh and run_q2.sh (sourced). Prepares, outside the repo working tree:
#   BANK_DIR   original regime banks, extracted from git commit $BANK_COMMIT
#   CKPT_ROOT  the paper's 20 run folders (base + delta checkpoints, meta, logs)
#   WORK_DIR   scratch cwd with a `dataset` symlink, so the eval code's ./checkpoints/<task> side
#              effects never touch the repo's checkpoints/ folder.
# Requires REPO_ROOT to be set by the caller.

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/myenv01/bin/python}"
GPU_ID="${GPU_ID:-1}"
BANK_COMMIT="${BANK_COMMIT:-9f98750}"
CKPT_COMMIT="${CKPT_COMMIT:-bb1589c}"
CKPT_ROOT="${CKPT_ROOT:-/tmp/claude-1009/-home-brucebi-Projects-REIN-RES-LLM/84949b34-6d4f-4e9c-bdad-80c093960c42/scratchpad/ckpt/checkpoints}"
WORK_DIR="${WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/rein_reviewer_XXXXXX")}"

BANKS=(
  regime_bank_load_freeNews_reneweconomy_pvmag_watt.npz
  regime_bank_price_freeNews_reneweconomy_pvmag_watt.npz
  regime_bank_gdelt_gas_nl_2025.npz
  regime_bank_gdelt_2024.npz
)

mkdir -p "$WORK_DIR"
ln -sfn "$REPO_ROOT/dataset" "$WORK_DIR/dataset"
BANK_DIR="$WORK_DIR/banks"
mkdir -p "$BANK_DIR"
for b in "${BANKS[@]}"; do
  if [[ ! -s "$BANK_DIR/$b" ]]; then
    git -C "$REPO_ROOT" show "$BANK_COMMIT:_shared_refine_cache/v4/$b" > "$BANK_DIR/$b"
  fi
done

n_runs=0
if [[ -d "$CKPT_ROOT" ]]; then
  n_runs=$(find "$CKPT_ROOT" -path '*best_delta*' -name model.pt | grep -v time_mmd | wc -l)
fi
if [[ "$n_runs" -lt 20 ]]; then
  echo "[prepare] $CKPT_ROOT has $n_runs/20 delta checkpoints; extracting from git commit $CKPT_COMMIT"
  mkdir -p "$CKPT_ROOT"
  mapfile -t files < <(git -C "$REPO_ROOT" ls-tree -r --name-only "$CKPT_COMMIT" -- checkpoints \
    | grep -v -E '^checkpoints/time_mmd' | grep -v -E '\.png$')
  tmp_extract="$(mktemp -d)"
  git -C "$REPO_ROOT" archive "$CKPT_COMMIT" "${files[@]}" | tar -x -C "$tmp_extract"
  cp -r "$tmp_extract/checkpoints/." "$CKPT_ROOT/"
  rm -rf "$tmp_extract"
fi
