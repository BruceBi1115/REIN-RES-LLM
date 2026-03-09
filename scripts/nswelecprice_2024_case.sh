#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
set -euo pipefail

# =======================
# 1) Core run config
# =======================
PYTHON_BIN="${PYTHON_BIN:-}"
ENTRY="run.py"

pick_python_bin() {
  local requested="$1"
  local cand resolved
  local -a candidates=()

  if [[ -n "$requested" ]]; then
    candidates+=("$requested")
  fi
  candidates+=("python" "python3")

  local env_root
  for env_root in "$HOME/miniconda3/envs" "$HOME/anaconda3/envs"; do
    if [[ -d "$env_root" ]]; then
      while IFS= read -r cand; do
        candidates+=("$cand")
      done < <(ls "$env_root"/*/bin/python 2>/dev/null || true)
    fi
  done

  local -A seen=()
  for cand in "${candidates[@]}"; do
    resolved="$(command -v "$cand" 2>/dev/null || true)"
    if [[ -z "$resolved" ]]; then
      continue
    fi
    if [[ -n "${seen[$resolved]:-}" ]]; then
      continue
    fi
    seen[$resolved]=1

    if "$resolved" - <<'PY' >/dev/null 2>&1
import importlib
for m in ["pandas", "torch", "transformers", "peft", "openai"]:
    importlib.import_module(m)
PY
    then
      echo "$resolved"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(pick_python_bin "$PYTHON_BIN")"; then
  echo "[ERROR] No suitable Python found for this project." >&2
  echo "        Need modules: pandas, torch, transformers, peft, openai" >&2
  echo "        You can run with an explicit interpreter, e.g.:" >&2
  echo "        PYTHON_BIN=/path/to/env/bin/python bash scripts/nswelecload_new.sh" >&2
  exit 1
fi
echo "[env] Using PYTHON_BIN=$PYTHON_BIN"

TIME_COL="SETTLEMENTDATE"
VALUE_COL="RRP"
UNIT="$/MWh"
DESCRIPTION="This dataset records the electricity price data in Australia NSW from 2024, collected from National electricity market."
REGION="Australia, NSW"

TRAIN_FILE="dataset/2024NSWelecprice/2024NSWelecprice_trainset.csv"
VAL_FILE="dataset/2024NSWelecprice/2024NSWelecprice_valset.csv"
TEST_FILE="dataset/2024NSWelecprice/2024NSWelecprice_testset.csv"

NEWS_TEXT_COL="content"
NEWS_TIME_COL="date"

DELTA_EPOCHS="40"
BASE_EPOCHS="40"
# NEWS_WINDOW_DAYS="7"
NEWS_TOPM="999"
NEWS_TOPK="999"
BATCH_SIZE="1"
GPU_ID="${GPU_ID:-0}"
DEFAULT_POLICY="all"

TEMPLATE_POOL_2="configs/deltaWithNews_template.yaml"

# Keep task settings aligned with your existing NSW scripts.
RESIDUAL_LOSS="smooth_l1"
REWARD_METRIC="mae"
STRIDE="1"
HORIZON="720"
PATCH_DROPOUT="0"
HEAD_DROPOUT="0.1"
STAGE="base"
DELTA_VAL_MODE="${DELTA_VAL_MODE:-each_epoch}"  # each_epoch | end_only | none
DELTA_CLIP="${DELTA_CLIP:-0.5}"
# Memory-safe defaults for 8B + SFT on limited VRAM.
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# pure TS base backbone (scheme2)
BASE_BACKBONES=(
  # "dlinear"
  "mlp"
)
BASE_HIDDEN_DIM="256"
BASE_MOVING_AVG="25"
BASE_DROPOUT="0.0"
BASE_LOSS="smooth_l1"
BASE_LR="1e-3"
BASE_WEIGHT_DECAY="1e-5"

# News utility-rerank settings
UTILITY_RERANK_ENABLE="1"
UTILITY_KEYWORD_WEIGHT="0.35"
UTILITY_RECENCY_WEIGHT="0.25"
UTILITY_RATE_WEIGHT="0.35"
UTILITY_SENTIMENT_WEIGHT="0.05"
UTILITY_RECENCY_TAU_HOURS="24"
UTILITY_MMR_ENABLE="1"
UTILITY_MMR_LAMBDA="0.8"
UTILITY_DEDUP_THRESHOLD="0.95"
UTILITY_KEEP_TOPK="-1"
UTILITY_MIN_SCORE="-1.0"
UTILITY_SHOW_IN_PROMPT="1"

# Delta news extension hooks
NEWS_REFINE_MODE="${NEWS_REFINE_MODE:-local}"                 # local | api
DELTA_INCLUDE_STRUCTURED_NEWS="${DELTA_INCLUDE_STRUCTURED_NEWS:-1}"  # 1 to append structured fields
NEWS_STRUCTURED_MODE="${NEWS_STRUCTURED_MODE:-heuristic}"     # off | heuristic | api
CASE_RETRIEVAL_ENABLE="${CASE_RETRIEVAL_ENABLE:-1}"           # 1 to enable retrieval hook
CASE_RETRIEVAL_TOPK="${CASE_RETRIEVAL_TOPK:-3}"
CASE_RETRIEVAL_MODE="${CASE_RETRIEVAL_MODE:-price_event}"      # off | price | price_event
CASE_RETRIEVAL_ALPHA_PRICE="${CASE_RETRIEVAL_ALPHA_PRICE:-0.85}"
CASE_RETRIEVAL_ALPHA_EVENT="${CASE_RETRIEVAL_ALPHA_EVENT:-0.15}"
CASE_RETRIEVAL_MIN_TOP_SCORE="${CASE_RETRIEVAL_MIN_TOP_SCORE:-0.12}"
CASE_RETRIEVAL_MIN_CANDIDATES="${CASE_RETRIEVAL_MIN_CANDIDATES:-2}"
CASE_RETRIEVAL_MIN_DIR_AGREE="${CASE_RETRIEVAL_MIN_DIR_AGREE:-0.45}"
CASE_RETRIEVAL_MAX_EVENT_MISMATCH="${CASE_RETRIEVAL_MAX_EVENT_MISMATCH:-0.80}"
CASE_RETRIEVAL_FEATURE_DIM="${CASE_RETRIEVAL_FEATURE_DIM:-12}"
CASE_RETRIEVAL_GATE_ONLY="${CASE_RETRIEVAL_GATE_ONLY:-0}"      # 1 => retrieval only assists gate/confidence
CASE_RETRIEVAL_RUN_ABLATIONS="${CASE_RETRIEVAL_RUN_ABLATIONS:-1}"
CASE_RETRIEVAL_ABLATION_SPLIT="${CASE_RETRIEVAL_ABLATION_SPLIT:-val}"  # val | test | both
CASE_RETRIEVAL_STRONG_NEWS_THRESH="${CASE_RETRIEVAL_STRONG_NEWS_THRESH:-0.6}"

DELTA_CF_LAMBDA="0.05"
DELTA_CF_MARGIN="0.05"
DELTA_GATE_REG_LAMBDA="0.05"
# =======================
# 2) Sweep spaces (same style as your original)
# =======================
TASK_NAMES=(
  "[2024-nswelecPrice-case]"
)

NEWS_CHOICES=(
  # "dataset/Rated_Sum_V7_FNT_2019_2020_WAtt2019_combined.json"
  # "dataset/FNT_2019_2020_combined.json"
  "dataset/news_2024_2025.json"
)

RUN_OR_NOT=(
  "1"
)

TEMPLATE_POOLS=(
  "$TEMPLATE_POOL_2"
)

LOOKBACK_WINDOWS=(
  "1"
)

SCHEDULERS=(
  "1"
)

LRS=(
  "5e-6"
  # "1e-5"
)

GRAD_ACCS=(
  # "1"
  "8"
  # "16"
)

# =======================
# 3) Shared args
# =======================
COMMON_ARGS=(
  --delta_cf_lambda "$DELTA_CF_LAMBDA"
  --delta_cf_margin "$DELTA_CF_MARGIN"
  --delta_gate_reg_lambda "$DELTA_GATE_REG_LAMBDA"
  --time_col "$TIME_COL"
  --value_col "$VALUE_COL"
  --unit "$UNIT"
  --description "$DESCRIPTION"
  --region "$REGION"
  --train_file "$TRAIN_FILE"
  --val_file "$VAL_FILE"
  --test_file "$TEST_FILE"
  --news_text_col "$NEWS_TEXT_COL"
  --news_time_col "$NEWS_TIME_COL"
  # --news_window_days "$NEWS_WINDOW_DAYS"
  --news_topM "$NEWS_TOPM"
  --news_topK "$NEWS_TOPK"
  --batch_size "$BATCH_SIZE"
  --gpu "$GPU_ID"
  --reward_metric "$REWARD_METRIC"
  --default_policy "$DEFAULT_POLICY"
  --utility_rerank_enable "$UTILITY_RERANK_ENABLE"
  --utility_keyword_weight "$UTILITY_KEYWORD_WEIGHT"
  --utility_recency_weight "$UTILITY_RECENCY_WEIGHT"
  --utility_rate_weight "$UTILITY_RATE_WEIGHT"
  --utility_sentiment_weight "$UTILITY_SENTIMENT_WEIGHT"
  --utility_recency_tau_hours "$UTILITY_RECENCY_TAU_HOURS"
  --utility_mmr_enable "$UTILITY_MMR_ENABLE"
  --utility_mmr_lambda "$UTILITY_MMR_LAMBDA"
  --utility_dedup_threshold "$UTILITY_DEDUP_THRESHOLD"
  --utility_keep_topk "$UTILITY_KEEP_TOPK"
  --utility_min_score "$UTILITY_MIN_SCORE"
  --utility_show_in_prompt "$UTILITY_SHOW_IN_PROMPT"
  --news_refine_mode "$NEWS_REFINE_MODE"
  --delta_include_structured_news "$DELTA_INCLUDE_STRUCTURED_NEWS"
  --news_structured_mode "$NEWS_STRUCTURED_MODE"
  --case_retrieval_enable "$CASE_RETRIEVAL_ENABLE"
  --case_retrieval_topk "$CASE_RETRIEVAL_TOPK"
  --case_retrieval_mode "$CASE_RETRIEVAL_MODE"
  --case_retrieval_alpha_price "$CASE_RETRIEVAL_ALPHA_PRICE"
  --case_retrieval_alpha_event "$CASE_RETRIEVAL_ALPHA_EVENT"
  --case_retrieval_min_top_score "$CASE_RETRIEVAL_MIN_TOP_SCORE"
  --case_retrieval_min_candidates "$CASE_RETRIEVAL_MIN_CANDIDATES"
  --case_retrieval_min_dir_agree "$CASE_RETRIEVAL_MIN_DIR_AGREE"
  --case_retrieval_max_event_mismatch "$CASE_RETRIEVAL_MAX_EVENT_MISMATCH"
  --case_retrieval_feature_dim "$CASE_RETRIEVAL_FEATURE_DIM"
  --case_retrieval_gate_only "$CASE_RETRIEVAL_GATE_ONLY"
  --case_retrieval_run_ablations "$CASE_RETRIEVAL_RUN_ABLATIONS"
  --case_retrieval_ablation_split "$CASE_RETRIEVAL_ABLATION_SPLIT"
  --case_retrieval_strong_news_thresh "$CASE_RETRIEVAL_STRONG_NEWS_THRESH"
  --residual_loss "$RESIDUAL_LOSS"
  --stage "$STAGE"
  --delta_val_mode "$DELTA_VAL_MODE"
  --delta_clip "$DELTA_CLIP"
)

# =======================
# 4) Run combinations
# =======================
for i in "${!TASK_NAMES[@]}"; do
  for k in "${!LOOKBACK_WINDOWS[@]}"; do
    for j in "${!NEWS_CHOICES[@]}"; do
      run_or_not="${RUN_OR_NOT[$i]}"
      task="${TASK_NAMES[$i]}"
      tpool="${TEMPLATE_POOLS[$i]}"

      for base_backbone in "${BASE_BACKBONES[@]}"; do
        for lr in "${LRS[@]}"; do
          for sch in "${SCHEDULERS[@]}"; do
            for grad_acc in "${GRAD_ACCS[@]}"; do
              run_task="${task}_${base_backbone}"
              args=( --taskName "$run_task" --rl_use "0" "${COMMON_ARGS[@]}" )

              args+=( --news_path "${NEWS_CHOICES[$j]}" )
              if [[ -n "$tpool" ]]; then
                args+=( --template_pool "$tpool" )
              fi

              args+=( --news_window_days "${LOOKBACK_WINDOWS[$k]}" )
              args+=( --head_mlp )
              args+=( --patch_dropout "$PATCH_DROPOUT" )
              args+=( --head_dropout "$HEAD_DROPOUT" )
              args+=( --stride "$STRIDE" )
              args+=( --horizon "$HORIZON" )
              args+=( --base_epochs "$BASE_EPOCHS" )
              args+=( --delta_epochs "$DELTA_EPOCHS" )
              args+=( --base_backbone "$base_backbone" )
              args+=( --base_hidden_dim "$BASE_HIDDEN_DIM" )
              args+=( --base_moving_avg "$BASE_MOVING_AVG" )
              args+=( --base_dropout "$BASE_DROPOUT" )
              args+=( --base_loss "$BASE_LOSS" )
              args+=( --base_lr "$BASE_LR" )
              args+=( --base_weight_decay "$BASE_WEIGHT_DECAY" )
              args+=( --grad_accum "$grad_acc" )
              args+=( --lr "$lr" )
              args+=( --scheduler "$sch" )
              if [[ "$LOAD_IN_4BIT" == "1" ]]; then
                args+=( --load_in_4bit )
              fi
              if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
                args+=( --gradient_checkpointing )
              fi

              if [[ "$run_or_not" == "1" ]]; then
                echo "==> Running: ${run_task} (base_backbone=${base_backbone})"
                "$PYTHON_BIN" "$ENTRY" "${args[@]}"
              fi

              # Optional cleanup for disk space after each run:
              # rm -rf "checkpoints/${run_task}/best_base_${run_task}"
              # rm -rf "checkpoints/${run_task}/best_delta_${run_task}"
            done
          done
        done
      done
    done
  done
done
