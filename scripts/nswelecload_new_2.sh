#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
VALUE_COL="TOTALDEMAND"
UNIT="megawatts"
DESCRIPTION="This dataset records the electricity load demand data in Australia NSW from 2019 to 2020, collected from National electricity market."
REGION="Australia, NSW"

TRAIN_FILE="dataset/2019-2020NSWelecload/elecload_2019-2020_trainset.csv"
VAL_FILE="dataset/2019-2020NSWelecload/elecload_2019-2020_valset.csv"
TEST_FILE="dataset/2019-2020NSWelecload/elecload_2019-2020_testset.csv"

NEWS_TEXT_COL="summary"
NEWS_TIME_COL="publication_time"

DELTA_EPOCHS="2"
BASE_EPOCHS="20"
# NEWS_WINDOW_DAYS="7"
NEWS_TOPM="999"
NEWS_TOPK="999"
BATCH_SIZE="1"
GPU_ID="${GPU_ID:-0}"
DEFAULT_POLICY="base"

TEMPLATE_POOL_2="configs/deltaWithNews_template.yaml"

# Keep task settings aligned with your existing NSW scripts.
RESIDUAL_LOSS="smooth_l1"
REWARD_METRIC="mae"
STRIDE="1"
HORIZON="48"
PATCH_DROPOUT="0"
HEAD_DROPOUT="0.1"
STAGE="all"
DELTA_VAL_MODE="${DELTA_VAL_MODE:-none}"  # each_epoch | end_only | none
DELTA_MODE="${DELTA_MODE:-kernel_tokens}"  # kernel-only experiment
DELTA_FUSION_MODE="${DELTA_FUSION_MODE:-mul_z}"  # add | mul_z | mul_raw
# Stability-first defaults for multiplicative fusion.
DELTA_MUL_SCALE="${DELTA_MUL_SCALE:-0.1}"
DELTA_MUL_COEFF_MIN="${DELTA_MUL_COEFF_MIN:-0.80}"
DELTA_MUL_COEFF_MAX="${DELTA_MUL_COEFF_MAX:-1.20}"
DELTA_CLIP="${DELTA_CLIP:-0.5}"
# Memory-safe defaults for 8B + SFT on limited VRAM.
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
KERNEL_REL_NORM_THRESH="${KERNEL_REL_NORM_THRESH:-0.05}"
KERNEL_REL_IMPROVE_RATIO="${KERNEL_REL_IMPROVE_RATIO:-0.10}"
KERNEL_REL_IMPROVE_ABS="${KERNEL_REL_IMPROVE_ABS:-0.0}"
KERNEL_A_MAX="${KERNEL_A_MAX:-2.0}"
KERNEL_AMP_BINS="${KERNEL_AMP_BINS:-21}"
KERNEL_CACHE_FILE="${KERNEL_CACHE_FILE:-sft_kernel_cache.json}"
KERNEL_AMP_TABLE_FILE="${KERNEL_AMP_TABLE_FILE:-kernel_amp_table.json}"

# kernel lr
KERNEL_SFT_LR="${KERNEL_SFT_LR:-1e-5}"

KERNEL_GEN_MAX_NEW_TOKENS="${KERNEL_GEN_MAX_NEW_TOKENS:-128}"
KERNEL_API_ENABLE="${KERNEL_API_ENABLE:-0}"
KERNEL_API_MODEL="${KERNEL_API_MODEL:-gpt-4o}"
KERNEL_API_TEMPERATURE="${KERNEL_API_TEMPERATURE:-0.1}"
KERNEL_API_MAX_CALLS="${KERNEL_API_MAX_CALLS:-200}"
KERNEL_API_UNCERTAIN_BAND="${KERNEL_API_UNCERTAIN_BAND:-0.02}"
KERNEL_API_LOW_AMP_BIN="${KERNEL_API_LOW_AMP_BIN:-2}"
KERNEL_API_LOG_EVERY="${KERNEL_API_LOG_EVERY:-10}"
KERNEL_API_CACHE_FILE="${KERNEL_API_CACHE_FILE:-sft_kernel_api_cache.json}"
KERNEL_API_KEY="${KERNEL_API_KEY:-${OPENAI_API_KEY:-}}"
KERNEL_API_KEY_FILE="${KERNEL_API_KEY_FILE:-.secrets/gpt4o_api_key.txt}"
if [[ -z "$KERNEL_API_KEY" && -f "$KERNEL_API_KEY_FILE" ]]; then
  KERNEL_API_KEY="$(tr -d ' \t\r\n' < "$KERNEL_API_KEY_FILE")"
fi
if [[ "$KERNEL_API_ENABLE" == "1" && "$KERNEL_CACHE_FILE" == "sft_kernel_cache.json" ]]; then
  KERNEL_CACHE_FILE="sft_kernel_cache_api.json"
fi

# pure TS base backbone (scheme2)
BASE_BACKBONES=(
  "dlinear"
  # "mlp"
)
BASE_HIDDEN_DIM="256"
BASE_MOVING_AVG="25"
BASE_DROPOUT="0.0"
BASE_LOSS="smooth_l1"
BASE_LR="1e-3"
BASE_WEIGHT_DECAY="1e-5"

# News utility-rerank settings (used by kernel sample construction/inference prompt building)
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

# =======================
# 2) Sweep spaces (same style as your original)
# =======================
TASK_NAMES=(
  "[3]NSW_19_20_LOAD_gateCF"
)

NEWS_CHOICES=(
  "dataset/Rated_Sum_V7_FNT_2019_2020_WAtt2019_combined.json"
  # "dataset/FNT_2019_2020_combined.json"
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
  --time_col "$TIME_COL"
  --value_col "$VALUE_COL"
  --unit "$UNIT"
  --description "$DESCRIPTION"
  --region "$REGION"
  --dayFirst
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
  --residual_loss "$RESIDUAL_LOSS"
  --stage "$STAGE"
  --delta_val_mode "$DELTA_VAL_MODE"
  --delta_mode "$DELTA_MODE"
  --delta_fusion_mode "$DELTA_FUSION_MODE"
  --delta_mul_scale "$DELTA_MUL_SCALE"
  --delta_mul_coeff_min "$DELTA_MUL_COEFF_MIN"
  --delta_mul_coeff_max "$DELTA_MUL_COEFF_MAX"
  --delta_clip "$DELTA_CLIP"
  --kernel_rel_norm_thresh "$KERNEL_REL_NORM_THRESH"
  --kernel_rel_improve_ratio "$KERNEL_REL_IMPROVE_RATIO"
  --kernel_rel_improve_abs "$KERNEL_REL_IMPROVE_ABS"
  --kernel_a_max "$KERNEL_A_MAX"
  --kernel_amp_bins "$KERNEL_AMP_BINS"
  --kernel_cache_file "$KERNEL_CACHE_FILE"
  --kernel_amp_table_file "$KERNEL_AMP_TABLE_FILE"
  --kernel_sft_lr "$KERNEL_SFT_LR"
  --kernel_gen_max_new_tokens "$KERNEL_GEN_MAX_NEW_TOKENS"
  --kernel_api_enable "$KERNEL_API_ENABLE"
  --kernel_api_key "$KERNEL_API_KEY"
  --kernel_api_model "$KERNEL_API_MODEL"
  --kernel_api_temperature "$KERNEL_API_TEMPERATURE"
  --kernel_api_max_calls "$KERNEL_API_MAX_CALLS"
  --kernel_api_uncertain_band "$KERNEL_API_UNCERTAIN_BAND"
  --kernel_api_low_amp_bin "$KERNEL_API_LOW_AMP_BIN"
  --kernel_api_log_every "$KERNEL_API_LOG_EVERY"
  --kernel_api_cache_file "$KERNEL_API_CACHE_FILE"
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
              if [[ "$DELTA_MODE" == "kernel_tokens" ]]; then
                run_task="${run_task}_kernelTok"
              fi
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
