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
STRIDE="48"
HORIZON="48"
PATCH_DROPOUT="0"
HEAD_DROPOUT="0.1"
STAGE="all"
DELTA_VAL_MODE="${DELTA_VAL_MODE:-each_epoch}"  # each_epoch | end_only | none
DELTA_CLIP="${DELTA_CLIP:-1.0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
NEWS_GATE_FLOOR="${NEWS_GATE_FLOOR:-0.0}"
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
NEWS_API_ENABLE="${NEWS_API_ENABLE:-0}"                       # 1 => use OpenAI API for refine/structured hooks
NEWS_API_MODEL="${NEWS_API_MODEL:-gpt-5.1}"
NEWS_API_KEY_PATH="${NEWS_API_KEY_PATH:-api_key.txt}"
NEWS_API_BASE_URL="${NEWS_API_BASE_URL:-}"
NEWS_API_TIMEOUT_SEC="${NEWS_API_TIMEOUT_SEC:-30}"
NEWS_API_MAX_RETRIES="${NEWS_API_MAX_RETRIES:-2}"
if [[ "$NEWS_API_ENABLE" == "1" ]]; then
  NEWS_REFINE_MODE="${NEWS_REFINE_MODE:-api}"                 # local | api
  NEWS_STRUCTURED_MODE="${NEWS_STRUCTURED_MODE:-api}"         # off | heuristic | api
else
  NEWS_REFINE_MODE="${NEWS_REFINE_MODE:-local}"               # local | api
  NEWS_STRUCTURED_MODE="${NEWS_STRUCTURED_MODE:-heuristic}"   # off | heuristic | api
fi
DELTA_INCLUDE_STRUCTURED_NEWS="${DELTA_INCLUDE_STRUCTURED_NEWS:-1}"  # 1 to append structured fields
CASE_RETRIEVAL_ENABLE="${CASE_RETRIEVAL_ENABLE:-0}"           # default off for conv-focused runs
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
NEWS_CONV_ENABLE="${NEWS_CONV_ENABLE:-1}"
NEWS_CONV_MAX_ITEMS="${NEWS_CONV_MAX_ITEMS:-8}"
NEWS_CONV_TEXT_MAX_TOKENS="${NEWS_CONV_TEXT_MAX_TOKENS:-96}"
NEWS_CONV_CHANNELS="${NEWS_CONV_CHANNELS:-64}"
NEWS_CONV_DROPOUT="${NEWS_CONV_DROPOUT:-0.1}"
NEWS_CONV_GATE_SCALE="${NEWS_CONV_GATE_SCALE:-1.0}"
NEWS_CONV_RUN_ABLATIONS="${NEWS_CONV_RUN_ABLATIONS:-1}"
NEWS_CONV_ABLATION_SPLIT="${NEWS_CONV_ABLATION_SPLIT:-val}"  # val | test | both
DELTA_MODEL_VARIANT="${DELTA_MODEL_VARIANT:-tiny_news_ts}"
TINY_NEWS_PRESET="${TINY_NEWS_PRESET:-tinyllama}"  # distilbert | gpt2 | tinyllama | custom
TINY_NEWS_LOADER="${TINY_NEWS_LOADER:-auto}"        # auto | encoder | causal_lm
TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-}"
TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-}"
TINY_NEWS_HIDDEN_SIZE="${TINY_NEWS_HIDDEN_SIZE:-256}"
TINY_NEWS_TEXT_TRAINABLE="${TINY_NEWS_TEXT_TRAINABLE:-0}"

case "$TINY_NEWS_PRESET" in
  distilbert)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-distilbert-base-uncased}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-distilbert-base-uncased}"
    ;;
  gpt2)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-gpt2}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-gpt2}"
    ;;
  tinyllama)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
    ;;
  custom)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-distilbert-base-uncased}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-$TINY_NEWS_MODEL}"
    ;;
  *)
    echo "[WARN] Unknown TINY_NEWS_PRESET=$TINY_NEWS_PRESET; fallback to distilbert" >&2
    TINY_NEWS_PRESET="distilbert"
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-distilbert-base-uncased}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-distilbert-base-uncased}"
    ;;
esac

DELTA_CF_LAMBDA="0.01"
DELTA_CF_MARGIN="0.05"
DELTA_GATE_REG_LAMBDA="0.01"
DELTA_NULL_LAMBDA="0.01"
DELTA_LORA_LR_SCALE="0.5"
DELTA_HEAD_LR_SCALE="1.0"
DELTA_AUX_LAMBDA="0.05"
# =======================
# 2) Sweep spaces (same style as your original)
# =======================
TASK_NAMES=(
  "[2024-nswelecPrice-tinynews]"
)

NEWS_CHOICES=(
  # "dataset/Rated_Sum_V7_FNT_2019_2020_WAtt2019_combined.json"
  # "dataset/FNT_2019_2020_combined.json"
  "dataset/news_2024_2025.json"
  # "dataset/empty.json"
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
  --delta_aux_lambda "$DELTA_AUX_LAMBDA"
  --delta_null_lambda "$DELTA_NULL_LAMBDA"
  --delta_lora_lr_scale "$DELTA_LORA_LR_SCALE"
  --delta_head_lr_scale "$DELTA_HEAD_LR_SCALE"
  --news_gate_floor "$NEWS_GATE_FLOOR"
  --early_stop_patience "$EARLY_STOP_PATIENCE"
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
  --news_api_model "$NEWS_API_MODEL"
  --news_api_key_path "$NEWS_API_KEY_PATH"
  --news_api_base_url "$NEWS_API_BASE_URL"
  --news_api_timeout_sec "$NEWS_API_TIMEOUT_SEC"
  --news_api_max_retries "$NEWS_API_MAX_RETRIES"
  --delta_include_structured_news "$DELTA_INCLUDE_STRUCTURED_NEWS"
  --news_structured_mode "$NEWS_STRUCTURED_MODE"
  --news_conv_enable "$NEWS_CONV_ENABLE"
  --news_conv_max_items "$NEWS_CONV_MAX_ITEMS"
  --news_conv_text_max_tokens "$NEWS_CONV_TEXT_MAX_TOKENS"
  --news_conv_channels "$NEWS_CONV_CHANNELS"
  --news_conv_dropout "$NEWS_CONV_DROPOUT"
  --news_conv_gate_scale "$NEWS_CONV_GATE_SCALE"
  --news_conv_run_ablations "$NEWS_CONV_RUN_ABLATIONS"
  --news_conv_ablation_split "$NEWS_CONV_ABLATION_SPLIT"
  --delta_model_variant "$DELTA_MODEL_VARIANT"
  --tiny_news_model_preset "$TINY_NEWS_PRESET"
  --tiny_news_model "$TINY_NEWS_MODEL"
  --tiny_news_tokenizer "$TINY_NEWS_TOKENIZER"
  --tiny_news_hidden_size "$TINY_NEWS_HIDDEN_SIZE"
  --tiny_news_text_trainable "$TINY_NEWS_TEXT_TRAINABLE"
  --tiny_news_loader "$TINY_NEWS_LOADER"
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
