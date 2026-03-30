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

TIME_COL="date"
VALUE_COL="RRP"
UNIT=""
DESCRIPTION="This dataset records the electricity price data in Australia NSW from 2024, collected from National electricity market."
REGION="Australia, NSW"

TRAIN_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_trainset.csv"
VAL_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_valset.csv"
TEST_FILE="dataset/2024NSWelecPRICE/2024NSWelecPRICE_testset.csv"

NEWS_TEXT_COL="content"
NEWS_TIME_COL="date"

DELTA_EPOCHS="100"
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
SELECT_METRIC="mae"
STRIDE="1"
STAGE="all"
HORIZONS=(
  "48"
  "96"
  "192"
  "336"
  "720"
)
PATCH_DROPOUT="0"
HEAD_DROPOUT="0.1"

DELTA_VAL_MODE="${DELTA_VAL_MODE:-each_epoch}"  # each_epoch | end_only | none
DELTA_CLIP="${DELTA_CLIP:-1.0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
NEWS_GATE_FLOOR="${NEWS_GATE_FLOOR:-0.0}"
# Stable DELTA default for LOAD: use additive true-residual correction with
# in-model factorized gate/sign/magnitude heads.
FINAL_GATE_ENABLE="${FINAL_GATE_ENABLE:-1}"   # legacy compatibility flag; factorized DELTA now gates inside the model
DISABLE_ALL_GATES="${DISABLE_ALL_GATES:-0}"   # 1 => bypass internal delta gate + legacy outer gate + text gate
DELTA_GATE_INIT_BIAS="${DELTA_GATE_INIT_BIAS:--2.0}"  # negative => in-model gate starts conservative
DELTA_INTERNAL_GATE_IN_MODEL="${DELTA_INTERNAL_GATE_IN_MODEL:-1}"
DELTA_FREEZE_FEATURE_MODULES="${DELTA_FREEZE_FEATURE_MODULES:-1}"  # keep delta adaptation focused on the residual head
DELTA_NON_DEGRADE_LAMBDA="${DELTA_NON_DEGRADE_LAMBDA:-1.0}"
DELTA_NON_DEGRADE_MARGIN="${DELTA_NON_DEGRADE_MARGIN:-0.0}"  # best current ablation: guard against degradation, but do not require extra margin
DELTA_SIGN_LAMBDA="${DELTA_SIGN_LAMBDA:-0.05}"   # explicitly teach DELTA whether residual should be positive or negative
DELTA_SIGN_EPS="${DELTA_SIGN_EPS:-0.03}"
DELTA_GATE_EPS="${DELTA_GATE_EPS:-0.05}"

DELTA_SIGN_TAU="${DELTA_SIGN_TAU:-1.0}"
DELTA_SIGN_MODE="${DELTA_SIGN_MODE:-signnet_binary}"  # signnet_binary only
DELTA_SIGN_EXTERNAL_EPOCHS="${DELTA_SIGN_EXTERNAL_EPOCHS:-80}"
DELTA_SIGN_EXTERNAL_HIDDEN="${DELTA_SIGN_EXTERNAL_HIDDEN:-192}"
DELTA_SIGN_EXTERNAL_DROPOUT="${DELTA_SIGN_EXTERNAL_DROPOUT:-0.2}"
DELTA_SIGN_EXTERNAL_LR="${DELTA_SIGN_EXTERNAL_LR:-3e-4}"
DELTA_SIGN_EXTERNAL_WEIGHT_DECAY="${DELTA_SIGN_EXTERNAL_WEIGHT_DECAY:-5e-4}"
DELTA_SIGN_EXTERNAL_GRAD_CLIP="${DELTA_SIGN_EXTERNAL_GRAD_CLIP:-1.0}"
DELTA_SIGN_EXTERNAL_PATIENCE="${DELTA_SIGN_EXTERNAL_PATIENCE:-5}"
DELTA_SIGN_EXTERNAL_SELECT_METRIC="${DELTA_SIGN_EXTERNAL_SELECT_METRIC:-balanced_acc}"  # acc | balanced_acc | loss
DELTA_SIGN_EXTERNAL_MIN_DELTA="${DELTA_SIGN_EXTERNAL_MIN_DELTA:-1e-4}"
DELTA_SIGN_EXTERNAL_LR_FACTOR="${DELTA_SIGN_EXTERNAL_LR_FACTOR:-0.5}"
DELTA_SIGN_EXTERNAL_LR_PATIENCE="${DELTA_SIGN_EXTERNAL_LR_PATIENCE:-1}"
DELTA_SIGN_EXTERNAL_MIN_LR="${DELTA_SIGN_EXTERNAL_MIN_LR:-1e-5}"
DELTA_SIGN_EXTERNAL_CALIBRATE_BIAS="${DELTA_SIGN_EXTERNAL_CALIBRATE_BIAS:-1}"
DELTA_SIGN_EXTERNAL_BIAS_CLIP="${DELTA_SIGN_EXTERNAL_BIAS_CLIP:-1.5}"
DELTA_SIGN_EXTERNAL_NEWS_DROPOUT="${DELTA_SIGN_EXTERNAL_NEWS_DROPOUT:-0}"
DELTA_SIGN_EXTERNAL_USE_NEWS_WEIGHTING="${DELTA_SIGN_EXTERNAL_USE_NEWS_WEIGHTING:-0}"
DELTA_SIGN_EXTERNAL_USE_RESIDUAL_WEIGHTING="${DELTA_SIGN_EXTERNAL_USE_RESIDUAL_WEIGHTING:-0}"
DELTA_SIGN_EXTERNAL_USE_POS_WEIGHT="${DELTA_SIGN_EXTERNAL_USE_POS_WEIGHT:-1}"
DELTA_SIGN_EXTERNAL_POS_WEIGHT_FLOOR="${DELTA_SIGN_EXTERNAL_POS_WEIGHT_FLOOR:-0.5}"
DELTA_SIGN_EXTERNAL_POS_WEIGHT_CLIP="${DELTA_SIGN_EXTERNAL_POS_WEIGHT_CLIP:-3.0}"
DELTA_SIGN_EXTERNAL_TAU="${DELTA_SIGN_EXTERNAL_TAU:-1.0}"
DELTA_SIGN_EXTERNAL_VARIANT="${DELTA_SIGN_EXTERNAL_VARIANT:-mlp}"  # mlp | dual_stream_tcn
DELTA_SIGN_EXTERNAL_TEXT_DIM="${DELTA_SIGN_EXTERNAL_TEXT_DIM:-64}"
DELTA_SIGN_EXTERNAL_TEXT_MAX_LEN="${DELTA_SIGN_EXTERNAL_TEXT_MAX_LEN:-64}"
DELTA_GATE_LOSS_WEIGHT="${DELTA_GATE_LOSS_WEIGHT:-0.2}"
DELTA_SIGN_LOSS_WEIGHT="${DELTA_SIGN_LOSS_WEIGHT:-0.1}"
DELTA_MAG_LOSS_WEIGHT="${DELTA_MAG_LOSS_WEIGHT:-0.5}"
DELTA_MAG_TARGET="${DELTA_MAG_TARGET:-log1p}"  # raw | log1p
DELTA_MAG_MAX="${DELTA_MAG_MAX:-0.0}"          # <=0 disables clamp
DELTA_RESIDUAL_WEIGHT_SCALE="${DELTA_RESIDUAL_WEIGHT_SCALE:-1.0}"
GATE_NULL_LAMBDA="${GATE_NULL_LAMBDA:-0.0}"
CF_PSEUDO_MARGIN="${CF_PSEUDO_MARGIN:-0.01}"
CF_PSEUDO_TEMP="${CF_PSEUDO_TEMP:-0.2}"
CF_PSEUDO_HARD="${CF_PSEUDO_HARD:-0}"

# pure TS base backbone (scheme2)
BASE_BACKBONES=(
  "mlp"
  # "mlp"
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

# Delta news extension hooks
NEWS_API_ENABLE="${NEWS_API_ENABLE:-1}"                       # 1 => use OpenAI API for news refine hooks
NEWS_API_MODEL="${NEWS_API_MODEL:-gpt-5.1}"
NEWS_API_KEY_PATH="${NEWS_API_KEY_PATH:-api_key.txt}"
NEWS_API_BASE_URL="${NEWS_API_BASE_URL:-}"
NEWS_API_TIMEOUT_SEC="${NEWS_API_TIMEOUT_SEC:-30}"
NEWS_API_MAX_RETRIES="${NEWS_API_MAX_RETRIES:-2}"
if [[ "$NEWS_API_ENABLE" == "1" ]]; then
  NEWS_REFINE_MODE="${NEWS_REFINE_MODE:-api}"                 # local | api
  NEWS_STRUCTURED_MODE="${NEWS_STRUCTURED_MODE:-api}"   # off | heuristic | api
else
  NEWS_REFINE_MODE="${NEWS_REFINE_MODE:-local}"               # local | api
  NEWS_STRUCTURED_MODE="${NEWS_STRUCTURED_MODE:-off}"   # off | heuristic | api
fi
NEWS_REFINE_CACHE_ENABLE="${NEWS_REFINE_CACHE_ENABLE:-1}"
NEWS_STRUCTURED_CACHE_ENABLE="${NEWS_STRUCTURED_CACHE_ENABLE:-1}"

NEWS_DOC_CACHE_PATH="${NEWS_DOC_CACHE_PATH:-}"  # optional unified cache file to reuse directly
# NEWS_DOC_CACHE_PATH="checkpoints/_shared_refine_cache/news_doc_cache_news_2024_2025.json"

NEWS_REFINE_PREWARM_MAX_BATCHES="${NEWS_REFINE_PREWARM_MAX_BATCHES:--1}"
NEWS_REFINE_SHOW_PROGRESS="${NEWS_REFINE_SHOW_PROGRESS:-1}"
NEWS_STRUCTURED_SHOW_PROGRESS="${NEWS_STRUCTURED_SHOW_PROGRESS:-1}"
DELTA_INCLUDE_STRUCTURED_NEWS="${DELTA_INCLUDE_STRUCTURED_NEWS:-0}"  # prompt path is disabled in DELTA; keep structured text off in stable runs
DELTA_STRUCTURED_ENABLE="${DELTA_STRUCTURED_ENABLE:-1}"  # allow DELTA to consume structured event features directly
DELTA_STRUCTURED_FEATURE_DIM="${DELTA_STRUCTURED_FEATURE_DIM:-12}"

DELTA_MODEL_VARIANT="${DELTA_MODEL_VARIANT:-tiny_news_ts}"
DELTA_TEXT_FUSE_LAMBDA="${DELTA_TEXT_FUSE_LAMBDA:-5.0}"
TINY_NEWS_PRESET="${TINY_NEWS_PRESET:-distilbert}"  # distilbert | gpt2 | bert_base | roberta_base | deberta_v3_base | custom
TINY_NEWS_LOADER="${TINY_NEWS_LOADER:-auto}"        # auto | encoder | causal_lm
TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-}"
TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-}"
TINY_NEWS_HIDDEN_SIZE="${TINY_NEWS_HIDDEN_SIZE:-256}"
TINY_NEWS_TEXT_TRAINABLE="${TINY_NEWS_TEXT_TRAINABLE:-0}"
DELTA_TEXT_DIRECT_ENABLE="${DELTA_TEXT_DIRECT_ENABLE:-1}"
DELTA_TEXT_GATE_INIT_BIAS="${DELTA_TEXT_GATE_INIT_BIAS:--2.0}"
DELTA_TEXT_CLIP="${DELTA_TEXT_CLIP:-1.5}"
DELTA_TEXT_MAX_LEN="${DELTA_TEXT_MAX_LEN:-160}"

# Keep the article-level refined-news branch off in the stable baseline.
DELTA_DOC_DIRECT_ENABLE="${DELTA_DOC_DIRECT_ENABLE:-0}"
DELTA_DOC_FUSE_LAMBDA="${DELTA_DOC_FUSE_LAMBDA:-0.0}"
DELTA_DOC_GATE_INIT_BIAS="${DELTA_DOC_GATE_INIT_BIAS:--2.0}"
DELTA_DOC_CLIP="${DELTA_DOC_CLIP:-1.0}"
DELTA_DOC_MAX_LEN="${DELTA_DOC_MAX_LEN:-96}"
DELTA_DOC_MAX_DOCS="${DELTA_DOC_MAX_DOCS:-4}"



case "$TINY_NEWS_PRESET" in
  distilbert)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-distilbert-base-uncased}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-distilbert-base-uncased}"
    ;;
  gpt2)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-gpt2}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-gpt2}"
    ;;
  bert_base)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-bert-base-uncased}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-bert-base-uncased}"
    ;;
  roberta_base)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-roberta-base}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-roberta-base}"
    ;;
  deberta_v3_base)
    TINY_NEWS_MODEL="${TINY_NEWS_MODEL:-microsoft/deberta-v3-base}"
    TINY_NEWS_TOKENIZER="${TINY_NEWS_TOKENIZER:-microsoft/deberta-v3-base}"
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

DELTA_CF_LAMBDA="0.0"
DELTA_CF_MARGIN="0.05"
DELTA_GATE_REG_LAMBDA="0.0"
DELTA_NULL_LAMBDA="0.0"
DELTA_RESIDUAL_MODE="${DELTA_RESIDUAL_MODE:-additive}"     # additive | relative
DELTA_RELATIVE_DENOM_FLOOR="${DELTA_RELATIVE_DENOM_FLOOR:-0.0}"
DELTA_RELATIVE_RATIO_CLIP="${DELTA_RELATIVE_RATIO_CLIP:-0.5}"
DELTA_HEAD_LR_SCALE="1.0"
DELTA_AUX_LAMBDA="0.0"
DELTA_WARMUP_EPOCHS="${DELTA_WARMUP_EPOCHS:-2}"
DELTA_CURRICULUM_EPOCHS="${DELTA_CURRICULUM_EPOCHS:-6}"
DELTA_NULL_WARMUP_STEPS="${DELTA_NULL_WARMUP_STEPS:-1200}"
DELTA_NULL_RAMP_STEPS="${DELTA_NULL_RAMP_STEPS:-1200}"
# =======================
# 2) Sweep spaces (same style as your original)
# =======================
TASK_NAME_BASE="${TASK_NAME_BASE:-[2024-nswelecPRICE-tinynews]}"
TASK_NAME_SUFFIX="${TASK_NAME_SUFFIX:-}"
TASK_NAMES=(
  "${TASK_NAME_BASE}${TASK_NAME_SUFFIX}"
)

NEWS_CHOICES=(
  # "dataset/Rated_Sum_V7_FNT_2019_2020_WAtt2019_combined.json"
  # "dataset/FNT_2019_2020_combined.json"
  "dataset/news_2024_2025_elecprice.json"
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
  --news_gate_enable "$FINAL_GATE_ENABLE"
  --disable_all_gates "$DISABLE_ALL_GATES"
  --delta_gate_init_bias "$DELTA_GATE_INIT_BIAS"
  --delta_internal_gate_in_model "$DELTA_INTERNAL_GATE_IN_MODEL"
  --delta_freeze_feature_modules "$DELTA_FREEZE_FEATURE_MODULES"
  --delta_non_degrade_lambda "$DELTA_NON_DEGRADE_LAMBDA"
  --delta_non_degrade_margin "$DELTA_NON_DEGRADE_MARGIN"
  --delta_sign_lambda "$DELTA_SIGN_LAMBDA"
  --delta_sign_eps "$DELTA_SIGN_EPS"
  --delta_gate_eps "$DELTA_GATE_EPS"
  --delta_sign_tau "$DELTA_SIGN_TAU"
  --delta_sign_mode "$DELTA_SIGN_MODE"
  --delta_sign_external_epochs "$DELTA_SIGN_EXTERNAL_EPOCHS"
  --delta_sign_external_hidden "$DELTA_SIGN_EXTERNAL_HIDDEN"
  --delta_sign_external_dropout "$DELTA_SIGN_EXTERNAL_DROPOUT"
  --delta_sign_external_lr "$DELTA_SIGN_EXTERNAL_LR"
  --delta_sign_external_weight_decay "$DELTA_SIGN_EXTERNAL_WEIGHT_DECAY"
  --delta_sign_external_grad_clip "$DELTA_SIGN_EXTERNAL_GRAD_CLIP"
  --delta_sign_external_patience "$DELTA_SIGN_EXTERNAL_PATIENCE"
  --delta_sign_external_select_metric "$DELTA_SIGN_EXTERNAL_SELECT_METRIC"
  --delta_sign_external_min_delta "$DELTA_SIGN_EXTERNAL_MIN_DELTA"
  --delta_sign_external_lr_factor "$DELTA_SIGN_EXTERNAL_LR_FACTOR"
  --delta_sign_external_lr_patience "$DELTA_SIGN_EXTERNAL_LR_PATIENCE"
  --delta_sign_external_min_lr "$DELTA_SIGN_EXTERNAL_MIN_LR"
  --delta_sign_external_calibrate_bias "$DELTA_SIGN_EXTERNAL_CALIBRATE_BIAS"
  --delta_sign_external_bias_clip "$DELTA_SIGN_EXTERNAL_BIAS_CLIP"
  --delta_sign_external_news_dropout "$DELTA_SIGN_EXTERNAL_NEWS_DROPOUT"
  --delta_sign_external_use_news_weighting "$DELTA_SIGN_EXTERNAL_USE_NEWS_WEIGHTING"
  --delta_sign_external_use_residual_weighting "$DELTA_SIGN_EXTERNAL_USE_RESIDUAL_WEIGHTING"
  --delta_sign_external_use_pos_weight "$DELTA_SIGN_EXTERNAL_USE_POS_WEIGHT"
  --delta_sign_external_pos_weight_floor "$DELTA_SIGN_EXTERNAL_POS_WEIGHT_FLOOR"
  --delta_sign_external_pos_weight_clip "$DELTA_SIGN_EXTERNAL_POS_WEIGHT_CLIP"
  --delta_sign_external_tau "$DELTA_SIGN_EXTERNAL_TAU"
  --delta_sign_external_variant "$DELTA_SIGN_EXTERNAL_VARIANT"
  --delta_sign_external_text_dim "$DELTA_SIGN_EXTERNAL_TEXT_DIM"
  --delta_sign_external_text_max_len "$DELTA_SIGN_EXTERNAL_TEXT_MAX_LEN"
  --delta_gate_loss_weight "$DELTA_GATE_LOSS_WEIGHT"
  --delta_sign_loss_weight "$DELTA_SIGN_LOSS_WEIGHT"
  --delta_mag_loss_weight "$DELTA_MAG_LOSS_WEIGHT"
  --delta_mag_target "$DELTA_MAG_TARGET"
  --delta_mag_max "$DELTA_MAG_MAX"
  --delta_residual_weight_scale "$DELTA_RESIDUAL_WEIGHT_SCALE"
  --gate_null_lambda "$GATE_NULL_LAMBDA"
  --cf_pseudo_margin "$CF_PSEUDO_MARGIN"
  --cf_pseudo_temp "$CF_PSEUDO_TEMP"
  --cf_pseudo_hard "$CF_PSEUDO_HARD"
  --delta_aux_lambda "$DELTA_AUX_LAMBDA"
  --delta_null_lambda "$DELTA_NULL_LAMBDA"
  --delta_residual_mode "$DELTA_RESIDUAL_MODE"
  --delta_relative_denom_floor "$DELTA_RELATIVE_DENOM_FLOOR"
  --delta_relative_ratio_clip "$DELTA_RELATIVE_RATIO_CLIP"
  --delta_warmup_epochs "$DELTA_WARMUP_EPOCHS"
  --delta_curriculum_epochs "$DELTA_CURRICULUM_EPOCHS"
  --delta_null_warmup_steps "$DELTA_NULL_WARMUP_STEPS"
  --delta_null_ramp_steps "$DELTA_NULL_RAMP_STEPS"
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
  --select_metric "$SELECT_METRIC"
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
  --news_refine_mode "$NEWS_REFINE_MODE"
  --news_refine_cache_enable "$NEWS_REFINE_CACHE_ENABLE"
  --news_structured_cache_enable "$NEWS_STRUCTURED_CACHE_ENABLE"
  --news_refine_show_progress "$NEWS_REFINE_SHOW_PROGRESS"
  --news_structured_show_progress "$NEWS_STRUCTURED_SHOW_PROGRESS"
  --news_api_model "$NEWS_API_MODEL"
  --news_api_key_path "$NEWS_API_KEY_PATH"
  --news_api_base_url "$NEWS_API_BASE_URL"
  --news_api_timeout_sec "$NEWS_API_TIMEOUT_SEC"
  --news_api_max_retries "$NEWS_API_MAX_RETRIES"
  --delta_include_structured_news "$DELTA_INCLUDE_STRUCTURED_NEWS"
  --news_structured_mode "$NEWS_STRUCTURED_MODE"
  --delta_structured_enable "$DELTA_STRUCTURED_ENABLE"
  --delta_structured_feature_dim "$DELTA_STRUCTURED_FEATURE_DIM"
  --delta_model_variant "$DELTA_MODEL_VARIANT"
  --tiny_news_model_preset "$TINY_NEWS_PRESET"
  --tiny_news_model "$TINY_NEWS_MODEL"
  --tiny_news_tokenizer "$TINY_NEWS_TOKENIZER"
  --tiny_news_hidden_size "$TINY_NEWS_HIDDEN_SIZE"
  --tiny_news_text_trainable "$TINY_NEWS_TEXT_TRAINABLE"
  --tiny_news_loader "$TINY_NEWS_LOADER"
  --delta_text_direct_enable "$DELTA_TEXT_DIRECT_ENABLE"
  --delta_text_fuse_lambda "$DELTA_TEXT_FUSE_LAMBDA"
  --delta_text_gate_init_bias "$DELTA_TEXT_GATE_INIT_BIAS"
  --delta_text_clip "$DELTA_TEXT_CLIP"
  --delta_text_max_len "$DELTA_TEXT_MAX_LEN"
  --delta_doc_direct_enable "$DELTA_DOC_DIRECT_ENABLE"
  --delta_doc_fuse_lambda "$DELTA_DOC_FUSE_LAMBDA"
  --delta_doc_gate_init_bias "$DELTA_DOC_GATE_INIT_BIAS"
  --delta_doc_clip "$DELTA_DOC_CLIP"
  --delta_doc_max_len "$DELTA_DOC_MAX_LEN"
  --delta_doc_max_docs "$DELTA_DOC_MAX_DOCS"
  --residual_loss "$RESIDUAL_LOSS"
  --stage "$STAGE"
  --delta_val_mode "$DELTA_VAL_MODE"
  --delta_clip "$DELTA_CLIP"
)

normalize_cache_stem() {
  local raw="$1"
  local stem
  stem="$(basename "$raw")"
  stem="${stem%.*}"
  stem="$(printf '%s' "$stem" | sed -E 's/[^0-9A-Za-z]+/_/g; s/^_+//; s/_+$//' | tr '[:upper:]' '[:lower:]')"
  if [[ -z "$stem" ]]; then
    stem="news"
  fi
  printf '%s\n' "$stem"
}

verify_news_doc_cache_file() {
  local news_path="$1"
  local cache_path="$2"
  local verify_output
  if [[ ! -f "$cache_path" ]]; then
    return 1
  fi
  if verify_output="$("$PYTHON_BIN" scripts/verify_refined_news_cache.py \
    --news "$news_path" \
    --cache "$cache_path" \
    --allow-missing 2>&1)"; then
    if [[ -n "$verify_output" ]]; then
      printf '%s\n' "$verify_output" >&2
    fi
    return 0
  fi
  printf '%s\n' "$verify_output" >&2
  return 1
}

resolve_news_doc_cache_write_path() {
  local news_path="$1"
  if [[ -n "$NEWS_DOC_CACHE_PATH" ]]; then
    printf '%s\n' "$NEWS_DOC_CACHE_PATH"
    return
  fi
  local stem
  stem="$(normalize_cache_stem "$news_path")"
  printf 'checkpoints/_shared_refine_cache/news_doc_cache_%s.json\n' "$stem"
}

prepare_news_cache_mode() {
  local news_path="$1"
  local stem doc_cache_path legacy_refine_path legacy_structured_path
  stem="$(normalize_cache_stem "$news_path")"
  doc_cache_path="$(resolve_news_doc_cache_write_path "$news_path")"
  legacy_refine_path="checkpoints/_shared_refine_cache/refine_news_cache_${stem}.json"
  legacy_structured_path="checkpoints/_shared_refine_cache/structured_news_cache_${stem}.json"

  CURRENT_NEWS_DOC_CACHE_PATH="$doc_cache_path"
  CURRENT_NEWS_REFINE_CACHE_READ_PATH=""
  CURRENT_NEWS_STRUCTURED_CACHE_READ_PATH=""
  CURRENT_NEWS_REFINE_PREWARM=1
  CURRENT_NEWS_STRUCTURED_PREWARM=1
  CURRENT_NEWS_DOC_CACHE_EXPLICIT=0
  CURRENT_NEWS_CACHE_MODE="build_mode"

  if [[ -n "$NEWS_DOC_CACHE_PATH" ]]; then
    CURRENT_NEWS_REFINE_CACHE_READ_PATH="$doc_cache_path"
    CURRENT_NEWS_STRUCTURED_CACHE_READ_PATH="$doc_cache_path"
    CURRENT_NEWS_REFINE_PREWARM=0
    CURRENT_NEWS_STRUCTURED_PREWARM=0
    CURRENT_NEWS_DOC_CACHE_EXPLICIT=1
    CURRENT_NEWS_CACHE_MODE="read_only"
    echo "[NEWS_CACHE] mode=read_only cache_path=$doc_cache_path source=explicit"
    if [[ ! -f "$doc_cache_path" ]]; then
      echo "[NEWS_CACHE][ERROR] explicit cache path does not exist: $doc_cache_path" >&2
      return 1
    fi
    if ! verify_news_doc_cache_file "$news_path" "$doc_cache_path"; then
      echo "[NEWS_CACHE][ERROR] explicit cache path is incompatible with current framework identity rules: $doc_cache_path" >&2
      echo "[NEWS_CACHE][ERROR] Repair the cache or point NEWS_DOC_CACHE_PATH to a validated cache file." >&2
      return 1
    fi
    return
  fi

  if [[ -f "$doc_cache_path" ]]; then
    if verify_news_doc_cache_file "$news_path" "$doc_cache_path"; then
      CURRENT_NEWS_REFINE_CACHE_READ_PATH="$doc_cache_path"
      CURRENT_NEWS_STRUCTURED_CACHE_READ_PATH="$doc_cache_path"
      CURRENT_NEWS_REFINE_PREWARM=0
      CURRENT_NEWS_STRUCTURED_PREWARM=0
      CURRENT_NEWS_CACHE_MODE="read_only"
      echo "[NEWS_CACHE] mode=read_only cache_path=$doc_cache_path source=auto_discovered"
      return
    fi

    echo "[NEWS_CACHE][ERROR] auto-discovered cache is incompatible with current framework identity rules: $doc_cache_path" >&2
    echo "[NEWS_CACHE][ERROR] Repair or delete this cache file before rerunning." >&2
    return 1
  fi

  mkdir -p "$(dirname "$doc_cache_path")"
  if [[ -f "$legacy_refine_path" ]]; then
    rm -f "$legacy_refine_path"
  fi
  if [[ -f "$legacy_structured_path" ]]; then
    rm -f "$legacy_structured_path"
  fi

  echo "[NEWS_CACHE] mode=build_mode news_path=$news_path"
  echo "[NEWS_CACHE] target_doc_cache=$doc_cache_path"
  echo "[NEWS_CACHE] cleared legacy_refine_cache=$legacy_refine_path"
  echo "[NEWS_CACHE] cleared legacy_structured_cache=$legacy_structured_path"
}

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
              for horizon in "${HORIZONS[@]}"; do
                run_task="${task}_${base_backbone}_h${horizon}"
                prepare_news_cache_mode "${NEWS_CHOICES[$j]}" || exit 1
                args=( --taskName "$run_task" "${COMMON_ARGS[@]}" )

                args+=( --news_api_enable "$NEWS_API_ENABLE" )
                args+=( --news_doc_cache_path "$CURRENT_NEWS_DOC_CACHE_PATH" )
                args+=( --news_doc_cache_explicit "$CURRENT_NEWS_DOC_CACHE_EXPLICIT" )
                args+=( --news_refine_cache_path "$CURRENT_NEWS_DOC_CACHE_PATH" )
                args+=( --news_refine_cache_read_path "$CURRENT_NEWS_REFINE_CACHE_READ_PATH" )
                args+=( --news_structured_cache_path "$CURRENT_NEWS_DOC_CACHE_PATH" )
                args+=( --news_structured_cache_read_path "$CURRENT_NEWS_STRUCTURED_CACHE_READ_PATH" )
                args+=( --news_refine_prewarm "$CURRENT_NEWS_REFINE_PREWARM" )
                args+=( --news_refine_prewarm_max_batches "$NEWS_REFINE_PREWARM_MAX_BATCHES" )
                args+=( --news_structured_prewarm "$CURRENT_NEWS_STRUCTURED_PREWARM" )
                args+=( --news_path "${NEWS_CHOICES[$j]}" )
                if [[ -n "$tpool" ]]; then
                  args+=( --template_pool "$tpool" )
                fi

                args+=( --news_window_days "${LOOKBACK_WINDOWS[$k]}" )
                args+=( --head_mlp )
                args+=( --patch_dropout "$PATCH_DROPOUT" )
                args+=( --head_dropout "$HEAD_DROPOUT" )
                args+=( --stride "$STRIDE" )
                args+=( --horizon "$horizon" )
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
                if [[ "$run_or_not" == "1" ]]; then
                  echo "==> Running: ${run_task} (base_backbone=${base_backbone}, horizon=${horizon})"
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
done
