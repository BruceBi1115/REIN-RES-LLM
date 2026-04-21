#!/usr/bin/env bash

usage_forecast_runner() {
  local script_name="${RUN_SCRIPT_NAME:-scripts/run_<dataset>.sh}"
  cat <<EOF
Usage:
  bash ${script_name} [--session <session-name>] [--target <session[:window[.pane]]>] [--no-attach] [-- <extra run.py args>]

Examples:
  bash ${script_name}
  bash ${script_name} --session forecast
  bash ${script_name} --session forecast --target forecast:0.1
  bash ${script_name} -- --epochs 5

Notes:
  - Dataset-specific defaults live in ${script_name}.
  - If --session is provided, the command is sent to that tmux target and then attached.
  - Arguments after -- are passed through to run.py unchanged.
EOF
}

set_default() {
  local name="$1"
  local default_value="$2"
  if [[ -z "${!name+x}" || -z "${!name}" ]]; then
    printf -v "$name" "%s" "$default_value"
  fi
}

set_array_default() {
  local name="$1"
  shift

  if ! declare -p "$name" >/dev/null 2>&1; then
    local -n ref="$name"
    ref=("$@")
    return
  fi

  local declaration
  declaration="$(declare -p "$name" 2>/dev/null || true)"
  if [[ "$declaration" != declare\ -a* ]]; then
    echo "[ERROR] $name must be a bash array." >&2
    exit 1
  fi
}

require_nonempty_var() {
  local name="$1"
  if [[ -z "${!name+x}" || -z "${!name}" ]]; then
    echo "[ERROR] Required config '$name' is missing." >&2
    exit 1
  fi
}

require_var_defined() {
  local name="$1"
  if [[ -z "${!name+x}" ]]; then
    echo "[ERROR] Required config '$name' is missing." >&2
    exit 1
  fi
}

require_array_set() {
  local name="$1"
  local declaration
  declaration="$(declare -p "$name" 2>/dev/null || true)"
  if [[ "$declaration" != declare\ -a* ]]; then
    echo "[ERROR] Required array config '$name' is missing." >&2
    exit 1
  fi

  local -n ref="$name"
  if [[ "${#ref[@]}" -eq 0 ]]; then
    echo "[ERROR] Required array config '$name' is empty." >&2
    exit 1
  fi
}

parse_forecast_runner_args() {
  TMUX_SESSION=""
  TMUX_TARGET=""
  TMUX_ATTACH=1
  EXTRA_RUN_ARGS=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session)
        if [[ $# -lt 2 ]]; then
          echo "[ERROR] --session requires a value." >&2
          usage_forecast_runner
          exit 1
        fi
        TMUX_SESSION="$2"
        shift 2
        ;;
      --target)
        if [[ $# -lt 2 ]]; then
          echo "[ERROR] --target requires a value." >&2
          usage_forecast_runner
          exit 1
        fi
        TMUX_TARGET="$2"
        shift 2
        ;;
      --no-attach)
        TMUX_ATTACH=0
        shift
        ;;
      --help|-h)
        usage_forecast_runner
        exit 0
        ;;
      --)
        shift
        EXTRA_RUN_ARGS=("$@")
        break
        ;;
      *)
        echo "[ERROR] Unknown argument: $1" >&2
        usage_forecast_runner
        exit 1
        ;;
    esac
  done
}

maybe_run_in_tmux() {
  if [[ -z "$TMUX_SESSION" || "${RUN_FORECAST_SKIP_TMUX:-0}" == "1" ]]; then
    return
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux is not installed or not on PATH." >&2
    exit 1
  fi

  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "[ERROR] tmux session '$TMUX_SESSION' does not exist." >&2
    exit 1
  fi

  local target="$TMUX_SESSION"
  if [[ -n "$TMUX_TARGET" ]]; then
    target="$TMUX_TARGET"
  fi

  if ! tmux display-message -p -t "$target" '#{session_name}' >/dev/null 2>&1; then
    echo "[ERROR] tmux target '$target' does not exist." >&2
    exit 1
  fi

  local repo_root="${REPO_ROOT:-}"
  if [[ -z "$repo_root" ]]; then
    repo_root="$(cd -- "${SCRIPT_DIR:-.}/.." && pwd)"
  fi

  local script_path="${RUN_SCRIPT_PATH:-}"
  if [[ -z "$script_path" ]]; then
    echo "[ERROR] RUN_SCRIPT_PATH is not set." >&2
    exit 1
  fi

  local -a cmd=(env RUN_FORECAST_SKIP_TMUX=1 bash "$script_path")
  if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
    cmd+=(-- "${EXTRA_RUN_ARGS[@]}")
  fi

  local quoted_cmd quoted_repo_root send_cmd
  printf -v quoted_cmd "%q " "${cmd[@]}"
  quoted_cmd="${quoted_cmd% }"
  printf -v quoted_repo_root "%q" "$repo_root"
  send_cmd="cd $quoted_repo_root && $quoted_cmd"

  echo "==> Sending command to tmux target '$target'"
  echo "==> $send_cmd"
  tmux send-keys -t "$target" "$send_cmd" C-m

  if [[ "$TMUX_ATTACH" == "0" ]]; then
    exit 0
  fi

  if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "$TMUX_SESSION"
  fi

  exec tmux attach-session -t "$TMUX_SESSION"
}

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
for module_name in ["matplotlib", "openai", "pandas", "torch", "transformers"]:
    importlib.import_module(module_name)
PY
    then
      echo "$resolved"
      return 0
    fi
  done
  return 1
}

prepare_chronological_timeseries_splits() {
  "$PYTHON_BIN" - <<'PY'
import os
import pandas as pd

raw_file = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine.csv"
source_train = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_trainset.csv"
source_val = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_valset.csv"
source_test = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_testset.csv"
out_train = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_trainset.csv"
out_val = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_valset.csv"
out_test = "dataset/NAS_14ticker_22_23_combine/NAS_14ticker_22_23_combine_chrono_testset.csv"
time_col = "date"

def parse_time_series(series):
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], dayfirst=True, errors="coerce")
    return parsed

for path in (raw_file, source_train, source_val, source_test):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

src_train = pd.read_csv(source_train)
src_val = pd.read_csv(source_val)
src_test = pd.read_csv(source_test)
full_df = pd.read_csv(raw_file)

expected_total = len(src_train) + len(src_val) + len(src_test)
if len(full_df) != expected_total:
    raise ValueError(
        f"NAS_14ticker split source mismatch: full={len(full_df)} split_sum={expected_total}"
    )

parsed = parse_time_series(full_df[time_col])
if parsed.isna().any():
    bad_rows = full_df.loc[parsed.isna(), [time_col]].head(5).to_dict("records")
    raise ValueError(f"Failed to parse {time_col} in raw file. examples={bad_rows}")

full_sorted = full_df.copy()
full_sorted["_parsed_time"] = parsed
full_sorted = full_sorted.sort_values("_parsed_time", kind="mergesort").reset_index(drop=True)
full_sorted = full_sorted.drop(columns=["_parsed_time"])

n_train = len(src_train)
n_val = len(src_val)
train_df = full_sorted.iloc[:n_train].copy()
val_df = full_sorted.iloc[n_train:n_train + n_val].copy()
test_df = full_sorted.iloc[n_train + n_val:].copy()

for path, df in ((out_train, train_df), (out_val, val_df), (out_test, test_df)):
    df.to_csv(path, index=False)

def _span(df):
    ts = parse_time_series(df[time_col])
    return str(ts.min()), str(ts.max()), int(len(df))

tr_min, tr_max, tr_n = _span(train_df)
va_min, va_max, va_n = _span(val_df)
te_min, te_max, te_n = _span(test_df)
print(
    "[DATASET] prepared chronological splits "
    f"train={tr_n}[{tr_min} -> {tr_max}] "
    f"val={va_n}[{va_min} -> {va_max}] "
    f"test={te_n}[{te_min} -> {te_max}]"
)
PY
}

init_common_defaults() {
  set_default ENTRY "run.py"
  set_default SAVE_DIR "./checkpoints"
  set_default RUN_NAME ""
  set_default STAGE "all"
  set_default SEED "2024"
  set_default GPU_ID "0"

  set_default NEWS_TEXT_COL "content"
  set_default NEWS_TIME_COL "date"
  set_default NEWS_TZ ""
  set_default NEWS_WINDOW_DAYS "1"

  set_default HISTORY_LEN "48"
  set_default STRIDE "1"
  set_default PATCH_LEN "4"
  set_default PATCH_STRIDE "4"
  set_default BATCH_SIZE "16"
  set_default DELTA_EPOCHS "30"
  set_default BASE_EPOCHS "40"
  set_default WEIGHT_DECAY "1e-5"
  set_default WARMUP_RATIO "0.1"
  set_default EARLY_STOP_PATIENCE "5"
  set_default SELECT_METRIC "mae"
  set_default NORMALIZATION_MODE "robust_quantile"
  set_default NORM_QUANTILE_LOW "0.25"
  set_default NORM_QUANTILE_HIGH "0.75"
  set_default ZSCORE_EPS "1e-6"
  set_default VOLATILITY_BIN_TIERS "10"
  set_default DAY_FIRST "0"

  set_default BASE_HIDDEN_DIM "256"
  set_default BASE_MOVING_AVG "25"
  set_default BASE_DROPOUT "0.0"
  set_default BASE_LOSS "smooth_l1"
  set_default BASE_LR "1e-3"
  set_default BASE_WEIGHT_DECAY "1e-5"

  set_default NEWS_API_ENABLE "1"
  set_default NEWS_API_MODEL "gpt-5.1"
  set_default NEWS_API_KEY_PATH ".secrets/api_key.txt"
  set_default NEWS_API_BASE_URL ""
  set_default NEWS_API_TIMEOUT_SEC "30"
  set_default NEWS_API_MAX_RETRIES "2"

  set_default DELTA_V3_REGIME_BANK_PATH ""
  set_default DELTA_V3_REFINED_BANK_BUILD "0"
  set_default DELTA_V3_SCHEMA_VARIANT "load"
  set_default DELTA_V3_TEXT_ENCODER_MODEL_ID "intfloat/e5-small-v2"
  set_default DELTA_V3_TEXT_ENCODER_MAX_LENGTH "256"
  set_default DELTA_V3_REGIME_TAU_DAYS "5.0"
  set_default DELTA_V3_REGIME_EMA_ALPHA "0.5"
  set_default DELTA_V3_REGIME_EMA_WINDOW "5"
  set_default DELTA_V3_ARCH "patchtst_regime_modulation"
  set_default DELTA_V3_HIDDEN_SIZE "128"
  set_default DELTA_V3_NUM_LAYERS "2"
  set_default DELTA_V3_NUM_HEADS "4"
  set_default DELTA_V3_PATCH_LEN "8"
  set_default DELTA_V3_PATCH_STRIDE "4"
  set_default DELTA_V3_DROPOUT "0.1"
  set_default DELTA_V3_USE_BASE_HIDDEN "1"
  set_default DELTA_V3_SLOW_WEIGHT "1.0"
  set_default DELTA_V3_SHAPE_WEIGHT "1.0"
  set_default DELTA_V3_SPIKE_WEIGHT "1.0"
  set_default DELTA_V3_SPIKE_GATE_THRESHOLD "0.8"
  set_default DELTA_V3_SPIKE_K "3.0"
  set_default DELTA_V3_SPIKE_TARGET_PCT "0.10"
  set_default DELTA_V3_SPIKE_GATE_LOSS_WEIGHT "0.25"
  set_default DELTA_V3_NEWS_BLANK_PROB "0.3"
  set_default DELTA_V3_CONSISTENCY_WEIGHT "0.05"
  set_default DELTA_V3_COUNTERFACTUAL_WEIGHT "0.1"
  set_default DELTA_V3_COUNTERFACTUAL_MARGIN "0.02"
  set_default DELTA_V3_INACTIVE_RESIDUAL_WEIGHT "0.5"
  set_default DELTA_V3_SPIKE_BIAS_L2 "1e-3"
  set_default DELTA_V3_ACTIVE_MASS_THRESHOLD "0.7"
  set_default DELTA_V3_LAMBDA_MIN "0.05"
  set_default DELTA_V3_LAMBDA_TS_CAP "0.30"
  set_default DELTA_V3_LAMBDA_NEWS_CAP "0.12"
  set_default DELTA_V3_LAMBDA_MAX "0.60"
  # Per-backbone lambda_max overrides (plan P0.2). Used inside the main loop to select
  # an effective cap for each backbone; empty means "use DELTA_V3_LAMBDA_MAX".
  set_default DELTA_V3_LAMBDA_MAX_MLP ""
  set_default DELTA_V3_LAMBDA_MAX_DLINEAR ""
  set_default DELTA_V3_LAMBDA_MAX_NLINEAR ""
  set_default DELTA_V3_LAMBDA_MAX_PATCHTST ""
  set_default DELTA_V3_SHAPE_GAIN_CAP "0.50"
  set_default DELTA_V3_SHAPE_GAIN_L2_WEIGHT "0.01"
  set_default DELTA_V3_HARD_GATE_MASS_THRESHOLD "0.0"
  # M5: direction-aware aux loss (cosine agreement between delta contribution and true residual).
  set_default DELTA_V3_DIRECTION_WEIGHT "0.05"
  # M1: residual-history channel (1=on, 0=off). Adds a detrended history channel to the delta encoder.
  set_default DELTA_V3_RESIDUAL_HISTORY_CHANNEL "1"
  set_default DELTA_V3_SPIKE_BIAS_CAP "0.75"
  set_default DELTA_V3_SELECTION_COUNTERFACTUAL_GAIN_MIN "0.01"
  set_default DELTA_V3_SELECTION_LAMBDA_SATURATION_MAX_PCT "0.35"
  set_default DELTA_V3_HARD_RESIDUAL_FRAC "0.6"
  set_default DELTA_V3_HARD_RESIDUAL_PCT "0.10"
  set_default DELTA_V3_PRETRAIN_EPOCHS "12"
  set_default DELTA_V3_PRETRAIN_LR "1e-3"
  set_default DELTA_V3_SCHEDULER "warmup_cosine"
  set_default DELTA_V3_WARMUP_PCT "0.05"
  set_default DELTA_V3_MIN_LR_RATIO "0.05"
  set_default DELTA_V3_PRETRAIN_WARMUP_PCT "0.10"
  set_default DELTA_V3_PRICE_WINSOR_LOW "0.005"
  set_default DELTA_V3_PRICE_WINSOR_HIGH "0.995"
  set_default DELTA_V3_GRAD_CLIP "1.0"
  set_default DELTA_V3_EVAL_PERMUTATION_SEED "2024"
  set_default DELTA_V3_SELECT_METRIC "mae"

  set_default SPIKE_CLIP_THRESHOLD "0.0" 

  set_default ID_COL ""
  set_default FREQ_MIN ""
  set_default TASK_NAME_SUFFIX ""

  set_array_default BASE_BACKBONES "mlp"
  set_array_default HORIZONS "48"
  set_array_default LRS "1e-4"
  set_array_default SCHEDULERS "1"
  set_array_default GRAD_ACCS "1"
  set_array_default DELTA_V3_ACTIVE_MASS_THRESHOLDS "$DELTA_V3_ACTIVE_MASS_THRESHOLD"

}

validate_dataset_config() {
  require_nonempty_var DATASET_KEY
  require_nonempty_var TIME_COL
  require_nonempty_var VALUE_COL
  require_var_defined UNIT
  require_nonempty_var DESCRIPTION
  require_nonempty_var REGION
  require_nonempty_var TRAIN_FILE
  require_nonempty_var VAL_FILE
  require_nonempty_var TEST_FILE
  if [[ "${STAGE:-all}" == "base" ]]; then
    require_var_defined DEFAULT_NEWS_PATH
  else
    require_nonempty_var DEFAULT_NEWS_PATH
  fi
  require_nonempty_var BATCH_SIZE
  require_nonempty_var TASK_NAME_BASE

  require_array_set BASE_BACKBONES
  require_array_set HORIZONS
  require_array_set LRS
  require_array_set SCHEDULERS
  require_array_set GRAD_ACCS
  require_array_set DELTA_V3_ACTIVE_MASS_THRESHOLDS

  if [[ -n "${PRE_RUN_HOOK:-}" ]] && ! declare -F "$PRE_RUN_HOOK" >/dev/null 2>&1; then
    echo "[ERROR] PRE_RUN_HOOK '$PRE_RUN_HOOK' is not defined." >&2
    exit 1
  fi
}

build_run_args() {
  local task_name="$1"
  local base_backbone="$2"
  local horizon="$3"
  local lr="$4"
  local scheduler="$5"
  local grad_acc="$6"
  local news_path="$7"

  RUN_ARGS=(
    --taskName "$task_name"
    --stage "$STAGE"
    --seed "$SEED"
    --gpu "$GPU_ID"
    --save_dir "$SAVE_DIR"
    --dataset_key "$DATASET_KEY"
    --train_file "$TRAIN_FILE"
    --val_file "$VAL_FILE"
    --test_file "$TEST_FILE"
    --time_col "$TIME_COL"
    --value_col "$VALUE_COL"
    --unit "$UNIT"
    --description "$DESCRIPTION"
    --region "$REGION"
    --history_len "$HISTORY_LEN"
    --horizon "$horizon"
    --stride "$STRIDE"
    --batch_size "$BATCH_SIZE"
    --epochs "$DELTA_EPOCHS"
    --base_epochs "$BASE_EPOCHS"
    --lr "$lr"
    --weight_decay "$WEIGHT_DECAY"
    --grad_accum "$grad_acc"
    --scheduler "$scheduler"
    --warmup_ratio "$WARMUP_RATIO"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --select_metric "$SELECT_METRIC"
    --normalization_mode "$NORMALIZATION_MODE"
    --norm_quantile_low "$NORM_QUANTILE_LOW"
    --norm_quantile_high "$NORM_QUANTILE_HIGH"
    --zscore_eps "$ZSCORE_EPS"
    --volatility_bin_tiers "$VOLATILITY_BIN_TIERS"
    --patch_len "$PATCH_LEN"
    --patch_stride "$PATCH_STRIDE"
    --base_backbone "$base_backbone"
    --base_hidden_dim "$BASE_HIDDEN_DIM"
    --base_moving_avg "$BASE_MOVING_AVG"
    --base_dropout "$BASE_DROPOUT"
    --base_loss "$BASE_LOSS"
    --base_lr "$BASE_LR"
    --base_weight_decay "$BASE_WEIGHT_DECAY"
    --news_path "$news_path"
    --news_text_col "$NEWS_TEXT_COL"
    --news_time_col "$NEWS_TIME_COL"
    --news_window_days "$NEWS_WINDOW_DAYS"
    --news_api_enable "$NEWS_API_ENABLE"
    --news_api_model "$NEWS_API_MODEL"
    --news_api_key_path "$NEWS_API_KEY_PATH"
    --news_api_base_url "$NEWS_API_BASE_URL"
    --news_api_timeout_sec "$NEWS_API_TIMEOUT_SEC"
    --news_api_max_retries "$NEWS_API_MAX_RETRIES"
    --delta_v3_refined_bank_build "$DELTA_V3_REFINED_BANK_BUILD"
    --delta_v3_schema_variant "$DELTA_V3_SCHEMA_VARIANT"
    --delta_v3_text_encoder_model_id "$DELTA_V3_TEXT_ENCODER_MODEL_ID"
    --delta_v3_text_encoder_max_length "$DELTA_V3_TEXT_ENCODER_MAX_LENGTH"
    --delta_v3_regime_tau_days "$DELTA_V3_REGIME_TAU_DAYS"
    --delta_v3_regime_ema_alpha "$DELTA_V3_REGIME_EMA_ALPHA"
    --delta_v3_regime_ema_window "$DELTA_V3_REGIME_EMA_WINDOW"
    --delta_v3_arch "$DELTA_V3_ARCH"
    --delta_v3_hidden_size "$DELTA_V3_HIDDEN_SIZE"
    --delta_v3_num_layers "$DELTA_V3_NUM_LAYERS"
    --delta_v3_num_heads "$DELTA_V3_NUM_HEADS"
    --delta_v3_patch_len "$DELTA_V3_PATCH_LEN"
    --delta_v3_patch_stride "$DELTA_V3_PATCH_STRIDE"
    --delta_v3_dropout "$DELTA_V3_DROPOUT"
    --delta_v3_use_base_hidden "$DELTA_V3_USE_BASE_HIDDEN"
    --delta_v3_slow_weight "$DELTA_V3_SLOW_WEIGHT"
    --delta_v3_shape_weight "$DELTA_V3_SHAPE_WEIGHT"
    --delta_v3_spike_weight "$DELTA_V3_SPIKE_WEIGHT"
    --delta_v3_spike_gate_threshold "$DELTA_V3_SPIKE_GATE_THRESHOLD"
    --delta_v3_spike_k "$DELTA_V3_SPIKE_K"
    --delta_v3_spike_target_pct "$DELTA_V3_SPIKE_TARGET_PCT"
    --delta_v3_spike_gate_loss_weight "$DELTA_V3_SPIKE_GATE_LOSS_WEIGHT"
    --delta_v3_news_blank_prob "$DELTA_V3_NEWS_BLANK_PROB"
    --delta_v3_consistency_weight "$DELTA_V3_CONSISTENCY_WEIGHT"
    --delta_v3_counterfactual_weight "$DELTA_V3_COUNTERFACTUAL_WEIGHT"
    --delta_v3_counterfactual_margin "$DELTA_V3_COUNTERFACTUAL_MARGIN"
    --delta_v3_inactive_residual_weight "$DELTA_V3_INACTIVE_RESIDUAL_WEIGHT"
    --delta_v3_spike_bias_l2 "$DELTA_V3_SPIKE_BIAS_L2"
    --delta_v3_active_mass_threshold "$DELTA_V3_ACTIVE_MASS_THRESHOLD"
    --delta_v3_lambda_min "$DELTA_V3_LAMBDA_MIN"
    --delta_v3_lambda_ts_cap "$DELTA_V3_LAMBDA_TS_CAP"
    --delta_v3_lambda_news_cap "$DELTA_V3_LAMBDA_NEWS_CAP"
    --delta_v3_lambda_max "$DELTA_V3_LAMBDA_MAX"
    --delta_v3_shape_gain_cap "$DELTA_V3_SHAPE_GAIN_CAP"
    --delta_v3_shape_gain_l2_weight "$DELTA_V3_SHAPE_GAIN_L2_WEIGHT"
    --delta_v3_hard_gate_mass_threshold "$DELTA_V3_HARD_GATE_MASS_THRESHOLD"
    --delta_v3_direction_weight "$DELTA_V3_DIRECTION_WEIGHT"
    --delta_v3_residual_history_channel "$DELTA_V3_RESIDUAL_HISTORY_CHANNEL"
    --delta_v3_spike_bias_cap "$DELTA_V3_SPIKE_BIAS_CAP"
    --delta_v3_selection_counterfactual_gain_min "$DELTA_V3_SELECTION_COUNTERFACTUAL_GAIN_MIN"
    --delta_v3_selection_lambda_saturation_max_pct "$DELTA_V3_SELECTION_LAMBDA_SATURATION_MAX_PCT"
    --delta_v3_hard_residual_frac "$DELTA_V3_HARD_RESIDUAL_FRAC"
    --delta_v3_hard_residual_pct "$DELTA_V3_HARD_RESIDUAL_PCT"
    --delta_v3_pretrain_epochs "$DELTA_V3_PRETRAIN_EPOCHS"
    --delta_v3_pretrain_lr "$DELTA_V3_PRETRAIN_LR"
    --delta_v3_scheduler "$DELTA_V3_SCHEDULER"
    --delta_v3_warmup_pct "$DELTA_V3_WARMUP_PCT"
    --delta_v3_min_lr_ratio "$DELTA_V3_MIN_LR_RATIO"
    --delta_v3_pretrain_warmup_pct "$DELTA_V3_PRETRAIN_WARMUP_PCT"
    --delta_v3_price_winsor_low "$DELTA_V3_PRICE_WINSOR_LOW"
    --delta_v3_price_winsor_high "$DELTA_V3_PRICE_WINSOR_HIGH"
    --delta_v3_grad_clip "$DELTA_V3_GRAD_CLIP"
    --delta_v3_eval_permutation_seed "$DELTA_V3_EVAL_PERMUTATION_SEED"
    --delta_v3_select_metric "$DELTA_V3_SELECT_METRIC"
    --spike_clip_threshold "$SPIKE_CLIP_THRESHOLD"
  )

  if [[ -n "$RUN_NAME" ]]; then
    RUN_ARGS+=( --run_name "$RUN_NAME" )
  fi
  if [[ -n "$ID_COL" ]]; then
    RUN_ARGS+=( --id_col "$ID_COL" )
  fi
  if [[ -n "$FREQ_MIN" ]]; then
    RUN_ARGS+=( --freq_min "$FREQ_MIN" )
  fi
  if [[ -n "$NEWS_TZ" ]]; then
    RUN_ARGS+=( --news_tz "$NEWS_TZ" )
  fi
  if [[ -n "$DELTA_V3_REGIME_BANK_PATH" ]]; then
    RUN_ARGS+=( --delta_v3_regime_bank_path "$DELTA_V3_REGIME_BANK_PATH" )
  fi
  if [[ "$DAY_FIRST" == "1" ]]; then
    RUN_ARGS+=( --dayFirst )
  fi
}

run_forecast_dataset_main() {
  parse_forecast_runner_args "$@"
  maybe_run_in_tmux
  init_common_defaults
  validate_dataset_config

  PYTHON_BIN="${PYTHON_BIN:-}"
  if ! PYTHON_BIN="$(pick_python_bin "$PYTHON_BIN")"; then
    echo "[ERROR] No suitable Python found for this project." >&2
    echo "        Need modules: matplotlib, openai, pandas, torch, transformers" >&2
    echo "        You can run with an explicit interpreter, e.g.:" >&2
    echo "        PYTHON_BIN=/path/to/env/bin/python bash ${RUN_SCRIPT_NAME:-scripts/run_<dataset>.sh}" >&2
    exit 1
  fi
  echo "[env] Using PYTHON_BIN=$PYTHON_BIN"

  NEWS_PATH="${NEWS_PATH:-$DEFAULT_NEWS_PATH}"
  set_array_default NEWS_CHOICES "$NEWS_PATH"

  if [[ -n "${PRE_RUN_HOOK:-}" ]]; then
    "$PRE_RUN_HOOK"
  fi

  echo "[config] dataset=$DATASET_KEY stage=$STAGE task_base=$TASK_NAME_BASE"
  for news_path in "${NEWS_CHOICES[@]}"; do
    for base_backbone in "${BASE_BACKBONES[@]}"; do
      # Plan P0.2: backbone-aware lambda_max. When the per-backbone override is non-empty,
      # it supersedes DELTA_V3_LAMBDA_MAX for this backbone only; restored after the run.
      local _lambda_max_saved="$DELTA_V3_LAMBDA_MAX"
      case "$base_backbone" in
        mlp)      [[ -n "${DELTA_V3_LAMBDA_MAX_MLP:-}"      ]] && DELTA_V3_LAMBDA_MAX="$DELTA_V3_LAMBDA_MAX_MLP" ;;
        dlinear)  [[ -n "${DELTA_V3_LAMBDA_MAX_DLINEAR:-}"  ]] && DELTA_V3_LAMBDA_MAX="$DELTA_V3_LAMBDA_MAX_DLINEAR" ;;
        nlinear)  [[ -n "${DELTA_V3_LAMBDA_MAX_NLINEAR:-}"  ]] && DELTA_V3_LAMBDA_MAX="$DELTA_V3_LAMBDA_MAX_NLINEAR" ;;
        patchtst) [[ -n "${DELTA_V3_LAMBDA_MAX_PATCHTST:-}" ]] && DELTA_V3_LAMBDA_MAX="$DELTA_V3_LAMBDA_MAX_PATCHTST" ;;
      esac
      for lr in "${LRS[@]}"; do
        for scheduler in "${SCHEDULERS[@]}"; do
          for grad_acc in "${GRAD_ACCS[@]}"; do
            for horizon in "${HORIZONS[@]}"; do
              for active_mass_threshold in "${DELTA_V3_ACTIVE_MASS_THRESHOLDS[@]}"; do
                local task_name
                DELTA_V3_ACTIVE_MASS_THRESHOLD="$active_mass_threshold"
                task_name="${TASK_NAME_BASE}__lr${lr}__ga${grad_acc}__sch${scheduler}${TASK_NAME_SUFFIX}"
                build_run_args "$task_name" "$base_backbone" "$horizon" "$lr" "$scheduler" "$grad_acc" "$news_path"
                if [[ ${#EXTRA_RUN_ARGS[@]} -gt 0 ]]; then
                  RUN_ARGS+=( "${EXTRA_RUN_ARGS[@]}" )
                fi

                echo "==> Running dataset=$DATASET_KEY backbone=$base_backbone horizon=$horizon lr=$lr grad_acc=$grad_acc lambda_max=$DELTA_V3_LAMBDA_MAX active_mass_threshold=$DELTA_V3_ACTIVE_MASS_THRESHOLD stage=$STAGE"
                "$PYTHON_BIN" "$ENTRY" "${RUN_ARGS[@]}"
              done
            done
          done
        done
      done
      DELTA_V3_LAMBDA_MAX="$_lambda_max_saved"
    done
  done
}
