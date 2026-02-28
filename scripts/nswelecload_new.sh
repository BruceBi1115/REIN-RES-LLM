#!/usr/bin/env bash
set -euo pipefail

# =======================
# 1) Core run config
# =======================
PYTHON_BIN="python"
ENTRY="run.py"

TIME_COL="SETTLEMENTDATE"
VALUE_COL="TOTALDEMAND"
UNIT="megawatts"
DESCRIPTION="This dataset records the electricity load demand data in Australia NSW from 2019 to 2020, collected from National electricity market."
REGION="Australia, NSW"

TRAIN_FILE="dataset/2019-2020NSWelecload/elecload_2019-2020_trainset.csv"
VAL_FILE="dataset/2019-2020NSWelecload/elecload_2019-2020_valset.csv"
TEST_FILE="dataset/2019-2020NSWelecload/elecload_2019-2020_testset.csv"

NEWS_TEXT_COL="summary_response"
NEWS_TIME_COL="publication_time"
KEYWORD_PATH="keywords/kw_2.txt"

DELTA_EPOCHS="20"
BASE_EPOCHS="20"
KEYWORD_NUMBER="20"
NEWS_WINDOW_DAYS="1"
NEWS_TOPM="999"
NEWS_TOPK="10"
BATCH_SIZE="1"
RL_ALGO="lints"
RL_CYCLE_STEPS="1"
SELECT_POLICY_BY="epoch"

TEMPLATE_POOL_2="configs/deltaWithNews_template.yaml"
NEWS_PATH="dataset/FNT_2019_2020_combined.json"

# Keep task settings aligned with your existing NSW scripts.
RESIDUAL_LOSS="smooth_l1"
REWARD_METRIC="mae"
STRIDE="48"
HORIZON="48"
PATCH_DROPOUT="0"
HEAD_DROPOUT="0.1"
STAGE="delta"

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
BASE_WEIGHT_DECAY="1e-4"

# -----------------------
# New denoise/gating defaults for upgraded code
# -----------------------
NEWS_GATE_ENABLE="1"
NEWS_GATE_TEMPERATURE="1.0"
NEWS_GATE_FLOOR="0.02"
GATE_LAMBDA="0.10"
GATE_NULL_LAMBDA="0.05"

CF_PSEUDO_MARGIN="0.005"
CF_PSEUDO_TEMP="0.08"
CF_PSEUDO_HARD="0"
CF_MIN_WEIGHT="0.10"

DELTA_CURRICULUM_EPOCHS="6"
DELTA_GRAD_CLIP="1.0"
DELTA_VIOLATION_CAP="1.0"
DELTA_NON_DEGRADE_LAMBDA="0.2"
DELTA_NON_DEGRADE_MARGIN="0.0"

DELTA_LORA_LR_SCALE="1.0"
DELTA_HEAD_LR_SCALE="1.0"
DELTA_OTHER_LR_SCALE="0.5"

DELTA_GATE_INIT_BIAS="-1.0"
DELTA_CLIP="2.5"
DELTA_NEWS_TAIL_TOKENS="180"
DELTA_REL_FLOOR="0.0"
REL_LAMBDA="0.0"

# =======================
# 2) Sweep spaces (same style as your original)
# =======================
TASK_NAMES=(
  "[Model2.2fix]NSW_19_20_LOAD_gateCF"
)

NEWS_CHOICES=(
  "dataset/Rated_Sum_V7_FNT_2019_2020_WAtt2019_combined.json"
)

RUN_OR_NOT=(
  "1"
)

RL_USES=(
  "0"
)

NEWS_PATHS=(
  "$NEWS_PATH"
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

NEWS_DROPOUTS=(
  "0.4"
)

NULL_LAMBDAS=(
  # "0.0"
  "0.01"
)

MARGIN_LAMBDAS=(
  "0.5"
  # "1.0"
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
  --keyword_path "$KEYWORD_PATH"
  --keyword_number "$KEYWORD_NUMBER"
  --news_window_days "$NEWS_WINDOW_DAYS"
  --news_topM "$NEWS_TOPM"
  --news_topK "$NEWS_TOPK"
  --batch_size "$BATCH_SIZE"
  --rl_algo "$RL_ALGO"
  --reward_metric "$REWARD_METRIC"
  --rl_cycle_steps "$RL_CYCLE_STEPS"
  --select_policy_by "$SELECT_POLICY_BY"
  --residual_loss "$RESIDUAL_LOSS"
  --stage "$STAGE"

  --news_gate_enable "$NEWS_GATE_ENABLE"
  --news_gate_temperature "$NEWS_GATE_TEMPERATURE"
  --news_gate_floor "$NEWS_GATE_FLOOR"
  --gate_lambda "$GATE_LAMBDA"
  --gate_null_lambda "$GATE_NULL_LAMBDA"

  --cf_pseudo_margin "$CF_PSEUDO_MARGIN"
  --cf_pseudo_temp "$CF_PSEUDO_TEMP"
  --cf_pseudo_hard "$CF_PSEUDO_HARD"
  --cf_min_weight "$CF_MIN_WEIGHT"

  --delta_curriculum_epochs "$DELTA_CURRICULUM_EPOCHS"
  --delta_grad_clip "$DELTA_GRAD_CLIP"
  --delta_violation_cap "$DELTA_VIOLATION_CAP"
  --delta_non_degrade_lambda "$DELTA_NON_DEGRADE_LAMBDA"
  --delta_non_degrade_margin "$DELTA_NON_DEGRADE_MARGIN"

  --delta_lora_lr_scale "$DELTA_LORA_LR_SCALE"
  --delta_head_lr_scale "$DELTA_HEAD_LR_SCALE"
  --delta_other_lr_scale "$DELTA_OTHER_LR_SCALE"

  --delta_gate_init_bias "$DELTA_GATE_INIT_BIAS"
  --delta_clip "$DELTA_CLIP"
  --delta_news_tail_tokens "$DELTA_NEWS_TAIL_TOKENS"
  --delta_rel_floor "$DELTA_REL_FLOOR"
  --rel_lambda "$REL_LAMBDA"
)

# =======================
# 4) Run combinations
# =======================
for i in "${!TASK_NAMES[@]}"; do
  for k in "${!LOOKBACK_WINDOWS[@]}"; do
    for j in "${!NEWS_CHOICES[@]}"; do
      run_or_not="${RUN_OR_NOT[$i]}"
      task="${TASK_NAMES[$i]}"
      rl_use="${RL_USES[$i]}"
      npath="${NEWS_PATHS[$i]}"
      tpool="${TEMPLATE_POOLS[$i]}"

      for base_backbone in "${BASE_BACKBONES[@]}"; do
        for null_lambda in "${NULL_LAMBDAS[@]}"; do
          for margin_lambda in "${MARGIN_LAMBDAS[@]}"; do
            for lr in "${LRS[@]}"; do
              for sch in "${SCHEDULERS[@]}"; do
                for grad_acc in "${GRAD_ACCS[@]}"; do
                  for news_dropout in "${NEWS_DROPOUTS[@]}"; do
                    run_task="${task}_${base_backbone}"
                    args=( --taskName "$run_task" --rl_use "$rl_use" "${COMMON_ARGS[@]}" )

                    if [[ -n "$npath" ]]; then
                      args+=( --news_path "${NEWS_CHOICES[$j]}" )
                    fi
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

                    args+=( --news_dropout "$news_dropout" )
                    args+=( --grad_acc "$grad_acc" )
                    args+=( --delta_null_lambda "$null_lambda" )
                    args+=( --delta_margin_lambda "$margin_lambda" )
                    args+=( --lr "$lr" )
                    args+=( --scheduler "$sch" )

                    if [[ -n "$run_or_not" ]]; then
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
    done
  done
done
