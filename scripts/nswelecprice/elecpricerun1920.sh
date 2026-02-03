#!/usr/bin/env bash
set -euo pipefail

# =======================
# 1) 把“原来命令行里的值”全部抽到这里
# =======================
PYTHON_BIN="python"
ENTRY="run.py"
TIME_COL="SETTLEMENTDATE"
VALUE_COL="RRP"
UNIT="$/MWh"
DESCRIPTION="This dataset records the electricity price data in Australia NSW from 2019 to 2020, collected from National electricity market."
REGION="Australia, NSW"


TRAIN_FILE="dataset/2019-2020NSWelecprice/2019To2020NSWData_trainset.csv"
VAL_FILE="dataset/2019-2020NSWelecprice/2019To2020NSWData_valset.csv"
TEST_FILE="dataset/2019-2020NSWelecprice/2019To2020NSWData_testset.csv"


NEWS_TEXT_COL="summary_response"
NEWS_TIME_COL="publication_time"
KEYWORD_PATH="keywords/kw_2.txt"

EPOCHS="50"
KEYWORD_NUMBER="20"
NEWS_WINDOW_DAYS="1"
NEWS_TOPM="999"
NEWS_TOPK="10"
BATCH_SIZE="1"

RL_ALGO="lints"
# RL_ALGO="linucb"

RL_CYCLE_STEPS="1"
SELECT_POLICY_BY="epoch"

TEMPLATE_POOL_1="configs/deltaEmptyNews_template.yaml"
TEMPLATE_POOL_2="configs/deltaWithNews_template.yaml"

NEWS_PATH="dataset/FNT_2019_2020_combined.json"

RESIDUAL_LOSS="mae"
REWARD_METRIC="mae"

#test different news lookback length in days
LOOKBACK_WINDOWS=(
  "1"
  "1"
  "1"
  "5"
  "5"
  "5"
  # "10"
  # "14"
)

# =======================
# 2) 四个 task 的“差异项”用数组列出来（for 循环跑）
# =======================
TASK_NAMES=(
  "[DeltaEmptyNews]NSW_2015_2016"
  "[DeltaWithNews]NSW_2015_2016"
  "[DeltaWithNews]NSW_2019_2020"
  "[Full]NSW_2019_2020"
)

STAGE="delta"

NEWS_CHOICES=(
  # ""
  "dataset/Rated_Sum_V7_FNT_2019_2020_WAtt2019_combined.json"
  # "dataset/V0_Watt_NoSum_news_2015_2020.json"
  # "dataset/FNT_traffic_news.json"
  # "dataset/Sum_V6_news_2015_2016.json"
  # "dataset/NoSum_Watt_2015_2016.json"
  # "dataset/Sum_V5_news_2015_2016.json"
  # "dataset/V0_Watt_RBA_15_16.json"
  # "dataset/Sum_V4_news_2015_2016.json"
  
)

RUN_OR_NOT=(
    ""
    ""
    "1"
    ""
)

scheduler=(
  # "0"
  "1"
)
# 每个 task 对应的 rl_use
RL_USES=(
  "0"  # Pure
  "0"  # NoNews
  "0"  # NoRL
  "1"  # Default
)

# 每个 task 是否带 news_path（空串表示不传这个参数）
NEWS_PATHS=(
  ""         # Pure: no --news_path
  "$NEWS_PATH"         # NoNews: no --news_path
  "$NEWS_PATH"  # NoRL: with --news_path
  "$NEWS_PATH"  # Default: with --news_path
)

# 每个 task 是否带 template_pool（空串表示不传这个参数）
TEMPLATE_POOLS=(
  "$TEMPLATE_POOL_1"  
  ""                     # NoNews: no --template_pool
  "$TEMPLATE_POOL_2"  
  ""                     # Default: no --template_pool
)
null=(
  "0.05"
  # "0.5"
  # "0.1"
  # # "0.05"
  # "0.005"

)

margin=(
  "1"
  # "2"
  # "0.5"
  # "0.1"
)

# adv=(
#   "0.05"
#   # "0.1"
#   # "0.5"
#   # "1"
#   # # "0.01"
#   # "0.005"
# )

LRs=(
  "1e-5"
  # "5e-5"
  # "1e-4"
  # "5e-4"
)



# =======================
# 3) 公共参数（完全来自你原来命令行里的值）
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
  --epochs "$EPOCHS"
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
)

# =======================
# 4) for 循环跑四种 task
# =======================
for i in "${!TASK_NAMES[@]}"; do

  for k in "${!LOOKBACK_WINDOWS[@]}"; do
    for j in "${!NEWS_CHOICES[@]}"; do
      run_or_not="${RUN_OR_NOT[$i]}"
      task="${TASK_NAMES[$i]}"
      rl_use="${RL_USES[$i]}"
      npath="${NEWS_PATHS[$i]}"
      tpool="${TEMPLATE_POOLS[$i]}"

      args=( --taskName "$task" --rl_use "$rl_use" "${COMMON_ARGS[@]}" )

      # 只有需要时才追加可选参数（保持与你原脚本行为一致）
      if [[ -n "$npath" ]]; then
        # args+=( --news_path "$npath" )
        args+=( --news_path "${NEWS_CHOICES[$j]}" )
      fi
      if [[ -n "$tpool" ]]; then
        args+=( --template_pool "$tpool" )
      fi

      # OVERRIDE
      args+=( --news_window_days "${LOOKBACK_WINDOWS[$k]}" )
      args+=( --head_mlp)
      args+=( --patch_dropout 0)
      args+=( --head_dropout 0.1)
      args+=( --news_dropout 0.4)

      
      for null_lambda in "${null[@]}"; do
        for margin_lambda in "${margin[@]}"; do
          for lr in "${LRs[@]}"; do
            for sch in "${scheduler[@]}"; do
              args+=( --delta_null_lambda "$null_lambda")
              args+=( --delta_margin_lambda "$margin_lambda")
              # args+=( --delta_adv_margin "$adv_margin")
              args+=( --lr "$lr")
              args+=( --scheduler "$sch")
              if [[ -n "$run_or_not" ]]; then
                echo "==> Running: $task"
                "$PYTHON_BIN" "$ENTRY" "${args[@]}"
              fi
              # 测试后，删除保存的模型，为了节省空间的不得已之举
              # rm -rf "checkpoints/${task}/best_base_${task}"
              rm -rf "checkpoints/${task}/best_delta_${task}"
            done
          done
        done
      done
      # args+=( --delta_null_lambda 0.05)
      # args+=( --delta_margin_lambda 1.0)
      # args+=( --delta_adv_margin 0.05)

      # if [[ -n "$run_or_not" ]]; then
      #   echo "==> Running: $task"
      #   "$PYTHON_BIN" "$ENTRY" "${args[@]}"
      # fi
      # # 测试后，删除保存的模型，为了节省空间的不得已之举
      # # rm -rf "checkpoints/${task}/best_base_${task}"
      # rm -rf "checkpoints/${task}/best_delta_${task}"
    done
  done
done
