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
DESCRIPTION="This dataset records the electricity price data in Australia NSW from 2015 to 2016, collected from National electricity market."
REGION="Australia, NSW"
# DAYFIRST="F"

TRAIN_FILE="dataset/2015-2020NSWelecprice/2015To2020NSWData_trainset.csv"
VAL_FILE="dataset/2015-2020NSWelecprice/2015To2020NSWData_valset.csv"
TEST_FILE="dataset/2015-2020NSWelecprice/2015To2020NSWData_testset.csv"

# TRAIN_FILE="dataset/2015-2016NSWelecprice/2015To2016NSWData_trainset.csv"
# VAL_FILE="dataset/2015-2016NSWelecprice/2015To2016NSWData_valset.csv"
# TEST_FILE="dataset/2015-2016NSWelecprice/2015To2016NSWData_testset.csv"


TRAIN_FILE="dataset/FNT-traffic/traffic_hourly_trainset.csv"
VAL_FILE="dataset/FNT-traffic/traffic_hourly_valset.csv"
TEST_FILE="dataset/FNT-traffic/traffic_hourly_testset.csv"

NEWS_TEXT_COL="full_article"
NEWS_TIME_COL="publication_time"
KEYWORD_PATH="keywords/kw_2.txt"

EPOCHS="50"
KEYWORD_NUMBER="20"
NEWS_WINDOW_DAYS="1"
NEWS_TOPM="999"
NEWS_TOPK="1"
BATCH_SIZE="1"

RL_ALGO="lints"
# RL_ALGO="linucb"

RL_CYCLE_STEPS="1"
SELECT_POLICY_BY="epoch"

TEMPLATE_POOL_1="configs/deltaEmptyNews_template.yaml"
TEMPLATE_POOL_2="configs/deltaWithNews_template.yaml"
# NEWS_PATH="dataset/Sum_V0_rba_media_releases_2015_2016.json"
# NEWS_PATH="dataset/empty.json"
# NEWS_PATH="dataset/Sum_V4_news_2015_2016.json"
# NEWS_PATH="dataset/V0_Watt_RBA_15_16.json"
NEWS_PATH="dataset/V0_Watt_Summarized_news_2015_2020.json"

RESIDUAL_LOSS="mae"
REWARD_METRIC="mae"

#test different news lookback length in days
LOOKBACK_WINDOWS=(
  "1"
  # "5"
  # "5"
  # "5"
  # "5"
  # "5"
  # "10"
  # "14"
)

# =======================
# 2) 四个 task 的“差异项”用数组列出来（for 循环跑）
# =======================
TASK_NAMES=(
  "[DeltaEmptyNews]NSW_2015_2016"
  "[DeltaWithNews]NSW_2015_2016"
  "[DeltaWithNews]NSW_2015_2016"
  "[Full]NSW_2015_2020"
)

STAGE="all"

NEWS_CHOICES=(
  "dataset/V0_Watt_Summarized_news_2015_2020.json"
  # "dataset/V0_Watt_NoSum_news_2015_2020.json"
  # "dataset/FNT_traffic_news.json"
  # "dataset/Sum_V6_news_2015_2016.json"
  # "dataset/NoSum_Watt_2015_2016.json"
  # "dataset/Sum_V5_news_2015_2016.json"
  # "dataset/V0_Watt_RBA_15_16.json"
  # "dataset/Sum_V4_news_2015_2016.json"
  # "dataset/empty.json"
)

RUN_OR_NOT=(
    ""
    ""
    "1"
    ""
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

# =======================
# 3) 公共参数（完全来自你原来命令行里的值）
# =======================
COMMON_ARGS=(
  --time_col "$TIME_COL"
  --value_col "$VALUE_COL"
  --unit "$UNIT"
  --description "$DESCRIPTION"
  --region "$REGION"
  --dayFirst "$DAYFIRST"
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
