#!/usr/bin/env bash
set -euo pipefail

# =======================
# 1) 把“原来命令行里的值”全部抽到这里
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
# RL_ALGO="linucb"

RL_CYCLE_STEPS="1"
SELECT_POLICY_BY="epoch"
DEFAULT_POLICY="smart"
DELTA_FREEZE_FEATURE_MODULES="0"
DELTA_AUTO_ALPHA="0"
DELTA_ALPHA_CANDIDATES="1.0"
NEWS_RL_ENABLE="1"
NEWS_RL_ALGO="auto"
NEWS_RL_K_CHOICES="1,2,3,5,7,10"
NEWS_RL_ALLOW_OVER_TOPK="0"
NEWS_RL_EPSILON="0.05"
NEWS_RL_PREFILTER_MULT="4"
NEWS_RL_POOL_CAP="128"
NEWS_RL_REWARD_CLIP="3.0"
NEWS_RL_RECENCY_TAU_HOURS="24"
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
UTILITY_RANK_LAMBDA="0.2"
UTILITY_RANK_MARGIN="0.10"


TEMPLATE_POOL_2="configs/deltaWithNews_template.yaml"

NEWS_PATH="dataset/FNT_2019_2020_combined.json"

RESIDUAL_LOSS="mae"
REWARD_METRIC="mae"

STRIDE="48"
HORIZON="48"
PATCH_DROPOUT="0"
HEAD_DROPOUT="0.1"

STAGE="all"

#test different news lookback length in days
LOOKBACK_WINDOWS=(
  "1"
  # "1"
  # "1"
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
  "[Model2.1]NSW_19_20_LOAD"
)



NEWS_CHOICES=(
  # ""
  "dataset/Rated_Sum_V7_FNT_2019_2020_WAtt2019_combined.json"
  # "dataset/Rated_Sum_V7_FNT_2019_2020_WAtt2019_combined.json"
  # ""
  # "dataset/V0_Watt_NoSum_news_2015_2020.json"
  # "dataset/FNT_traffic_news.json"
  # "dataset/Sum_V6_news_2015_2016.json"
  # "dataset/NoSum_Watt_2015_2016.json"
  # "dataset/Sum_V5_news_2015_2016.json"
  # "dataset/V0_Watt_RBA_15_16.json"
  # "dataset/Sum_V4_news_2015_2016.json"
  
)


#---------basics----------
# 每个 task 是否真的要跑（空串表示不跑），用来快速控制开关
RUN_OR_NOT=(
  "1"
)
# 每个 task 对应的 rl_use
RL_USES=(
  "0"
)
# 每个 task 是否带 news_path（空串表示不传这个参数）
NEWS_PATHS=(
  "$NEWS_PATH"
)
# 每个 task 是否带 template_pool（空串表示不传这个参数）
TEMPLATE_POOLS=(          
  "$TEMPLATE_POOL_2"                      
)
#-------------------


#--------- iteration ---------
scheduler=(
  # "0"
  "1"
)


NEWS_DROPOUT=(
  "0.4"
  # "0.5"
  # "0.4"
  # "0.7"
  # "0.0"
)

null=(
  "0.00"
  # "0.05"
  # "0.5"
  # "0.1"
  # "0.005"
  # "0.001"
  # "0.0001"
  # "0.00001"

)

margin=(
  "10"
  # "10"
  # "20"
  # "50"
  # "100"
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
GRAD_ACCS=(
  "1"
  # "2"
  # "4"
  # "8"
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
  --keyword_number "$KEYWORD_NUMBER"
  --news_window_days "$NEWS_WINDOW_DAYS"
  --news_topM "$NEWS_TOPM"
  --news_topK "$NEWS_TOPK"
  --batch_size "$BATCH_SIZE"
  --rl_algo "$RL_ALGO"
  --reward_metric "$REWARD_METRIC"
  --rl_cycle_steps "$RL_CYCLE_STEPS"
  --select_policy_by "$SELECT_POLICY_BY"
  --default_policy "$DEFAULT_POLICY"
  --news_rl_enable "$NEWS_RL_ENABLE"
  --news_rl_algo "$NEWS_RL_ALGO"
  --news_rl_k_choices "$NEWS_RL_K_CHOICES"
  --news_rl_allow_over_topk "$NEWS_RL_ALLOW_OVER_TOPK"
  --news_rl_epsilon "$NEWS_RL_EPSILON"
  --news_rl_prefilter_mult "$NEWS_RL_PREFILTER_MULT"
  --news_rl_pool_cap "$NEWS_RL_POOL_CAP"
  --news_rl_reward_clip "$NEWS_RL_REWARD_CLIP"
  --news_rl_recency_tau_hours "$NEWS_RL_RECENCY_TAU_HOURS"
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
  --delta_freeze_feature_modules "$DELTA_FREEZE_FEATURE_MODULES"
  --delta_auto_alpha "$DELTA_AUTO_ALPHA"
  --delta_alpha_candidates "$DELTA_ALPHA_CANDIDATES"
  --utility_rank_lambda "$UTILITY_RANK_LAMBDA"
  --utility_rank_margin "$UTILITY_RANK_MARGIN"
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
      args+=( --patch_dropout "$PATCH_DROPOUT")
      args+=( --head_dropout "$HEAD_DROPOUT")
      args+=( --stride "$STRIDE")
      args+=( --horizon "$HORIZON")
      args+=( --base_epochs "$BASE_EPOCHS")
      args+=( --delta_epochs "$DELTA_EPOCHS")
      
      
      for null_lambda in "${null[@]}"; do
        for margin_lambda in "${margin[@]}"; do
          for lr in "${LRs[@]}"; do
            for sch in "${scheduler[@]}"; do
              for grad_acc in "${GRAD_ACCS[@]}"; do
                for news_dropout in "${NEWS_DROPOUT[@]}"; do
                  args+=( --news_dropout "$news_dropout")
                  args+=( --grad_acc "$grad_acc")
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
