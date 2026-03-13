#!/bin/bash

############################
# 固定不变的参数
############################
GPU=1
TIME_COL="date"
VALUE_COL="OT"
UNIT="$/MWh"
DESC="This dataset records the electricity transformer temperature."
REGION="China"

TRAIN_FILE="dataset/ETTm2/ETTm2_trainset.csv"
VAL_FILE="dataset/ETTm2/ETTm2_valset.csv"
TEST_FILE="dataset/ETTm2/ETTm2_testset.csv"

NEWS_PATH="dataset/Summarized_news_2015_2020.json"
NEWS_TEXT_COL="summary_response"
NEWS_TIME_COL="date"

EPOCHS=30
KEYWORD_NUM=20
NEWS_WINDOW=7
NEWS_TOPM=20
NEWS_TOPK=5
BATCH_SIZE=1
SELECT_METRIC="mse"

BASE_TASK="TestEttm2withElecpriceNewsfrom2015To2020"

############################
# 实验配置
# name | use_news
############################
CONFIGS=(
  "Normal|1"
  "NoNews|0"
)

############################
# 主循环
############################
for CFG in "${CONFIGS[@]}"; do
  IFS="|" read TAG USE_NEWS <<< "$CFG"

  if [ "$TAG" == "Normal" ]; then
    TASK_NAME="${BASE_TASK}"
  else
    TASK_NAME="[${TAG}]${BASE_TASK}"
  fi

  CMD="python run.py \
    --taskName \"${TASK_NAME}\" \
    --time_col ${TIME_COL} \
    --value_col ${VALUE_COL} \
    --gpu ${GPU} \
    --unit \"${UNIT}\" \
    --description \"${DESC}\" \
    --region ${REGION} \
    --train_file ${TRAIN_FILE} \
    --val_file ${VAL_FILE} \
    --test_file ${TEST_FILE} \
    --epochs ${EPOCHS} \
    --news_window_days ${NEWS_WINDOW} \
    --news_topM ${NEWS_TOPM} \
    --news_topK ${NEWS_TOPK} \
    --batch_size ${BATCH_SIZE} \
    --select_metric ${SELECT_METRIC} \
  "

  if [ "$USE_NEWS" -eq 1 ]; then
    CMD+=" \
      --news_path ${NEWS_PATH} \
      --news_text_col ${NEWS_TEXT_COL} \
      --news_time_col ${NEWS_TIME_COL} \
    "
  fi

  echo "========================================"
  echo "Running: ${TASK_NAME}"
  echo "USE_NEWS=${USE_NEWS}"
  echo "========================================"

  eval $CMD
done
