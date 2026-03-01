# REIN-LLM-Forecast 技术框架说明

## 1. 目标与核心设计

该项目是一个两阶段时序预测框架，核心思想是把预测拆成:

- `base` 阶段: 用纯时序模型学习稳定的基础预测 `base_pred`。
- `delta` 阶段: 用 LLM + LoRA 学习新闻驱动的残差修正 `delta_pred`。
- 最终预测: `final_pred = base_pred + delta_pred`（先在 z-space 里相加，再反归一化）。

主入口文件:

- [`run.py`](/home/brucebi/Projects/REIN-LLM-Forecast/run.py)
- [`src/base_delta_decoouple_trainer.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/base_delta_decoouple_trainer.py)

---

## 2. 端到端执行链路

### 2.1 运行阶段控制

`run.py` 解析参数后调用 `main_train(args)`，支持:

- `stage=base`: 仅训练/测试基线模型
- `stage=delta`: 读取已有 base checkpoint，只训练 delta
- `stage=all`: 先跑 `base`，再跑 `delta`

### 2.2 数据与环境初始化

`setup_env_and_data()` 主要做 6 件事:

1. 读取 `train/val/test`（CSV 或 Parquet）。
2. 从训练集计算全局 z-score 统计量: `mu_global`、`sigma_global`。
3. 用 `SlidingDataset` 构建滑窗样本（窗口 `L`，预测步长 `H`）。
4. 加载新闻 JSON，并过滤空新闻文本。
5. 加载关键词文件与 prompt 模板。
6. 计算 train/val/test 的波动率分箱（volatility bin）。

关键代码:

- [`src/data_construction/data.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/data_construction/data.py)
- [`src/news_rules.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/news_rules.py)
- [`src/data_construction/prompt.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/data_construction/prompt.py)

---

## 3. 预测逻辑（核心）

## 3.1 Base 阶段（纯时序）

模型位于:

- [`src/base_backbone.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/base_backbone.py)

可选 backbone:

- `DLinearBackbone`
- `MLPBackbone`

输入输出:

- 输入: `history_z`，形状 `(B, L)`
- 输出: `base_pred_z`，形状 `(B, H)`

监督信号:

- `target_z`（由全局 z-score 标准化得到）

损失:

- `mse` / `mae` / `smooth_l1`（可配）

checkpoint:

- `checkpoints/<task>/best_base_<task>/`
- 包含 `base_backbone.pt`、`meta.json`、可选 `trainer_state.pt`

## 3.2 Delta 阶段（LLM 残差学习）

### 3.2.1 batch 构建（`build_batch_inputs`）

函数输出:

- 文本 token: `input_ids`, `attn`
- 时序 patch: `ts_patches`, `ts_patch_mask`
- 监督目标: `targets_z`
- 新闻相关标签: `rel_labels`
- 可选新闻 RL 元数据

每个样本的处理流程:

1. `history/target` 转换到 z-space。
2. 从 `history_z` 切 patch（`patch_len`, `patch_stride`）。
3. 根据 `target_time` 在 lookback 窗口内取候选新闻。
4. 按策略选新闻（默认 `smart`），可再做 utility rerank。
5. 可选新闻 bandit 再决定 `K` 和具体条目。
6. 模板填充: `HISTORY + NEWS + 元信息`，生成 prompt。

### 3.2.2 Delta 模型前向（`TSForecastRegressor`）

模型实现:

- [`src/model2.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/model2.py)

技术路径:

1. patch encoder: `MLP + gate + dropout + patch position embedding`。
2. 把文本 embedding 与 patch embedding 拼接成 soft tokens 输入 LLM。
3. 取 hidden states，做最后 K 层可学习加权融合。
4. 对 patch token 做注意力池化（delta 模式可把文本并入 KV）。
5. delta head 输出 `raw_delta`。
6. 内部门控与缩放: `delta_gate * exp(delta_log_scale)`，可选 `tanh clip`。
7. relevance head 输出 `rel_logits`。

`head_mode`:

- `base`: 走 base head
- `delta`: 走 delta head + gate 路径

### 3.2.3 Base + Delta 组合预测

在 delta 训练/验证/测试中统一使用:

- `base_pred_z = base_backbone(history_z)`（冻结）
- `delta_pred_z = delta_model(prompt_with_news, head_mode="delta")`
- `pred_z = base_pred_z + delta_pred_z`
- `pred_raw = pred_z * sigma_global + mu_global`

同时构造一个 counterfactual 分支:

- `delta_pred_null = delta_model(prompt_without_news, head_mode="delta")`

用于训练约束（判断新闻是否真的有帮助）。

### 3.2.4 Delta 损失系统

总损失是加权组合:

- `loss_res`: 拟合残差目标 `target_z - base_pred_z`
- `loss_null`: 约束无新闻时 delta 输出接近 0
- `loss_margin`: 约束 `err_null >= err_real + margin`
- `loss_non_degrade`: 约束有新闻预测不劣于 base
- `loss_gate`: 对 real 分支 `rel_logits` 做伪标签 BCE
- `loss_gate_null`: 对 null 分支 `rel_logits` 压低
- `loss_rank`: 约束 `rel_real - rel_null` 的排序间隔
- `loss_rel_supervise`: relevance 监督 BCE
- `loss_contrast`: real-vs-null 的对比项

训练调度机制:

- delta warmup（前若干 epoch 关闭部分约束）
- curriculum（约束权重随 epoch 递增）
- null loss warmup + ramp
- cold start 采样权重
- grad accumulation + gradient clipping

---

## 4. 各模块职责

## 4.1 入口与实验脚本

- [`run.py`](/home/brucebi/Projects/REIN-LLM-Forecast/run.py)
  - 管理训练参数、模型参数、新闻参数、RL 参数
  - 控制 `stage` 流程

- [`scripts/nswelecload_new.sh`](/home/brucebi/Projects/REIN-LLM-Forecast/scripts/nswelecload_new.sh)
  - NSW 任务实验脚本
  - 组合超参并批量调用 `python run.py`

## 4.2 训练编排

- [`src/base_delta_decoouple_trainer.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/base_delta_decoouple_trainer.py)
  - 数据初始化与 loader 构建
  - prompt/patch/news batch 构建
  - base 训练、delta 训练、验证、早停、测试
  - 结果落盘（log/csv/json/checkpoint）

## 4.3 模型层

- [`src/base_backbone.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/base_backbone.py)
  - `DLinearBackbone`: 趋势-季节分解 + 线性预测
  - `MLPBackbone`: 多层感知机基线

- [`src/model2.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/model2.py)
  - LLM + LoRA + patch encoder 的残差回归器
  - 内置 base/delta 双头 + relevance head
  - checkpoint 保存/加载（tokenizer、adapter、regressor、meta）

## 4.4 新闻检索与重排

- [`src/news_rules.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/news_rules.py)
  - 新闻加载、时间窗候选检索
  - 多策略选择（`keywords` / `polarity` / `smart` 等）
  - utility 重排（关键词覆盖 + 时效 + rate + sentiment）
  - 可选 MMR 去冗余

## 4.5 RL 组件

- [`src/RL/rl_bandit.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/RL/rl_bandit.py)
  - `LinTS` / `LinUCB` / `RewardNormalizer`

- [`src/RL/features.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/RL/features.py)
  - 上下文特征编码（时序统计、新闻密度、训练态）

- `NewsRLBanditSelector`（定义在 trainer 内）
  - 针对单个 prompt，学习“选多少条新闻 K”和“选哪些新闻”
  - 在线接收 `err_null - err_real` 奖励更新

说明:

- 模板/策略 bandit 在当前 trainer 中被固定关闭（`use_bandit=False`）。
- 新闻 item/K bandit 仍可单独启用（`news_rl_enable=1`）。

## 4.6 工具与状态

- [`src/utils/utils.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/utils/utils.py)
  - seed、device、波动率分箱
  - 画图与测试结果记录

- [`src/ValidationState.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/ValidationState.py)
  - 验证指标 EMA 状态

- [`src/data_construction/DataStatistic.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/data_construction/DataStatistic.py)
  - prompt 和 news 数量统计

---

## 5. 单样本技术预测公式

给定历史窗口 `x_{t-L:t-1}`:

1. 标准化:
   `x_z = (x - mu_global) / sigma_global`
2. 基线预测:
   `y_base_z = BaseBackbone(x_z)`
3. 构造新闻 prompt + patch，送入 delta 模型:
   `y_delta_z = DeltaLLM(prompt, patches, head_mode="delta")`
4. 合成:
   `y_hat_z = y_base_z + y_delta_z`
5. 反标准化:
   `y_hat = y_hat_z * sigma_global + mu_global`

这就是项目在 residual 评估和测试阶段的最终预测逻辑。

---

## 6. 训练产物与输出文件

每次任务运行（`checkpoints/<task>/`）会生成:

- `best_base_<task>/`：base 最优 checkpoint
- `best_delta_<task>/`：delta 最优 checkpoint
- `residual_pair.json`：最佳 base/delta 组合信息
- `*.log`：训练日志
- `prompts_*.json`：训练 prompt 记录
- `test_answers_*.json`：测试样本预测详情
- `true_pred_*.csv`：逐点预测与真值

全局测试汇总:

- [`results/test_results.csv`](/home/brucebi/Projects/REIN-LLM-Forecast/results/test_results.csv)
