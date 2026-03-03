# REIN-LLM-Forecast 技术框架说明

## 1. 目标与核心设计

该项目是一个两阶段时序预测框架，核心思想是把预测拆成:

- `base` 阶段: 用纯时序模型学习稳定的基础预测 `base_pred`。
- `delta` 阶段: 用 LLM + LoRA 学习新闻驱动的残差修正 `delta_pred`。
- 最终预测: 由 `delta_fusion_mode` 融合 `base_pred` 与 `delta_pred`（支持加法或乘法融合），再反归一化。

在当前实现中，`delta` 阶段支持两种运行模式（由 `--delta_mode` 控制）:

- `regression`（兼容旧方案）: 使用 `delta_head` 直接回归残差向量。
- `kernel_tokens`（当前主实验方案）: LLM 只生成离散参数 token；由 impact-kernel 引擎把参数还原为 `delta` 向量。

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

`setup_env_and_data()` 主要做 7 件事:

1. 读取 `train/val/test`（CSV 或 Parquet）。
2. 从训练集计算全局 z-score 统计量: `mu_global`、`sigma_global`。
3. 用 `SlidingDataset` 构建滑窗样本（窗口 `L`，预测步长 `H`）。
4. 加载新闻 JSON，并过滤空新闻文本。
5. 加载 prompt 模板（当前主流程不再从文件加载 `policy_kw`）。
6. 计算 train/val/test 的波动率分箱（volatility bin）。
7. 在日志开头打印完整参数快照（敏感字段自动脱敏）。

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

### 3.2.1 共有输入构建（新闻 + 时序）

`build_batch_inputs` 负责常规 prompt/patch 构建，主要产物:

- 文本 token: `input_ids`, `attn`
- 时序 patch: `ts_patches`, `ts_patch_mask`
- 监督目标: `targets_z`
- 新闻相关标签: `rel_labels`

该构建器同时服务于 `regression` 兼容模式；`kernel_tokens` 模式有独立的 kernel prompt 生成函数。

### 3.2.2 `regression` 兼容模式（旧路径）

模型来自 [`src/model2.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/model2.py)，通过 `delta_head` 回归残差:

- `base_pred_z = base_backbone(history_z)`
- `delta_pred_z = delta_model(..., head_mode="delta")["pred"]`
- `pred_z = base_pred_z + delta_pred_z`

该路径保留了历史损失系统（`loss_res/loss_null/loss_margin/...`），用于兼容旧实验；当前主实验不使用该路径。

### 3.2.3 `kernel_tokens` 主模式（当前主实验）

核心思想:

- LLM 不直接输出残差向量。
- LLM 只生成离散参数 token 序列。
- impact-kernel 引擎将参数还原为 `delta` 向量。
- 最终预测由 `delta_fusion_mode` 融合 `base_pred_z` 与 `delta`（`add` / `mul_z` / `mul_raw`）。

离散参数空间:

- `REL ∈ {0,1}`
- `SIGN ∈ {UP,DOWN}`
- `SHAPE ∈ {SPIKE, STEP_DECAY, RAMP_DECAY}`
- `LAG ∈ [0..6]`
- `HALF_LIFE ∈ {1,2,4,8,16,32}`
- `DUR ∈ [0..6]`
- `AMP_BIN ∈ [0..20]`

核函数定义（`tau=0..H-1`, `t=max(0, tau-lag)`）:

- `SPIKE`: `k = exp(-t/lambda)`
- `STEP_DECAY`: `k = 1 (t<dur) else exp(-(t-dur)/lambda)`
- `RAMP_DECAY`: `k = t/dur (t<dur, dur>=1) else exp(-(t-dur)/lambda)`

`HALF_LIFE -> lambda` 换算:

- `lambda = half_life / ln(2)`（满足 `exp(-half_life/lambda)=0.5`）。

幅度映射:

- 训练集 residual 统计生成 `AMP table`（分位数）。
- 推理时通过 `AMP_BIN` 查表取幅度。
- `delta` 最终 clip 区间由 `--delta_clip` 控制（对称 `[-delta_clip, +delta_clip]`；`<=0` 视为不裁剪）。

### 3.2.4 Kernel 自动标注（不依赖 teacher）

`kernel_fitter` 对每个样本拟合离散参数:

1. `raw_residual = target_z - base_pred`
2. 若残差范数低于阈值，置 `REL=0`（其余参数默认，`delta=0`）
3. 否则先定 `SIGN`（由残差均值符号）
4. 对 `SHAPE/LAG/HALF_LIFE/DUR` 网格搜索
5. 每组参数用闭式投影求幅度:
   `a* = clip((y·k)/(k·k), 0, a_max)`
6. 计算改进量门槛:
   - `baseline_err = ||y||^2`
   - `improve = baseline_err - best_err`
   - 若 `improve < max(rel_improve_abs, rel_improve_ratio * baseline_err)`，回退为 `REL=0`
7. 取通过门槛的最优组合并量化成 `AMP_BIN`

补充约束:

- 对 `has_news=0` 样本，构建阶段强制 `REL=0`（`force_rel0=True`），避免“无新闻却有修正”的伪标签。

标签 token 形式:

- `<REL_1> <SIGN_UP> <SHAPE_STEP_DECAY> <LAG_2> <HL_8> <DUR_3> <AMP_12>`

### 3.2.5 API 边界样本复标（可选）

当前实现支持在自动标注后接入外部 API（默认关闭），仅对不确定样本做 `REL/SIGN` 复标，以提升标签质量并控制成本。

触发样本（任一满足）:

- `abs(rel_norm - kernel_rel_norm_thresh) <= kernel_api_uncertain_band`
- 自动标注为 `REL=1` 且 `AMP_BIN <= kernel_api_low_amp_bin`

API 使用策略:

- 目标: 仅做“是否相关（REL）/方向（SIGN）”校验，保留自动拟合的 `SHAPE/LAG/HL/DUR/AMP`。
- 合并策略（保守）:
  - 允许将 `REL` 从 1 下调到 0。
  - 当 `REL` 仍为 1 时可修正 `SIGN`。
  - 不自动把 `REL=0` 上调为 `REL=1`，避免过修正。
- 预算控制: `kernel_api_max_calls`
- 进度日志: 每 `kernel_api_log_every` 次调用输出一次
- 结果缓存: `checkpoints/<task>/sft_kernel_api_cache.json`

### 3.2.6 Kernel 训练与推理路径

训练:

- 构建 `(prompt, label_tokens)` 数据，写入 `sft_kernel_cache.json`
- 仅训练 LoRA（Causal LM loss，labels 对 input 区域 mask 为 `-100`）
- 训练循环使用 `tqdm` 显示 `[DELTA][KERNEL_SFT] Epoch x/y` 进度条与实时 loss
- 每个 epoch 结束后立即在验证集评估（`evaluate_metrics_kernel_tokens`）
- 按 `reward_metric` 维护 best checkpoint，并支持 `early_stop_patience` 提前停止
- 最终测试前优先重载 `best_delta_<task>`，避免“最后一轮覆盖最优轮”

推理:

1. 构建 kernel prompt，并在末尾追加 `"[Output Tokens]"` 对齐训练模板
2. LLM `generate()` 参数 token 序列
3. 解析 token（解析器容错不完整尖括号，如 `REL_1>`），失败时回退 `REL=0`
4. `impact_kernel.params_to_delta(...)` 生成 `delta`
5. 按 `--delta_fusion_mode` 融合为最终 z 预测:
   - `add`: `pred_z = base_pred_z + delta`
   - `mul_z`: `pred_z = base_pred_z * clip(1 + s * delta, c_min, c_max)`
   - `mul_raw`: 先 `base_raw = inv_z(base_pred_z)`，再 `pred_raw = base_raw * clip(1 + s * delta, c_min, c_max)`，最后回到 z-space

运行期诊断（日志）:

- `[DELTA][KERNEL][GEN][VAL/TEST]`：`token_hit_rate`、`rel1_rate`、`amp_nonzero_rate`、`empty_gen_rate`、`mean_abs_delta`
- `[DELTA][KERNEL][EVAL]` / `[DELTA][KERNEL][VAL]`：每轮验证指标（`val_loss/val_mse/val_mae`）
- 评估阶段进度条：`[EVAL][KERNEL|RESIDUAL|BACKBONE|SINGLE][VAL/TEST]`

排障经验:

- 若出现 `amp_nonzero_rate=0` 且 `mean_abs_delta=0`，通常表示 AMP token 未成功生成/解析，模型退化为 `delta=0`（预测退回 base）。
- 首先检查 `--kernel_gen_max_new_tokens` 是否过小（当前 NSW 脚本默认已提升到 `96`）。
- 解析器已增加兜底: 若生成文本含 `AMP_x (x>0)` 但缺失 `REL`，会按 `REL=1` 处理，减少“有幅度却被当作无效修正”的解析偏差。

实现文件:

- [`src/sft/impact_kernel.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/impact_kernel.py)
- [`src/sft/kernel_fitter.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/kernel_fitter.py)
- [`src/sft/sft_kernel_dataset.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/sft_kernel_dataset.py)
- [`src/base_delta_decoouple_trainer.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/base_delta_decoouple_trainer.py)

---

## 4. 各模块职责

## 4.1 入口与实验脚本

- [`run.py`](/home/brucebi/Projects/REIN-LLM-Forecast/run.py)
  - 管理训练参数、模型参数、新闻参数、RL 参数
  - 控制 `stage` 流程
  - 关键开关:
    - `--delta_mode {regression,kernel_tokens}`
    - `--delta_fusion_mode {add,mul_z,mul_raw}`
    - `--delta_mul_scale`
    - `--delta_mul_coeff_min`
    - `--delta_mul_coeff_max`
  - `kernel_tokens` 参数:
    - `--kernel_amp_bins`
    - `--kernel_rel_norm_thresh`
    - `--kernel_rel_improve_ratio`
    - `--kernel_rel_improve_abs`
    - `--kernel_a_max`
    - `--kernel_sft_lr`
    - `--kernel_gen_max_new_tokens`
    - `--kernel_cache_file`
    - `--kernel_amp_table_file`
    - `--kernel_api_enable`
    - `--kernel_api_key`
    - `--kernel_api_model`
    - `--kernel_api_temperature`
    - `--kernel_api_max_calls`
    - `--kernel_api_uncertain_band`
    - `--kernel_api_low_amp_bin`
    - `--kernel_api_log_every`
    - `--kernel_api_cache_file`
  - 关键词说明:
    - 已移除 `--keyword_path`；主流程不再读取关键词文件。
    - `utility_keyword_weight` 参数仍保留兼容定义，但在当前 NSW kernel 主流程里 `policy_kw=[]`，关键词覆盖分量默认不产生贡献。
  - 评估进度条参数:
    - `--eval_progress_bar {0,1}`
    - `--eval_progress_leave {0,1}`
  - 说明:
    - 当前 `run.py` 未单独暴露 `--sft_batch_size/--sft_max_seq_len`。
    - kernel SFT 的 batch size 默认由 trainer 内部 `getattr(args, "sft_batch_size", 1)` 读取，脚本默认等效为 `1`。

- [`scripts/nswelecload_new.sh`](/home/brucebi/Projects/REIN-LLM-Forecast/scripts/nswelecload_new.sh)
  - NSW 任务实验脚本
  - 组合超参并批量调用 `python run.py`
  - 当前默认固定单卡: `export CUDA_VISIBLE_DEVICES=1`（可按机器修改）
  - 当前默认（kernel-only）:
    - `STAGE=all`
    - `DELTA_MODE=kernel_tokens`
    - `DELTA_FUSION_MODE=mul_z`
    - `DELTA_MUL_SCALE=0.5`
    - `DELTA_MUL_COEFF_MIN=0.90`
    - `DELTA_MUL_COEFF_MAX=1.10`
    - `DELTA_CLIP=0.5`
    - `STRIDE=1`
    - `NEWS_TOPK=999`
    - `DEFAULT_POLICY=all`
    - `RL_USE=0`（脚本固定关闭 RL/bandit 训练分支）
    - `DELTA_EPOCHS=20`（kernel LoRA 最大训练轮数；可能因 early stop 提前结束）
    - `LOAD_IN_4BIT=1` + `GRADIENT_CHECKPOINTING=1`（显存保守配置）
    - `KERNEL_REL_IMPROVE_RATIO=0.10`（自动标注改进门槛）
    - `KERNEL_REL_IMPROVE_ABS=0.0`
    - `KERNEL_GEN_MAX_NEW_TOKENS=128`（避免 token 截断导致 AMP 缺失）
    - `KERNEL_API_ENABLE=0`（默认不调用外部 API）
    - `taskName` 自动追加 `_kernelTok`，避免与旧日志混写
    - 当 `KERNEL_API_ENABLE=1` 且未手动改 cache 名时，脚本会自动切到 `sft_kernel_cache_api.json`，避免误命中旧缓存

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
  - 内置 base/delta 双头 + relevance head（regression 兼容路径）
  - checkpoint 保存/加载（tokenizer、adapter、regressor、meta）

## 4.4 新闻检索与重排

- [`src/news_rules.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/news_rules.py)
  - 新闻加载、时间窗候选检索
  - 多策略选择（`all` / `polarity` / `smart` 等；`keywords` 策略入口保留但当前主流程未注入关键词列表）
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
- NSW 默认脚本固定 `rl_use=0` 且走 `kernel_tokens` 主路径，因此不会进入 bandit 决策训练分支。
- `news_rl_*` 参数在 `run.py` 仍保留兼容定义，但在当前 NSW kernel 默认运行拓扑中不生效。

## 4.6 工具与状态

- [`src/utils/utils.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/utils/utils.py)
  - seed、device、波动率分箱
  - 画图与测试结果记录

- [`src/ValidationState.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/ValidationState.py)
  - 验证指标 EMA 状态

- [`src/data_construction/DataStatistic.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/data_construction/DataStatistic.py)
  - prompt 和 news 数量统计

- [`src/sft/impact_kernel.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/impact_kernel.py)
  - 参数 token 解析与校验、冲击核计算、AMP table 管理

- [`src/sft/kernel_fitter.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/kernel_fitter.py)
  - residual -> 离散参数自动拟合（网格搜索 + 幅度投影）

- [`src/sft/sft_kernel_dataset.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/sft_kernel_dataset.py)
  - kernel prompt 构建、参数 token 标签构建、kernel SFT 数据格式化

---

## 5. 单样本技术预测公式

给定历史窗口 `x_{t-L:t-1}`:

1. 标准化:
   `x_z = (x - mu_global) / sigma_global`
2. 基线预测:
   `y_base_z = BaseBackbone(x_z)`
3. `delta` 分支:
   - `regression`: `y_delta_z = DeltaLLM(..., head_mode="delta")`
   - `kernel_tokens`: `tokens = LLM.generate(prompt)`，`y_delta_z = ImpactKernel(tokens, amp_table)`
4. 合成（由 `delta_fusion_mode` 决定）:
   - `add`: `y_hat_z = y_base_z + y_delta_z`
   - `mul_z`: `y_hat_z = y_base_z * clip(1 + s * y_delta_z, c_min, c_max)`
   - `mul_raw`: 先转 raw 做乘法，再转回 z-space
5. 反标准化:
   `y_hat = y_hat_z * sigma_global + mu_global`

两种 `delta_mode` 都先生成 `y_delta_z`；最终如何与 `y_base_z` 合成由 `delta_fusion_mode` 控制。

---

## 6. 训练产物与输出文件

每次任务运行（`checkpoints/<task>/`）会生成:

- `best_base_<task>/`：base 最优 checkpoint
- `best_delta_<task>/`：delta 最优 checkpoint
- `residual_pair.json`：最佳 base/delta 组合信息
- `<task>_<base_backbone>_kernel_*.log`：训练日志（`kernel_tokens` 模式会在文件名追加关键 kernel/fusion 参数）
- `prompts_*.json`：训练 prompt 记录
- `test_answers_*.json`：测试样本预测详情
- `true_pred_*.csv`：逐点预测与真值
- `sft_kernel_cache.json`：kernel token SFT 样本缓存（kernel 模式）
- `kernel_amp_table.json`：AMP_BIN 到幅度值映射（kernel 模式）
- `sft_kernel_api_cache.json`：API 复标缓存（启用 API 时）
- `sft_kernel_cache_api.json`：启用 API 且使用脚本默认自动切换时的样本缓存文件名

全局测试汇总:

- [`results/test_results.csv`](/home/brucebi/Projects/REIN-LLM-Forecast/results/test_results.csv)

---

## 7. 当前默认运行拓扑（NSW 脚本）

以 [`scripts/nswelecload_new.sh`](/home/brucebi/Projects/REIN-LLM-Forecast/scripts/nswelecload_new.sh) 当前默认配置为例:

1. `stage=all`：默认先训练/验证 base，再进入 delta 阶段。
2. `delta_mode=kernel_tokens`：默认走参数 token + impact-kernel 推理链路。
3. `delta_fusion_mode=mul_z`：默认在 z-space 使用乘法融合（可切换到 `add` 或 `mul_raw`）。
4. 默认单卡运行（`CUDA_VISIBLE_DEVICES=1`），并启用 `4bit + gradient checkpointing`。
5. `rl_use=0`：默认不启用 RL/bandit 训练分支。
6. `delta_epochs=20`：kernel LoRA 训练上限为 20 轮；每轮验证并支持 early stop。
7. `taskName` 默认追加 `_kernelTok`，避免与旧 regression 日志/目录冲突。
8. `kernel_gen_max_new_tokens=128`：减少参数 token 截断风险（尤其是 `AMP_*`）。
9. 默认不启用 API 复标（`kernel_api_enable=0`）；开启后只对边界样本调用并走缓存复用。
10. 评估阶段默认开启终端进度条（`eval_progress_bar=1`，`eval_progress_leave=0`）。
11. 日志文件名按 `taskName + base_backbone + kernel关键参数` 生成，便于直接从文件名识别实验配置。
