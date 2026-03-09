# Technical Review: REIN-LLM-Forecast（2024 NSW ElecPrice Case, dlinear）

## 1. Review Scope

本评审基于以下运行与代码：

- 运行日志：`checkpoints/[2024-nswelecPrice-case]_dlinear/[2024-nswelecPrice-case]_dlinear_dlinear.log`
- 启动脚本：`scripts/nswelecprice_2024_case.sh`
- 入口与训练主干：`run.py`、`src/base_delta_decoouple_trainer.py`
- 模型实现：`src/base_backbone.py`、`src/model2.py`
- 数据与提示词构建：`src/data_construction/data.py`、`src/data_construction/prompt.py`
- 新闻与检索：`src/news_rules.py`、`src/delta_news_hooks.py`、`src/delta_case_retrieval.py`

---

## 2. Experiment Snapshot

### 2.1 数据规模（L=48, H=48, stride=48）

- Train rows: 9600, windows: 199
- Val rows: 4800, windows: 99
- Test rows: 3168, windows: 65

### 2.2 本次关键配置

- Task: `[2024-nswelecPrice-case]_dlinear`
- Stage: `all`（先 base 后 delta）
- Base backbone: `dlinear`
- Base epochs: 40（early stop 到第20轮）
- Delta epochs: 40（early stop 到第5轮）
- Value column: `RRP`
- Unit: `$/MWh`
- News source: `dataset/news_2024_2025.json`
- News columns: `date` + `content`
- News policy: `all`
- Case retrieval: `enable=1, mode=price_event, topk=3`
- LoRA: Llama-3-8B, `r=8, alpha=32, dropout=0.05`
- Global z-score（来自 train）：
  - `mu_global = 129.026443`
  - `sigma_global = 628.777100`

---

## 3. End-to-End Pipeline（实验全流程）

1. `scripts/nswelecprice_2024_case.sh` 组织参数并调用 `python run.py ...`。
2. `run.py` 解析参数并进入 `src/base_delta_decoouple_trainer.main(args)`。
3. `setup_env_and_data()` 读取 train/val/test CSV，计算全局 z-score。
4. `make_loader()` 按 `(L,H,stride)` 切窗构建 DataLoader。
5. 读取新闻 JSON，按 `news_time_col/news_text_col` 清洗文本。
6. 加载模板 `configs/deltaWithNews_template.yaml`。
7. **Base 阶段**：训练 `DLinearBackbone`，输入 `history_z (B,L)` 输出 `base_pred_z (B,H)`，每轮在 val 上评估并保存 best。
8. **Delta 阶段初始化**：加载 best base（冻结）+ LLaMA/LoRA + `TSForecastRegressor`。
9. 构建 train-only case bank（本次 `n_cases=199`）。
10. 每个 delta batch 先由 base backbone 得到 `base_pred`。
11. 同批次构建两套输入：
    - with-news（真实新闻上下文）
    - no-news（counterfactual 空新闻）
12. with-news 过程：候选筛选 -> policy 选取 -> utility 重排 -> news RL 选择 -> refine -> structured event。
13. 若开启检索：query case 对 case bank 检索，生成 retrieval features 与 validity/confidence 元信息。
14. Delta 前向：文本 token + patch token 共送 LLM，做层融合与池化，输出 `delta_pred` 和 `rel_logits`。
15. 最终预测：
    - `gate = bounded_sigmoid(rel_logits)`
    - `pred_final_z = base_pred_z + gate * delta_pred_z`
    - 反归一化到原值域并统计指标。

---

## 4. 单样本预测是如何生成的

1. 从滑窗取历史 48 点和未来 48 点标签。
2. 用 train 全局 `mu/sigma` 做 z-score。
3. Base 先预测 `base_pred_z`（无新闻）。
4. 历史 z 序列生成 history 文本（含 slope）。
5. 以 `target_time` 为锚点，检索过去 `news_window_days=1` 天新闻。
6. 新闻经筛选/重排/截断，附加 structured event（heuristic）。
7. 拼接 prompt（history + news + output instruction）。
8. 历史 z 切 patch（`patch_len=4`，48点=>12 patch）。
9. Prompt token 与 patch token 一起输入 LLM。
10. Delta head 生成残差修正 `delta_pred_z`。
11. Relevance head 生成 `rel_logits`，经过门控得到 `gate`。
12. 最终：
    - `final_pred_z = base_pred_z + gate * delta_pred_z`
    - `final_pred = final_pred_z * sigma + mu`

---

## 5. 本次实验结果解读

### 5.1 Base 训练

- 最优 val 出现在 epoch16：`val_mae = 91.335106`
- epoch20 触发 early stop（patience=4）

### 5.2 Delta 训练

- epoch1~5 的 val MAE：`92.837763 -> 91.449093 -> 91.438855 -> 91.433438 -> 91.428947`
- 对比 base-only val（91.335106），delta 阶段始终未超过 base
- epoch5 触发 early stop

### 5.3 `[ABLATION]` 到底在做什么（VAL）

#### 5.3.1 触发时机

训练结束后，加载 best delta，再对同一数据集做 4 组“检索注入方式”对照：

- `case_retrieval_run_ablations=1` 才会执行
- `case_retrieval_ablation_split=val` 本次只跑验证集

#### 5.3.2 四个模式

- `no_retrieval`
  - 强制关闭检索
  - 用作基线
- `price_only`
  - 开启检索，仅按状态向量相似度排序
- `price_event`
  - 在 price 相似度上融合结构化事件相似度重排
- `gate_only`
  - 检索流程同 `price_event`
  - 但检索特征只影响 gate/confidence，不注入 delta 主干特征

#### 5.3.3 日志字段解释

- `metric`：本次比较主指标（由 `reward_metric` 决定；本 run 为 MAE）
- `delta_vs_no`：相对 `no_retrieval` 的差值（负值更好）
- `coverage`：检索判定有效（`retrieval_valid=True`）的样本占比
- `mae_valid`：有效检索子集上的 MAE
- `mae_rejected`：被拒绝检索子集上的 MAE
- `strong_valid/strong_rej`：仅“强新闻样本”子集上的 MAE

#### 5.3.4 本次 ablation 数值

- `no_retrieval`: 92.771487（基线）
- `price_only`: 92.770754（`-0.000733`）
- `price_event`: 92.770353（`-0.001135`，四者中最佳）
- `gate_only`: 92.771058（`-0.000429`）

可读结论：

1. 全局差异极小（千分之一量级），检索分支对总 MAE 的边际收益很弱。
2. `price_event` 相比 `price_only` 略好，说明事件重排有轻微正向增益。
3. `gate_only` 逊于 `price_event`，说明“检索特征注入 delta 主干”仍有微弱贡献。
4. `coverage` 从 `0.657` 提到 `0.929`，表示 price_event 策略下更多样本被判为检索有效。
5. `strong_valid/strong_rej` 为 `nan`，这是因为本次数据里新闻没有 `rate` 字段，强新闻标签基本不可用。

### 5.4 Test 最终

- `[TEST][FINAL] mse(raw)=721256.694098, mae(raw)=176.461609`
- `[TEST][BASE_ONLY] mse(raw)=721377.949785, mae(raw)=175.887803`

对比结论：

- MSE 小幅改善：`-121.255687`（约 `-0.0168%`）
- MAE 退化：`+0.573806`（约 `+0.3262%`）

总体上，这次 delta 在测试集上没有带来稳定收益（尤其 MAE 变差）。

---

## 6. 关键技术风险（按优先级）

1. **参数覆盖导致“看起来开了、实际没生效”**
   - `delta_margin_lambda/delta_adv_margin` 会被 `delta_cf_lambda/delta_cf_margin` 覆盖。
   - 本次配置里 `delta_cf_lambda=0.0`，导致 margin类约束几乎没参与。

2. **`gate_lambda` 可能未真正生效**
   - 训练逻辑优先读 `delta_gate_reg_lambda`，本次是 0。

3. **best-delta 日志语义与真实性能可能不一致**
   - 首次保存 best delta 时，`best_metric` 可能仍沿用 base 阈值。

4. **日志损失名与实际 loss 类型有语义偏差**
   - 日志里多处写 `zMSE`，实现中常用 `L1/smooth_l1`。

5. **数据集新闻缺少 `rate` 字段**
   - `rel_label` 与 strong-news 统计可解释性下降，`strong_*` 结果失效（`nan`）。

6. **数据切窗前未强制排序**
   - 若原始 CSV 顺序异常，窗口会混入时序错位样本。

7. **配置日志误脱敏**
   - 非敏感字段（如 `tokenizer`、`token_budget`）被打码，不利于复现。

---

## 7. 建议的最小修复路线

1. 合并并规范 counterfactual 参数命名，避免双命名覆盖。
2. 明确 `gate_lambda` 与 `delta_gate_reg_lambda` 的优先级逻辑。
3. 修正 best-delta 保存与日志语义（best 值与真实 checkpoint 对齐）。
4. 统一日志中的 loss 命名，按真实实现输出。
5. 对无 `rate` 新闻源增加 fallback 的强新闻定义，避免 `strong_* = nan`。
6. 在 `SlidingDataset` 中恢复时间排序，防止潜在错窗。
7. 调整敏感字段掩码规则，只屏蔽真正敏感键。

---

## 8. 产物清单（本次 run）

- Base ckpt: `checkpoints/[2024-nswelecPrice-case]_dlinear/best_base_[2024-nswelecPrice-case]_dlinear`
- Delta ckpt: `checkpoints/[2024-nswelecPrice-case]_dlinear/best_delta_[2024-nswelecPrice-case]_dlinear`
- Case bank: `checkpoints/[2024-nswelecPrice-case]_dlinear/case_bank_train_[2024-nswelecPrice-case]_dlinear.json`
- Prompt dump: `checkpoints/[2024-nswelecPrice-case]_dlinear/prompts_[2024-nswelecPrice-case]_dlinear_dlinear.json`
- Test answers: `checkpoints/[2024-nswelecPrice-case]_dlinear/test_answers_[2024-nswelecPrice-case]_dlinear_dlinear.json`
- True vs Pred CSV: `checkpoints/[2024-nswelecPrice-case]_dlinear/true_pred_[2024-nswelecPrice-case]_dlinear_dlinear.csv`
- Plot: `checkpoints/[2024-nswelecPrice-case]_dlinear/PredVsTrue_[2024-nswelecPrice-case]_dlinear.png`
