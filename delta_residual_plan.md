# Delta 残差修正能力提升方案

> 前提：**不重新 refine news**，只从损失、采样、门控、正则、超参、架构小改动层面寻找提升空间。
> 所有结论来自对 `checkpoints/` 日志和 `results/test_results.csv` 的直接观察。

---

## 一、现状诊断（从日志看出的关键问题）

### 1.1 横向对比（history=48, horizon=48, lr=1e-4, active_mass_thr=0.7）

| Dataset/Backbone | Base MSE | +Delta MSE | MSE Skill | Active_Mass mean/median | Active_days% | 关键观察 |
|---|---|---|---|---|---|---|
| traffic · MLP | 361.6M | 363.3M | **−0.47%** | 0.80 / 0.099 | 30.6% | base 强→delta 被迫退化为恒等 |
| traffic · DLinear | 550.8M | 453.3M | **+17.7%** | 0.80 / 0.099 | 30.6% | base 弱→delta 空间大，news 被真正用上 |
| bitcoin · MLP | 262.4M | 264.4M | **−0.75%** | 0.11 / 0.035 | **0.0%** | news bank 基本为空，delta 强加残差=加噪 |
| bitcoin · DLinear | 7.0M | 7.6M | **−8.8%** | 0.11 / 0.035 | 0.0% | 同上，更糟 |

### 1.2 三个最强的可诊断信号

**A. Counterfactual gap：news 是否真的被使用**

`TEST][COUNTERFACTUAL] active_mae vs blank_active` 的差距直接反映 delta 对 news 的依赖度：

- traffic · DLinear：`active_mae=13759, blank_active=18428` → **gap 4669**（news 被强烈使用）
- traffic · MLP：`active_mae=16687, blank_active=16620` → **gap −67**（news 几乎无作用，blank 掉 news 甚至更好）
- bitcoin：active 样本在 test 集几乎不存在（`nan`），counterfactual 完全失效

结论：**MLP 上 delta 学到了"忽略 news"的捷径**。这是 base 残差太小 + news 信号弱的联合效应。

**B. shape_gain 饱和到 cap**

所有 epoch 的 `KNOB_HIST shape_active` 直方图显示 `shape_gain` 几乎全部样本都挤在 1.1999…（即 `shape_gain_cap=1.2`）。这说明：
- shape head 的输出被 cap 夹死，梯度被截断
- shape_gain 已经不再是一个自由参数，成了常数 1.2

**C. lambda 的 bimodal 分布正常，但 active cap 被频繁触达**

- `lambda_inactive` 522/570 样本挤在 0.05 地板（符合预期）
- `lambda_active` 最后 bin (0.42–0.45) 频次 87–96，明显堆积在 `lambda_max=0.45` 天花板

说明 active 样本想要更大的 lambda 权重，但被现有 cap 限制了。

### 1.3 其他可观察细节

- `active_mass median=0.003`（traffic）说明即使在 active_mass_thr 以上，仍有大量样本的主 topic 相关度很低——硬门限没有把"弱相关"样本滤掉
- `inactive_gap_pct` 在 traffic · MLP 上为 −0.28（负值说明 inactive 样本上 delta 反而加入了小量噪声）
- Bitcoin 上 `actionable_news=4/5219`，`active_days=0/366`——这不是模型的问题，是数据/门限问题，任何改进都该**先把 bitcoin 当独立 case 处理**

---

## 二、改进方案（按投入产出比排序）

### P0 — 立即可试，改动最小

#### P0.1 放开 shape_gain_cap，并加入软正则

**问题**：shape_gain 饱和在 1.2，没有学习空间。

**措施**：
- `--delta_v3_shape_gain_cap` 默认 0.30（代码定义）但脚本传的是 `0.20`——先把 cap 提到 **0.5 或 0.7**
- 额外加一个朝 1.0 的 L2 软正则项（`shape_gain_l2_to_one_weight` 新参）——允许大 shape_gain，但要付代价，这样模型会把容量留给真正需要 reshape 的样本

**预期收益**：KNOB_HIST 应该从"全部挤在 cap"变成"大部分接近 1.0，少数上冲"——这才是健康分布。

#### P0.2 调高 lambda_max，让 active 样本有更大修正幅度

当前 `lambda_max=0.45`，且 active 直方图最后一个 bin 堆积严重。

**措施**：提到 **0.60**（run.py 默认本来就是 0.60，是脚本里被压到 0.45），对 DLinear 这种弱 base 尤其有用。

**注意**：MLP 这种强 base 不宜放太大，否则会把更多 base 有效预测拉偏。建议 **backbone-specific**：MLP 用 0.40，DLinear 用 0.60。

#### P0.3 提高 inactive_residual_weight（惩罚 inactive 上的 delta 噪声）

当前 `--delta_v3_inactive_residual_weight=0.1`，inactive_gap_pct 仍有 −0.28 的负偏。

**措施**：提到 **0.3–0.5**。让模型在 mass<τ 的样本上必须把 delta 压得极接近 0。

**预期**：解决"MLP-traffic 里 delta 在 inactive 上反而加噪"的问题，直接改善整体 MAE。

#### P0.4 加硬门控（hard gate）：mass < τ 时 delta = 0

当前是软门控（lambda 连续调节）。对 bitcoin 这种 active_days=0% 的 case，软门控不够——训练/测试时都应该在数据层面强制 delta=0。

**措施**：加 `--delta_v3_hard_gate_mass_threshold`，默认 0.0（不启用）；当设为 e.g. 0.2 时，凡是 active_mass<0.2 的样本训练时 loss 只看 base 不反传到 delta，推理时 delta 输出置零。

**预期收益**：
- bitcoin 直接从 "delta 退化 −8.8%" 变成 "delta 退回到 base 水平"（保底）
- 整体 "Delta_Helped_Rate" 更诚实，不虚高

---

### P1 — 中等改动，针对 counterfactual gap 小的 case

#### P1.1 强化 counterfactual / contrastive loss

当前 `counterfactual_weight=0.1`，`blank_gain_z` 只有 0.005–0.02（几乎没 gap）。

**措施**：
- 把 `counterfactual_weight` 提到 **0.3–0.5**
- 引入**三元对比损失**：`‖δ(x, news) − δ(x, blank)‖` 要 > `‖δ(x, news) − δ(x, perm_news)‖` 一个固定 margin
- 在 inactive 样本上加镜像惩罚：`‖δ(x, news) − δ(x, blank)‖ ≈ 0` （news 存在与否不应该改变 inactive 样本的预测）

**预期**：让 delta 对 news 输入产生真实敏感度，避免 MLP-traffic 上的"忽略 news 捷径"。

#### P1.2 残差加权损失（以 |residual| 为采样/加权依据）

当前 hard_residual_sampler：`hit_rate=0.6, top_pct=0.1`（60% batch 来自 top10% 残差）。这个机制很好，但可以更狠：

**措施**：
- 把 loss 直接写成 `L_delta = Σ w_i · (delta_pred_i − residual_i)²`，其中 `w_i ∝ |residual_i|^γ` (γ=0.5–1.0)
- 或者采用**curriculum**：前 20% epoch 用 top_pct=0.3（广覆盖），后 80% 收缩到 top_pct=0.1（聚焦硬样本）

**预期**：让 delta 的学习精力更聚焦在"base 确实出错的那些时刻"，而不是在 base 已经对的样本上浪费容量。

#### P1.3 残差历史作为 delta 的额外输入特征

Delta 当前吃 `base_hidden + news_features`，但**没有看到 base 最近几步的残差**。给它加一个滑窗残差特征（过去 7–14 天的 base 误差）能让 delta 学到"base 在这种 context 下历来低估/高估"。

**措施**：在 `Delta_v3` 的 input 里拼一个 `recent_residual_window`（需要在 dataloader 里算出）。这是新 feature，不涉及 news。

**预期**：特别有助于 bitcoin/gas 这种 news 覆盖极低但 base 存在系统性偏差的场景。

---

### P2 — 较大改动，结构层面

#### P2.1 Backbone-aware 训练 schedule

数据很清楚：MLP 和 DLinear 上 delta 的最优超参**不一样**。

**措施**：在脚本里按 `base_backbone` 切换一组预设：

```
MLP 预设:   lambda_max=0.35, counterfactual_weight=0.5, inactive_weight=0.5
DLinear 预设: lambda_max=0.60, counterfactual_weight=0.2, inactive_weight=0.2
NLinear/PatchTST: 初始值用 MLP 预设
```

论文讨论里这点可以直接写成"base-aware residual correction calibration"。

#### P2.2 双头 delta：envelope + shape

当前 delta 一步直接输出 horizon 长度的残差。可以拆成：

- **Envelope head**：输出一个幅度 `amp_t`（标量或低维）——相当于"今天整体要修正多少"
- **Shape head**：输出归一化形状 `s_t ∈ R^H`，`‖s_t‖_2 = 1`
- 最终：`δ_t = amp_t · s_t`

**好处**：
- Envelope 更容易被 news 直接驱动（news 告诉你"今天异常强度"）
- Shape 更依赖历史形态
- 更容易可视化/做 ablation（论文里图表漂亮）

#### P2.3 分位数残差头（quantile delta）

MSE/MAE 目标让 delta 在尖峰处 under-predict（因为极端值是少数）。让 delta 同时预测 q10/q50/q90 三个分位，主预测用 q50，额外把 q90−q10 作为"不确定性"输出给 downstream（或者论文里展示 calibration）。

这一条兼顾"更好的 spike 捕捉 + 论文里多一个维度的 story"。

---

## 三、验证协议（避免超参调优退化成 cherry-picking）

**所有改动验证遵循**：

1. **冻结 dataset splits**：固定 train/val/test，不重新切
2. **验证集驱动**：在 val 上选超参，**test 只看一次**
3. **对照组**：每个改动必须与 "base-only"、"base+delta_v0（当前配置）"同场对比
4. **多 seed**：关键改动跑 3 个 seed，看均值和方差
5. **per-dataset 表**：不求通用最优，每个数据集可以有自己的最佳 delta 配置（配合 P2.1 的 backbone-aware）

**最小的 ablation 矩阵**（第一轮）：

| Exp | shape_cap | λ_max | cf_w | inact_w | hard_gate | backbone |
|---|---|---|---|---|---|---|
| baseline | 0.20 | 0.45 | 0.10 | 0.10 | off | both |
| P0.1 | 0.70 | 0.45 | 0.10 | 0.10 | off | both |
| P0.2 | 0.20 | 0.60 | 0.10 | 0.10 | off | DLinear only |
| P0.3 | 0.20 | 0.45 | 0.10 | 0.40 | off | both |
| P0.4 | 0.20 | 0.45 | 0.10 | 0.10 | 0.20 | both (bitcoin 重点看) |
| P0.all | 0.70 | 0.60 | 0.10 | 0.40 | 0.20 | both |

6 个实验 × 2 backbones × 4 datasets = 48 runs，单 GPU 一天内可完成。

---

## 四、论文层面的副产品

即使某些改进在某个数据集上提升不大，这套日志指标本身（counterfactual gap、blank_gain_z、inactive_gap、shape/spike KNOB_HIST 直方图）在论文里是**非常强的"分析段落"素材**——能证明：

1. Delta 确实在 active 样本上使用 news（counterfactual gap 图）
2. Delta 在 inactive 样本上保持沉默（inactive_gap_pct → 0）
3. shape/spike head 的动态分布随训练演化（KNOB_HIST 热图）

这些图在目前任何 time-series LLM 论文里都**很少见**，是你框架的独特卖点，单独写一个"Analysis of Residual Correction Behavior" 小节。

---

## 五、下一步行动建议

1. **先跑 P0.1–P0.4 四个独立实验**（traffic 上最快有结果，2 个 backbone × 4 改动 = 8 runs）
2. **确认 P0.3（inactive 权重）能消除 MLP-traffic 上的 delta 负贡献**——这是最具体可验证的 hypothesis
3. **P0.4 hard_gate 用 bitcoin 验证**——预期 skill 从 −8.8% 回升到 0 附近
4. 如果 P0 组合效果好，再进入 P1
5. P2 是论文深度相关的改动，在 P0/P1 收尾后再考虑

若同意此计划，我可以从 P0.1（shape_cap + 软正则）开始实施。
