# 框架改进建议：提升新闻文本对预测效果的贡献

基于对框架代码、实验日志、数据特征的全面分析，以下是针对"显著提升预测效果（尤其是新闻文本处理）"的改进建议。

---

## 一、现状诊断

### 1.1 实验结果总览

| 实验 | MAE | Base MAE | Skill_MAE | Helped% | 备注 |
|------|-----|----------|-----------|---------|------|
| LOAD gated_add | 486.32 | 497.68 | **2.28%** | 63.2% | 当前最优 |
| LOAD cross_attn | 494.54 | 497.68 | 0.63% | 53.2% | 明显弱于 gated_add |
| PRICE (DistilBERT) | 95.46 | 96.00 | **0.56%** | 63.9% | 提升极小 |
| PRICE (GPT-2) | 95.46 | 96.00 | **0.56%** | 64.0% | 与 DistilBERT 几乎一致 |
| PRICE grid 各配置 | 95.43-95.46 | 96.00 | 0.56-0.59% | 64-65% | 调参无显著变化 |

### 1.2 训练日志关键发现

**LOAD 模型**：
- 训练 29 个 epoch 后收敛，val skill_mae 从 -10.7%（epoch 1，模型在伤害预测）稳步提升至 +3.88%
- sign_acc 从 0.49 → 0.67，说明模型确实在学习残差方向
- 测试集 skill_mae=2.28%，低于验证集 3.88%，存在一定过拟合

**PRICE 模型**：
- val skill_mae 在 epoch 1 就达到 0.62%，仅微幅改善至 epoch 10 的 0.82%
- 测试集 skill_mae 进一步缩水至 0.56%
- `true_|res|=2.76`（PRICE残差的z-score绝对值远大于 LOAD 的 0.07），而 `|delta|=0.05` 仅为残差的 1.8%
- **关键**: val 上的 helped=74% 远高于 test 上的 64%，说明模型在验证集上学到的修正模式在测试集上泛化不佳

**PRICE vs LOAD 根本差异**：
- PRICE CV=5.08（测试集），极端尖峰值达 17,500，11.7% 的值为负。残差分布极度偏斜
- LOAD CV=0.20，分布平稳
- PRICE 的 `true_|res|=2.76` vs LOAD 的 `true_|res|=0.07`：PRICE 残差的 z-score 是 LOAD 的 **39 倍**，模型只能预测残差的 1.8%，几乎可以忽略

### 1.3 新闻数据特征

- LOAD 和 PRICE **共用同一新闻源**（743 篇文章），内容来自 WattClarity 电力市场博客
- 平均 3,308 字符/篇，均为宏观电力市场分析文章
- 测试期间 74 天仅 48 天有新闻，平均 3.2 篇/天
- **新闻内容高度同质化**：均为 NEM 电力市场综合分析，缺少细粒度的供需/天气/设备事件

---

## 二、核心问题分析

### 问题 1：新闻与预测目标之间缺乏因果联系

当前新闻全部来自 WattClarity 博客的市场回顾与分析文章。这类文章的特点是：
- **事后回顾为主**：大多是对已发生事件的分析总结，而非对未来走势的前瞻性信息
- **宏观叙事**：讨论政策、可再生能源趋势等长期话题，对 30 分钟粒度的负荷/价格波动缺乏解释力
- **同质化严重**：743 篇来自同一来源，视角单一

**证据**：更换编码器（DistilBERT→GPT-2）和各种超参数调整对 PRICE 结果几乎无影响（95.43-95.46），说明**瓶颈在新闻内容本身，而非模型对新闻的处理方式**。

### 问题 2：时间对齐粒度与新闻频率严重不匹配

TemporalTextTower 对 history_len=48 的每个 30 分钟步都尝试对齐新闻。但新闻以天为粒度发布（日均 3 篇），导致：
- 绝大多数历史步对齐到**完全相同的新闻文本**（同一天内的 48 个半小时步看到同样的新闻）
- 384 次 encoder forward（8 batch × 48 步）中大量是重复计算
- 模型被迫从不变的文本中区分变化的时序，只能靠时序分支自身能力

### 问题 3：文本融合对极端值无能为力

PRICE 残差的分布是重尾的（2% 的值 >300，0.6% 的值 >1000），这些极端值主要由以下因素驱动：
- 供需瞬时失衡（发电机跳闸、负荷骤增）
- 天气极端事件（热浪、暴风）
- 市场机制（bidding strategy、dispatch constraints）

这些信息**不在宏观新闻文章中**。模型的 confidence 机制（bias=-2.0 初始化）学会了对这些样本输出低置信度来避免伤害，但也放弃了修正这些样本的机会。

### 问题 4：Unified Trunk 对文本信号的利用不够直接

从 `UnifiedResidualTrunk.forward()` 看，文本信号经过以下链路：
```
文本 → TemporalTextTower (768→64) → gated_add 融入 ts_feat → pool → delta_fuse (concat×3→hidden) → unified_trunk (hidden×4→hidden→2层MLP) → heads
```

文本信号经过至少 6 次线性变换和非线性激活后才到达预测头，信息衰减严重。对比 LOAD 的 2.28% vs PRICE 的 0.56%，说明只有在残差足够小且分布平稳（LOAD）时，这个微弱的文本信号才能起作用。

### 问题 5：cross_attention 融合模式在 LOAD 上反而更差

LOAD cross_attn (skill_mae=0.63%) 明显弱于 gated_add (skill_mae=2.28%)。分析原因：
- cross_attention 将 ts_feat 作为 query、temporal_text_patch_context 作为 key/value，但文本 patch 在时间维度上变化极少（问题 2），attention 退化为近似常量偏置
- gated_add 的门控机制（sigmoid gate, bias=-2.0）能更好地"关闭"无信息的文本通道

---

## 三、改进建议

### A. 数据侧改进（影响最大，不改代码）

#### A1. 引入细粒度事件新闻源

**问题**：当前 743 篇 WattClarity 博客文章以宏观分析为主，缺乏细粒度市场事件。

**建议**：收集以下类型的数据作为补充新闻源：
- **AEMO Market Notices**：澳大利亚电力市场运营商发布的实时市场通知（发电机跳闸、传输约束、需求预警等），与价格尖峰有直接因果关系。数据量远大于当前 743 篇
- **BOM 天气预报/预警**：澳大利亚气象局发布的极端天气预警，直接影响电力负荷和可再生能源出力
- **社交媒体/Twitter 的实时电力市场讨论**：更高频的信息来源

**预期效果**：对 PRICE 可能从 0.56% 提升至 2-5%（参考 LOAD 的 2.28%）。AEMO market notices 与价格尖峰有直接因果链接，这是当前新闻源最缺乏的。

**工作量**：纯数据收集，不需要任何代码改动。JSON 格式保持 `{title, date, content, url}` 即可。

#### A2. PRICE 和 LOAD 使用不同的新闻源

**问题**：当前两个数据集共用同一新闻文件。PRICE 的极端波动需要供给侧事件信息，LOAD 需要需求侧/天气信息。

**建议**：
- PRICE: AEMO market notices + dispatch 数据摘要 + generation outage reports
- LOAD: BOM 天气预报 + 大型活动日历 + 公共假期信息

#### A3. 增加新闻标注

**建议**：在新闻 JSON 中添加结构化标注字段：
```json
{
  "title": "...",
  "date": "...",
  "content": "...",
  "event_type": "generator_outage|weather_extreme|demand_surge|...",
  "affected_region": "NSW",
  "price_direction_hint": "up|down|neutral",
  "magnitude_hint": "high|medium|low"
}
```

这些标注可以通过 LLM 批量生成（利用已有的 API refine 管道），然后作为额外的 structured features 输入 DELTA。

---

### B. 文本编码方式改进（需改代码）

#### B1. 改进 TemporalTextTower 的时间对齐策略

**问题**：逐步对齐导致 48 步中大量重复编码，且无法区分"新闻刚发布"和"新闻已经发布 23 小时"。

**建议**：
- **时间衰减加权**：对每个历史步匹配到的新闻，根据 `(step_time - news_time)` 的时间差施加指数衰减权重。刚发布的新闻权重高，旧新闻权重低
- **去重编码 + 位置映射**：先对每个样本的唯一新闻文本做一次编码，然后通过查表 + 时间位置编码将结果映射到 48 个步。避免重复 encoder forward

**预期效果**：
- 编码效率：从 48 次/样本降至 ~3 次/样本（因为 topk=3 篇新闻）
- 预测质量：时间衰减权重让模型能区分新鲜信息和过期信息

#### B2. 新闻 Summary 级别的直接残差预测

**问题**：文本信号经过过多层变换后衰减。

**建议**：添加一个从 text_summary 直接到 horizon 维度的 shortcut head：
```
text_residual_head: text_summary (hidden_size) → horizon
```
在最终残差预测中加入：`final_delta = main_delta + alpha * text_residual_head(text_summary)`，其中 `alpha` 是一个可学习的标量（初始化为 0）。这让文本信号有一条直达预测的路径，梯度更强。

#### B3. 文本编码器选择

**发现**：GPT-2 和 DistilBERT 在 PRICE 上效果一致，说明编码器不是瓶颈。

**建议**：统一使用 DistilBERT（参数少 40%，速度快 2x），把节省的计算量用于更多 epoch 或更大的 hidden_size。

---

### C. 模型结构改进（需改代码）

#### C1. 条件性残差修正（Conditional Residual Correction）

**问题**：当前模型对所有样本用同一套头做修正。但 PRICE 数据中 98% 的正常值和 2% 的极端值需要完全不同的修正策略。

**建议**：基于已有的 RegimeRouter 扩展一个"修正策略路由"：
- **Normal regime**：小幅修正，高置信度，主要靠时序模式
- **Spike regime**：大幅修正，低置信度，需要事件新闻支撑
- **Negative regime**：需要供给过剩信号（太阳能过剩等）

路由依据：history_z 的波动率、结构化特征中的事件类型、text_strength。

**关键区别**：当前的 RegimeRouter 只在 `plan_c_mvp` 架构下使用。建议在 `summary_gated` + `unified` 模式下也引入轻量级的条件路由。

#### C2. 残差目标的自适应缩放

**问题**：PRICE 的残差 `true_|res|=2.76`（z-score）但 `|delta|=0.05`，模型只学到了真实残差的 1.8%。这说明 loss 函数对大残差样本的梯度被平均掉了。

**建议**：
- **分位数加权损失**：对残差绝对值在 top-10% 的样本给更高的 loss 权重（例如 3x），迫使模型学习大残差修正
- **对数空间残差**：将 relative 模式的残差目标做 `sign(r) * log1p(|r|)` 变换，压缩极端值范围，让模型更容易学习
- **当前 `delta_mag_target=log1p` 只作用于 magnitude target**，建议对整个残差目标空间做类似处理

#### C3. 文本融合时机前移

**问题**：文本在 patch 级别融合后，信息被 pool 操作平均化。

**建议**：在 `delta_fuse` 之前，让文本直接参与 residual_context 的构建：
- 当前：`residual_base = delta_fuse(concat[pooled, fused_news_context, text_summary * text_strength])`
- 建议：让 text_summary 也参与 unified_trunk 的输入（已经在做，但只是拼接后过 MLP），增加一个 cross-attention 让 text_summary 直接 attend 到 history_z 和 base_pred_z，产生更有针对性的文本条件信息

---

### D. 训练策略改进（部分不需要改代码）

#### D1. 新闻对比学习预训练（需改代码）

**问题**：TemporalTextTower 的编码器冻结（`unfreeze_last_n=0`），step_proj 和 patch_proj 只通过残差 loss 的间接梯度学习，信号太弱。

**建议**：在 DELTA 训练开始前，用对比学习预训练 TemporalTextTower：
- **正样本**：同一时间窗口的新闻和对应的时序模式
- **负样本**：不同时间窗口的新闻和时序模式
- **目标**：让 text embedding 和 time-series embedding 在语义空间中对齐

这能让 TemporalTextTower 在 DELTA 训练前就学到"哪些新闻内容与哪种时序模式相关"。

#### D2. 课程学习（不需要改代码，仅调参）

**建议**：
- **delta_warmup_epochs**：前 5 个 epoch 将 `temporal_text_fuse_lambda` 从 0 线性增加到 1.0，让模型先学好时序模式，再逐步引入文本信号
- 当前日志显示 LOAD 前 7 个 epoch skill_mae 为负（模型在伤害预测），可能是因为文本信号一开始就全量加入导致干扰

**可以通过调整已有的 `--delta_warmup_epochs` 参数实现**。

#### D3. 增加 early_stop_patience（不需要改代码）

**发现**：
- LOAD 在 epoch 29 达到最佳（29/100 epoch，patience=5），之后 5 个 epoch 未提升就停止了
- PRICE 在 epoch 10 就被 early stop 了

**建议**：
- 对 LOAD，增大 patience 到 8-10，让模型有更多探索空间
- 对 PRICE，patience=5 足够（因为模型在 epoch 7-10 就已经饱和）

#### D4. 增大 DELTA 学习率（不需要改代码）

**发现**：当前 lr=5e-6 对于冻结编码器的 DELTA 训练来说非常保守。模型的可训练参数主要是轻量级 MLP head（patch_proj、delta_fuse、unified_trunk 等），不需要这么小的学习率。

**建议**：尝试 `lr=1e-4` 或 `5e-5`，配合 warmup。这可以让模型更快学到有效的文本融合模式，减少训练轮数。

---

### E. 评估与诊断改进

#### E1. 新闻有效性消融实验（不需要改代码）

**建议**：跑一组完全关闭新闻的消融实验（`--delta_temporal_text_enable 0 --delta_structured_enable 0`），获得纯时序的 DELTA 基线。当前只有 Base（无 DELTA）的基线，没有"有 DELTA 但无新闻"的基线。

这能回答一个关键问题：**当前的改善（LOAD 2.28%）有多少来自 DELTA 的时序建模，多少真正来自新闻信息？**

#### E2. 按新闻密度分层评估

**发现**：日志已有 `[no_news]`、`[sparse_news]` 分层，但有趣的是：
- LOAD: no_news MAE=534.9，sparse_news MAE=471.9 — 有新闻的样本效果好 12%
- PRICE: no_news MAE=61.6，sparse_news MAE=105.5 — 有新闻的样本反而 MAE 更高

**解读**：PRICE 有新闻时 MAE 更高不代表新闻有害，而是因为有新闻的时段（工作日白天）恰好是价格更波动的时段。需要用 **相对于 base 的改善量** 而非绝对 MAE 来评估。

**建议**：在 FINAL_DIAG 中增加 `[no_news] skill_mae` 和 `[sparse_news] skill_mae`，而不只是绝对 MAE。

#### E3. 极端值专项评估

**建议**：对 PRICE 增加针对极端值的评估：
- `[spike] n=? base_mae=? final_mae=?`（价格 > 300 的样本）
- `[negative] n=? base_mae=? final_mae=?`（价格 < 0 的样本）
- `[normal] n=? base_mae=? final_mae=?`（其余样本）

---

## 四、优先级排序

| 优先级 | 建议 | 预期效果 | 是否需要改代码 | 工作量 |
|--------|------|---------|---------------|--------|
| **P0** | A1. 引入 AEMO Market Notices 新闻源 | PRICE 可能提升 3-10x | 否（数据收集） | 中 |
| **P0** | E1. 无新闻消融实验 | 明确新闻贡献量 | 否（改脚本参数） | 小 |
| **P1** | D2. 课程学习（warmup text fusion） | 减少早期干扰 | 否（调参） | 小 |
| **P1** | D4. 增大 DELTA lr | 加速收敛 | 否（调参） | 小 |
| **P1** | C2. 残差目标自适应缩放 | PRICE 提升 | 是 | 中 |
| **P1** | B1. TemporalText 去重编码 | 效率+质量双提升 | 是 | 中 |
| **P2** | B2. Text shortcut head | 文本信号直达预测 | 是 | 小 |
| **P2** | C1. 条件性残差修正 | PRICE 极端值处理 | 是 | 大 |
| **P2** | A3. 新闻标注 | 提升 structured features | 否（LLM批量生成） | 中 |
| **P3** | D1. 对比学习预训练 | 文本对齐质量 | 是 | 大 |
| **P3** | E2/E3. 评估指标完善 | 诊断能力 | 是 | 小 |

---

## 五、总结

**最关键的发现**：当前框架在新闻处理方面的技术管道（TemporalTextTower、gated fusion、unified trunk）设计合理，LOAD 上 2.28% 的 Skill_MAE 和 63% 的 Helped Rate 证明了管道的有效性。**PRICE 提升小的根本原因不在模型架构，而在新闻数据本身**：

1. **新闻与价格尖峰无因果关系**：WattClarity 博客不报道 30 分钟粒度的供需事件
2. **新闻频率与时序粒度不匹配**：日均 3 篇 vs 48 个半小时步
3. **残差分布极端**：PRICE 残差 z-score 是 LOAD 的 39 倍，超出了 gated text fusion 的有效修正范围

因此，**最高优先级的改进是数据侧的：引入与 PRICE 有直接因果关系的细粒度事件数据源（AEMO Market Notices）**。模型侧的改进（B1/B2/C1/C2）可以进一步提升效果，但在当前新闻数据下，天花板已经被数据质量限制了。
