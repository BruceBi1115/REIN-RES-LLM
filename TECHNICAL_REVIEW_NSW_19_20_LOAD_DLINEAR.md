# TECHNICAL REVIEW (Current Project State)

Last updated: 2026-03-13  
Scope: `tinynews + base/delta + case retrieval` current code path (no conv branch in active forward path)

## 1) Executive Summary

- 当前主实验（NSW 2024 electricity price）已经进入“稳定小幅提升”阶段：  
  - Base-only test MAE: **118.4445**  
  - Final (Base+Delta) test MAE: **117.8560**  
  - 绝对提升: **0.5885 MAE**（约 **0.50%**）
- 提升主要来自 **case retrieval + delta residual correction**，而不是文本 token 直接建模。
- 当前代码中，预测主干并未直接使用大模型做生成式预测；OpenAI API 主要用于 **news refine / structured event extraction**（可缓存）。
- 风险点不是“模型不会学”，而是“增益上限偏低”：delta 主要在 base 周围做小修正，提升容易被门控/先验约束压住。

## 2) What Is Actually Running Now

### 2.1 Core Pipeline

1. Base stage: 纯时间序列 backbone（当前脚本默认 `mlp`）训练并保存 best base。  
2. Case bank build: 若启用 retrieval，用 train split + base 预测构建 case bank。  
3. Refine cache prewarm: 预热新闻 refine cache，减少 delta 阶段 API 成本与等待。  
4. Delta stage: 训练 residual correction（支持 additive/relative，两者当前脚本默认 relative）。  
5. Test + ablation: 运行最终测试和 retrieval 消融。

### 2.2 Model Path (Important)

- 当前 `TinyNewsTSRegressor` 的 forward 实际用的是：
  - `ts_patches` -> patch MLP encoder -> pooled TS feature
  - base head / delta head / delta gate / retrieval bias
- `input_ids`、`attention_mask` 在 forward 里被显式丢弃（不参与预测计算）。
- 这意味着：
  - 文本并不直接进入预测头；
  - 文本影响是间接的，主要通过 refine -> structured/retrieval 特征 -> delta 修正链路。

### 2.3 API Usage

- API 模型（脚本默认 `gpt-5.1`）用于：
  - `refine_news_text`：压缩为固定模板方向摘要；
  - `extract_structured_events`（若 `news_structured_mode=api`）；
- cache 机制已生效，格式为标准 JSON array（`cache_key + refined_news`），并在日志中持续记录 hit/miss。

## 3) Current Script Snapshot (tinynews main script)

`scripts/nswelecprice_2024_tinynews.sh` 当前关键设置：

- Stage: `all`
- Base backbone: `mlp`
- Delta residual mode: `relative`
- Case retrieval: `enable=1`, `mode=price_event`, `topk=3`
- Ablation split: `test`
- Refine mode: 默认 API（如果 key 可用）
- Conv 相关开关：当前主路径无 conv forward 贡献

## 4) Latest Observed Results

From `checkpoints/[2024-nswelecPrice-tinynews]_mlp/[2024-nswelecPrice-tinynews]_mlp_mlp.log`:

- Best delta validation (epoch 6):  
  - Delta val MAE: **62.383731**  
  - Base-only val MAE: **62.610653**  
  - Delta vs base: **-0.226922**
- Final test:
  - `[TEST][FINAL]` MAE: **117.855973**
  - `[TEST][BASE_ONLY]` MAE: **118.444495**

From test ablation:

- `no_retrieval`: **118.436602**
- `price_only`: **117.861588** (improve 0.5750)
- `price_event`: **117.855973** (best, improve 0.5806)
- `gate_only`: **117.856918** (几乎与 full 持平)
- `random_retrieval`: **118.474581** (负贡献)

Interpretation:

- 检索本身是有效信号（price / price_event 明显优于 no_retrieval）。
- 随机检索会恶化表现，说明不是“多加一个分支就有用”，而是“相似性质量决定收益”。
- `gate_only` 接近 full，说明当前阶段 retrieval 对“门控/可信度”贡献大于对“delta数值形状”贡献。

## 5) Why Improvement Is Still Limited

1. Delta correction幅度被多重稳定项限制（gate、null regularization、knn blending等），策略偏保守。  
2. Delta 训练目标本质仍是“围绕 base 的小修正”，不是重建主预测。  
3. 检索先验有效但上限有限，尤其在高波动样本上，top-k 先验可能过于平滑。  
4. 当前文本作用路径是“摘要->结构化->检索特征”，不是端到端语义建模，信息利用深度受限。  

## 6) Highest-Impact Next Steps (Practical)

### P0 (先做，低风险高收益)

1. 做“高波动子集”参数分桶：对高波动样本放宽 gate 与 knn 抑制，低波动维持保守。  
2. 调整 knn blend：降低默认 alpha，改为更依赖 retrieval confidence 的自适应 alpha。  
3. 提高 retrieval 特征的信息密度：增加“候选分歧度、残差分位数、event一致性”输入，而不是仅均值统计。

### P1 (中等改动)

1. 两阶段 delta：先学 gate/relevance，再学 correction magnitude（冻结/半冻结策略）。  
2. 检索候选重排增加“时段结构约束”（同小时/同weekday权重更高）。  
3. 对 `relative` 模式增加 regime-aware denominator floor（不是全局常数 floor）。

### P2 (研究向)

1. 从 case bank 的 top-k 序列构建轻量 cross-attn 聚合器，替代手工统计特征。  
2. 针对 retrieval fail 样本加入 hard-negative 对比训练，提升 reject 机制与权重标定。

## 7) Reproducibility Notes

- 当前结果对应的是“最新代码 + 当前 tinynews 脚本配置”；历史 run 的可比性会受以下因素影响：
  - 是否命中 refine cache；
  - case bank是否重建；
  - train/val/test split 是否完全一致；
  - 随机种子与 checkpoint 选择策略。
- 如果要做严谨对比，建议固定：
  1. cache 文件；
  2. case bank 文件；
  3. seed；
  4. 单一脚本快照（commit hash）。

## 8) Bottom Line

- 当前框架已经验证：**case retrieval 对 test 有稳定正贡献**。  
- 但系统仍偏“稳健保守型 delta”，所以提升不是爆发式。  
- 想把 MAE 再明显压低，关键不是继续加模块，而是做“按样本难度/波动状态的自适应放权与检索质量提升”。
