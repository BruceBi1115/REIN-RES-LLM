# 技术分析报告：REIN-RES-LLM 框架

> 一个两阶段残差预测框架：一个冻结的时序 **base** 骨干 + 一个新闻条件化的 **delta_v3** 残差头。Delta 通过乘性调制（而非加性修正）base 预测。

---

## 1. 系统总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  原始 CSV ─┐                                                              │
│           │                                                              │
│           ├──→ [阶段 0] 数据管道 → batch dict                            │
│           │                                                              │
│           ├──→ [阶段 1] Base 训练（之后冻结）                             │
│           │          ↓                                                   │
│           │      base_pred_z, base_hidden                                │
│           │                                                              │
│  新闻 JSON ├──→ [阶段 2a] Schema Refine v2 → refined.jsonl               │
│           │                                                              │
│           └──→ [阶段 2b] Regime Bank 构建 → .npz                         │
│                                ↓                                         │
│                          [阶段 3] Delta V3 训练                          │
│                                ↓                                         │
│                          final_pred = base + λ·residual                  │
└──────────────────────────────────────────────────────────────────────────┘
```

5 个顺序管道由 5 个明确定义的数据接口连接（详见 §3–§7）。

---

## 2. 关键形状图例

以下符号贯穿全报告。`B` = batch size，`L` = 历史长度（默认 48），`H` = 预测 horizon（48/96/192/336/720），`D` = hidden size（128）。

| 符号 | 含义 | 典型值 |
|---|---|---|
| `B` | Batch size | 16 |
| `L` | 历史长度（步数） | 48 |
| `H` | 预测 horizon（步数） | 48–720 |
| `D` | Hidden dim（delta encoder） | 128 |
| `P` | Patch 长度 | 8 |
| `S` | Patch stride | 4 |
| `N_p` | Patch 数量 | `1 + (L-P)/S` = 11 |
| `R` | Regime 向量维度 | 5 |
| `T` | Topic tag 数量 | 15 |
| `E` | 文本 embedding 维度 | 384 |

---

## 3. 节点 1 — 原始数据加载器

**文件**：[src/data_construction/data.py](src/data_construction/data.py)
**类**：`SlidingDataset`（第 36 行）

### 3.1 输入

原始 CSV/Parquet 文件，至少包含两列：

| 列 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `date`（通过 `--time_col` 配置） | str | `"2024-01-01 00:30:00"` | 支持 day-first 或 ISO 格式 |
| `value`（通过 `--value_col` 配置） | float | `492.3`（负荷），`45.6`（价格） | 目标时间序列 |
| `id`（可选，通过 `--id_col` 配置） | str | `"NSW"` | 仅多序列数据集需要 |

**采样频率**：30 分钟（NEM 市场标准）。48 步 = 24 小时。

### 3.2 输出：`batch` dict（每次迭代）

由 `SlidingDataset.__getitem__` 生成，`DataLoader` 聚合：

```python
{
    "history_value":  torch.Tensor,  # shape (B, L),  dtype=float32, 原始单位（MW 或 $/MWh）
    "target_value":   torch.Tensor,  # shape (B, H),  dtype=float32, 原始单位
    "history_times":  list[list[str]],    # B × L 时间戳 "YYYY-MM-DD HH:MM:SS"
    "target_times":   list[list[str]],    # B × H 时间戳
    "target_time":    list[str],          # 长度 B —— 每个样本的首个目标时间戳
    "series_id":      list[str],          # 长度 B —— 例如 ["NSW","NSW",...]
}
```

**样例**（单个样本，L=48，H=48）：
```
history_value = [492.3, 485.1, ..., 510.9]         # 48 个值
target_value  = [508.4, 515.2, ..., 468.1]          # 48 个值
history_times = ["2024-01-01 00:30:00", ..., "2024-01-02 00:00:00"]
target_times  = ["2024-01-02 00:30:00", ..., "2024-01-03 00:00:00"]
target_time   = "2024-01-02 00:30:00"
series_id     = "NSW"
```

---

## 4. 节点 2 — 归一化（原始 → Z-space）

**文件**：[src/base/common.py](src/base/common.py)
**函数**：`_compute_global_zstats_from_train_df`（第 182 行），`_z_batch_tensors`（第 451 行）

### 4.1 全局统计量计算（训练前执行一次）

**输入**：训练集 DataFrame 的完整 `value_col`（numpy 数组，shape `(N_train,)`）

**输出**：`global_zstats` dict（保存到 checkpoint meta 以保证可复现）：

```python
{
    "normalization_mode": "robust_quantile",  # 或 "zscore"
    "center_global":       float,   # robust_quantile 用中位数；zscore 用均值
    "scale_global":        float,   # robust_quantile 用 IQR/1.349；zscore 用标准差
    "quantile_low":        0.25,
    "quantile_high":       0.75,
    "quantile_low_value":  float,   # 原始 Q25 值
    "quantile_high_value": float,   # 原始 Q75 值
    # 向后兼容的别名：
    "mu_global": float, "sigma_global": float, "center": float, "scale": float,
}
```

**默认模式：`robust_quantile`** —— 使用 Q25 / 中位数 / Q75，抵抗 spike 污染。对 NSW Price，原始 `σ_std ≈ 635`，但 robust scale 要小得多（通常 20–40），这是抵御极端 spike 的主要防线。

### 4.2 逐 batch 归一化

**输入**：`batch` dict（来自节点 1）+ `global_zstats`
**输出**：

```python
history_z:    torch.Tensor   # (B, L),  dtype=float32, z-space 归一化
targets_z:    torch.Tensor   # (B, H),  dtype=float32, z-space 归一化
metas:        list[dict]     # 每个样本一份 global_zstats 拷贝
```

公式：`history_z = (history_value − center_global) / scale_global`

---

## 5. 节点 3 — Base 骨干

**文件**：[src/base_backbone.py](src/base_backbone.py)
**类**：`MLPBackbone`（第 51 行），`DLinearBackbone`（第 26 行）

### 5.1 输入

```python
history_z: torch.Tensor     # (B, L), float32, z-space
```

### 5.2 输出

两种调用签名：

**不返回 hidden**（默认推理）：
```python
pred: torch.Tensor          # (B, H), float32, z-space
```

**返回 hidden**（供 delta 做条件输入）：
```python
pred:   torch.Tensor        # (B, H),      float32, z-space
hidden: torch.Tensor        # (B, D_base), float32 —— 内部表征

# D_base 取决于 backbone：
#   MLPBackbone: D_base = hidden_dim（默认 256）
#   DLinearBackbone: D_base = 2L = 96  （seasonal 与 trend 特征拼接）
```

### 5.3 训练目标（阶段 1）

- **损失**：`F.smooth_l1_loss(pred, targets_z)`，在 z-space 中计算
- **训练轮数**：40（默认），lr=1e-3，SmoothL1 损失
- **Checkpoint 保存路径**：`checkpoints/{taskName}/best_base_backbone/`
  - `base_backbone.pt`（state dict）
  - `meta.json`（backbone 类型、L/H、global_zstats 供后续复用）

阶段 1 完成后，该模型在所有下游使用中被 **冻结**（`requires_grad=False`，`eval()` 模式）。

---

## 6. 节点 4 — 新闻精炼管道（离线，每个数据集执行一次）

**文件**：[src/delta_v3/schema_refine_v2.py](src/delta_v3/schema_refine_v2.py)

### 6.1 输入：原始新闻 JSONL

每行 = 一篇原始文章：
```json
{
  "id": "wattclarity_20240115_abc",
  "title": "NSW battles through third heatwave day...",
  "date": "2024-01-15 08:00:00",
  "url": "https://...",
  "content": "文章正文全文（纯文本，可能较长）"
}
```

### 6.2 处理：LLM（GPT）分类

每篇文章通过结构化 prompt 发送给 LLM，返回固定 JSON schema。

### 6.3 输出：`refined.jsonl`

每行：

```json
{
  "doc_key":       "wattclarity_20240115_abc",
  "published_at":  "2024-01-15 08:00:00",
  "is_actionable": true,                          // false = 下游永久忽略
  "topic_tags":    ["heatwave", "supply_tight"],  // 封闭词表多标签（15 个）
  "regime_vec": {
      "tightness":         -0.6,   // [-1, +1]
      "demand_outlook":     0.8,   // [-1, +1]
      "renewable_surplus": -0.2,   // [-1, +1]
      "volatility_tone":    0.7,   // [0, +1]
      "policy_in_effect":   0.0    // {0, 1}
  },
  "horizon_days": 7,           // 整数 [1, 14]
  "confidence":   0.85,        // [0, 1] LLM 自评置信度
  "summary":     "Heatwave expected to last 3 days with record demand..."   // ≤60 词
}
```

**封闭词表**（定义于 [schema_refine_v2.py:13-36](src/delta_v3/schema_refine_v2.py#L13)）：
- `TOPIC_TAGS`（15 个）：supply_tight, supply_surplus, heatwave, cold_snap, holiday, outage, fuel_shock, renewable_surge, renewable_drought, interconnector_limit, policy_active, market_intervention, retrospective, routine, other
- `REGIME_KEYS`（5 个）：tightness, demand_outlook, renewable_surplus, volatility_tone, policy_in_effect

---

## 7. 节点 5 — Regime Bank（离线聚合）

**文件**：[src/delta_v3/regime_bank.py](src/delta_v3/regime_bank.py)
**函数**：`build_regime_bank`（第 94 行）

### 7.1 输入

- `refined.jsonl`（来自节点 4）
- 日期范围 `[date_start, date_end]`
- 文本编码器：`intfloat/e5-small-v2`（384 维句嵌入）
- 超参数：`tau_days=5`，`ema_alpha=0.5`

### 7.2 算法（按日聚合）

```
对每一天 d ∈ [date_start, date_end]：
    in_force = { 文章 : article.published_at ≤ d ≤ article.published_at + horizon_days
                        且 is_actionable == True }
    若 in_force 为空则跳过

    对 in_force 中每篇文章：
        age = (d - published_at).days
        weight[article] = confidence × exp(-age / tau_days)

    relevance_mass[d] = sum(weight)  # 未归一化的"信号强度"
    对 weight 归一化；加权平均 regime_vec、topic_mass、text_emb

沿日期方向做 EMA 平滑（alpha=0.5）
```

### 7.3 输出：`.npz` 文件

保存于 `checkpoints/_shared_refine_cache/v4/regime_bank_{dataset}.npz`：

```python
{
    "dates":              ndarray['<U10'],   # shape (N_days,) —— "YYYY-MM-DD" 字符串
    "regime_vec":         float32,           # shape (N_days, 5)
    "topic_tag_mass":     float32,           # shape (N_days, 15)
    "text_emb":           float32,           # shape (N_days, 384)
    "relevance_mass":     float32,           # shape (N_days,) —— 关键门控信号
    "in_force_doc_count": int64,             # shape (N_days,) —— 当日生效文章数
    "topic_tags":         ndarray['<U32'],   # shape (15,) —— tag 名称查找表
    "regime_keys":        ndarray['<U32'],   # shape (5,)  —— regime key 查找表
}
```

### 7.4 运行时查询 API

`RegimeBank.lookup(target_date)` 返回 5 元组：

```python
regime_vec:          ndarray (5,)
topic_tag_mass:      ndarray (15,)
text_emb:            ndarray (384,)
relevance_mass:      ndarray (1,)       # 封装成数组的标量
in_force_doc_count:  ndarray (1,)       # 封装成数组的标量
```

若日期在 bank 范围之外 → 全 0（视为"无新闻 regime"）。

---

## 8. 节点 6 — Regime Pack 构建器（逐 batch）

**文件**：[src/delta_v3/trainer.py](src/delta_v3/trainer.py)
**函数**：`_build_regime_pack`

### 8.1 输入

- `batch` dict（来自节点 1，使用 `target_time`）
- `RegimeBank` 实例

### 8.2 输出：`regime_pack`（传入 delta 模型）

```python
{
    "regime_vec":      torch.Tensor,   # (B, 5),   float32
    "topic_tag_mass":  torch.Tensor,   # (B, 15),  float32
    "text_emb":        torch.Tensor,   # (B, 384), float32
    "relevance_mass":  torch.Tensor,   # (B,) 或 (B, 1), float32 —— 硬门控信号
}
```

**另加辅助 tensor**（仅用于日志，不参与 forward）：
```python
regime_active_used:   list[int]   # 长度 B —— 若当日新闻有效则为 1
in_force_docs_used:   list[int]   # 长度 B —— 每个样本的生效文章数
```

---

## 9. 节点 7 — Delta V3 前向传播

**文件**：[src/delta_v3/model.py](src/delta_v3/model.py)
**类**：`DeltaV3Regressor`

### 9.1 输入签名

```python
model.forward(
    history_z:      torch.Tensor,        # (B, L)
    history_times:  list[list[str]],     # B × L 时间戳（用于日历嵌入）
    base_pred_z:    torch.Tensor,        # (B, H) —— 来自冻结 base
    base_hidden:    torch.Tensor | None, # (B, D_base) —— 可选的 base 条件
    regime_pack:    dict,                # （来自节点 6）
) → dict of tensors
```

### 9.2 内部管道

#### 步骤 1：TS 编码器（`PatchTSTTSEncoder`）
```
输入：  history_z (B, L), history_times, base_hidden (B, D_base)
 ├─ patchify          → (B, N_p, P)        其中 N_p=11, P=8
 ├─ patch_proj        → (B, N_p, D)        D=128
 ├─ + 日历嵌入        （DOW[0..6], HOD[0..47], holiday[0..1]）
 ├─ 前置 base_hidden 作为 token → (B, N_p+1, D)
 ├─ Transformer（2 层，4 头）
 └─ LayerNorm

输出：
  ts_tokens:  (B, N_p+1, D)   —— token 级特征
  ts_summary: (B, D)          —— 在 token 维上均值池化
```

#### 步骤 2：Regime 编码器（`RegimeProjector`）
```
输入：  concat(regime_vec, topic_tag_mass, text_emb) → (B, 5+15+384) = (B, 404)
 ├─ LazyLinear(404→128) + GELU + LayerNorm
 ├─ Linear(128→128)    + GELU + LayerNorm
 └─ × active_mask（当 relevance_mass ≤ 0.7 时硬置零）

输出：
  regime_repr: (B, D)   —— 新闻不活跃时为零向量
```

#### 步骤 3：TS-only 残差头
```
slow_ts:              (B,)      —— 每样本的标量水平偏移  [SlowHead(ts_summary)]
shape_ts:             (B, H)    —— 中心化形状（零均值）  [ShapeHead(ts_tokens)]
spike_ts:             (B, H)    —— spike 幅度           [SpikeHead delta 分支]
spike_gate_logits_ts: (B, H)    —— spike gate 预激活值  [SpikeHead gate 分支]
```

#### 步骤 4：调制旋钮（`RegimeModulationHeads`）
```
lambda_ts          = λ_min + λ_ts_cap × sigmoid(ts_trust_mlp(ts_summary))
                                              (B,) ∈ [0.05, 0.35]

lambda_news_delta  = λ_news_cap × tanh(trust_delta_mlp(regime_repr) × active)
                                              (B,) ∈ [−0.12, +0.12]

lambda_base        = clamp(lambda_ts + lambda_news_delta, λ_min, λ_max)
                   × soft_gate(relevance_mass − 0.7)       # 平滑地抑制不活跃样本
                                              (B,) ∈ [0.05, 0.45]

shape_gain         = 1.0 + 0.20 × tanh(shape_gain_mlp(regime_repr) × active)
                                              (B,) ∈ [0.80, 1.20]

spike_bias         = 0.75 × tanh(spike_prior_mlp(regime_repr) × active)
                                              (B,) ∈ [−0.75, +0.75]
```

**关键安全不变量**：当 `relevance_mass ≤ 0.7`（无新闻）时，`active = 0` → `shape_gain → 1`，`spike_bias → 0`，`lambda_base → λ_min`。模型优雅地退化为仅 TS 的 delta。

#### 步骤 5：组合（新闻触及残差的唯一方式是乘性）
```
spike_gate_logits  = spike_gate_logits_ts + spike_bias.unsqueeze(-1)   # (B, H)
spike_gate         = sigmoid(spike_gate_logits)                         # (B, H)

shape_z            = shape_ts * shape_gain.unsqueeze(-1)                # (B, H)
residual_z         = slow_ts.unsqueeze(-1)                              # (B, H)
                     + shape_z
                     + spike_gate * spike_ts

pred_z             = base_pred_z + lambda_base.unsqueeze(-1) * residual_z
```

### 9.3 输出：`out` dict

```python
{
    # 主预测
    "pred_z":               torch.Tensor,   # (B, H) —— z-space 中的最终预测
    "residual_hat":         torch.Tensor,   # (B, H) —— pred_z − base_pred_z（用于 loss）

    # TS 残差组件（调制前）
    "slow_ts":              torch.Tensor,   # (B,)    标量水平偏移
    "shape_ts":             torch.Tensor,   # (B, H)  原始 shape
    "spike_ts":             torch.Tensor,   # (B, H)  原始 spike 幅度
    "shape_z":              torch.Tensor,   # (B, H)  已调制 shape = shape_ts * shape_gain

    # 门控诊断
    "spike_gate_logits_ts": torch.Tensor,   # (B, H)  调制前 logits
    "spike_gate_logits":    torch.Tensor,   # (B, H)  logits + spike_bias
    "spike_gate":           torch.Tensor,   # (B, H)  sigmoid(logits)
    "spike_gate_hard":      torch.Tensor,   # (B, H)  (spike_gate ≥ 0.8).float()

    # 调制旋钮（用于日志/诊断）
    "lambda_base":          torch.Tensor,   # (B,)  ∈ [0.05, 0.45]
    "lambda_ts":            torch.Tensor,   # (B,)  ∈ [0.05, 0.35]
    "lambda_news_delta":    torch.Tensor,   # (B,)  ∈ [-0.12, 0.12]
    "shape_gain":           torch.Tensor,   # (B,)  ∈ [0.80, 1.20]
    "shape_gain_raw":       torch.Tensor,   # (B,)  MLP 原始输出（tanh 前）
    "spike_bias":           torch.Tensor,   # (B,)  ∈ [-0.75, 0.75]
    "trust_logit":          torch.Tensor,   # (B,)  MLP 原始输出（sigmoid 前）

    # 新闻激活元信息
    "active_mask":          torch.Tensor,   # (B,) {0, 1} —— relevance_mass > 0.7 时为 1
    "relevance_mass":       torch.Tensor,   # (B,)  每样本的标量信号强度
}
```

---

## 10. 节点 8 — 残差目标分解

**文件**：[src/delta_v3/targets.py](src/delta_v3/targets.py)
**类**：`ResidualTargetDecomposer`

将真实残差（true − base_pred）分解为与三个头对应的三个解耦组件。

### 10.1 输入
```python
true_residual_z:   torch.Tensor   # (B, H) —— targets_z - base_pred_z
target_times:      list[list[str]] # B × H —— 用于日历基线查找
```

### 10.2 输出：`target_parts` dict
```python
{
    "slow_target":  torch.Tensor,   # (B,)    真实水平偏移
    "shape_target": torch.Tensor,   # (B, H)  真实 shape 偏离
    "spike_target": torch.Tensor,   # (B, H)  真实 spike 幅度
    "spike_mask":   torch.Tensor,   # (B, H)  {0, 1} —— |残差| > k×σ 的二值掩码
}
```

分解器使用从训练集学习到的预计算 **日历基线** `(7 DOW × 48 HOD)`，加上基于分位数的 spike 阈值（load 用 k=3，price 用 k=4）。

---

## 11. 节点 9 — 损失计算

**文件**：[src/delta_v3/trainer.py](src/delta_v3/trainer.py)
**函数**：`_main_point_loss`（第 401 行），以及 `_compute_losses` 中的辅助损失

### 11.1 点预测损失（按 schema 区分）

**Load 变体**（`schema.variant = "load"`）：
```python
loss_point = F.smooth_l1_loss(pred_z, targets_z)   # z-space
```

**Price 变体**（`schema.variant = "price"`）：
```python
pred_raw = denormalize(pred_z)
true_raw = targets_raw
pred_w   = clamp(pred_raw, low, high)              # winsorize (0.5%/99.5%)
true_w   = clamp(true_raw, low, high)
quantile_loss = mean(pinball(pred_raw, true_raw, q)) for q in [0.1, 0.5, 0.9]
loss_point    = F.smooth_l1_loss(pred_w, true_w) + 0.1 * quantile_loss
```

### 11.2 完整损失组合

```
total_loss = loss_point                                   # 主预测损失
           + slow_weight   × MSE(slow_ts,  slow_target)   # 1.0
           + shape_weight  × MSE(shape_z,  shape_target)  # 1.0
           + spike_weight  × (
               BCE(spike_gate_logits, spike_mask)
             + gate_loss_weight × extra_gate_reg          # 0.25
           )
           + consistency_weight × ‖pred(news_on) − pred(news_off)‖²  # 0.05
                                 × (1 − active_mask)                 # 仅在不活跃时生效
           + counterfactual_weight × orthogonality_term              # 0.1
           + inactive_residual_weight × ‖residual_z‖²                # 0.1（仅 load）
           + spike_bias_l2 × ‖spike_bias‖²                           # 1e-3
```

---

## 12. 节点 10 — 评估与诊断

**文件**：[src/delta_v3/trainer.py](src/delta_v3/trainer.py)
**函数**：`evaluate_delta_v3`（第 449 行）

### 12.1 每样本三次前向

1. **Normal**：完整 regime_pack
2. **Blank 反事实**：`regime_pack` 全零 → 新闻不活跃时应与 normal 一致
3. **Permuted 反事实**：日期打乱 → 新闻活跃时应明显退化

### 12.2 输出：`diag` dict（`results/test_results.csv` 的一行）

主要指标：

| 指标 | 公式 | 解读 |
|---|---|---|
| `mse`, `mae` | 原始空间误差 | 主要性能指标 |
| `base_mse`, `base_mae` | 仅 base 的误差 | 参考基线 |
| `skill_score_mae` | `1 − mae / base_mae` | 正 = delta 有帮助 |
| `delta_helped_rate` | `P(|err_delta| < |err_base|)` | 被改善的样本比例 |
| `delta_helped_rate_top10pct_residual` | 在 base 误差最大 10% 样本上的 help rate | 尾部行为的关键指标 |
| `top10pct_residual_mae` | 最难 10% 样本的 MAE | 尾部绝对误差 |
| `regime_active_pct` | `P(relevance_mass > 0.7)` | 新闻覆盖率 |
| `lambda_base_mean` | 所有样本的平均 λ | 期望 ∈ [0.1, 0.4] |
| `lambda_saturation_pct` | `P(λ ≥ 0.98 × λ_max)` | 应 < 0.35 |
| `shape_gain_mean` | 平均 shape 幅度调制 | 不活跃时 ~1.0 |
| `spike_bias_mean` | 平均 spike logit 偏移 | 不活跃时为 0 |
| `inactive_blank_gap_pct` | `(inactive_normal − inactive_blank)/inactive_blank × 100` | 必须在 ±2% 内 |
| `active_blank_gain_z` | 新闻活跃子集上的 z-space MAE 差 | 正 = 新闻有帮助 |
| `active_permuted_gain_z` | 日期打乱后的 z-space MAE 差 | 正 = 收益真实 |
| `spike_gate_hit_rate` | gate ≥ 0.8 的步数比例 | 通常 2–7% |
| `spike_target_hit_rate` | 真实 spike_mask = 1 的步数比例 | 通常 3–10% |

外加按 `active ∈ {0, 1}` 分层的 `lambda_base`、`shape_gain`、`spike_bias` 直方图。

### 12.3 输出文件格式

[results/test_results.csv](results/test_results.csv) 列：

```
Task, MSE, MAE, Base_MSE, Base_MAE,
Skill_Score_MSE, Skill_Score_MAE,
Delta_Helped_Rate, Delta_Helped_Rate_Top10Pct,
Top10Pct_Residual_MAE,
Regime_Active_Pct, Regime_Days_Mean, Regime_Docs_Mean,
Counterfactual_Blank_Active_MAE, Counterfactual_Blank_Inactive_MAE, Counterfactual_Permuted_Active_MAE,
Inactive_Blank_Gap_Pct,
Lambda_Base_Mean, Shape_Gain_Mean, Spike_Bias_Mean,
Relevance_Mass_Mean,
Spike_Gate_Hit_Rate, Spike_Target_Hit_Rate
```

每行对应一次（数据集 × horizon × seed）实验。

---

## 13. 端到端数据流总览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  原始数据                                                                    │
│  ┌───────────────────────┐      ┌───────────────────────┐                    │
│  │ CSV: (date, value)    │      │ JSONL: (title, content│                    │
│  │ 30 分钟粒度           │      │         date, url)    │                    │
│  └──────────┬────────────┘      └──────────┬────────────┘                    │
│             │ 节点 1                        │ 节点 4（LLM 精炼）              │
│             ▼                              ▼                                 │
│  ┌───────────────────────┐      ┌───────────────────────┐                    │
│  │ batch dict            │      │ refined.jsonl         │                    │
│  │ history_value (B,L)   │      │ + topic_tags          │                    │
│  │ target_value (B,H)    │      │ + regime_vec (5)      │                    │
│  └──────────┬────────────┘      │ + horizon_days        │                    │
│             │ 节点 2（z 归一化） └──────────┬────────────┘                    │
│             ▼                              │ 节点 5（按日聚合）               │
│  ┌───────────────────────┐                 ▼                                 │
│  │ history_z (B,L)       │      ┌───────────────────────┐                    │
│  │ targets_z (B,H)       │      │ regime_bank.npz       │                    │
│  └──────────┬────────────┘      │ (N_days, 5+15+384+1)  │                    │
│             │ 节点 3（Base）     └──────────┬────────────┘                    │
│             ▼                              │ 节点 6（逐 batch 查询）          │
│  ┌───────────────────────┐                 ▼                                 │
│  │ base_pred_z (B,H)     │      ┌───────────────────────┐                    │
│  │ base_hidden (B,D_b)   │      │ regime_pack dict:     │                    │
│  └──────────┬────────────┘      │  regime_vec (B,5)     │                    │
│             │                   │  topic_mass (B,15)    │                    │
│             │                   │  text_emb (B,384)     │                    │
│             │                   │  relevance_mass (B,)  │                    │
│             │                   └──────────┬────────────┘                    │
│             │          ┌──────── ┘                                           │
│             └──────────┤                                                     │
│                        │ 节点 7（Delta V3 前向）                              │
│                        ▼                                                     │
│              ┌──────────────────────────────┐                                │
│              │ out dict:                    │                                │
│              │  pred_z (B, H)               │                                │
│              │  residual_hat (B, H)         │                                │
│              │  lambda_base, shape_gain,    │                                │
│              │  spike_bias, spike_gate ...  │                                │
│              └──────────┬───────────────────┘                                │
│                         │ 节点 8（目标分解）                                   │
│                         │ 节点 9（损失）                                       │
│                         ▼                                                    │
│              ┌──────────────────────────────┐                                │
│              │ loss tensor (标量)            │                                │
│              └──────────┬───────────────────┘                                │
│                         │ 节点 10（评估与指标）                                 │
│                         ▼                                                    │
│              ┌──────────────────────────────┐                                │
│              │ results/test_results.csv     │                                │
│              │ debug_*.csv, diag JSON       │                                │
│              └──────────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. 文件级产物清单

一次完整管道运行产生的所有文件，按阶段分组：

| 阶段 | 路径 | 格式 | 用途 |
|---|---|---|---|
| 前置 | `{train,val,test}_file` | CSV/Parquet | 原始时间序列（用户提供） |
| 前置 | `{news_path}.jsonl` | JSONL | 原始新闻语料（用户提供） |
| 节点 4 | `checkpoints/_shared_refine_cache/v4/refined_*.jsonl` | JSONL | LLM 精炼后的文章 |
| 节点 5 | `checkpoints/_shared_refine_cache/v4/regime_bank_*.npz` | NumPy npz | 按日聚合的 regime |
| 阶段 1 | `checkpoints/{taskName}/best_base_backbone/base_backbone.pt` | PyTorch | 冻结的 base 权重 |
| 阶段 1 | `checkpoints/{taskName}/best_base_backbone/meta.json` | JSON | Backbone 配置 + zstats |
| 阶段 3 | `checkpoints/{taskName}/best_delta_v3_{taskName}/delta_v3.pt` | PyTorch | Delta V3 权重 |
| 阶段 3 | `checkpoints/{taskName}/best_delta_v3_{taskName}/cfg.json` | JSON | Delta 配置快照 |
| 评估 | `results/test_results.csv` | CSV（追加） | 汇总测试指标 |
| 评估 | `checkpoints/{taskName}/test_delta_residual_debug_*.csv` | CSV | 逐样本旋钮/预测详情 |

---

## 15. 超参快速参考（Load vs Price）

两种变体共享大部分超参。关键差异：

| 设置 | Load | Price | 原因 |
|---|---:|---:|---|
| `schema.variant` | `load` | `price` | 选择损失函数 |
| 主损失 | z-space SmoothL1 | 原始空间 winsorized SmoothL1 + pinball | Price 有 σ=635 重尾 |
| `spike_k` | 3.0 | 4.0 | Price spike 需更严阈值 |
| `spike_target_pct` | 0.10 | 0.06 | 训练集中 price spike 更少 |
| `inactive_residual_weight` | 0.1 | 0.0 | Price 无需额外静音 |
| `spike_bias_cap` | 0.75 | 0.30（部分配置更严） | 限制新闻对 price spike logit 的影响 |
| Base backbone | mlp | dlinear | 经验选择 |
| lr | 1e-4 | 5e-6 | Price 更敏感 |

其余参数（`λ_min`、`λ_ts_cap`、`λ_news_cap`、`λ_max`、`shape_gain_cap` 等）在两种变体间统一。

---

## 16. 设计不变量（框架保证什么）

1. **新闻仅通过乘性耦合**：新闻永远不会加入自由残差项，只能调制 `λ_base`、`shape_gain`、`spike_gate_logit`。最坏情况的伤害有界：shape 拉伸 `±30%` + 有界 gate 上的 logit 偏移。

2. **优雅退化**：当 `relevance_mass ≤ 0.7` 时，三个旋钮全部收缩到近似恒等，`λ_base → 0.05`。输出退化为 `base_pred + ~5% × TS_residual`。

3. **Spike 概率 ≠ 幅度**：新闻只能通过 `spike_bias` 调整 gate，从而升降 `P(spike)`；spike 幅度 `spike_ts` 仅看时间序列，阻止了"凭空幻觉出多少美元"。

4. **标签来源纪律**：残差子目标（slow/shape/spike）从时序本身派生（通过日历基线与分位数阈值），而非 LLM 输出字段。LLM 输出只调制 modulation，从不充当回归目标。

5. **可复现性**：`global_zstats` 烘焙在 base checkpoint meta 中；delta 重载时使用保存的统计量，而非重新计算，因此 base 与 delta 两阶段的 z-space 字节级一致。
