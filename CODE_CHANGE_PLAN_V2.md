# 代码改动计划 V2

基于 IMPROVEMENT_SUGGESTIONS.md 中的建议，按用户选定范围（A3, B全部, C全部, D1）制定的详细代码改动计划。

---

## 前置说明：关于非 30 分钟粒度的新闻

当前框架的时间对齐逻辑（`_build_temporal_text_series_for_sample`）是 **按 timestamp 比较** 的——对每个历史步 `step_ts`，选取 `doc_ts <= step_ts` 的新闻。这个逻辑本身**不依赖新闻的发布频率**，理论上支持任意粒度的新闻输入。

但以下几处需要注意适配：

1. **`news_window_days`**（当前默认 1 天）：如果新闻是周报或更低频，window 需要放大，否则很多样本匹配不到新闻
2. **TemporalTextTower 的时间冗余问题**：如果新闻是日频，48 个半小时步都对齐到同一批新闻（当前已是如此）。如果新闻是小时频或更高频，则每个步可能对齐到不同新闻，此时 B1 的去重编码策略需要动态判断重复
3. **`per_step_topk`**（当前默认 10）：高频新闻下每步匹配到的候选可能很多，topk 需要重新调优
4. **时间衰减权重（B1 新增）**：衰减常数 `tau` 应与新闻频率匹配——日频新闻用 24h tau，小时频用 2-4h tau

**以下计划中，涉及时间粒度敏感的地方会用 `[粒度适配]` 标签标出。**

---

## A3. 利用 Refined Cache 中已有的结构化标注

### 现状分析

框架**已经有**完整的结构化特征管道：

```
refined cache (per-doc)
  → structured_events {relevance, direction, strength, persistence, confidence, event_type}
    → _structured_events_to_feature_vec() → 12 维向量 [5 数值 + 7 类型 one-hot]
      → model._build_structured_pack() → structured_proj(12 → hidden) → structured_summary
        → 进入 delta_fuse / news_context / route_scalars
```

**但存在以下改进空间**：

### A3-1. 特征向量维度不足，信息丢失

**问题**：当前 `_structured_events_to_feature_vec` 将多篇新闻的结构化事件 **合并** 为单个 dict 后再提取特征。合并逻辑（`merge_structured_events`）取加权平均，丢失了以下信息：
- **多篇新闻的方向一致性**：3 篇都说 "up" vs 1 篇 "up" + 2 篇 "down" 有很不同的含义
- **事件计数**：匹配到 3 篇有事件的新闻 vs 1 篇，前者的信号更强
- **最强事件的 strength**：平均后稀释了最重要事件的信号

**改动位置**：`src/refine/cache.py :: _structured_events_to_feature_vec()`

**改动内容**：扩展特征维度（从 12 → 20 或可配置），在现有 5+7 的基础上追加：
```python
# 新增特征（索引 12-19，如果 dim 允许）：
# [12] direction_agreement:  所有 doc 的 direction 一致性（-1 到 1），全一致=±1，矛盾=0
# [13] event_count:          有事件的 doc 数量 / total docs（归一化到 0-1）
# [14] max_strength:         所有 doc 中最大的 strength
# [15] max_confidence:       所有 doc 中最大的 confidence
# [16] avg_relevance:        所有 doc 的 relevance 均值
# [17] strength_variance:    所有 doc 的 strength 方差（信号一致性指标）
# [18] has_outage_or_weather: 二值，是否包含 outage/weather 类型事件（对 PRICE 关键）
# [19] recency_weight:       最近一篇有事件新闻的时间衰减权重
```

**改动方式**：
- 修改 `_structured_events_to_feature_vec()` 的输入签名，额外接收 `doc_events: list[dict]`（当前已在 `structured_doc_events_list` 中传递）
- 扩展 `--delta_structured_feature_dim` 默认值从 12 → 20

**上游改动**：`src/base/common.py` line 1130 处调用 `_structured_events_to_feature_vec` 时，需要将 `structured_doc_events` 也传入：
```python
# 当前：
_structured_events_to_feature_vec(
    structured_events,
    dim=int(max(1, getattr(args, "delta_structured_feature_dim", 12))),
)
# 改为：
_structured_events_to_feature_vec(
    structured_events,
    dim=int(max(1, getattr(args, "delta_structured_feature_dim", 20))),
    doc_events=structured_doc_events,
)
```

**模型侧自动适配**：`model2.py` 的 `structured_feat_dim` 由 `--delta_structured_feature_dim` 控制，`_build_structured_pack` 和 `structured_proj` 自动适应新维度，**无需改模型代码**，仅需在脚本中更新参数。

### A3-2. 脚本参数更新

**文件**：`scripts/run_tinynews_experiment.sh`

```bash
# 当前
DELTA_STRUCTURED_FEATURE_DIM="${DELTA_STRUCTURED_FEATURE_DIM:-12}"
# 改为
DELTA_STRUCTURED_FEATURE_DIM="${DELTA_STRUCTURED_FEATURE_DIM:-20}"
```

### A3 涉及文件

| 文件 | 改动 |
|------|------|
| `src/refine/cache.py` | `_structured_events_to_feature_vec()` 扩展输入和维度 |
| `src/base/common.py` | 调用处传入 `doc_events` |
| `scripts/run_tinynews_experiment.sh` | `DELTA_STRUCTURED_FEATURE_DIM` 默认值更新 |

---

## B1. 改进 TemporalTextTower 的时间对齐策略

### 当前问题

`_build_temporal_text_series_for_sample()` 对 48 个历史步逐步匹配新闻。日频新闻下，48 步大多匹配到相同的 topk 篇文章 → encoder 重复编码相同文本 ~48 次。且不区分"新闻刚发布 10 分钟"和"新闻发布了 23 小时"。

### 改动方案：去重编码 + 时间衰减映射

**核心思想**：
1. 对每个样本，先提取唯一的新闻文本列表（去重）
2. 对唯一文本各编码一次 → 得到 embedding 字典
3. 对 48 个历史步，用 timestamp 比较找到匹配的新闻，用 **时间衰减权重** 加权已编码的 embedding

#### 改动 1：`src/base/common.py :: _build_temporal_text_series_for_sample()`

**当前签名与输出**：
```python
def _build_temporal_text_series_for_sample(
    *, history_times, news_metas, news_docs, tokenizer, max_tokens, per_step_topk
) -> list[str]  # 每步一个拼接后的文本字符串
```

**新增返回（或改为新函数）**：在原有的文本字符串列表基础上，额外返回时间衰减信息：
```python
# 新增返回：
# temporal_text_weight: list[float]  # 每步的时间衰减权重（0-1），无新闻=0，刚发布=1
# temporal_text_unique_map: dict     # {doc_text_hash: doc_text} 去重映射
```

**建议实现方式**：保留原函数不动（避免影响其他调用方），新增一个 `_build_temporal_text_series_with_decay()` 函数：

```python
def _build_temporal_text_series_with_decay(
    *,
    history_times: list[str],
    news_metas: list[dict],
    news_docs: list[str],
    tokenizer,
    max_tokens: int,
    per_step_topk: int,
    decay_tau_hours: float = 24.0,  # [粒度适配] 衰减半衰期
) -> tuple[list[str], list[float]]:
    """
    Returns:
      texts: list[str]   — 每步的新闻文本（同原函数）
      weights: list[float] — 每步的时间衰减权重，范围 [0, 1]
    """
    # 实现：
    # 1. 复用原有的 doc_ts <= step_ts 过滤逻辑
    # 2. 对匹配到的新闻，计算 exp(-(step_ts - doc_ts) / tau)
    # 3. 返回加权后的 weight（最近新闻的衰减值）
```

#### 改动 2：`src/temporal_text.py :: TemporalTextTower`

**新增方法**：`encode_unique_then_map()`

```python
def encode_unique_then_map(
    self,
    unique_text_ids: torch.Tensor,      # (N_unique, tok_len)
    unique_text_attn: torch.Tensor,     # (N_unique, tok_len)
    step_to_unique_idx: torch.Tensor,   # (B, hist_len) 每步指向 unique 中的 index，-1 表示无新闻
    step_decay_weight: torch.Tensor,    # (B, hist_len) 时间衰减权重
    *,
    device, dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1. 对 unique texts 编码一次 → (N_unique, step_dim)
    2. 通过 step_to_unique_idx 查表 → (B, hist_len, step_dim)
    3. 乘以 step_decay_weight → 带时间感知的 step_feat
    """
```

**保留原 `encode_step_series`**：向后兼容，当未提供 unique 映射时走原有路径。

#### 改动 3：`src/base/common.py :: _tokenize_temporal_text_series()`

**新增配套函数**：`_tokenize_temporal_text_unique()`

```python
def _tokenize_temporal_text_unique(
    temporal_text_series: list[list[str]],
    *,
    tokenizer,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对 batch 内所有唯一文本做一次 tokenize。
    Returns:
      unique_ids:         (total_unique, max_len)
      unique_attn:        (total_unique, max_len)
      step_to_unique_idx: (B, hist_len) 每步→unique 的索引，-1=无文本
      step_mask:          (B, hist_len) 与原有 step_mask 兼容
    """
    # 实现：
    # 1. 收集所有非空文本，去重
    # 2. tokenize 唯一文本列表
    # 3. 构建 step→unique 映射 tensor
```

#### 改动 4：`src/model2.py :: forward()`

在 `head_mode == "delta"` 分支，`_encode_temporal_text_features()` 内部判断：如果 batch 提供了 unique 映射信息，走 `encode_unique_then_map` 路径；否则走原有 `encode_step_series` 路径。

#### `[粒度适配]`

- `decay_tau_hours` 参数新增到 CLI：`--delta_temporal_text_decay_tau_hours`，默认 24.0
- 日频新闻：tau=24 合适
- 小时频新闻：建议 tau=4-8
- 实时新闻（分钟级）：建议 tau=1-2
- 周报：建议 tau=168（7天）

### B1 涉及文件

| 文件 | 改动 |
|------|------|
| `src/base/common.py` | 新增 `_build_temporal_text_series_with_decay()`，新增 `_tokenize_temporal_text_unique()` |
| `src/temporal_text.py` | 新增 `encode_unique_then_map()` 方法 |
| `src/model2.py` | `_encode_temporal_text_features()` 新增路径分支 |
| `run.py` | 新增 `--delta_temporal_text_decay_tau_hours` 参数 |
| `scripts/run_tinynews_experiment.sh` | 新增 `DELTA_TEMPORAL_TEXT_DECAY_TAU_HOURS` |

---

## B2. 新闻 Summary 级别的直接残差预测（Text Shortcut Head）

### 当前问题

文本信号从 TemporalTextTower → gated_add → pool → delta_fuse → unified_trunk → heads，经过 6+ 层变换，梯度弱化严重。

### 改动方案

#### 改动 1：`src/model2.py :: __init__()`

在 `__init__` 中新增一个 text shortcut head：

```python
# 在现有 delta head 初始化区域（约 line 165-180）之后
self.text_shortcut_head = nn.Linear(self.hidden_size, self.horizon)
nn.init.zeros_(self.text_shortcut_head.weight)
nn.init.zeros_(self.text_shortcut_head.bias)
self.text_shortcut_alpha = nn.Parameter(torch.zeros(1))  # 初始化为 0，训练中逐渐学习
```

#### 改动 2：`src/model2.py :: forward()`

在 `head_mode == "delta"` 的最终预测组装处（约 line 675）：

```python
# 当前（unified arch）：
pred = confidence * direction_score * magnitude

# 改为：
pred = confidence * direction_score * magnitude
# text shortcut：直接从 text_summary 预测残差偏移
if temporal_text_summary is not None and temporal_text_strength is not None:
    text_shortcut = self.text_shortcut_head(temporal_text_summary)
    text_shortcut = text_shortcut * temporal_text_strength  # 无新闻时自动关闭
    pred = pred + self.text_shortcut_alpha * text_shortcut
```

**关键设计**：
- `text_shortcut_alpha` 初始化为 0 → 开始时不影响训练，逐渐学习打开
- 乘以 `temporal_text_strength` → 无新闻的样本自动跳过
- 这条路径只经过 1 层 linear，梯度直接传到 text_summary，显著加强文本学习信号

#### 改动 3：其他 `residual_arch` 路径

对 `residual_arch == "current"` 路径（line 682-698），同样在最终 pred 上追加 text shortcut：
```python
# line 698 之后（delta_sign_mode == "none" 路径）或类似位置
# pred = ... (原有计算)
if temporal_text_summary is not None and temporal_text_strength is not None:
    text_shortcut = self.text_shortcut_head(temporal_text_summary)
    pred = pred + self.text_shortcut_alpha * (text_shortcut * temporal_text_strength)
```

#### 改动 4：checkpoint 兼容

`save_checkpoint()` 和 `load_checkpoint()` 无需特殊处理——PyTorch `state_dict` 自动包含新参数。加载旧 checkpoint 时新参数保持初始化值（`strict=False` 如果需要）。

### B2 涉及文件

| 文件 | 改动 |
|------|------|
| `src/model2.py` | `__init__` 新增 head + alpha；`forward` 各路径追加 shortcut |

---

## B3. 文本编码器统一为 DistilBERT

### 现状

PRICE 配置用 GPT-2，LOAD 用 DistilBERT。实验证明两者效果一致（PRICE 上 95.46 vs 95.46），GPT-2 参数多 40% 且慢。

### 改动方案

**仅改脚本**，不改代码。

**文件**：`scripts/run_tinynews_experiment.sh`

```bash
# nsw_price 区块中（约 line 497）
# 当前：
TEMPORAL_TEXT_MODEL_ID="${TEMPORAL_TEXT_MODEL_ID:-distilbert-base-uncased}"
# 此处实际通过 TASK_NAME_BASE 中的 "gpt2" 标识传入 GPT-2
# 确认脚本中哪里覆盖了此值
```

**注意**：检查脚本中是否有 profile 或 grid search 区域覆盖了 `TEMPORAL_TEXT_MODEL_ID` 为 GPT-2。确保 PRICE 配置统一使用 `distilbert-base-uncased`。

### B3 涉及文件

| 文件 | 改动 |
|------|------|
| `scripts/run_tinynews_experiment.sh` | PRICE 配置区 `TEMPORAL_TEXT_MODEL_ID` 确认/修改 |

---

## C1. 条件性残差修正（Conditional Residual Correction）

### 当前问题

所有样本用同一套 unified_trunk → direction/magnitude/confidence heads。PRICE 的 98% 正常值和 2% 极端尖峰需要不同修正策略，但当前模型只能用 confidence 来"缩放"——对极端值输出低 confidence，等于放弃修正。

### 改动方案：轻量级 Regime-Aware Head

在 `unified` 架构下引入轻量级的条件修正，**不需要引入完整的 RegimeRouter + ExpertMixture**（那套只在 `plan_c_mvp` 下使用且已证明效果一般）。

#### 改动 1：`src/unified_trunk.py :: UnifiedResidualTrunk`

新增一个 **regime embedding** 分支，在 trunk 输出后根据输入特征选择不同的"修正力度模式"：

```python
class UnifiedResidualTrunk(nn.Module):
    def __init__(self, *, hidden_size, horizon, history_len, dropout, num_regimes=3):
        # ... 现有代码 ...
        
        # 新增：regime classifier
        self.num_regimes = num_regimes  # 3: normal, spike, negative
        self.regime_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, num_regimes),
        )
        # 每个 regime 一个 magnitude bias 和 confidence bias
        self.regime_mag_bias = nn.Parameter(torch.zeros(num_regimes, horizon))
        self.regime_conf_bias = nn.Parameter(torch.zeros(num_regimes, horizon))
```

#### 改动 2：`src/unified_trunk.py :: forward()`

```python
def forward(self, *, residual_context, history_z, base_pred_z, text_summary, text_strength):
    # ... 现有 trunk 计算 ...
    hidden = self.trunk(fused + ctx)
    
    # 新增：regime soft routing
    regime_logits = self.regime_classifier(hidden.detach())  # detach 避免 regime 影响 trunk 学习
    regime_probs = torch.softmax(regime_logits, dim=-1)      # (B, num_regimes)
    
    # 原有 head 输出
    direction_logits = self.direction_head(hidden)
    magnitude_raw = self.magnitude_head(hidden)
    confidence_logits = self.confidence_head(hidden)
    
    # 新增：regime-aware bias
    mag_bias = torch.einsum('br,rh->bh', regime_probs, self.regime_mag_bias)
    conf_bias = torch.einsum('br,rh->bh', regime_probs, self.regime_conf_bias)
    magnitude_raw = magnitude_raw + mag_bias
    confidence_logits = confidence_logits + conf_bias
    
    # 原有后处理
    direction_score = torch.tanh(direction_logits)
    magnitude = F.softplus(magnitude_raw)
    confidence = torch.sigmoid(confidence_logits)
    
    return {
        # ... 原有 keys ...
        "regime_logits": regime_logits,
        "regime_probs": regime_probs,
    }
```

#### 改动 3：Regime 辅助损失

**文件**：`src/delta/core.py`

新增 regime classification 的辅助损失。由于没有显式标签，用**残差绝对值的分位数**作为伪标签：

```python
def _build_regime_pseudo_labels(delta_target: torch.Tensor, args) -> torch.Tensor:
    """
    根据残差绝对值划分 regime：
    - 0 (normal):   |residual| < p75
    - 1 (spike):    |residual| >= p75 且 residual > 0
    - 2 (negative): |residual| >= p75 且 residual < 0
    """
    abs_res = delta_target.abs().mean(dim=1)  # (B,)
    threshold = abs_res.quantile(0.75)
    mean_sign = delta_target.mean(dim=1)
    labels = torch.zeros_like(abs_res, dtype=torch.long)
    labels[(abs_res >= threshold) & (mean_sign > 0)] = 1
    labels[(abs_res >= threshold) & (mean_sign <= 0)] = 2
    return labels
```

**文件**：`src/delta/stage.py`

在 DELTA 训练循环中，将 regime 辅助损失加入总损失：

```python
# 在 loss 计算处（与现有 residual loss 合并）
if 'regime_logits' in model_output and model_output['regime_logits'] is not None:
    regime_labels = _build_regime_pseudo_labels(delta_target, args)
    regime_loss = F.cross_entropy(model_output['regime_logits'], regime_labels)
    loss = loss + 0.1 * regime_loss  # 权重可配置
```

#### 改动 4：`run.py` 新增参数

```python
parser.add_argument('--delta_num_regimes', type=int, default=3)
parser.add_argument('--delta_regime_loss_weight', type=float, default=0.1)
```

### C1 涉及文件

| 文件 | 改动 |
|------|------|
| `src/unified_trunk.py` | 新增 regime classifier, bias, forward 修改 |
| `src/delta/core.py` | 新增 `_build_regime_pseudo_labels()` |
| `src/delta/stage.py` | 训练循环中加入 regime 辅助损失 |
| `src/model2.py` | 将 `num_regimes` 参数传递给 `UnifiedResidualTrunk` |
| `run.py` | 新增 `--delta_num_regimes`, `--delta_regime_loss_weight` |

---

## C2. 残差目标的自适应缩放

### 当前问题

PRICE 的 `true_|res|=2.76`（z-score）但 `|delta|=0.05`，模型只修正了 1.8%。loss 函数对大残差样本的梯度被平均到小残差样本中。

### 改动方案

#### 改动 1：分位数加权损失

**文件**：`src/delta/core.py`

新增残差加权函数：

```python
def _build_residual_importance_weights(
    delta_target: torch.Tensor,
    *,
    top_pct: float = 0.10,
    top_weight: float = 3.0,
    normal_weight: float = 1.0,
) -> torch.Tensor:
    """
    对残差绝对值 top-10% 的样本给更高的 loss 权重。
    Returns: (B,) 权重 tensor
    """
    abs_res = delta_target.abs().mean(dim=1)  # (B,)
    threshold = abs_res.quantile(1.0 - top_pct)
    weights = torch.where(
        abs_res >= threshold,
        torch.full_like(abs_res, top_weight),
        torch.full_like(abs_res, normal_weight),
    )
    return weights / weights.mean()  # 归一化，保持 loss 量级不变
```

#### 改动 2：在训练循环中应用权重

**文件**：`src/delta/stage.py`

在 DELTA loss 计算处：

```python
# 当前（伪代码）：
# loss = residual_loss(pred, target)  # 均匀对待所有样本
# 改为：
importance_w = _build_residual_importance_weights(
    delta_target,
    top_pct=float(getattr(args, 'delta_residual_top_pct', 0.10)),
    top_weight=float(getattr(args, 'delta_residual_top_weight', 3.0)),
)
# loss = (per_sample_loss * importance_w).mean()
```

**具体位置**：需要找到 `src/delta/stage.py` 中计算 DELTA loss 的位置，将 per-sample loss 乘以 importance_w。

#### 改动 3：对数空间残差目标（可选）

**文件**：`src/delta/core.py :: _build_delta_targets()`

在 relative 模式下，对残差目标做对数压缩：

```python
# 当前（relative mode, line 164）：
delta_target = ((target_raw - base_raw) / scale_raw).detach()

# 新增选项 delta_target_transform=log_compress：
if target_transform == "log_compress":
    delta_target = torch.sign(delta_target) * torch.log1p(delta_target.abs())
```

这样 `|r|=10` 变成 `log(11)=2.4`，`|r|=100` 变成 `log(101)=4.6`，极端值被大幅压缩，模型更容易学习。

#### 改动 4：`run.py` 新增参数

```python
parser.add_argument('--delta_residual_top_pct', type=float, default=0.10,
                    help='top percentile of residuals to upweight')
parser.add_argument('--delta_residual_top_weight', type=float, default=3.0,
                    help='weight multiplier for top-percentile residuals')
parser.add_argument('--delta_target_transform', type=str, default='none',
                    choices=['none', 'log_compress'],
                    help='transform applied to delta targets')
```

### C2 涉及文件

| 文件 | 改动 |
|------|------|
| `src/delta/core.py` | 新增 `_build_residual_importance_weights()`，`_build_delta_targets()` 增加 log_compress |
| `src/delta/stage.py` | 训练循环中应用 importance weights |
| `run.py` | 新增 3 个参数 |

---

## C3. 文本融合时机前移

### 当前问题

文本在 patch 级别融合后被 pool 平均化，且在 unified_trunk 中只是与其他信号简单拼接后过 MLP。

### 改动方案：Text-TS Cross-Attention 在 Trunk 内部

#### 改动 1：`src/unified_trunk.py :: __init__()`

新增一个轻量级 cross-attention 层：

```python
class UnifiedResidualTrunk(nn.Module):
    def __init__(self, *, hidden_size, horizon, history_len, dropout, 
                 num_regimes=3, text_cross_attn_enable=False):
        # ... 现有代码 ...
        
        # 新增：text → history/base cross attention
        self.text_cross_attn_enable = text_cross_attn_enable
        if text_cross_attn_enable:
            self.text_history_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.text_attn_ln = nn.LayerNorm(hidden_size)
```

#### 改动 2：`src/unified_trunk.py :: forward()`

```python
def forward(self, *, residual_context, history_z, base_pred_z, text_summary, text_strength):
    # ... 现有 projection ...
    history_pack = self.history_proj(...)
    base_pack = self.base_proj(...)
    text_pack = self.text_proj(...)
    
    # 新增：让 text_pack 作为 query，attend 到 [history_pack, base_pack]
    if self.text_cross_attn_enable and text_strength is not None:
        # 构造 key/value：history 和 base 信息
        kv = torch.stack([history_pack, base_pack], dim=1)  # (B, 2, H)
        text_q = text_pack.unsqueeze(1)                      # (B, 1, H)
        attn_out, _ = self.text_history_attn(
            query=text_q, key=kv, value=kv, need_weights=False
        )
        # 用 text_strength 门控
        gate = text_strength.clamp(0, 1)
        text_pack = self.text_attn_ln(text_pack + gate * attn_out.squeeze(1))
    
    fused = self.input_fuse(torch.cat([ctx, history_pack, base_pack, text_pack], dim=-1))
    # ... 后续不变 ...
```

**设计理由**：
- text_summary 作为 query 去 attend history 和 base_pred 信息，产生**条件性的文本表示**："在这种历史模式 + base 预测下，新闻意味着什么"
- 用 `text_strength` 门控：无新闻时 attention 输出被抑制
- 只增加 1 层 MHA（4 heads），参数量很小

#### 改动 3：参数传递

**文件**：`src/model2.py :: __init__()`

```python
# 构建 UnifiedResidualTrunk 时传入新参数
if self.residual_arch == "unified":
    self.unified_trunk = UnifiedResidualTrunk(
        hidden_size=self.hidden_size,
        horizon=self.horizon,
        history_len=self.history_len,
        dropout=float(head_dropout),
        num_regimes=int(num_regimes),
        text_cross_attn_enable=bool(text_cross_attn_in_trunk),
    )
```

**文件**：`run.py`

```python
parser.add_argument('--delta_text_cross_attn_in_trunk', type=int, default=1,
                    choices=[0, 1], help='enable cross-attention in unified trunk')
```

### C3 涉及文件

| 文件 | 改动 |
|------|------|
| `src/unified_trunk.py` | `__init__` 新增 cross-attn；`forward` 新增 attention 路径 |
| `src/model2.py` | 传递新参数给 `UnifiedResidualTrunk` |
| `run.py` | 新增 `--delta_text_cross_attn_in_trunk` |

---

## D1. 新闻对比学习预训练

### 目标

在 DELTA 训练前，通过对比学习让 TemporalTextTower 的 `step_proj` 和 `patch_proj` 学到"新闻内容与时序模式的对应关系"。

### 改动方案

#### 改动 1：新增预训练模块

**新文件**：`src/temporal_text_pretrain.py`

```python
class TemporalTextContrastivePretrain:
    """
    对比学习预训练 TemporalTextTower 的 step_proj 和 patch_proj。
    
    正样本对：(news_embedding, ts_pattern) 来自同一时间窗口
    负样本对：(news_embedding, ts_pattern) 来自不同时间窗口
    
    Loss: InfoNCE
    """
    
    def __init__(self, temporal_text_tower, hidden_size, temperature=0.07):
        self.tower = temporal_text_tower  # 复用 DELTA 模型的 tower
        self.temperature = temperature
        # 时序模式编码器（轻量级，只用于预训练）
        self.ts_encoder = nn.Sequential(
            nn.Linear(history_len, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        # 投影头（预训练后丢弃）
        self.text_proj_head = nn.Linear(hidden_size, 128)
        self.ts_proj_head = nn.Linear(hidden_size, 128)
    
    def compute_loss(self, text_summary, ts_pattern):
        """
        text_summary: (B, hidden_size) — 来自 tower.summarize_patch_context
        ts_pattern:   (B, history_len) — 历史时序 z-score
        
        InfoNCE loss: 同一样本的 text/ts 对应为正，batch 内其他为负
        """
        text_emb = F.normalize(self.text_proj_head(text_summary), dim=-1)
        ts_emb = F.normalize(self.ts_proj_head(self.ts_encoder(ts_pattern)), dim=-1)
        
        logits = text_emb @ ts_emb.T / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)
```

#### 改动 2：预训练调度

**文件**：`src/delta/stage.py :: train_delta_stage()`

在 DELTA 训练循环之前插入预训练阶段：

```python
def train_delta_stage(args, bundle, best_base_path, best_base_metric):
    # ... 现有初始化 ...
    
    # 新增：对比学习预训练
    pretrain_epochs = int(getattr(args, 'delta_text_pretrain_epochs', 0))
    if pretrain_epochs > 0 and delta_model.temporal_text_enable:
        _pretrain_temporal_text_contrastive(
            delta_model=delta_model,
            train_loader=train_loader,
            base_backbone=base_backbone,
            args=args,
            device=device,
            epochs=pretrain_epochs,
            global_zstats=global_zstats_bundle,
            news_df=news_df,
            # ... 其他必要参数 ...
        )
    
    # ... 原有 DELTA 训练循环 ...
```

#### 改动 3：预训练的数据构建

预训练需要 (文本, 时序模式) 对。可以复用现有的 `build_delta_batch_inputs` 和 base_backbone：

```python
def _pretrain_temporal_text_contrastive(
    delta_model, train_loader, base_backbone, args, device, epochs, 
    global_zstats, news_df, ...
):
    """
    1. 遍历 train_loader
    2. 对每个 batch：
       - 用 base_backbone 得到 base_pred_z
       - 用 build_delta_batch_inputs 得到 temporal_text_ids/attn
       - 通过 tower.encode_step_series → summarize_patch_context 得到 text_summary
       - history_z 作为时序模式
    3. 计算 InfoNCE loss，只更新 step_proj 和 patch_proj 的参数
    """
    # 冻结 encoder（已经冻结），只训练 step_proj + patch_proj
    pretrain_params = list(delta_model.temporal_text_tower.step_proj.parameters()) + \
                      list(delta_model.temporal_text_tower.patch_proj.parameters())
    optimizer = AdamW(pretrain_params, lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(epochs):
        for batch in train_loader:
            # ... 构建文本和时序输入 ...
            loss = contrastive_module.compute_loss(text_summary, history_z)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

#### 改动 4：`run.py` 新增参数

```python
parser.add_argument('--delta_text_pretrain_epochs', type=int, default=0,
                    help='contrastive pretrain epochs for TemporalTextTower (0=skip)')
parser.add_argument('--delta_text_pretrain_lr', type=float, default=1e-3)
parser.add_argument('--delta_text_pretrain_temperature', type=float, default=0.07)
```

#### `[粒度适配]`

对比学习的正/负样本构造依赖"相同/不同时间窗口"的定义。对于低频新闻（周报），batch 内多个样本可能匹配到同一篇新闻 → 正样本对减少。建议：
- 如果新闻频率低，增大 batch_size 以保证 batch 内有足够不同的新闻
- 或在 InfoNCE 中用 hard negative mining，专门选择"新闻相似但时序模式不同"的负样本

### D1 涉及文件

| 文件 | 改动 |
|------|------|
| `src/temporal_text_pretrain.py` | **新文件**：对比学习模块 |
| `src/delta/stage.py` | `train_delta_stage()` 中插入预训练阶段 |
| `run.py` | 新增 3 个参数 |
| `scripts/run_tinynews_experiment.sh` | 新增对应变量 |

---

## 总览：全部改动文件清单

| 文件 | 涉及建议 | 改动类型 |
|------|---------|---------|
| `src/refine/cache.py` | A3 | 扩展特征向量 |
| `src/base/common.py` | A3, B1 | 特征调用 + 去重编码函数 |
| `src/temporal_text.py` | B1 | 新增 `encode_unique_then_map` |
| `src/model2.py` | B1, B2, C1, C3 | 编码路径 + shortcut head + 参数传递 |
| `src/unified_trunk.py` | C1, C3 | regime classifier + cross-attention |
| `src/delta/core.py` | C1, C2 | regime 伪标签 + importance weights + log_compress |
| `src/delta/stage.py` | C1, C2, D1 | 训练循环中 regime loss + importance weights + 预训练调度 |
| `src/temporal_text_pretrain.py` | D1 | **新文件** |
| `run.py` | A3, B1, C1, C2, C3, D1 | 新增 ~10 个 CLI 参数 |
| `scripts/run_tinynews_experiment.sh` | A3, B1, B3, C1, C2, D1 | 新增/修改配置变量 |

---

## 建议实施顺序

```
第 1 轮（基础改进，低风险）：
  B2 (text shortcut head)      — 改动最小，立即测试文本信号强度
  B3 (统一 DistilBERT)         — 仅改脚本
  A3 (扩展 structured features) — 利用已有缓存数据

第 2 轮（核心架构）：
  C2 (残差自适应缩放)           — 对 PRICE 影响最大
  C3 (文本融合前移)             — 与 B2 互补

第 3 轮（时间对齐优化）：
  B1 (去重编码 + 时间衰减)      — 改动较大但效率+质量双提升
  C1 (条件性残差修正)           — 依赖 C2 的残差缩放

第 4 轮（预训练）：
  D1 (对比学习预训练)           — 依赖 B1 完成后的 tower 结构
```

每轮完成后跑 LOAD + PRICE 对比实验，确认改进方向正确后再进入下一轮。
