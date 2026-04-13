# 框架分析与思考笔记

---

## 问题一：为什么 Load 数据集 delta 有效，而 Price 数据集 help rate > 50% 却 final MAE 反而比 base 差？

### 1.1 实验数据回顾

| 数据集 | Delta MAE | Base MAE | Skill Score | Help Rate | Help Rate Top10% |
|---|---:|---:|---:|---:|---:|
| NSW Load h48 | 482.17 | 490.66 | **+1.73%** | 54.5% | 80.1% |
| NSW Load h192 | 644.60 | 654.92 | **+1.57%** | 61.1% | 59.0% |
| NSW Load h720 | 716.93 | 727.86 | **+1.50%** | 60.1% | 45.0% |
| NSW Price h48 (watt_free) | 93.32 | 92.71 | **-0.66%** | 53.5% | 66.0% |
| NSW Price h48 (news_2024_2025) | 93.69 | 92.71 | **-1.05%** | 52.8% | 64.3% |

**核心矛盾：Price 的 help rate 过 50%（即超过一半的样本 delta 有帮助），但总体 MAE 反而比 base 更差。**

### 1.2 根因分析：误差分布的非对称性（"帮了多数，害了少数但害得很重"）

这是一个典型的 **"help rate 与 MAE 脱钩"** 现象，其本质在于：

**Help rate 是样本计数指标（多少个样本变好了），MAE 是误差幅度指标（平均每个样本差多少）。**

对于 Price 数据：
- Delta 在 53.5% 的样本上减小了误差（help），但每个样本平均只减小了一点点（比如减少 $2-5）
- Delta 在 46.5% 的样本上增大了误差（hurt），但其中少数极端样本被增大了很多（比如增加 $50-200）
- 因为 price 的分布有极长的右尾（σ = 635），少量"害了"的极端样本就足以拉高整体 MAE

对于 Load 数据：
- Load 分布近似对称、无极端尾部（σ 相对 μ 较小）
- Delta 帮助的样本和伤害的样本，误差变化的幅度大致对称
- 因此 help rate > 50% 就能直接转化为 MAE 改善

**用数学语言说：**
```
MAE_delta - MAE_base = E[|err_delta|] - E[|err_base|]
                     = E[|err_delta| - |err_base| | helped] * P(helped) 
                       + E[|err_delta| - |err_base| | hurt] * P(hurt)
```
- Load: 两项幅度接近 → P(helped) > 0.5 就能赢
- Price: hurt 项的条件期望远大于 helped 项 → 即使 P(helped) = 0.535，hurt 的少数大误差仍然主导

### 1.3 更深层的结构性原因

#### (a) Price 的残差结构不适合 slow + shape + spike 分解

Load 的残差（base_pred - true）有很强的日历结构：
- **slow（level shift）**：日间均值偏移，变化平缓，容易学
- **shape（日内形状）**：工作日/周末有非常稳定的用电模式（早峰、晚峰），calendar baseline 能解释大部分变异
- **spike**：极端负荷事件（热浪等），频率低但新闻信号明确

Price 的残差则完全不同：
- **slow**：价格的 level shift 由供需边际成本决定，变化剧烈且不可预测
- **shape**：价格的日内模式远不如负荷稳定——同一个周三，可能因为一次机组跳闸导致下午价格飙升 10 倍，也可能因为风力充足导致价格接近零
- **spike**：价格 spike 的幅度可达 $5,000-$15,000/MWh（而正常价格约 $50-100），这些极端值的 z-space 表示会产生巨大梯度

**结论：Load 的残差"形状好"（平滑、有规律），delta 的三头分解能有效建模；Price 的残差"形状差"（尖锐、随机），三头分解反而引入了不必要的结构约束。**

#### (b) Lambda_base 的安全上限在 price 上过于激进

从实验数据看：
- Load: `lambda_base_mean = 0.34` → delta 对预测的修正幅度约 34%
- Price: `lambda_base_mean = 0.24` → 修正幅度约 24%

尽管 price 的 lambda 已经更低，但问题是：对于 σ=635 的 price 数据，即使 24% 的修正比例，在 z-space 中微小的残差预测误差，反归一化后就变成了巨大的 raw-space 误差。具体来说：

```
假设某个样本的 z-space 残差预测偏差为 0.1
raw-space 误差 = 0.1 × 635 × 0.24 = 15.24 (美元)
```

这在 base MAE 约 $92 的尺度下是显著的。而 Load 的 σ 可能只有几百 MW，同样的 z-space 偏差造成的影响小得多。

#### (c) Spike 机制的不对称收益

Load spike：
- 频率约 8.3%（spike_target_hit_rate），delta 的 spike gate 命中率约 4.2%
- Spike 的幅度有限（MW），即使 gate 误报，造成的额外误差可控
- 但当 spike 命中时（热浪预警等），收益显著 → top10% help rate = 80%

Price spike：
- Spike 频率类似约 3.5%，gate 命中率约 2.8%
- Spike 的幅度极大（$），gate 误报一次的代价远超多次命中的收益
- 数据显示 `spike_bias_mean = -0.155`（price） vs `-0.308`（load），说明 price 模型的 spike bias 更保守，但仍不够保守

#### (d) 新闻信号对 Load vs Price 的因果强度不同

这是最根本的原因：**新闻对电力负荷的预测力远强于对电力价格的预测力。**

- **负荷 ← 天气 + 日历**：热浪新闻 → 空调负荷上升，这是近乎确定性的因果关系。新闻能可靠地告诉你"未来几天是高负荷 regime"
- **价格 ← 负荷 × 供给侧不确定性**：即使新闻正确预测了高负荷，价格还取决于（机组是否跳闸、可再生能源实际出力、竞价策略、市场规则）等大量额外因素。新闻只能提供模糊的"波动性可能增大"信号

在框架中，这体现为：
- Load 的 regime bank 能有效区分"高负荷 regime"和"正常 regime"，modulation heads 有可学习的信号
- Price 的 regime bank 即使捕捉了 tightness/demand_outlook 等信号，这些信号到 price 的映射是高度非线性且噪声极大的

### 1.4 小结

| 因素 | Load（有效） | Price（无效） |
|---|---|---|
| 误差分布 | 近似对称，无极端尾部 | 重尾分布，少数极端误差主导 MAE |
| 残差结构 | 平滑、有日历规律，三头分解匹配 | 尖锐、随机，三头分解过度约束 |
| z-space 放大 | σ 小，z-space 偏差影响可控 | σ=635，微小偏差放大为巨大 raw 误差 |
| 新闻→目标因果性 | 强（天气→负荷直接因果） | 弱（新闻→负荷→供需→价格，链条长且噪声大） |
| Spike 机制 | 误报代价小，命中收益大 | 误报代价极大，命中收益不对称 |

**一句话总结：Price 数据的核心问题是"重尾分布 + 弱因果信号"的组合——delta 在多数平静样本上微弱地帮忙，但在少数极端样本上因为信号不足而造成了不成比例的伤害。**

---

## 问题二：能否为 Delta 量身定制一个 Base 模型？

### 2.1 当前 Base 的局限

当前 base（MLP 或 DLinear）是"task-agnostic"的纯时序预测器：
- **MLP**: `history → FC → GELU → FC → GELU → head → prediction`，完全忽略时序结构
- **DLinear**: 分解为 seasonal + trend 后分别线性映射，保留了一定的时序结构但表达能力有限

这些 base 有一个关键问题：**它们对 delta 来说是黑盒。Delta 只能从 `base_pred_z` 和 `base_hidden` 中猜测 base 在哪里犯了错，但无法理解 base 为什么犯错。**

### 2.2 头脑风暴：Delta-Aware Base 设计

#### 思路 A：Uncertainty-Aware Base（不确定性感知型 base）

**核心想法：让 base 不仅输出预测值，还输出一个"我不确定"的信号，直接告诉 delta 在哪里需要帮忙。**

```
BaseWithUncertainty:
  Input:  history_z [B, L]
  Output: base_pred_z [B, H],  uncertainty_z [B, H],  hidden [B, D]
  
  # 内部结构
  shared_encoder = MLP/DLinear(history_z)  → hidden [B, D]
  pred_head      = Linear(hidden)          → base_pred_z [B, H]
  sigma_head     = Linear(hidden) + Softplus → uncertainty_z [B, H]
  
  # 训练：用 Gaussian NLL loss 而不是 SmoothL1
  loss = 0.5 * (log(sigma^2) + (y - pred)^2 / sigma^2)
```

**Delta 如何利用：**
- `uncertainty_z` 直接告诉 delta "这几个时间步 base 很不确定"
- Delta 的 `lambda_base` 不再是全局标量，而是可以按 step 调整：uncertainty 高的位置 lambda 大，低的位置 lambda 小
- 这相当于给 delta 一个自动的"注意力地图"，知道在哪里用力

**对 Price 数据的特殊好处：** Price 的 base 应该在 spike 期间输出高 uncertainty，这就自动把 delta 的注意力引导到了 spike 附近——而不是让 delta 浪费建模能力在 base 已经做得不错的平静时段。

#### 思路 B：Residual-Structured Base（残差结构对齐型 base）

**核心想法：让 base 的内部结构与 delta 的三头分解（slow/shape/spike）对齐，这样 delta 看到的残差天然就是"已经被结构化分解过的"。**

```
StructuredBase:
  Input:  history_z [B, L], calendar_features [B, L, C]
  
  # 显式分解
  trend_module   = MovingAvg + Linear(L→H)     → trend_pred [B, H]  (对应 slow)
  calendar_module = DOW×HOD lookup + Linear     → calendar_pred [B, H]  (对应 shape)
  residual_module = MLP(history_z - trend - cal) → spike_pred [B, H]  (对应 spike)
  
  base_pred_z = trend_pred + calendar_pred + spike_pred
  
  # 额外输出给 delta 的信息
  base_components = {
      "trend": trend_pred,      # delta 的 slow_head 知道 base 的 trend 预测
      "calendar": calendar_pred, # delta 的 shape_head 知道 base 的 calendar 预测  
      "residual": spike_pred,   # delta 的 spike_head 知道 base 试图捕捉什么
  }
```

**好处：**
- Delta 的三头不再"盲猜" base 的残差来自哪个成分
- 当 base 的 trend 预测准确但 calendar 偏了，shape_head 可以精准修正；反之亦然
- 避免了三个 head 互相干扰（slow_head 试图修正 spike 造成的误差等错误分配）

**对 Price 的特殊好处：** Price 数据上 base 的 spike 成分预测几乎必然很差（因为 spike 不可预测），但 trend 和 calendar 成分可能还行。分解后 delta 可以"放弃"修正 spike 部分（因为无法可靠预测），而专注于 trend 和 shape 的微调。

#### 思路 C：Adaptive Complexity Base（自适应复杂度型 base）

**核心想法：Base 不是固定架构，而是根据 local 数据特征自动调整复杂度。在"简单"时段用极简模型（给 delta 留出空间），在"复杂"时段用更强模型（减少 delta 的负担）。**

```
AdaptiveBase:
  Input:  history_z [B, L]
  
  # 两条路径
  simple_path = Linear(L→H)                    → pred_simple [B, H]
  complex_path = Transformer(history_z, 2层)     → pred_complex [B, H]
  
  # 门控（基于历史数据的波动性）
  complexity_signal = history_z.std(dim=-1)      # 或用更精细的特征
  gate = sigmoid(Linear(complexity_signal))       # [B, 1]
  
  base_pred_z = gate * pred_complex + (1-gate) * pred_simple
  
  # 输出 gate 给 delta
  base_complexity_gate = gate  # delta 知道 base 用了多少"力"
```

**逻辑：**
- 波动小的时段：simple_path 主导，base 输出平滑预测，delta 的 shape_head 有空间微调
- 波动大的时段：complex_path 主导，base 尽力捕捉模式，delta 只做边际修正
- Gate 信号传递给 delta，delta 可以据此调整自己的信任度

#### 思路 D：Multi-Horizon Decomposed Base（多尺度分解型 base）

**核心想法：利用 delta 需要预测不同 horizon（48/96/192/336/720）的特点，让 base 在不同时间尺度上有专门的子模块。**

```
MultiScaleBase:
  Input:  history_z [B, L]
  
  # 多尺度分解
  fast_branch  = Conv1D(kernel=4) + Linear    → captures 2-hour patterns
  medium_branch = Conv1D(kernel=24) + Linear   → captures 12-hour patterns  
  slow_branch  = Conv1D(kernel=48) + Linear    → captures 24-hour patterns
  
  # 按 horizon 加权
  if horizon <= 48:   weights = [0.5, 0.3, 0.2]
  elif horizon <= 192: weights = [0.2, 0.5, 0.3]
  else:                weights = [0.1, 0.3, 0.6]
  
  base_pred_z = sum(w * branch for w, branch in zip(weights, branches))
  base_scale_features = stack([fast, medium, slow])  # 给 delta 多尺度信息
```

### 2.3 推荐方案：Uncertainty-Aware Base（思路 A）+ Residual Structure Hints（思路 B 的简化版）

综合可行性和收益，推荐组合方案：

```python
class DeltaAwareBase(nn.Module):
    def __init__(self, history_len, horizon, hidden_dim=256):
        super().__init__()
        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # 预测头
        self.pred_head = nn.Linear(hidden_dim, horizon)
        # 不确定性头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, horizon),
            nn.Softplus(),
        )
        # 日历 hint（供 delta 使用）
        self.calendar_head = nn.Linear(hidden_dim, horizon)
    
    def forward(self, history_z, return_hidden=False):
        hidden = self.encoder(history_z)
        pred = self.pred_head(hidden)
        uncertainty = self.uncertainty_head(hidden) + 1e-4
        calendar_hint = self.calendar_head(hidden)
        
        if not return_hidden:
            return pred
        return pred, hidden, uncertainty, calendar_hint
    
    def compute_loss(self, pred, target, uncertainty):
        # Gaussian NLL: 让 base 学会表达不确定性
        nll = 0.5 * (torch.log(uncertainty**2) + (target - pred)**2 / uncertainty**2)
        return nll.mean()
```

**为什么这个组合最好：**
1. **对现有框架改动最小**：只需要扩展 base_backbone.py，delta 端只需要接受额外的 `uncertainty` 和 `calendar_hint` 输入
2. **Uncertainty 解决 Price 的核心问题**：Base 在 spike 时段输出高 uncertainty → delta 自动知道在哪里需要修正、在哪里应该保守
3. **Calendar hint 解决残差分配问题**：Delta 的 shape_head 可以直接对标 base 的 calendar 预测，减少三头之间的串扰
4. **训练稳定**：Gaussian NLL 是成熟的训练目标，不需要额外的 trick

---

## 问题三：推荐适合本框架的其他数据集

### 3.1 选择标准

根据框架结构，理想数据集需要满足：

1. **30min 或 1h 频率的时序数据**（框架 patch_len=8, stride=4 针对此设计）
2. **有对应的公开新闻/报告数据源**（regime bank 需要文本输入）
3. **新闻对时序有实际因果影响**（不然 delta 无法从新闻中获益）
4. **时序数据有一定的日历规律**（shape_head 的 DOW×HOD 分解能发挥作用）
5. **分布不要像 price 那样极端重尾**（避免 σ 放大问题）

### 3.2 推荐数据集

#### 推荐 1：欧洲天然气消费量（EU Natural Gas Demand）

**时序数据：**
- 来源：ENTSOG Transparency Platform（欧洲天然气传输系统运营商网络）
- 频率：小时级或日级天然气需求量（可聚合为逐小时）
- 区域：可选德国、荷兰、英国等单一国家
- 获取方式：ENTSOG 提供免费 API，数据格式标准化

**新闻数据：**
- 来源：Reuters / Bloomberg Energy / IEA 报告 / ACER 市场报告
- 内容类型：天气预报（供暖需求）、LNG 船期、储气量报告、俄罗斯管道流量变化、政策（如欧盟储气目标）
- 特点：**宏观新闻对天然气需求的影响链条短且明确**：寒潮 → 供暖需求 ↑ → 天然气消费 ↑

**为什么适合本框架：**
- 日历模式极强（冬季高、夏季低；工作日 vs 周末差异明显）→ shape_head 有用武之地
- 天气新闻是典型的 "regime modulator"：寒潮期间 = 高需求 regime，与 regime bank 设计完美匹配
- 分布比电价温和得多，残差结构接近 NSW Load
- 数据量充足：至少 3-5 年历史数据可获取

**预期表现：** 与 NSW Load 类似或更好，因为天气→天然气需求的因果关系比天气→电力负荷更直接（电力有可再生能源替代，天然气主要就是供暖）

#### 推荐 2：城市交通流量（Urban Traffic Volume）

**时序数据：**
- 来源：Caltrans PeMS（加州交通性能测量系统）或 UK Highways England
- 频率：5min 原始，可聚合为 30min
- 选取：单一高速公路传感器站点的车流量
- 获取方式：PeMS 提供免费下载，需注册账号

**新闻数据：**
- 来源：当地新闻 + 交通管理机构公告 + 活动日历
- 内容类型：道路施工公告、大型活动（体育赛事、音乐节）、天气预警（暴风雪/暴雨）、假日安排、事故报道
- 特点：这些新闻 **精确匹配 "macro regime" 概念** —— 施工 = "capacity reduced regime"、大型活动 = "demand spike regime"

**为什么适合本框架：**
- 日历模式极其规律（早晚高峰、周末模式）→ shape 分解的最佳场景
- 新闻信号清晰：施工公告直接导致该路段流量下降/转移
- 分布良好：流量数据近似正态，无极端尾部
- Regime 的 horizon_days 概念天然匹配：施工通常持续 3-14 天，活动是 1-3 天

**预期表现：** 很可能是三个推荐中最好的。交通流量的日历规律性极强（shape_head 最大化效果），且施工/活动新闻对流量的因果影响是最直接的。

#### 推荐 3：太阳能发电量（Solar Generation / Irradiance）

**时序数据：**
- 来源：AEMO（澳大利亚，已有框架支持的 NEM 市场）或 ENTSOE（欧洲）或 EIA（美国）
- 频率：30min（AEMO）或 1h（ENTSOE）
- 数据：区域太阳能总发电量或辐照度
- 获取方式：AEMO 数据与现有 NSW 数据同源，最容易获取

**新闻数据：**
- 来源：**可复用现有的 wattclarity 新闻语料库**（因为是同一市场 NEM）
- 额外来源：BOM（澳大利亚气象局）的天气预报、光伏行业报告
- 内容类型：天气预报（阴天/晴天/风暴）、大型光伏电站并网/检修、电网约束

**为什么适合本框架：**
- **最大优势：可直接复用现有的 regime bank 和 news pipeline**（同一市场 NEM 的新闻）
- 强烈的日历模式（白天发电、夜间为零；季节性显著）→ shape_head 效果好
- 天气新闻对太阳能发电有直接因果影响：阴天/暴风 → 发电量下降
- 分布比价格温和（有自然上下界：0 和晴天最大出力）
- 框架中 `schema_refine_v2` 的 topic_tags 已包含 `renewable_surplus/renewable_drought`，天然适配

**预期表现：** 中等偏好。太阳能的日内模式（bell curve）非常规律，但云层变化导致的分钟级波动可能比新闻的日级 regime 粒度更细。适合验证框架在 "新闻有中等信号强度" 场景下的表现。

### 3.3 三个推荐的对比总结

| 维度 | 天然气消费 | 交通流量 | 太阳能发电 |
|---|---|---|---|
| 数据获取难度 | 中（需注册 ENTSOG API） | 中（需注册 PeMS） | **低（复用 NEM/AEMO 数据源）** |
| 新闻获取难度 | 中（需爬取能源新闻） | 中（交通公告+活动日历） | **低（复用现有 wattclarity 语料）** |
| 新闻→时序因果强度 | **强** | **强** | 中 |
| 日历规律性 | 强 | **极强** | 强 |
| 分布友好度 | 好 | **好** | 好 |
| 框架适配改动量 | 中（需新增数据加载器） | 中（需新增数据加载器+schema调整） | **小（同市场同格式）** |
| 预期 delta 提升 | 大 | **最大** | 中 |

**如果只选一个先做**：选太阳能发电（改动最小，复用最多）快速验证；如果追求最佳效果展示，选交通流量。

---

## 附录：框架架构简述

```
┌─────────────────────────────────────────────────────────┐
│                      训练流程                            │
│                                                         │
│  Stage 1: Base Training (40 epochs)                     │
│  ┌──────────┐                                          │
│  │ history_z ├──→ MLP/DLinear ──→ base_pred_z          │
│  └──────────┘       ↓                                  │
│                  SmoothL1 Loss                          │
│                                                         │
│  Stage 2: Delta Training (30 epochs, base frozen)       │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────┐    │
│  │ history_z ├──│ PatchTST      ├──│ ts_tokens    │    │
│  │ + calendar│  │ Encoder       │  │ ts_summary   │    │
│  └──────────┘  └───────────────┘  └──────┬───────┘    │
│                                           │             │
│  ┌──────────┐  ┌───────────────┐         │             │
│  │ News     ├──│ Regime Bank   ├──→ regime_repr        │
│  │ Corpus   │  │ (offline)     │         │             │
│  └──────────┘  └───────────────┘         │             │
│                                           ▼             │
│               ┌───────────────────────────────┐        │
│               │ Three Residual Heads           │        │
│               │ slow_ts + shape_ts + spike_ts  │        │
│               └──────────────┬────────────────┘        │
│                              │                          │
│               ┌──────────────▼────────────────┐        │
│               │ Modulation Heads (×, not +)    │        │
│               │ lambda_base, shape_gain,       │        │
│               │ spike_bias (from regime_repr)  │        │
│               └──────────────┬────────────────┘        │
│                              │                          │
│               pred = base_pred + λ × residual           │
└─────────────────────────────────────────────────────────┘
```
