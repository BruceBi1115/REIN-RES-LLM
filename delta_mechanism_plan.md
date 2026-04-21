# Delta 阶段机制级改造方案（非参数级）

> **前提**：已完成 P0 参数级改造（见 [delta_residual_plan.md](delta_residual_plan.md)）。
> 本文的目标是**从机制层面重构 delta**，让残差修正有更强的信号和更稳定的训练动力学。
> 所有改动**不重新 refine news**，但会修改 delta 模型结构、输入通道、损失函数和训练流程。

---

## 当前结构的根本性局限

### 结构性诊断（阅读 [src/delta_v3/](src/delta_v3/) 源码后）

| # | 问题 | 位置 | 后果 |
|---|------|------|------|
| L1 | **Delta 编码器看不到 base 的历史残差** | [encoder_ts.py:88-108](src/delta_v3/encoder_ts.py#L88-L108) | Delta 只看到 `history_z`，和 base 吃的是同一份原始信号；它必须**从头推断** base 在哪里会出错，而不是**被告知** |
| L2 | **ShapeHead 是 flatten+linear，没有 horizon-position 感知** | [heads.py:20-32](src/delta_v3/heads.py#L20-L32) | news 对 horizon 的影响是时间非均匀的（事件后第 1 小时 ≠ 第 48 小时），但当前结构对所有 horizon 位置做同质处理 |
| L3 | **Regime bank 是日级静态查找** | [regime_bank.py:254-273](src/delta_v3/regime_bank.py#L254-L273) | 每个样本只拿到一个日级 news vector，无法让"不同 horizon 位置"查询"不同 news 元素" |
| L4 | **News 和 base-state 的联合建模只通过 base_hidden 拼接** | [model.py:69-83](src/delta_v3/model.py#L69-L83) | 损失里没有显式"news 应该让 delta 偏离 zero-correction"的对抗压力（只有 hinge 形式的 counterfactual） |
| L5 | **Base 阶段冻结后从不再更新** | [src/base/stage.py](src/base/stage.py) 训练后 freeze | Base 不知道"哪些系统性误差可以交给 delta"，delta 也无法逆向要求 base 调整行为 |
| L6 | **损失是对称 MSE/MAE，不看方向** | [trainer.py:1311-1322](src/delta_v3/trainer.py#L1311-L1322) | Delta 修正方向错但幅度小 vs 方向对幅度差，被同等惩罚——但**方向错**是更严重的商业错误 |
| L7 | **Lambda_base 是通过 main loss 间接学出来的调制器** | [modulation_heads.py:67-87](src/delta_v3/modulation_heads.py#L67-L87) | 它代表"TS 和 news 组合后该信多少 delta"，但没有**自我校准信号**告诉它"我这次预测到底靠不靠谱" |

---

## 六个机制级改造（按 leverage 排序）

每一条给出：**问题 → 改造 → 核心代码位置 → 工作量 → 预期收益**。

---

### M1 — Residual-history channel：让 delta 知道 base 的错误历史（🔥 最高优先级）

**问题**：Delta 的 TS encoder 只看 `history_z`，**完全不知道 base 在最近几步的实际误差**。这等于让一个"修改医生诊断的助手"去看病人的原始症状，而不是医生的历史误诊记录。

**改造**：为 delta encoder 添加第二通道 `base_residual_history`。

1. 训练集构建时（一次性、离线）：对 train/val/test 的 `history` 区间，用 stage-1 frozen base 前向，算出 `residual_history[t-L:t] = true_z[t-L:t] - base_pred_z[t-L:t]`
2. 在 dataloader 里和 `history_z` 一起输出 `residual_history`，shape `(B, L)`
3. [encoder_ts.py:72](src/delta_v3/encoder_ts.py#L72) 的 `patch_proj: nn.Linear(patch_len, hidden)` 改成 `nn.Linear(patch_len * 2, hidden)`（两通道拼接 patch）或者做两路 patch_proj 再相加
4. Test 时用 base test-window 的对应残差（用 rolling-forecast 思路一次性算出来即可）

**核心位置**：
- 新增：`src/data_construction/data.py` 里增加 residual_history 字段（需找 loader 实现）
- 修改：[src/delta_v3/encoder_ts.py](src/delta_v3/encoder_ts.py)（2 通道 patchify + proj）
- 修改：[src/delta_v3/model.py:69](src/delta_v3/model.py#L69) 传入 `residual_history`

**工作量**：中（~150 行代码，含 dataloader 改）

**预期收益**：高。直接解决"delta 靠猜"的问题。尤其对 MLP-traffic 这种 base 已经很准的 case，delta 能从"残差历史"里找到 base 的**剩余系统性偏差**。

---

### M2 — Position-query horizon decoder：news 能命中具体 horizon 位置

**问题**：[heads.py:20-32](src/delta_v3/heads.py#L20-L32) 的 `ShapeHead` 把 `ts_tokens` flatten 后用 LazyLinear 直接投到 horizon-dim。这意味着 horizon 第 1 位和第 48 位共享同一套表示，**无法表达"news 事件后 6 小时最强、24 小时衰减"这种时间结构**。

**改造**：将 `ShapeHead` 改为 cross-attention decoder。

```python
class ShapeHeadPQ(nn.Module):
    def __init__(self, hidden_size, horizon, n_layers=1):
        # H 个可学习的 position query
        self.pos_queries = nn.Parameter(torch.randn(horizon, hidden_size))
        # 每个 query 对 (ts_tokens + regime_tokens) 做 cross-attn
        self.xattn = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, memory_tokens):  # memory = [ts_tokens | regime_tokens]
        Q = self.pos_queries.unsqueeze(0).expand(memory_tokens.size(0), -1, -1)
        out = self.xattn(Q, memory_tokens)   # (B, H, hidden)
        shape = self.out(out).squeeze(-1)    # (B, H)
        return shape - shape.mean(dim=-1, keepdim=True)
```

**核心位置**：
- 新增/替换：[src/delta_v3/heads.py:20](src/delta_v3/heads.py#L20)
- 修改：[src/delta_v3/model.py:73](src/delta_v3/model.py#L73) 的 shape_head 调用改为传入 concat 后的 memory_tokens
- 同理可以升级 `SpikeHead` 的 `delta_proj` 用同样结构

**工作量**：中（~80 行，纯模型侧改动）

**预期收益**：中-高。对 traffic/load 这种有日内节律的 dataset 尤其明显；对 bitcoin 这种无强周期的 dataset 收益小一些。

---

### M3 — InfoNCE 对比损失：news 驱动的表示必须是"可辨别的"

**问题**：当前 counterfactual_loss 是 hinge（[trainer.py:1300-1303](src/delta_v3/trainer.py#L1300-L1303)）。一旦 `real_err < blank_err - margin`，梯度就消失了，**不再推动 delta 的 news representation 变得更精细**。结果就是 MLP-traffic 上 `blank_gain_z ≈ 0.005`——gap 小到几乎无意义。

**改造**：在 hinge 之外加 InfoNCE。

对 batch 内每个 active 样本 `i`：
- Positive：`(history_i, real_news_i)` 编码出的 regime_repr
- In-batch negatives：`{(history_i, real_news_j)} for j ≠ i` 以及 `(history_i, blank_news)`
- 损失：`-log(exp(sim(pos)/τ) / Σ_j exp(sim(neg_j)/τ))`

```python
# In trainer after computing regime_repr for real / blank / perm:
if cfg.contrastive_weight > 0:
    feats_real = F.normalize(regime_repr_real[active_mask], dim=-1)
    feats_neg = F.normalize(regime_repr_perm[active_mask], dim=-1)  # other-day news
    logits = feats_real @ feats_neg.T / cfg.contrastive_temperature
    labels = torch.arange(feats_real.size(0), device=device)
    contrastive_loss = F.cross_entropy(logits, labels)
```

**核心位置**：
- [src/delta_v3/trainer.py:1258-1303](src/delta_v3/trainer.py#L1258-L1303)（counterfactual 块扩展）
- [src/delta_v3/config.py](src/delta_v3/config.py)（新增 `contrastive_weight`、`contrastive_temperature`）

**工作量**：小（~40 行）

**预期收益**：中。对 traffic/load 这种 news 真正有信号的 case，能让 `blank_gain_z` 从 0.005 推到 0.05+。对 bitcoin 无效（news 本身太稀疏）。

---

### M4 — 阶段 3：Base+Delta 联合微调（打破"冻结 base"的人为瓶颈）

**问题**：当前 pipeline 是 [base 全训] → [freeze] → [delta 全训]。Base 从未接收 delta 侧的反馈——如果 base 存在一个**可以让出给 delta**的系统性误差，base 永远不会主动让位，delta 永远要从"硬抢"状态学。

**改造**：在 delta 收敛后增加 **stage-3 联合微调**：

1. 解冻 base backbone，设置 `base_lr = original_base_lr / 10`
2. Delta 的 lr 不变
3. 训练 N epoch（建议 N=10~20），loss 仍是 delta 侧的 composite loss，但现在梯度能流回 base
4. 早停阈值收紧（因为微调很快过拟合）

**核心位置**：
- [src/base_delta_decoouple_trainer.py](src/base_delta_decoouple_trainer.py) 或 delta trainer 末尾增加一个 joint-finetune 阶段
- 需要在 `base/stage.py` 里保留 base 的 optimizer state（或重建）
- 新增 args：`--joint_finetune_epochs`（默认 0 = 关闭）、`--joint_base_lr_scale`（默认 0.1）

**工作量**：中（~100 行，涉及 optimizer 重建 + checkpoint 协调）

**预期收益**：高（特别是对强 base + 弱 delta 的 case，如 MLP-traffic）。这个是论文里"base-delta co-adaptation"小节的直接素材。

---

### M5 — 方向敏感的辅助损失（direction-aware loss）

**问题**：[trainer.py:1311-1312](src/delta_v3/trainer.py#L1311-L1312) 的 `point_loss`（MSE/MAE）对"方向对但幅度差"和"方向反但幅度小"同等惩罚。但实际上 delta 的价值在于**把 base 预测朝正确方向拉**。

**改造**：增加一个辅助损失项：

```python
# True residual: what base got wrong
true_residual = targets_z - base_pred_z      # (B, H)
# Delta's contribution: how much it shifts prediction
delta_contribution = pred_z - base_pred_z    # (B, H)
# Reward directional alignment, penalize opposition
direction_loss = -F.cosine_similarity(delta_contribution, true_residual, dim=-1).mean()
```

在 loss 里加 `cfg.direction_weight * direction_loss`，推荐权重 0.05~0.1。

**核心位置**：
- [src/delta_v3/trainer.py:1310](src/delta_v3/trainer.py#L1310) 后
- [src/delta_v3/config.py](src/delta_v3/config.py) 新增 `direction_weight`

**工作量**：极小（~10 行）

**预期收益**：中。特别有助于"base 方向对但幅度估计偏保守"的场景。论文里是一个清晰的 ablation 点。

---

### M6 — 自校准 Confidence Head（"我这个 delta 到底靠不靠谱"）

**问题**：当前 `lambda_base` 是通过 main loss 反推出来的调制——它学到"什么时候该多加 delta"，但**不知道自己的 delta 是否真的比 base 好**。结果：当 delta 学坏时，lambda_base 也跟着学坏（lambda_base 和 delta 是同一套参数 co-trained）。

**改造**：加一个**自监督的 confidence head** `c_i ∈ (0,1)`：

1. 模型侧：新增 `self.confidence_mlp = _ScalarMLP(hidden_size)`，输出 `c_i = sigmoid(c_mlp(ts_summary + regime_repr))`
2. 前向：`pred_z = base_pred_z + c_i * lambda_base * residual_z`（c_i 再乘一层）
3. 训练时自监督目标：`c_i* = 1 if |pred_z_i - true_i| < |base_pred_z_i - true_i| else 0` （即 delta 帮了就是 1）
4. 损失：`calibration_loss = BCE(c_i, c_i*.detach())`

这样 `c_i` 学会预测"这次 delta 修正真的有帮助的概率"，**用它自己过去的成败经验**去调节当前输出。

**核心位置**：
- [src/delta_v3/modulation_heads.py](src/delta_v3/modulation_heads.py) 新增 `confidence_mlp`
- [src/delta_v3/model.py:83](src/delta_v3/model.py#L83) 前向改写
- [src/delta_v3/trainer.py:1310](src/delta_v3/trainer.py#L1310) 增加 `calibration_loss`
- 加 diagnostic：`confidence_mean`、`confidence_helped_calibration`（confidence 和实际 helped 的相关系数）

**工作量**：中（~120 行，涉及 model、trainer、diagnostics 三处）

**预期收益**：中-高。**对 bitcoin 最有效**：由于 news 稀疏，delta 经常乱修正，c_i 会学会"大部分情况输出 0"，等效于一个**可学习的、基于内容的 hard gate**。比当前静态 `hard_gate_mass_threshold` 更智能。

---

## 非核心补充（后置，按需启用）

| 代号 | 想法 | 适合的场景 |
|---|---|---|
| M7 | Event-lag attention bias：每条 news 学习一个 horizon-position 的衰减权重 | News 有明确时间滞后结构的场景（如宏观经济发布） |
| M8 | Wavelet/MA-based 频带分解替代 slow+shape+spike | 残差有多尺度结构的场景（gas, load） |
| M9 | Delta 的预训练任务改为"预测 base 未来 H 步误差"而非 derived targets | 有充足无标签 base 输出可用时 |

---

## 推荐落地顺序（phase 化）

### Phase A（2~3 天）：最高性价比，对论文最有贡献
1. **M1**（residual-history channel）——信号级解决"delta 没输入"
2. **M5**（direction-aware loss）——几乎零成本，单独可做一次 ablation

> 预期：traffic-MLP 的 skill_score 从 −0.47% 翻正；bitcoin-DLinear 的 −8.8% 减小到 −2%~0

### Phase B（3~5 天）：架构升级
3. **M2**（position-query shape head）——结构性升级，论文 figure 来源
4. **M6**（confidence head）——自校准机制，bitcoin 专治

> 预期：traffic-DLinear 的 skill_score 从 17.7% 提升到 22%+；bitcoin-* 变成 delta 无害（skill ≥ 0）

### Phase C（3~4 天）：训练流程升级
5. **M4**（joint fine-tune stage-3）——打破冻结瓶颈
6. **M3**（InfoNCE contrastive）——news 表示强化

> 预期：整体再提升 3~5%，但风险更高（需要早停调参）

### Phase D（选做）
7. M7~M9——按需，非论文核心章节

---

## 验证协议

- **每一个 M 单独 ablation**：与 "base+delta (current P0)" 对照，不要多项同时启用——否则论文里讲不清哪个在起作用
- **4 datasets × 2 backbones 矩阵**（traffic/load/gas/bitcoin × MLP/DLinear）
- **关键 metric**：
  - primary：test MAE, MSE, skill_score
  - secondary：`blank_gain_z`（news usage）, `confidence_helped_calibration`（M6）, `direction_cos`（M5）
- **3 seeds**，报 mean±std
- 若某个 M 在 4/8 个 case 里显著涨点，就写进论文；若只在 1/8 里涨点，放进 supplementary

---

## 对论文的加分点

1. **M1 + M5 是"机制级 insight"**：比参数调优更有学术重量——"delta 需要知道 base 的错误 + 方向"是一个可以推广的 principle
2. **M2 的 position-query 是 novel architectural contribution**：可以画出"news → horizon position"的 attention heatmap，visualize 非常强
3. **M6 的 confidence calibration**：可以做一张 reliability diagram（预测置信度 vs 实际命中率），这是 Bayesian TS forecasting 领域的标准 plot，少见用在 news-driven delta 上
4. **M4 的 joint finetune**：直接给出"sequential vs joint"的 ablation，这是 residual learning 文献里反复被讨论的话题

---

## 与 P0 参数改动的关系

P0（inactive_weight、lambda_max、shape_gain_cap、shape_gain_l2、hard_gate）是**基础设施**，M1-M6 是**建筑**。必须先跑完 P0 的 baseline，再在 P0 基础上做 M 系列 ablation，否则数字不可比。

建议先把 P0 跑完得到新的 baseline 数字，再从 **M1** 开始。
