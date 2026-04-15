# Delta Stage 改进方案 (updateplan.md)

> 目的：在不破坏 base → delta 两阶段范式的前提下，针对 [ANALYSIS_NOTES.md](ANALYSIS_NOTES.md)、[price_plan.md](price_plan.md)、[results/diagnose_news_price_signal.json](results/diagnose_news_price_signal.json) 已经暴露的短板，给出分轴、可独立落地的改造清单。每项都写明「根因指向 / 触点文件 / 预期影响 / 验证指标」。本文档只做方案，不改代码。
>
> 当前的定位：诊断脚本已经证明 news→price 的互信息显著（regime_vec MI=0.132, p=0.005；text_emb_pca8 MI=0.177, p=0.005），所以瓶颈**不是信号缺失，而是 delta 端的建模/优化机制**。以下改造围绕「让可用信号被 delta 正确吸收」展开。

---

## 0. 背景速览（为什么 delta 目前吃不到信号）

| 维度 | 当前状态 | 问题 |
|---|---|---|
| 优化器 | 仅 `AdamW(lr=1e-4)`，见 [src/delta/core.py:66-69](src/delta/core.py#L66-L69) | 无 LR 调度、无 warmup、无差分学习率 |
| 损失尺度 | Price 走 raw-space + z-space 归一（[trainer.py:467-473](src/delta_v3/trainer.py#L467-L473)），其它辅助损失仍是 z-space | 量级即便已经做了 rescale，price spike batch 的量级仍会瞬时压倒 consistency / counterfactual |
| slow head | 标量输出 (B,)（[heads.py:7-17](src/delta_v3/heads.py#L7-L17)） | 对一个 720 步 horizon 用一个标量描述「慢变」，表达力过弱；当 base 已经解决大部分 level，slow 会变成噪声 |
| shape head | `Flatten + LazyLinear + Linear(hidden→horizon)`（[heads.py:20-32](src/delta_v3/heads.py#L20-L32)） | 没有显式时序结构（RNN/TCN/Fourier/DOW×HOD lookup），长 horizon 极易过拟合训练集的特定 shape |
| spike head | gate × delta，gate logit 由 `SpikeHead + spike_bias`（[model.py:74-79](src/delta_v3/model.py#L74-L79)） | gate 初期概率极低 → spike 点 gradient 被「门关着」吃掉；spike mask 是由 detrended residual 的 top-10% 位置生成（[targets.py:140-146](src/delta_v3/targets.py#L140-L146)），不是市场真实 spike；门阈值 0.8 全局共用 |
| regime bank | 日级 EMA + 指数衰减 + 文档 confidence×age 加权（[regime_bank.py:170-198](src/delta_v3/regime_bank.py#L170-L198)） | 日级颗粒度丢掉了当天内的 30-min 市场事件；EMA 会把 spike 事件拖平；text_emb 是 in-force 加权平均，短期强信号被长 horizon_days 的文档稀释 |
| 训练目标对齐 | 三头共享 encoder（[model.py:69-74](src/delta_v3/model.py#L69-L74)），梯度共用 | Price 的 spike 大梯度会把 encoder 拧歪，shape/slow 隐式受害（与 price_plan 根因 F 一致） |

---

## 1. 学习率调度与优化器（最小改动、最高性价比）

### 1.1 三段式 LR：warmup → cosine → flat
- **触点**：[src/delta/core.py `_build_delta_optimizer`](src/delta/core.py#L66-L69)；新增 `_build_delta_scheduler`，在 [trainer.py:948](src/delta_v3/trainer.py#L948) 之后构造 `LinearLR(warmup=5% steps) → CosineAnnealingLR(T_max=90% steps)`。
- **理由**：当前 lr 前 2–3 个 epoch 还没把 modulation head（刚从 pretrain 复制一层）训稳，spike 梯度就已经轰进来了；`price_plan` 根因 F 的 shape_gain 贴 cap 现象就是这种「冷启动期被 spike 梯度拖偏」的典型表现。
- **验证指标**：`shape_gain_mean` 前 5 epoch 的波动幅度应从当前 ~0.4 的 std 降到 <0.1；`spike_bias_mean` 的 epoch 间翻号应消失。

### 1.2 差分学习率（param groups）
- **触点**：`_build_delta_optimizer` 拆分三组：
  - group A：`ts_encoder` + `regime_projector` → base lr（1e-4）
  - group B：`slow_head / shape_head / spike_head` → 0.5× base lr（这些是残差结构，容易过拟合）
  - group C：`modulation_heads`（lambda_* / shape_gain / spike_bias）→ 2× base lr（有 cap 约束、从 pretrain 初始化、需要快速适配下游）
- **理由**：modulation heads 是 delta 与 base 之间的「信任接口」，它的学习越快 consistency 越早稳。当前所有参数共用一个 lr，等 modulation 学到合理 λ 时 heads 已经严重拟合了残差中的噪声。
- **验证指标**：`lambda_base` 在 epoch 3 之前就能跨越 0.15 → 稳态，而不是像目前这样 epoch 10+ 才稳定。

### 1.3 EMA of weights for eval
- **触点**：trainer 里加 `torch.optim.swa_utils.AveragedModel`，每 epoch eval 前 swap 到 EMA 权重；delta 本身参数少（~100K–1M 级），EMA 几乎无开销。
- **理由**：price 任务 val 选点严重受单 epoch 尖峰 batch 影响（`price_plan` 根因 D）。EMA 等价于在时间维度给 val_mae 做低通滤波。
- **验证指标**：同一套超参下 best_epoch 的方差（重复 3 seed）应下降。

---

## 2. Slow / Shape / Spike 三头重构

### 2.1 Slow：从「标量」升级为「低频多尺度残差」
- **当前**：`slow_ts ∈ R^(B,)`（[heads.py:17](src/delta_v3/heads.py#L17)），然后在 [model.py:82](src/delta_v3/model.py#L82) 被 `unsqueeze(-1)` 广播到整个 horizon。
- **问题**：对 h=720 而言，"一个样本一个 slow 标量" 等于强行假设 level shift 在 30 天内是常数；负荷的「冷/热 regime」持续 3–7 天，价格的 level shift 更短。
- **方案**：
  - 输出改为一个 **低频基函数系数向量** `slow_coef ∈ R^(B, K)`（K=4–8），然后在 horizon 上用预定义 **低阶多项式 + 周日周期**（`[1, t/H, sin(2πt/48), cos(2πt/48), sin(2πt/(48·7)), cos(...)]`）线性组合出 `slow_ts ∈ R^(B,H)`。
  - 这等价于给 slow head 一个「带结构先验的样条」，只学 K 个系数而不是 H 个值，避免和 shape head 抢频段。
- **触点**：[src/delta_v3/heads.py `SlowHead`](src/delta_v3/heads.py#L7-L17)、[src/delta_v3/model.py:82](src/delta_v3/model.py#L82) 的广播、[targets.py:119 `slow = residual.mean(dim=-1)`](src/delta_v3/targets.py#L119) 需要改为投影到同一组基函数的系数。
- **预期**：Load h720 的 `slow_ts` 对 val 真实均值序列的 R² 从目前（标量 = 样本内均值）的 ~0.2 提升到 >0.5；shape head 的目标不再携带被 slow 混进去的低频分量，shape_gain 不再贴 cap。

### 2.2 Shape：引入日历感知 + 解耦 head（解决 shape_gain 贴 cap 与 spike 污染）
三个递进选项，按 ROI 排序：

**(a) DOW×HOD 残差形状库（最便宜）**
- 复用 [targets.py `compute_residual_calendar_baseline`](src/delta_v3/targets.py#L34-L95) 已经算好的 `dow_hod_mean (7,48)`。在 shape head 的输入侧把「horizon 上按 target_time 查表得到的 calendar baseline」作为一个 **额外通道**拼进 fused_tokens。
- **理由**：目前这个 7×48 矩阵只用于分解 target（算 slow/shape/spike 拆分），**模型自己看不见它**。把它作为 hint feed 给 shape_head，等于 ANALYSIS_NOTES 里「Residual-Structured Base 思路 B 的轻量版」。
- **触点**：[src/delta_v3/model.py:69](src/delta_v3/model.py#L69) 的 `ts_tokens`, [heads.py:20-32](src/delta_v3/heads.py#L20-L32) 的 `ShapeHead.proj`。

**(b) Fourier-head（对周期数据天然匹配）**
- 把 shape_head 的 `Linear(hidden, horizon)` 替换为 `Linear(hidden, 2·K)` → 输出 sin/cos 系数 → 在 horizon 上重建。K=8 可覆盖日周期 + 半日 + 周周期的主要模态。
- **理由**：Load/Price 的残差形状 90% 能量在 <24h 周期；直接在频域出解比在时域出 H 个独立值更鲁棒，且天然防止对某一天特定形状的过拟合。
- **预期**：h=720 时参数量从 hidden×720 骤降到 hidden×16，shape_head 过拟合训练集形状的倾向显著下降。

**(c) Encoder 浅层共享 + head 独立（解 price_plan 根因 F）**
- 当前三个 head 共享 `ts_encoder + ts_tokens + ts_summary`。把架构改成：共享 patch embedding 和前 1 层 Transformer，**后 1 层 Transformer 分叉成 slow-branch / shape-branch / spike-branch 三个 sub-encoder**（参数量增加 ~50%）。spike 的大梯度只污染它自己的分支。
- **触点**：[src/delta_v3/encoder_ts.py](src/delta_v3/encoder_ts.py)、model.py 的 forward 分派。
- **注意**：此项代价最高（参数量 ↑），建议只在 price 侧开启，用 config flag 门控。

### 2.3 Spike：伪标签重定义 + gate 预热 + 门阈 regime 自适应
这是目前最严重的 bug 群，`price_plan` 根因 E/B 已经指出。

**(a) Spike pseudo-label 改用 **目标 y 本身** 的 IQR spike，而非 base residual 的 top-10%**
- **当前**：[targets.py:138-146](src/delta_v3/targets.py#L138-L146) 把 detrended_residual 的 |·| top-10% 作为 spike 位置。
- **问题**：这让 spike_head 学的是「base 在哪里错」，而新闻信号只对「市场在哪里 spike」有因果（price_plan 根因 E）。
- **改造**：新增一条并行标签 `market_spike_mask = |y − median_y_window| > 3·IQR_y`，**两个 mask 取并集**作为 spike 位置；loss 在并集上算，但 `spike_gate_rate_loss`（[trainer.py:1044](src/delta_v3/trainer.py#L1044)）对齐到 market_spike_rate（不再对齐 residual-defined rate）。
- **预期**：price h48 的 help_rate_top10% 从 66%（watt_free）升到 >72%；spike_bias_mean 从当前 −0.155 附近的抖动稳定到一个固定符号。

**(b) Gate warm-up：前 N epoch 强制 gate=1，只学 spike_delta**
- **当前**：`out["spike_gate"] * out["spike_ts"]`（[model.py:82, trainer.py:1024](src/delta_v3/trainer.py#L1024)）——冷启动期 gate 几乎是 0，spike 位置的 gradient 几乎没回流到 spike_delta。
- **改造**：epoch < 3 时 forward 用 `detach(spike_gate) + 1·(1−epoch_progress)` 或直接把 gate 强制固定为 spike_target_pct；之后切回学习态。BCE gate loss 同步从第 3 epoch 起激活。
- **预期**：spike_head 在 epoch 3 结束就收敛到有用状态，省下后面 20+ epoch 的挣扎。

**(c) Regime-adaptive gate threshold**
- **当前**：[config.py:134](src/delta_v3/config.py#L134) 的 `spike_gate_threshold=0.8` 是全局常数。
- **改造**：用 `lambda_news_delta` / `spike_bias` 的 sign 自适应 —— 新闻明显偏高风险时阈值 0.6，新闻中性时 0.85，新闻明显稳态时 0.95。可以在 modulation_heads 里新增一个 `gate_thresh_mlp(regime_repr) ∈ [0.6, 0.95]`。
- **理由**：Price 侧 false-positive 的代价远大于 Load（ANALYSIS_NOTES §1.3(c)）；让新闻直接调节 FP/FN 阈值比改 bias 更直接。

### 2.4 新增：Uncertainty head（可选，与 ANALYSIS_NOTES §2.3 对齐）
- 在 delta 的 heads 里加一个 `sigma_head : hidden → R^H`（Softplus），损失换成 Gaussian NLL 的一部分：`0.5·(log σ² + (y−μ)²/σ²)`，权重 0.1。
- **意义**：Price h48 测试上输出 per-step σ，可用于（1）给 modulation head 的 λ 做 step 级调节（uncertainty 低时 lambda→0），（2）给后续 regime_bank 做主动标注。
- **优先级**：第二梯队，先等 2.3 三项落地后再评估收益。

---

## 3. Regime Bank 构造与特征工程

### 3.1 多尺度 regime bank（短/中/长 horizon 分离）
- **当前**：[regime_bank.py:170-193](src/delta_v3/regime_bank.py#L170-L193) 把所有文档按 `confidence × exp(−age/τ)` 加权混入**同一**日级向量，τ=5 天全局固定。
- **问题**：AEMO market notice（有效期 2h–2d）与 Reuters 政策分析（有效期 2–6 周）被同一个 τ 压平。price 需要的是前者，load/seasonal 需要的是后者，当前 bank 在两者之间取了一个两边都不讨好的均值。
- **改造**：build_regime_bank 输出三组向量：
  - `regime_short`（τ=1 天，只收 horizon_days≤2 的 actionable）
  - `regime_mid`（τ=5 天，horizon_days 3–7）
  - `regime_long`（τ=14 天，horizon_days ≥8）
- `encode_regime`（[model.py:40-54](src/delta_v3/model.py#L40-L54)）把三者拼接进 features；`relevance_mass` 分别输出三路，modulation_heads 各自看一路（spike_bias 看 short+mid，shape_gain 看 mid+long，lambda_base 看全部）。
- **预期**：price 侧 spike_bias 能捕捉 AEMO 公告这种短半衰期信号；load 侧 shape_gain 继续吃 weather 预报这种长 horizon 信号。

### 3.2 把 30-min 事件层信号独立出来（price 专用）
- 目前 bank 只有「日」粒度。price 真正的驱动是跨州联络线 / 机组跳闸 / 燃料价格日内 spike，这些在 AEMO Market Notices / NEMweb 上本来就是 **带时间戳的事件**。
- **改造**：新增 `intraday_event_track ∈ R^(H, D_ev)`，对齐到预测的 horizon 上（每 30min 一槽），对每一步查「未来 t 时刻是否有 in-force market event」。这条 track 直接拼到 shape_head / spike_head 的 fused_tokens 上，而不经过 daily bank 平均。
- **数据**：已有 [src/News_scrape/crawl_aemo_newsroom.py](src/News_scrape/crawl_aemo_newsroom.py)，扩展一个 market-notices 专用 crawler 即可。
- **预期**：直击 `price_plan` 根因 E、G；price help_rate_top10% 可望从 64–66% 升到 75%+。

### 3.3 修正 relevance_mass 的 threshold 语义（低成本）
- **当前**：[config.py:146](src/delta_v3/config.py#L146) 默认 `active_mass_threshold=0.7`，但 [model.py:53](src/delta_v3/model.py#L53) 用硬阈乘 regime_repr，模糊语义只在 lambda_base 的 soft_gate（[modulation_heads.py:86](src/delta_v3/modulation_heads.py#L86)）里存在。
- **问题**：硬阈会让 0.69 和 0.71 的两天有跳变，而实际 relevance_mass 的分布可能在 0.6–0.8 有密集带。
- **改造**：
  - 阈值改为**按数据集自适应**（取 train 期 relevance_mass 的 60% 分位数）
  - `encode_regime` 里的 `active` 乘子改成软 sigmoid（和 modulation_heads 一样的 soft_gate），彻底移除硬阈。
- **预期**：inactive_gap 从 0.28–1.66% 降到 <0.3%，news_blank 与 news_present 的过渡更平滑。

### 3.4 用 **基于 MI 的后验重加权** 替代 confidence
- **当前**：build_regime_bank 依赖 LLM refine 输出的 `confidence`（0–1 自评）作为权重。
- **问题**：LLM 自评 confidence 与它对 price MI 没有必然相关；`price_plan` 根因 A 已经点出「relevant ≠ predictive」。
- **改造**：用 diagnose_news_price_signal 的 MI 结果做 **re-weighting 层**——对每个 regime_key / topic_tag，计算它在 train 集上对 residual 的 MI，作为 global scaling factor 乘到 regime_vec / topic_tag_mass。低 MI 的 key 直接 zero out。
- **触点**：新增一个 `bank_posterior_reweight.py` 离线脚本，产出一个 weight vector，load 时乘进去即可；不动 bank 构造逻辑本身。
- **预期**：regime_vec 从 8 维有效信息通道缩到 3–4 维，但 SNR 显著上升；modulation_heads 的拟合更快、更稳。

### 3.5 对 text_emb 做 topic-aware pooling（替代当前的加权平均）
- **当前**：[regime_bank.py:193](src/delta_v3/regime_bank.py#L193) 是单加权平均 → 384 维。一个 demand_surge 文档和一个 renewable_drought 文档的 emb 会被平均成一个无意义的中点。
- **改造**：按 topic_tags 分桶后分别 mean-pool，输出 `text_emb_by_topic ∈ R^(T, 384)`，再对 regime_projector 的输入做 attention pooling。
- **预期**：regime_repr 的信息密度提升；ANALYSIS_NOTES §1.3(d) 里「price 的 regime→target 映射高度非线性」部分被缓解。

---

## 4. 损失函数与训练过程（解 price_plan 根因 C/D）

### 4.1 完全 z-space 统一 + 动态损失平衡
- **当前**：[trainer.py:463-473](src/delta_v3/trainer.py#L463-L473) price 用 raw + quantile + /scale_global 近似 z-space。但 `scale_global` 是全局 σ，对 price 这种重尾而言除出来的仍然不稳。
- **改造**：
  - 主损失彻底走 z-space smooth_l1（非 price 已是），price 的 winsor 也在 z-space 做。
  - 引入 **GradNorm** 或 **Uncertainty Weighting**（Kendall et al. 2018）自动平衡 slow / shape / spike / consistency / counterfactual 五项。当前都是硬编码权重，`price_plan` 根因 C 就是在说 consistency_weight=0.30 根本拉不住 spike 梯度。
- **预期**：inactive_gap 稳定 < 0.2%；blank_gain_z 不再逐 epoch 变负。

### 4.2 Val 切分重构（解 price_plan 根因 D）
- **当前**：val = 2024-09-13 → 10-19（NSW 初春，无 spike），test 含夏季 spike；select_metric="mae" 选出的 delta 到 test 翻车。
- **改造**（不改框架，只改 split 脚本）：
  - 方案 X：train 尾 → val：rolling origin，从 train 尾部按季节等比例抽出 val，确保 val 含至少 1 次 summer spike 周和 1 次 weekend low 周。
  - 方案 Y：复合 select_metric：`0.6·val_mae + 0.4·val_top10pct_residual_mae`，强制选能搞定尖峰的 checkpoint。
- **触点**：[split.py](split.py) + [trainer.py:1201 的 select 逻辑](src/delta_v3/trainer.py#L1201)。

### 4.3 Hard-sample importance sampling 升级
- **当前**：[src/delta_v3/importance.py `HardResidualSampler`](src/delta_v3/importance.py) 按残差 top 60%×10% 抽样（[config.py:159-160](src/delta_v3/config.py#L159-L160)）。
- **改造**：加一个 **news-active × high-residual 双条件**桶：
  - 桶 A（news-active ∧ high residual，30%）：delta 应该起作用但目前起不了的样本
  - 桶 B（news-inactive ∧ low residual，30%）：consistency 必须稳的样本
  - 桶 C（其它，40%）
- 训练时三桶等权重采样。避开单纯按 residual 幅度抽样带来的 spike 偏置。
- **预期**：help_rate 与 MAE 的脱钩现象（ANALYSIS_NOTES §1.2）被显式压下。

---

## 5. 分阶段落地清单（建议顺序）

| 阶段 | 动作 | 改动量 | 预期 ΔMAE (price h48) | 预期 ΔMAE (load h48) |
|---|---|---|---|---|
| **S1 — 调度** | §1.1 warmup+cosine，§1.2 param groups，§4.2 复合 select_metric | 小（<100 行） | −0.3 ~ −0.5 | −0.05 |
| **S2 — 三头结构** | §2.2(a) calendar hint，§2.3(a) y-based spike pseudo-label，§2.3(b) gate warm-up | 中（~300 行） | −0.8 ~ −1.5 | −0.1 ~ −0.2 |
| **S3 — Regime** | §3.1 多尺度 bank，§3.4 MI re-weighting，§3.3 自适应 threshold | 中（~250 行） | −0.5 ~ −1.0 | −0.1 |
| **S4 — 深层** | §2.2(c) encoder 解耦，§3.2 intraday event track，§2.1 slow 基函数 | 大（>500 行） | −1.0 ~ −2.0 | −0.1 |
| **S5 — 探索** | §2.4 uncertainty head，§4.1 GradNorm | 中 | 不确定 | 不确定 |

**推荐先做 S1 全部 + S2 (a)(b)**：这是性价比最高的一档，不碰模型骨架，只改 trainer / heads 的输入通道和训练调度，2 天工作量内可完成并回归。

---

## 6. 验证清单（每阶段必须跑的诊断）

1. `delta_mae / base_mae`：主目标，分别跑 Load (h48/192/720) 和 Price (h48)。
2. `help_rate_top10%_residual`：这是 [results/delta_v3_acceptance.md](results/delta_v3_acceptance.md) 的核心验收项，目前 load 80%/59%/45%，price 66%。
3. `inactive_gap`（news-blank vs news-present 在 inactive 样本上的差异）：应 <0.3%；price_plan 里目前 1.66% 是爆雷级。
4. `spike_bias_mean` epoch 间翻号次数：期望降为 0。
5. `shape_gain_mean` 贴 cap 比例：期望 <10%（当前 price 侧常 >50%）。
6. 对 §3.4 做完 MI re-weight 后重跑 [scripts/diagnose_news_price_signal.py](scripts/diagnose_news_price_signal.py)：regime_vec MI 应从 0.132 升到 >0.18。

---

## 7. 不建议在本轮做的事

- **换 base 架构**（ANALYSIS_NOTES §2 的 Uncertainty-Aware Base / Residual-Structured Base）：收益大但改动面广，需要重跑所有 baseline，留到 S5 之后。
- **日级聚合 price** / **pinball 区间预测**（price_plan 阶段 III 选项 1/2）：等 S1–S4 全部跑完若仍不达标再考虑，否则会丢失 30-min 粒度的应用价值。
- **换 LLM refine schema**：目前 schema 已经在 MI 诊断上显著，重做 refine 的成本（重跑全部 2000+ 文档）与收益不成比例；先用 §3.4 的后验 re-weight 低成本救场。
