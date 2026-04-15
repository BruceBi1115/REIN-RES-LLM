Price 持续不达标的根因与改造计划
1. 最新跑（watt_pv_mag, h48）关键证据
epoch	val_mae	base_mae	active_mae	blank_active	blank_gain_z	inactive_gap	λ_sat
1	32.82	31.87	35.36	33.90	−0.031	0.32%	0.000
2	33.62	31.87	36.49	34.03	−0.052	0.28%	0.011
3	34.71	31.87	37.91	34.16	−0.079	0.83%	0.024
4	35.16	31.87	38.32	34.20	−0.087	1.66%	0.022
三个关键不等式，指向同一个结论：

active_mae > base_mae：delta 在它最该帮忙的 active 样本上反而更差
blank_active < active_mae：把 news blank 掉，active 样本反而变好
blank_gain_z 逐 epoch 更负：news 越训练越有害
换新闻源（pv_magazine → watt_pv_mag，2163 docs + 347 actionable）没有改善。这说明问题不在新闻源，在框架机制。

2. 框架机制层面的 7 个根因假设
根因 A：active 样本筛选≠因果相关筛选
active_mass_threshold=1.2 用关联总质量筛 active day，但"光伏新闻与 30 分钟电价"是弱因果。一篇"某州装机容量上涨"的文章会拉高 relevance_mass，却对 test 期（NSW 夏季晚高峰尖峰）毫无预测力。框架误把相关 relevant当成可预测 predictive。

根因 B：冻结的 base 本身是"去 spike"的，让 delta 必须独吞 spike 责任
Base 用 z-space SmoothL1 训练，天然回归均值、平滑掉 spike。Delta 要同时完成：(1) 拉回 base 的过度平滑、(2) 对齐真实 spike、(3) 保持 inactive 样本不动。news 信号不足以支撑 (1)+(2)，delta 就用噪声去填。

根因 C：损失尺度不对称，raw-space price loss 主导梯度
Delta 主损失在 raw-space price（winsor [-48.9, 557.7]）上算 SmoothL1 + pinball；而 consistency/counterfactual 损失在 z-space。一个 spike 样本的 raw-loss 梯度可以是 consistency 梯度的 30–100 倍。consistency_weight=0.30 根本拉不住。→ SGD 默默把 delta 优化成"只管 spike、不管 inactive"。

根因 D：val / test 分布漂移
val = 2024-09-13 → 10-19（NSW 初春，价格平缓）
test = 2024-10-19 → 2025-01-01（夏季晚高峰 + 圣诞季，包含全年 top 10% spike 的大部分）
select_metric="mae" 在 val 选 checkpoint，但 val 几乎没有尖峰，等于在无 spike 场景选 spike 模型。选出来的 delta 到 test 必然翻车。

根因 E：spike_gate 的"标签"是 base 的残差，不是市场 spike
spike_abs_threshold = base_residual_sigma × 3 = 基于 base 在 train 上的误差定义 spike，不是基于市场真实 spike。spike head 其实在学"base 哪里错了"，而新闻对"base 错在哪"毫无直接信号，只有对"市场发生什么"有信号。→ 任务本身就错位。

根因 F：三头共享 encoder，spike 头的大梯度污染 slow/shape 头
slow + shape + spike 三个 head 共用 PatchTST 编码隐层。spike head 在 raw price 上算 loss → 大 spike 样本的梯度量级巨大 → 反向传播把共享隐层"拧"向 spike 敏感方向 → slow/shape head 的语义漂移 → shape_gain 持续贴到 cap（1.29+），spike_bias epoch 间翻号（-0.01/-0.43/-0.07/-0.33）。

根因 G：LLM refine schema 对 price 太粗
现 schema 的 5 个 regime key 对 load 合适（supply/demand 宏观面），但 price 的真正驱动是机组报价行为、跨州联络线潮流、强迫停机、燃料现货、夏季冷却需求——这些在通用能源新闻里出现少、被 refine prompt 主动过滤掉。→ regime_vec 在 price 任务上承载的信息熵接近 0，无论换多少新闻源都一样。

3. 改造计划（按"代价/收益"排序，分三阶段）
阶段 I：证伪式诊断（不改框架，只跑数据）
在动机制之前，先证实"news 对 price 到底有没有信号"。如果根本没有，后面的机制改动都是白费。

互信息测算：对 train 集做 MI(regime_vec[t], |residual_t→t+48|)。如果 MI < 0.02 bits，news→price 信号压根不存在，直接跳到阶段 III（改任务或放弃 news）。
Permutation test：把 regime_bank 的日期随机置换 100 次，重跑 delta 一小段。若真实 vs 置换的 active_mae 差异不显著（p>0.05），news 相当于随机噪声。
按 top/bot decile 分桶看 skill：分别看 news-active 日 vs news-inactive 日、test 的平价日 vs spike 日上 delta 的 skill。定位到底 delta 在哪个子群被害死。
阶段 II：机制修正（真改框架，按根因 A–F 对应）
对应根因	改动	预期影响
A	新增 price_relevance_filter：refine 阶段用 price-specific prompt 过滤文章（保留含「outage / bidding / demand peak / interstate flow / fuel」等 token 的），重建 regime bank	过滤后 active_pct 可能从 ~30% 降到 ~10%，但剩下的都是真正与 price 相关
B	Base 改用 pinball 损失（不对称分位损失），或在 base 训练时直接加入 spike-aware CRPS 项。让 base 自己能预测 spike，delta 只做微调	base_mae 会升一点点，但 active 样本上给 delta 留出合理 headroom
C	把主 delta loss 改到 z-space（或归一化后 scaled-SmoothL1），让 consistency / counterfactual / residual 三项梯度量级可比	立刻消除 inactive_gap 泄漏；spike_bias 震荡消失
D	val/test 切法改为时间块交叉：用 train 尾部（涵盖至少一次 spike 事件）做 val，或 rolling origin + test-augmented val	选出来的 delta 在 test 真实分布下稳健
E	spike pseudo-label 从"base 残差"改为"市场 y 的真实 spike"（用 y 本身的 IQR-based thresholding），让 spike head 学市场而非 base 的错	spike head 语义对齐；即便 news 弱也不会学错
F	三头解耦：slow/shape/spike 各自一个小 encoder head，或只共享浅层 + 各自 head 前加 detach。同时对 spike head 的上游 grad 做 norm-clip	shape_gain 不再贴 cap；slow head 输出恢复稳定
阶段 III：任务重定义（如果阶段 II 改完仍无增益）
若阶段 I 就证实 news→price MI ≈ 0，或阶段 II 改完 Skill 仍 ≤ 0.5%，应承认：news 对 30-min NSW price 的 short-horizon causal 链条太弱。此时有三个选项：

改任务颗粒度：30-min price → daily average price 或 daily peak price。新闻在日级聚合下 SNR 显著上升。
改任务对象：把 price 任务换成 daily 解释 delta 预测区间（概率预测、pinball loss），而非点预测 MAE。新闻对区间宽度的信号比对点值强。
换新闻类型：放弃通用能源新闻，改用 AEMO 官方 Market Notices（公告类、含停机 / 约束 / 警报）。这是 price 的一阶因果输入，之前任何通用新闻源都不会有。
4. 我的推荐路径
先做阶段 I 的第 1、2 项（2 天工作量，不改框架），用数据证伪"news→price 有信号"的假设。

如果 MI 显著 → 做阶段 II 的 C + F + D 三项（损失尺度、三头解耦、val 切法），这是成本最低但命中根因的三项。
如果 MI 不显著 → 直接跳阶段 III 第 3 项（换 AEMO Market Notices）；若无法获取，跳第 1 项（改到日级）。
要我先写一个阶段 I 的诊断脚本（MI + permutation test），还是直接按阶段 II 的 C+F+D 给你一个改造方案？