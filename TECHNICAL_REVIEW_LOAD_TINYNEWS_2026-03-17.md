# Technical Review：2024 NSW LOAD TinyNews DELTA

Last updated: 2026-03-17

## 1. 文档目的

这份 review 的作用有两个：

- 总结已经完成的 2024 NSW LOAD TinyNews DELTA 消融实验到底得出了什么结论
- 把这些结论和当前代码里的真实框架状态对齐，避免“实验结论还是旧的，但代码已经变了”带来的混淆

需要特别注意：

- 历史实验里曾经存在 `case retrieval`
- 当前代码里，这个机制已经被移除

所以这份文档会明确区分：

- 已完成实验告诉了我们什么
- 当前框架现在到底长什么样

## 2. 已完成实验得到的稳定结论

### 2.1 Base 参考线

历史完成实验中的 Base-only test 参考值：

- MAE = `501.103628`

### 2.2 已完成的关键 DELTA 运行

- `[2024-nswelecLOAD-tinynews-margin0]`
  - `delta_freeze_feature_modules = 1`
  - `delta_non_degrade_lambda = 1.0`
  - `delta_non_degrade_margin = 0.0`
  - 当时 retrieval 还存在
  - test MAE = `502.211778`

- `[2024-nswelecLOAD-tinynews-unfreeze]`
  - `delta_freeze_feature_modules = 0`
  - `delta_non_degrade_margin = 0.003`
  - 当时 retrieval 还存在
  - test MAE = `503.096264`

- `[2024-nswelecLOAD-tinynews-retrievaloff-current]`
  - 同样是保守 DELTA 配置
  - retrieval 关闭
  - test MAE = `502.203767`

### 2.3 结论一：冻结 DELTA feature modules 是正确的

`unfreeze` 明显更差。

解释：

- 当 DELTA 已经被收缩成一个比较保守的 residual 修正器后
- 继续训练 feature modules 带来的更多是漂移，而不是有价值的表达增强

结论：

- 保留 `delta_freeze_feature_modules = 1`

### 2.4 结论二：non-degrade 要开，但 margin 不应该设成正数

已完成实验表明更实用的组合是：

- `delta_non_degrade_lambda = 1.0`
- `delta_non_degrade_margin = 0.0`

解释：

- non-degrade 保护本身有价值
- 但要求 DELTA “不仅别变差，而且要明显优于 BASE” 这件事，在当前任务上并没有带来更好的 test 泛化

结论：

- 保留 non-degrade
- 保持 `margin = 0.0`

### 2.5 结论三：case retrieval 净收益极低

这是历史消融里最清晰的一个结论。

同一保守 DELTA 框架下：

- retrieval 开启：test MAE = `502.211778`
- retrieval 关闭：test MAE = `502.203767`

差值只有：

- 约 `0.008` MAE

解释：

- retrieval 确实在运行
- 但几乎没有带来可观的净收益
- 反而更像是在增加复杂度和噪声

结论：

- `case retrieval` 不值得继续保留
- 因此后来已经从框架中整体移除，而不是仅仅保持为一个低价值可选项

## 3. 当前代码里的真实框架状态

下面这一节描述的是“当前代码”而不是“旧实验”。

### 3.1 已被移除的机制

`case retrieval` 已经从当前框架中移除：

- 没有 case bank build 阶段
- 没有 retrieval feature 分支
- 没有 retrieval KNN prior
- 没有 retrieval 专用 CLI 参数
- 没有 retrieval 专用 wrapper 脚本

因此，今后所有结论都不应该再写成“当前框架主要问题是 retrieval”，因为它已经不在主动代码路径里了。更准确的说法应该是：

- 历史实验证明 retrieval 低价值
- 当前框架已经按这个结论完成了清理

### 3.2 当前新闻预处理链

现在新闻相关处理在 DELTA 开始前会走统一链路：

1. 按当前新闻文件加载原始新闻
2. 对每条唯一新闻文本做 `refine`
3. 紧接着做结构化标签抽取
4. 把两者写进一份联合 cache

当前联合 cache 形态是：

- `news_doc_cache_<newsfile>.json`

它是一个标准 JSON array，每个 object 会保存：

- 原新闻文本
- 标题
- 日期
- URL
- refined news
- structured events
- 各类 cache key 和元信息

### 3.3 当前 cache 选择逻辑

当前脚本的逻辑已经简化为：

- 如果提供 `NEWS_DOC_CACHE_PATH`，就直接读取这一个联合 cache
- 如果没提供，就按新闻文件名去 shared cache 目录里找对应的 `news_doc_cache_<newsfile>.json`
- 如果找不到，才自动重建

这比以前分开控制 refine/structured read path 和 rebuild 的方式简单很多，也更不容易误用。

### 3.4 当前时间处理

之前的“强制 UTC”已经去掉了。

当前原则是：

- 按输入数据集里的时间解释新闻和样本时间
- 不再偷偷把新闻时间统一改成 UTC

这属于框架正确性修复，不是单纯的调参。

## 4. DELTA Stage 的输入与输出

这一节专门回答：DELTA stage 到底吃什么、吐什么。

需要分成两层来看：

- 系统级输入/输出
- 模型级输入/输出

### 4.1 系统级输入

整个 DELTA stage 在运行时，实际依赖这些输入：

- Base stage 训练好的 backbone checkpoint
- 当前样本的历史数值序列
- 当前样本的目标序列
- 当前样本时间窗下选中的新闻
- 联合新闻 cache 里的 `refined_news` 与 `structured_events`
- 全局 z-score 统计量

也就是说，DELTA 不是“直接拿原始新闻就出预测”，而是：

- 先有 Base 预测
- 再根据新闻相关特征去学一个修正量

### 4.2 模型级输入

当前 `tiny_news_ts` 这条 DELTA 模型路径，真正会接收这些张量输入：

- `ts_patches`
  - 历史数值序列切成 patch 之后的张量
- `ts_patch_mask`
  - patch 对应的 mask
- `structured_feats`
  - 由 `structured_events` 映射出来的固定维度结构化特征向量
- `refined_news_input_ids`
  - 合并后的 refined news token ids
- `refined_news_attention_mask`
  - 对应 attention mask
- `refined_news_doc_input_ids`
  - 逐条 refined news doc 的 token ids
- `refined_news_doc_attention_mask`
  - 对应 doc attention mask
- `refined_news_doc_mask`
  - 哪些 doc 有效

训练时还会额外给：

- `targets`
  - 用来计算预测损失

但要注意：

- 这些输入是否真的会对最终预测起数值作用，还取决于相应分支有没有打开

### 4.3 当前主脚本下真正生效的 DELTA 输入

按当前 `scripts/nswelecLOAD_2024_tinynews.sh` 的默认配置：

- `delta_model_variant = tiny_news_ts`
- `delta_structured_enable = 1`
- `delta_doc_direct_enable = 0`
- `delta_text_direct_enable = 1`
- `delta_text_fuse_lambda = 0.0`
- `final_gate_enable = 1`

因此，当前主脚本里对预测真正有直接数值作用的 DELTA 输入，主要是：

- 历史时序 patch
- 结构化新闻特征 `structured_feats`

而下面这些虽然仍会被构造出来，但当前默认不构成实际数值修正主通路：

- prompt 文本本身
- doc direct 分支
- text direct 分支的实际修正值  
  原因是 `delta_text_fuse_lambda = 0.0`

### 4.4 模型级输出

当前 DELTA 模型前向主要会输出：

- `pred`
  - DELTA 分支预测出来的修正量
- `rel_logits`
  - 最终 gate 相关的 logits
- 若干诊断量
  - 例如 `delta_gate_mean`
  - `structured_weight_mean`
  - `text_delta_mean_abs`
  - `doc_delta_mean_abs`

其中最重要的一点是：

- `pred` 不是最终预测值
- `pred` 是“DELTA 分支自己的修正量”

### 4.5 阶段级输出

整个 DELTA stage 最终输出的，才是融合后的最终预测。

在 trainer 里，DELTA 分支输出会和 Base 预测融合：

- `additive` 模式下：
  - `final_pred = base_pred + gate * delta_pred`
- `relative` 模式下：
  - `final_pred = base_pred * (1 + gate * delta_ratio)`

而当前 LOAD 主脚本使用的是：

- `delta_residual_mode = additive`

所以当前可以直接理解成：

- Base 先给出一条预测曲线
- DELTA 再给出一条“该加多少 / 该减多少”的修正曲线
- final gate 决定这条修正最终放大或缩小多少

### 4.6 训练时额外约束

当前 DELTA stage 的训练并不只是最小化最终预测误差，它还包含几条约束：

- `non-degrade loss`
  - 尽量不要让 DELTA 把 Base 改坏
- `sign loss`
  - 显式约束残差方向，鼓励 DELTA 学对“该加还是该减”
- 内部保守 gate 初始化
  - 让 DELTA 一开始不要改得太猛

这也是当前框架被称作“保守 DELTA”的主要原因。

## 5. 当前 LOAD 主脚本默认开启了什么

当前 `scripts/nswelecLOAD_2024_tinynews.sh` 默认配置的核心风格是：

- Base + DELTA 两阶段
- DELTA 使用 `tiny_news_ts`
- final gate 开启
- internal delta gate 保守初始化
- feature modules 冻结
- non-degrade 打开
- sign supervision 打开
- `news_refine_mode = api`
- `news_structured_mode = api`
- `delta_structured_enable = 1`
- unified `NEWS_DOC_CACHE_PATH` 可直接复用现有新闻 cache

同时，当前默认没有打开的重机制包括：

- case retrieval
- doc direct 分支
- KNN prior

而 `text direct` 虽然在配置上是 enable，但因为：

- `delta_text_fuse_lambda = 0.0`

所以当前默认并不会把这条分支作为真正的数值修正主通道。

## 6. 当前最可靠的默认理解

如果只用一句话概括当前框架，可以写成：

**现在的 LOAD DELTA 框架，是一个“Base 预测 + 保守 residual 修正”的两阶段框架；新闻主要先被 API refine，再抽成结构化标签，随后以 structured feature 的形式进入 DELTA，而不是再通过 retrieval 或 prompt 文本去主导预测。**

## 7. 当前建议默认方向

当前最推荐保留的方向：

- `delta_freeze_feature_modules = 1`
- `delta_non_degrade_lambda = 1.0`
- `delta_non_degrade_margin = 0.0`
- `delta_sign_lambda > 0`
- `delta_structured_enable = 1`
- 使用 unified news-doc cache

当前不建议优先回头重新加回来的东西：

- case retrieval
- retrieval-like KNN prior
- doc direct 分支
- 单纯指望 prompt 模板改写来改善 DELTA

## 8. 下一步最有建设性的方向

在当前 retrieval-free 框架下，下一步最值得继续验证的是：

1. structured feature 这条主通路是否能稳定超过 BASE
2. final gate 与 sign / non-degrade 约束是否还需要继续细化
3. 当前新闻信号如果仍然不够强，问题究竟在“新闻本身”，还是在“structured 映射方式”

如果这条 retrieval-free、structured-first 的 DELTA 线最终仍然稳定打不过 BASE，那么更诚实的结论应该是：

- 当前 TinyNews 信号对 LOAD 任务的增益还不够强

而不是再回去重新把 retrieval 加回来。
