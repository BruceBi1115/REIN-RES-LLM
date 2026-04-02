# 框架工作逻辑说明

## 这份文档是给谁看的

这份文档面向想理解框架整体工作方式、但不想先扎进源码细节的人。

它描述的是当前仓库里的真实状态，而不是早期版本的旧设计。

## 一句话概括

这个框架当前做的事可以概括为：

- 先用 `Base` 模型只根据历史时间序列做一个基线预测
- 再用 `DELTA` 模型结合新闻信息去修正 Base 还没解释好的残差
- 如果开启 `SignNet`，就先额外学一个外部残差状态分类器：
  在 `additive` 模式下学残差正负，
  在 `relative` 模式下学 `amplify / neutral / shrink`

最终可以理解为：

- `additive` 模式：`最终预测 = Base 预测 + DELTA 修正`
- `relative` 模式：先定义
  `q = (target_raw - base_raw) / scale_raw`
  其中 `scale_raw = max(abs(base_raw), floor_or_eps)`
  然后：
  `最终预测 = base_raw + q_hat * scale_raw`

如果这时还启用了外部 `SignNet`，那么在当前版本里：

- `additive` 模式下，`SignNet` 主要决定 DELTA 修正更像正向还是负向
- `relative` 模式下，`SignNet` 输出 3 类状态：
  `amplify / neutral / shrink`
  并通过 softmax 概率生成一个状态分数
  `state_score = P(amplify) - P(shrink)`
  然后让 `DELTA` 预测的 `|q_hat|` 变成有符号 `q_hat = |q_hat| * state_score`

## 当前代码结构

现在主逻辑已经不再堆在一个大文件里，而是拆成了几个模块目录：

- `run.py`
  命令行入口，负责解析参数并启动训练。

- `src/base/`
  Base 阶段相关逻辑。
  这里负责环境准备、数据读取、Base 训练、Base 验证、主流程编排。

- `src/delta/`
  DELTA 阶段相关逻辑。
  这里负责 DELTA 的损失、训练、验证、测试和最终残差修正。

- `src/signnet/`
  SignNet 相关逻辑。
  这里负责 SignNet 模型定义、训练、校准和外部符号信号输出。

- `src/refine/`
  refined news cache 和 structured news cache 的读取、构建、校验、写回逻辑。

- `src/base_delta_decoouple_trainer.py`
  现在只是兼容导出层。
  它把新的 `base/delta` 模块重新导出给旧调用路径使用，但真实主逻辑已经不在这里实现。

## 当前框架在解决什么问题

如果只看历史数值，模型通常能学到：

- 周期性
- 趋势
- 季节性
- 工作日与周末差异

但很多真实扰动来自外部事件，比如：

- 天气
- 政策
- 市场消息
- 行业新闻
- 公司新闻

这些事件常常会让“只看历史”的预测出现系统性误差。

所以当前框架的核心思想不是“让新闻从零开始预测数值”，而是：

- Base 学正常规律
- 新闻帮助解释和修正异常偏差

## 一次完整运行会发生什么

### 1. 启动与参数整理

程序从 `run.py` 进入后，会在 `src/base/stage.py` 里先做统一初始化。

当前版本在日志开头会打印两张 ASCII 双栏表：

- 参数表
- cache 决策表

也就是说，训练一开始就会把：

- 当前参数取值
- refined news cache 的路径
- cache 是 `read_only`、`build_mode` 还是 `disabled`
- 是否检测到 API key

直接写进日志。

### 2. 读取时间序列数据

框架会读取：

- `train_file`
- `val_file`
- `test_file`

并且用训练集计算全局 z-score 统计量：

- `mu_global`
- `sigma_global`

后续 Base、SignNet、DELTA 里凡是用到 `z` 空间，都会基于这个训练集统计量。

当前日志里的 `[DATA_RANGE][TRAIN/VAL/TEST]` 只显示原始 split 的真实时间范围，不再混入补历史窗口时带上的上下文区间。

### 3. 构造样本

当前样本的核心字段包括：

- `history_value`
- `target_value`
- `history_times`
- `target_times`
- `target_time`
- `series_id`

如果数据是多序列面板数据，比如同一天有多个 ticker，那么必须传 `id_col`。

`id_col` 的作用是：

- 按不同序列分别切历史窗口和预测窗口
- 给 `val/test` 分别补各自序列的历史前缀
- 避免把多条序列错误地拼成一条

所以：

- 单序列数据可以不传 `id_col`
- 多序列数据必须传

### 4. Base 阶段

Base 阶段只看历史数值，不看新闻。

输入本质上是：

- `history_value` 的 z-score 版本 `history_z`

输出是：

- 对未来 `horizon` 步的基线预测 `base_pred_z`

Base 的作用很明确：

- 学“如果完全不看新闻，我会怎么预测”

Base 的 checkpoint 会在验证集上按设定指标做早停和选优。

### 5. 为 DELTA 准备新闻输入

对每个样本，框架会根据样本的目标时间和新闻窗口规则去选新闻。

当前逻辑大致是：

1. 先按时间窗取候选新闻
2. 再按策略筛选或 rerank
3. 然后把选中的新闻交给 refine/cache 逻辑

最终会整理出几类 DELTA 可直接使用的新闻输入：

- `structured_events`
  聚合后的结构化事件

- `structured_doc_events`
  文档级结构化事件
  这个对象现在主要用于调试、日志和 cache 侧保留文档粒度信息，不作为 DELTA 或 SignNet 的直接张量输入

- `structured_feats`
  结构化事件向量
  这是当前新闻真正进入 DELTA 和 SignNet 的主输入形式

- `news_counts`
  当前样本最终使用到的新闻数量
  这个值目前只保留给日志、诊断和部分 DELTA 侧权重逻辑，不再作为 SignNet 或 DELTA 模型的直接数值输入

换句话说，refine 后的文本结果现在不只是给结构化抽取用。
如果开启 `delta_temporal_text_enable`，这些文本还会进入共享 `TemporalTextTower`；
在默认 `summary_gated` 架构下，它会给 DELTA 提供 `text_summary + patch_context`，也会给外部 `SignNet` 提供 `text_summary`；
在 `delta_multimodal_arch=plan_c_mvp` 时，外部 `SignNet` 仍然只直接吃 `text_summary + text_strength`，但会在内部额外走一层 `RegimeRouter + ResidualExpertMixture` 做路由式方向推理。

## 关键节点输入清单

下面这部分专门回答四个问题：

- 这个节点吃什么输入
- 输入的格式是什么
- 哪些参数会影响这个节点
- 参数是怎么影响的，最终影响到框架的哪一部分

### 节点 1：启动与环境整理 `run.py -> setup_env_and_data()`

- 输入：命令行参数对象 `args`
- 关键输入格式：
  `taskName: str`，`save_dir: str`，`stage: str`
  `train_file/val_file/test_file: str`
  `time_col/value_col/id_col: str`
  `news_path/news_doc_cache_path: str`
- 关键参数：
  `taskName`
  `stage`
  `save_dir`
  `base_backbone`
  `news_path`
  `news_doc_cache_path`
  `news_refine_mode`
  `news_structured_mode`
  `news_api_key_path`
- 参数如何影响：
  `taskName` 和 `save_dir` 决定 checkpoint、日志、测试输出写到哪里
  `stage` 决定这次只跑 Base、只跑 DELTA，还是整条链一起跑
  `base_backbone` 会进入后续 Base 模型构造
  `news_path` 和 `news_doc_cache_path` 决定是否启用新闻链，以及 refined cache 是 `disabled`、`read_only` 还是 `build_mode`
  `news_refine_mode/news_structured_mode/news_api_key_path` 会影响是否需要 API adapter，以及在缺 key 时是否启动前直接报错
- 最终影响到：
  日志开头的参数表
  cache 决策表
  后续整个运行链是否允许读新闻、构建 cache、调用 API

### 节点 2：时间序列数据读取与切分拼接 `setup_env_and_data()`

- 输入：
  `train_df`
  `val_df`
  `test_df`
- 关键输入格式：
  每个表至少要有 `time_col` 和 `value_col`
  如果是多序列面板数据，还要有 `id_col`
  文件可以是 `csv` 或 `parquet`
- 关键参数：
  `train_file`
  `val_file`
  `test_file`
  `time_col`
  `value_col`
  `id_col`
  `dayFirst`
  `history_len`
  `horizon`
  `stride`
  `batch_size`
- 参数如何影响：
  `time_col/value_col/id_col` 决定框架从哪几列取时间、数值和序列身份
  `dayFirst` 决定时间解析策略，直接影响日期是否会被读反
  `history_len/horizon/stride` 决定样本窗口长度和滑动步长
  `id_col` 决定 `val/test` 补历史前缀时是按整表补，还是按每个序列单独补
  `batch_size` 决定 DataLoader 的 batch 规模
- 最终影响到：
  全局 z-score 统计量 `mu_global/sigma_global`
  样本条数
  每个样本的 `history_value/target_value`
  `val/test` 是否有合法窗口可评估

### 节点 3：新闻源加载 `load_news()`

- 输入：
  `news_path`
- 关键输入格式：
  一个 JSON 数组文件
  每条新闻通常包含 `title/date/url/content`
  其中 `date` 对应 `news_time_col`
  其中正文列对应 `news_text_col`
- 关键参数：
  `news_path`
  `news_time_col`
  `news_text_col`
  `news_tz`
  `dayFirst`
- 参数如何影响：
  `news_path` 决定是否启用新闻源
  `news_time_col` 决定从哪一列读发布时间
  `news_tz` 决定新闻时间是否本地化或转换时区
  `dayFirst` 会影响非 ISO 日期解析
- 最终影响到：
  `news_df` 的时间排序
  新闻 identity 去重结果
  后续 candidate news 检索和 cache 精确匹配

### 节点 4：refined news cache 决策与读取 `src/refine/cache.py`

- 输入：
  `news_df`
  `args`
  当前样本对应的 `news_metas`
- 关键输入格式：
  cache 记录按 `title + date + url` 做 identity
  单条 unified doc cache 记录里保留 `refined_news`、`structured_events`、`title/date/url/news_path` 等字段
- 关键参数：
  `news_doc_cache_path`
  `news_path`
  `news_refine_mode`
  `news_structured_mode`
  `news_api_model`
  `news_api_key_path`
  `dayFirst`
- 参数如何影响：
  `news_doc_cache_path` 显式指定 unified cache 文件
  `news_path` 会触发自动发现 `checkpoints/_shared_refine_cache/news_doc_cache_{新闻文件名}.json`
  `news_refine_mode/news_structured_mode` 决定 build 时走本地逻辑还是 API 逻辑
  `news_api_key_path` 和环境变量一起决定 API key 是否可用
  `dayFirst` 会影响 cache 记录里日期的标准化匹配
- 最终影响到：
  当前运行是 `read_only`、`build_mode` 还是 `disabled`
  missing 新闻是报错、跳过，还是允许新建条目
  refine 和 structured 结果是否会被写回 cache

### 节点 5：新闻筛选与样本级新闻构造 `build_batch_inputs()`

- 输入：
  `batch`
  `news_df`
  `templates`
  `tokenizer`
  `global_zstats`
- 关键输入格式：
  `batch["history_value"]`: `list[list[float]]`
  `batch["target_value"]`: `list[list[float]]`
  `batch["target_time"]`: 每个样本一个时间戳
  `news_df`: `DataFrame`
- 关键参数：
  `news_window_days`
  `news_topM`
  `news_topK`
  `utility_rerank_enable`
  `token_budget`
  `token_budget_news_frac`
  `delta_include_structured_news`
  `delta_structured_enable`
  `delta_structured_feature_dim`
  `news_refine_mode`
  `news_structured_mode`
  `news_dropout`
  `force_no_news`
  `delta_temporal_text_enable`
  `delta_temporal_text_source`
  `delta_temporal_text_max_len`
  `delta_temporal_text_per_step_topk`
- 参数如何影响：
  `news_window_days` 决定候选新闻回看窗口
  `news_topM` 决定初始 candidate cap
  `news_topK` 决定最终保留多少条新闻
  `utility_rerank_enable` 决定筛选后是否再做 utility rerank
  `token_budget * token_budget_news_frac` 决定 refine 文本合并时的 token budget
  `delta_include_structured_news` 决定 prompt 里是否把结构化事件展开成文本
  `delta_structured_enable` 决定是否真的去构造 `structured_feats`
  `delta_structured_feature_dim` 决定结构化向量最终维度
  `news_dropout` 只会在需要构造 prompt 文本时随机把 `news_str` 拿掉；在当前 DELTA / SignNet 主路径里通常不会改变真正送进模型的张量输入
  `force_no_news` 会强制当前 batch 走“无新闻”路径
  `delta_temporal_text_enable` 决定是否额外构造按 history step 对齐的文本辅助序列
  `delta_temporal_text_source` 决定这条文本辅助序列编码的是原新闻正文 `raw`，还是 doc-cache 里的 `refined` 摘要
  `delta_temporal_text_max_len / delta_temporal_text_per_step_topk` 决定每个时间步文本片段的截断长度，以及每步最多合并多少条已发生新闻摘要
- 最终影响到：
  `structured_events`
  `structured_doc_events`
  `structured_feats`
  `news_counts`
  `prompt_texts`
  `temporal_text_ids / temporal_text_attn / temporal_text_step_mask`
  当前样本最终能给 SignNet 提供的 `structured_feats`，以及在开启 temporal text 时能给 DELTA 提供的时间对齐文本辅助信号

### 节点 6：Base 输入与输出 `train_base_stage()`

- 输入：
  `history_z`
- 关键输入格式：
  `history_z: torch.FloatTensor`，形状大致是 `(B, L)`
  其中 `L = history_len`
- 关键参数：
  `base_backbone`
  `history_len`
  `horizon`
  `lr`
  `weight_decay`
  `scheduler`
  `patience`
- 参数如何影响：
  `base_backbone` 决定 Base 是 `dlinear` 还是 `mlp`
  `history_len` 决定输入长度
  `horizon` 决定一次输出多少未来步
  优化器和 scheduler 参数影响收敛速度、稳定性和 early stop 结果
- 最终影响到：
  `base_pred_z`
  Base checkpoint 选优
  后续 SignNet 和 DELTA 的残差学习基线

### 节点 7：SignNet 输入与输出 `_train_external_signnet()`

这一节最重要的一句话是：

- `SignNet` 现在真正进入模型的主输入是：
  `history_z`
  `base_pred_z`
  `structured_feats`
  以及可选的 temporal-text 特征

也就是说，当前版本里：

- 不输入 `history_raw`
- 不输入新闻数量 `news_counts`
- 不输入 prompt 文本
- 不直接输入原始 prompt token 序列

- 输入：
  `history_z`
  `base_pred_z`
  `structured_feats`
  可选的 `text_summary / text_strength`
- 关键输入格式：
  `history_z: torch.FloatTensor (B, L)`
  `base_pred_z: torch.FloatTensor (B, H)`
  `structured_feats: torch.FloatTensor (B, D)`
  `text_summary: torch.FloatTensor (B, hidden_text)` 或 `None`
  `text_strength: torch.FloatTensor (B, 1)` 或 `None`
  其中 `D = delta_structured_feature_dim`
- 关键参数：
  `delta_sign_external_epochs`
  `delta_sign_external_hidden`
  `delta_sign_external_dropout`
  `delta_sign_external_lr`
  `delta_sign_eps`
  `delta_sign_external_use_news_weighting`
  `delta_sign_external_use_residual_weighting`
  `delta_sign_external_use_pos_weight`
  `delta_sign_external_pos_weight_floor`
  `delta_sign_external_pos_weight_clip`
  `delta_sign_external_tau`
  `delta_multimodal_arch`
  `delta_multimodal_fuse_lambda`
- 参数如何影响：
  `delta_sign_external_hidden/dropout` 决定 SignNet 宽度和正则强度
  `delta_sign_external_epochs/lr` 决定训练时长和学习率
  `delta_sign_eps` 决定哪些残差位置被视为“符号有意义”
  `delta_sign_external_use_news_weighting` 会根据结构化新闻强弱调节样本权重
  `delta_sign_external_use_residual_weighting` 会根据残差幅度调节 horizon 位置权重
  `delta_sign_external_use_pos_weight` 以及 floor/clip 只在 additive 的二分类监督里有效
  `delta_sign_external_tau` 只在 additive 模式里影响 `logits -> tanh(sign_soft)` 的温度
  `delta_multimodal_arch` 决定 SignNet 走原来的 summary-based MLP，还是切到 Plan C 的 `RegimeRouter + ResidualExpertMixture` 版本
  `delta_multimodal_fuse_lambda` 决定 Plan C 下 expert mixture 对最终方向表征的注入强度
  `cleaned_residual_enable` 打开后，relative 模式下只会先对 `q_target` 做 horizon 平滑，再据此构造 3 类状态标签
  `cleaned_residual_smooth_alpha` 控制 cleaned residual 在 horizon 上的 EWMA 平滑强度
  `cleaned_residual_structured_mix` 主要影响 additive 模式的 cleaned residual 模板混合；relative 模式下不再把结构化模板直接写进监督目标
- 最终影响到：
  additive 下的 `sign_logits / sign_soft`
  relative 下的 `state_logits / state_score`
  DELTA 在 relative 模式下的状态选择稳定性

#### SignNet 的输入分别是什么意思

- `history_z`
  这是当前样本历史窗口的 z-score 序列。
  形状是 `(B, L)`，其中 `B` 是 batch size，`L = history_len`。
  它回答的是：“最近这段历史数值轨迹长什么样？”

- `base_pred_z`
  这是 Base 模型对未来 `H` 步的基线预测。
  形状是 `(B, H)`，其中 `H = horizon`。
  它回答的是：“如果完全不看新闻，Base 觉得未来会怎么走？”

- `structured_feats`
  这是当前样本新闻经过 structured extraction 之后得到的固定维度向量。
  形状是 `(B, D)`，其中 `D = delta_structured_feature_dim`。
  它回答的是：“当前新闻在方向、强度、持续性、相关性、置信度和事件类型上长什么样？”

- `text_summary`
  这是共享 `TemporalTextTower` 从 `temporal_text_ids / temporal_text_attn / temporal_text_step_mask` 编码后，再做 patch-level 聚合和 masked mean pool 得到的文本摘要向量。
  形状大致是 `(B, hidden_size)`，另外还会带一个 `(B, 1)` 的 `text_strength` 表示当前样本文本覆盖度。
  它回答的是：“如果把时间对齐新闻文本先编码成一条统一表征，这个样本现在最重要的文本上下文是什么，以及这条文本信号强不强？”

#### SignNet 的输入分别从哪里来

- `history_z`
  来自 `_z_batch_tensors(batch, args, global_zstats)`。
  原始来源是样本里的 `history_value`，再用训练集统计量做 z-score。

- `base_pred_z`
  来自已经训练好的 Base 模型。
  也就是先把 `history_z` 喂进 Base，再得到未来窗口的 `base_pred_z`。

- `structured_feats`
  来自 `build_delta_batch_inputs() -> build_batch_inputs()`。
  先选新闻，再读 refined/structured cache，再把 `structured_events` 数值化成固定长度向量。

- `text_summary`
  来自共享 `TemporalTextTower`。
  它的原始张量来源仍然是 `build_delta_batch_inputs()` 产出的 `temporal_text_*`，只是不会把 token 序列直接送进 SignNet，而是先在 tower 里压成一个 pooled summary。

#### 哪些参数会改变 SignNet 的输入本身

- `history_len`
  直接改变 `history_z` 的长度 `L`。

- `horizon`
  直接改变 `base_pred_z` 的长度 `H`。

- `delta_structured_enable`
  决定是否真的构造并使用结构化新闻向量。
  关掉后，`structured_feats` 会退化成无信息或零向量，SignNet 基本只剩历史和 Base 预测可看。

- `delta_structured_feature_dim`
  直接改变 `structured_feats` 的维度 `D`。

- `force_no_news`
  会强制当前 batch 走无新闻路径。
  这会让该 batch 的 `structured_feats` 变弱甚至接近零。

- `news_window_days / news_topM / news_topK / utility_rerank_enable`
  这些参数不会改变张量形状，但会改变当前样本最终选到哪些新闻，因此会改变 `structured_feats` 的数值内容。

- `news_refine_mode / news_structured_mode / news_doc_cache_path`
  这些参数会影响 refined/structured 结果从哪里来、是否能读到、是否跳过缺失新闻，因此也会影响 `structured_feats` 的具体值。

- `delta_temporal_text_enable`
  决定 SignNet 是否还能额外看到一条来自共享 `TemporalTextTower` 的文本支路。
  无论是默认架构还是 Plan C，直接进入 SignNet 的都主要是 `text_summary + text_strength`。

- `delta_temporal_text_source`
  决定这条 `text_summary` 是从原新闻正文 `raw` 还是从 `refined` 摘要编码出来的。

- `temporal_text_model_id`
  决定共享 `TemporalTextTower` 使用哪个 tokenizer/encoder，因此会改变 `text_summary` 的表征空间。

- `delta_temporal_text_max_len / delta_temporal_text_per_step_topk`
  不会改变 `history_z / base_pred_z / structured_feats` 的形状，但会改变 `text_summary` 的信息密度和每步聚合到多少文本。

- `delta_multimodal_arch / delta_multimodal_fuse_lambda`
  这组参数决定 SignNet 是只做普通 summary-based MLP 融合，还是在 pooled summary 之上再做 Plan C 的 regime routing 和 expert mixture。

#### SignNet 的输入不会再被什么影响

当前版本里，下面这些东西已经不会再直接进入 SignNet：

- prompt token
- 新闻数量 `news_counts`
- `temporal_text_ids / temporal_text_attn / temporal_text_step_mask` 这些逐 token 张量本身
- 单条 doc 级文本 embedding 列表

也就是说，SignNet 现在可以通过共享 tower 间接看到文本：

- 在默认 `summary_gated` 架构下，看到的是 pooled 过的 `text_summary`
- 在 `plan_c_mvp` 架构下，仍然直接看到 pooled `text_summary + text_strength`，但会再基于历史波动、Base 分散度、文本强度和结构化新闻强度去做 route-aware 推理

但它仍然不会直接吃原始新闻 token 序列本身。

### 节点 8：DELTA 样本输入 `build_delta_batch_inputs()`

这一节也可以先记一句话：

- `build_delta_batch_inputs()` 会产出很多字段
- 默认真正送进 DELTA 主模型前向的核心输入是：
  `ts_patches`
  `ts_patch_mask`
  `structured_feats`
- 如果开启 `delta_temporal_text_enable=1`，还会额外送：
  `temporal_text_ids`
  `temporal_text_attn`
  `temporal_text_step_mask`

- 输入：
  `build_batch_inputs()` 的输出
- 关键输入格式：
  `ts_patches: torch.FloatTensor (B, P, patch_len)`
  `ts_patch_mask: torch.FloatTensor (B, P)`
  `targets_z: torch.FloatTensor (B, H)`
  `structured_feats: torch.FloatTensor (B, D)`
  `temporal_text_ids: torch.LongTensor (B, L, T)`，其中 `T` 是单步文本最大 token 长度
  `temporal_text_attn: torch.LongTensor (B, L, T)`
  `temporal_text_step_mask: torch.LongTensor (B, L)`
  `news_counts: torch.FloatTensor (B,)`
  `prompt_texts: list[str]`
- 关键参数：
  `patch_len`
  `patch_stride`
  `history_len`
  `horizon`
  `delta_structured_feature_dim`
  `delta_temporal_text_enable`
  `delta_temporal_text_source`
  `delta_temporal_text_max_len`
  `delta_temporal_text_per_step_topk`
- 参数如何影响：
  `patch_len/patch_stride` 决定历史序列如何被切成 patch
  `history_len` 决定 patch 总覆盖长度
  `horizon` 决定 `targets_z` 长度
  `delta_structured_feature_dim` 决定结构化新闻特征维度
  `delta_temporal_text_enable` 决定是否真的生成时间对齐文本辅助输入
  `delta_temporal_text_source` 决定 `temporal_text_*` 来自原新闻正文还是 refined 摘要
  `delta_temporal_text_max_len / delta_temporal_text_per_step_topk` 决定 `temporal_text_*` 的信息密度
- 最终影响到：
  DELTA 模型真正吃到的张量形状
  sign、state、magnitude head 的输出形状

#### `build_delta_batch_inputs()` 产出的字段里，哪些是给 DELTA 真正前向用的

- `ts_patches`
  会直接送进 DELTA 主模型

- `ts_patch_mask`
  会直接送进 DELTA 主模型

- `structured_feats`
  会直接送进 DELTA 主模型

- `temporal_text_ids / temporal_text_attn / temporal_text_step_mask`
  只在 `delta_temporal_text_enable=1` 时直接送进 DELTA 主模型
  这是一条按历史时间步对齐的辅助文本分支

- `targets_z`
  不进入 DELTA 前向本身，但会进入训练损失

- `news_counts`
  当前不作为 DELTA 主模型的直接输入
  它主要用于日志、调试，以及少数辅助统计逻辑

- `prompt_texts`
  当前不进入 DELTA 主模型
  而且当前 DELTA / SignNet 这条 helper 调用会设置 `build_prompt_inputs=False`，所以这个字段通常只是兼容保留，不是主路径输入

- `structured_events / structured_doc_events`
  当前不直接作为 DELTA 张量输入
  它们主要保留给日志、调试、cache 追踪和结构化信息检查

### 节点 9：DELTA 模型前向 `TinyNewsTSRegressor.forward()`

这一节最重要的结论是：

- DELTA 默认真正吃的是“时间序列 patch + patch mask + structured news vector”
- 如果开启 `delta_temporal_text_enable`，它还会额外吃一条按 history step 对齐的新闻文本辅助序列
- 它不直接吃整段 prompt token，也不直接吃新闻数量

- 输入：
  `ts_patches`
  `ts_patch_mask`
  `structured_feats`
  可选的 `temporal_text_ids / temporal_text_attn / temporal_text_step_mask`
- 关键输入格式：
  `ts_patches: torch.FloatTensor (B, P, patch_len)`
  `ts_patch_mask: torch.FloatTensor (B, P)`
  `structured_feats: torch.FloatTensor (B, D)` 或 `None`
  `temporal_text_ids: torch.LongTensor (B, L, T)` 或 `None`
  `temporal_text_attn: torch.LongTensor (B, L, T)` 或 `None`
  `temporal_text_step_mask: torch.LongTensor (B, L)` 或 `None`
- 关键参数：
  `tiny_news_hidden_size`
  `delta_clip`
  `delta_sign_tau`
  `delta_mag_max`
  `delta_structured_enable`
  `delta_structured_feature_dim`
  `delta_temporal_text_enable`
  `delta_temporal_text_source`
  `temporal_text_model_id`
  `delta_temporal_text_dim`
  `delta_temporal_text_fuse_lambda`
  `delta_temporal_text_freeze_encoder`
  `delta_multimodal_arch`
  `delta_multimodal_fuse_lambda`
- 参数如何影响：
  `tiny_news_hidden_size` 决定 DELTA 主隐藏维度
  `delta_clip` 决定最终 delta 修正是否做 `tanh` 截断
  `delta_sign_tau` 决定内部 sign logits 的软符号温度
  `delta_mag_max` 决定 magnitude 上限
  `delta_structured_enable/delta_structured_feature_dim` 决定是否读取以及如何解释结构化新闻向量
  `delta_temporal_text_enable` 决定是否启用时间对齐文本辅助分支
  `delta_temporal_text_source` 决定时间对齐文本分支编码 `raw` 还是 `refined` 新闻文本
  `temporal_text_model_id` 决定这条文本辅助分支实际使用哪个 HF tokenizer/encoder；如果不单独指定，就回退到主 `tokenizer / base_model`
  `delta_temporal_text_dim / delta_temporal_text_fuse_lambda / delta_temporal_text_freeze_encoder` 决定这条文本辅助分支的投影维度、融合强度，以及编码器是冻结前向还是参与训练
  `delta_multimodal_arch` 决定 DELTA 走原来的 summary/gated 路径，还是切到 Plan C 的 regime-router expert-mixture 路径
  `delta_multimodal_fuse_lambda` 决定 Plan C 路径里各个 residual expert 对最终 residual_context 的注入强度
- 最终影响到：
  `pred`
  `sign_logits`
  `state_logits`
  `magnitude`
  `delta_init`

#### DELTA 的这些真正输入分别是什么意思

- `ts_patches`
  这是把历史 z-score 序列按 `patch_len` 和 `patch_stride` 切开的 patch 张量。
  形状是 `(B, P, patch_len)`，其中 `P` 是 patch 数。
  它回答的是：“把历史轨迹切成多个局部片段之后，每段局部模式长什么样？”

- `ts_patch_mask`
  这是 patch 有效位掩码。
  形状是 `(B, P)`。
  它回答的是：“哪些 patch 是真实数据，哪些是 padding 补出来的？”

- `structured_feats`
  这是新闻结构化特征向量。
  形状是 `(B, D)`。
  它回答的是：“当前新闻侧提供了什么方向、强度、持续性和事件类型信息？”

- `temporal_text_ids / temporal_text_attn / temporal_text_step_mask`
  这是可选的时间对齐文本辅助输入。
  它不是“整段 prompt”，而是对每个历史时间步单独整理“在该时间点之前已经发生的新闻文本”，再把这些文本序列送进共享 `TemporalTextTower` 编码后按 patch 对齐融合进 DELTA。

#### 这条 temporal text 分支到底是怎么工作的

- 第一步，`build_batch_inputs()` 仍然会先做样本级新闻选择。
  也就是说，temporal text 分支不是重新遍历全量 `news_df`；它复用的是当前样本已经筛好的 `selected_news_metas`，以及与之对齐的原新闻正文或 refined 文本。
  所以 `news_window_days / news_topM / news_topK / utility_rerank_enable / force_no_news` 这些参数，仍然会先决定这条文本分支“看得到哪些新闻”。

- 第二步，`_build_temporal_text_series_for_sample()` 会遍历每个 history step 的时间戳 `step_ts`。
  对于当前这个 `step_ts`，它只保留 `doc_ts <= step_ts` 的新闻文档。
  这意味着每个历史步只能看到“在那个时刻之前已经发生”的新闻，不会偷看未来新闻。

- 第二步半，要先看 `delta_temporal_text_source`：
  如果设成 `refined`，这条支路编码的是 doc-cache 里的 refined 新闻摘要；
  如果设成 `raw`，编码的就是当前样本已选中的原新闻正文。

- 第三步，对当前 step 可见的文档，它会按时间从近到远排序，取最近的 `delta_temporal_text_per_step_topk` 条，
  再用 `_merge_refined_news_docs()` 合成一段短文本，并按 `delta_temporal_text_max_len` 做 tokenizer-level truncation。
  如果某个 history step 没有任何可见新闻，这个 step 对应的文本就是空字符串。

- 第四步，`_tokenize_temporal_text_series()` 会把整批样本的 step 文本序列 pad 成：
  `temporal_text_ids: (B, L, T)`
  `temporal_text_attn: (B, L, T)`
  `temporal_text_step_mask: (B, L)`
  这里第二维的 `L` 本质上就是 history 长度。
  `temporal_text_step_mask[b, l] = 1` 表示这个历史步最终确实有非空文本 token。

- 第五步，进入共享 `TemporalTextTower` 之后，
  它会先把 `(B, L, T)` 展平成 `(B * L, T)`，
  再送进 `temporal_text_model_id` 对应的 HF `AutoModel` 文本编码器。
  编码后，它不是取生成结果，而是对 token hidden states 做 attention-mask mean pooling，
  从而得到“每个 history step 一个文本向量”，然后再投影到 `delta_temporal_text_dim`。

- 第六步，`TemporalTextTower` 会继续用和时间序列分支相同的 `patch_len / patch_stride`，
  把 step-level 文本向量继续聚成 patch-level 文本上下文，并再 pooled 成一个样本级 `text_summary`。
  所以这条分支最终不是直接对齐 horizon，也不是直接对齐整段 history，而是和 `ts_patches` 在 patch 轴上对齐。

- 第七步，要看 `delta_multimodal_arch`：

  - 如果是默认的 `summary_gated`，
    patch-level 文本上下文会先过一个 gate，再做 residual add：
    `ts_feat = ts_feat + lambda * gate(ts_feat, text_patch_context) * text_patch_context`

  - 如果是 `plan_c_mvp`，
    patch-level 文本上下文本身不再直接进入一个额外的 token-level 交互块。
    当前实现会先照常得到 `pooled_ts`、`fused_news_context`、`text_summary * text_strength`，
    再把它们拼成一个基础 `residual_base`；
    然后根据历史波动、文本强度、结构化新闻强度和组合后的 news strength 构造 route scalars，
    送进 `RegimeRouter` 输出 `none / trend / event / reversal / sparse` 五类路由概率；
    再由 `ResidualExpertMixture` 对多个 residual experts 做加权混合，得到新的 `residual_context`。
    这里的 `none` 路由本质上是 abstention 信号，会在后面压低修正置信度。

- 第八步，`text_summary` 现在也不只影响 patch 表征。
  无论哪种架构，它都会进入 DELTA 的 `residual_context`；
  在 `plan_c_mvp` 下，还会额外形成一个 `route_summary`，并通过 `route_mag_head` 和 abstention-aware `confidence_head` 去调节最终 `magnitude`。
  所以 temporal text 对 DELTA 的作用，已经从“辅助 patch-level 融合”扩展成：
  patch 融合 + pooled residual context + route-aware magnitude/confidence 调节。

- 这也解释了为什么它和 `structured_feats` 不是互斥关系：
  `temporal text` 分支是在 patch level 融合文本上下文，
  `text_summary` 会在 pooled/residual level 提供文本摘要上下文，
  `structured_feats` 分支则是在 pooled level 提供数值化新闻上下文。
  当前实现里，两条新闻支路可以同时打开。

- 还要特别区分：
  这条分支不等于 prompt 分支。
  它不直接读取整段 prompt token，也不直接把 merged prompt 文本送进 DELTA 主干。
  在当前 DELTA / SignNet 主路径里，`build_delta_batch_inputs()` 通常会关掉 prompt token 构造，但 temporal-text 分支仍然可以照常工作。

- `news_dropout` 主要影响 prompt 文本构造，通常不会把这条 temporal-text 张量支路一起丢掉。
  但 SignNet 现在可以通过共享 `TemporalTextTower` 间接吃到这条分支的 pooled `text_summary`。

#### DELTA 的这些真正输入分别从哪里来

- `ts_patches`
  来自 `build_batch_inputs()` 里的 `_make_patches(history_z, patch_len, patch_stride)`。
  本质上它来源于样本的 `history_value`，先做 z-score，再切 patch。

- `ts_patch_mask`
  和 `ts_patches` 同时生成。
  当历史长度不整齐或需要 pad 时，用来告诉模型哪些 patch 有效。

- `structured_feats`
  来自新闻筛选、refine、structured extraction 之后的数值化结果。
  本质上是 `structured_events -> feature vector`。

- `temporal_text_*`
  来自 `build_batch_inputs()` 里的 `_build_temporal_text_series_for_sample()` 和 `_tokenize_temporal_text_series()`。
  它会把“每个历史时间步之前可见的 refined news 摘要”整理成 `(B, L, T)` 张量，再在 DELTA 前向里编码并按 patch 聚合。

#### 哪些参数会改变 DELTA 的输入本身

- `history_len`
  先决定历史窗口长度，再间接影响能切出多少 patch。

- `patch_len`
  直接决定每个 patch 的长度。

- `patch_stride`
  直接决定 patch 之间的滑动步长，也会改变 patch 数量 `P`。

- `delta_structured_enable`
  决定 DELTA 是否接收结构化新闻输入。

- `delta_structured_feature_dim`
  直接决定 `structured_feats` 的维度 `D`。

- `news_window_days / news_topM / news_topK / utility_rerank_enable / force_no_news`
  这些参数不会改变 DELTA 时间序列 patch 的形状，但会改变 `structured_feats` 的具体值。

- `delta_temporal_text_enable`
  决定 DELTA 是否额外接收时间对齐文本辅助输入。

- `delta_temporal_text_max_len / delta_temporal_text_per_step_topk`
  不改变 patch 形状，但会改变 `temporal_text_*` 的内容密度和每步能看到多少文本信息。

#### DELTA 当前明确不直接吃什么

当前版本里，下面这些东西不会直接送进 `TinyNewsTSRegressor.forward()`：

- `history_z` 原始整段序列
- `history_raw`
- `base_pred_z`
- `prompt_texts`
- raw 新闻文本
- `news_counts`
- `structured_events` 原始 dict
- `structured_doc_events` 原始列表

需要特别注意：

- 如果开启 `delta_temporal_text_enable`，模型会额外直接吃“按 history step 对齐后的新闻文本 token 序列”
- 但它仍然不会直接吃“整段 prompt token”或“整段 merged 新闻文本”

### 节点 10：DELTA 训练损失与最终融合 `train_delta_stage()`

- 输入：
  `base_pred_z`
  `delta_pred`
  `sign_soft`
  `targets_z`
  可选的 `external_signnet`
- 关键输入格式：
  这些张量大多是 `torch.FloatTensor (B, H)`
- 关键参数：
  `residual_loss`
  `delta_residual_mode`
  `delta_relative_denom_floor`
  `delta_relative_ratio_clip`
  `cleaned_residual_enable`
  `cleaned_residual_smooth_alpha`
  `cleaned_residual_structured_mix`
  `delta_sign_external_tau`
- 参数如何影响：
  `residual_loss` 决定最终主损失用 `mse/mae/smooth_l1` 哪一种
  `delta_residual_mode` 决定最终融合是 `additive` 还是 `relative`
  `delta_relative_denom_floor` 决定相对残差目标 `q = (target_raw - base_raw) / scale_raw` 中 `scale_raw = max(abs(base_raw), floor_or_eps)` 的 floor
  `delta_relative_ratio_clip` 决定相对百分比 `q_hat` 是否截断；`<=0` 表示禁用
  `cleaned_residual_enable` 打开后，relative 模式只会把平滑后的 `q_target` 用于 3 类状态标签构造
  `cleaned_residual_smooth_alpha` 控制 cleaned residual 的 horizon 平滑
  `cleaned_residual_structured_mix` 主要影响 additive 模式的模板混合
  `delta_sign_external_tau` 会影响外部 SignNet 输出给 DELTA 时的软符号强度
- 最终影响到：
  DELTA 训练方向
  在 `relative` 模式下，当前 DELTA 的反向传播来自：
  `loss_final + loss_relative_mag`
  如果 `delta_sign_mode=internal`，还会再加 `loss_relative_state_ce`
  也就是说，旧版那些 sign/magnitude/null/counterfactual 辅助损失已经不再参与训练
  `final_pred`
  在 `additive` 模式下：`final_pred = base_pred + delta_corr`
  在 `relative` 模式下：先预测 `|q_hat|` 和状态分数，再得到有符号 `q_hat`
  然后：`final_pred_raw = base_pred_raw + q_hat * scale_raw`
  验证集选优
  测试集最终结果

### 节点 11：日志与调试输出

- 输入：
  `args`
  当前 batch 的统计值
  当前 epoch 的评估结果
- 关键输入格式：
  标量统计、路径字符串、时间范围字符串、结构化新闻调试记录
- 关键参数：
  `taskName`
  `stage`
  `delta_structured_enable`
  `delta_sign_mode`
  `residual_debug_csv_path`
- 参数如何影响：
  `taskName/stage` 决定日志和 debug 文件命名
  `delta_structured_enable` 会决定日志是否强调 structured 分支
  `delta_sign_mode` 会决定日志里是否出现 external signnet 相关信息
- 最终影响到：
  你能否从日志里快速定位
  当前机制到底开没开
  cache 是怎么决策的
  每个阶段到底在吃什么输入

## refined news cache 的当前规则

这部分是当前版本变化最大、也最需要记住的地方。

### 1. unified doc cache 的主键

当前 unified refined news cache 不再按原始新闻文本做主键。

现在统一按新闻 identity 做主键，identity 由三部分组成：

- `title`
- `date`
- `url`

也就是说，当前 cache 匹配规则是：

`title + 发布时间 + url`

而不是旧版本里那种“正文一样就算同一条”的逻辑。

### 2. 新闻源本身也按同一 identity 规则处理

当前 `load_news()` 读入新闻 JSON 后，会做几件事：

- 解析发布时间
- 统一排序
- 对完全相同的 `title + date + url` 做去重

去重时会尽量保留信息更完整的那一行。

### 3. cache 运行模式

当前 refined news cache 有三种模式：

- `disabled`
  没有 `news_path`，新闻 cache 整体关闭

- `read_only`
  显式指定了现成 cache，或者按新闻文件名自动发现到了现成 cache

- `build_mode`
  没有现成 cache，需要边运行边生成

自动发现时，脚本通常会去找类似：

- `checkpoints/_shared_refine_cache/news_doc_cache_{新闻文件名}.json`

### 4. read_only 模式下的行为

如果进入 `read_only`：

- 不做 prewarm
- 不写回 unified doc cache
- 不写回 refine cache
- 不写回 structured cache

当前版本对缺失 cache 新闻的处理也已经收紧并改过一次：

- 不再偷偷现场补算
- 而是跳过这条缺失新闻，并打印 warning

换句话说，当前 read-only 的真实行为是：

- 有 cache 就读
- 缺条目就忽略该新闻
- 不在 read-only 运行中生成新条目

### 5. build_mode 模式下的行为

只有在没有现成 cache 时，框架才会进入 `build_mode`。

这时才允许：

- 做 refine
- 做 structured extraction
- 写回 unified doc cache

如果这时配置要求使用 API，而系统里又没有可用 API key，框架会直接报错。

### 6. cache 排序

当前 unified doc cache 在保存时会按下面顺序排序：

- `date`
- `title`
- `url`

也就是从早到晚。

### 7. cache 允许部分缺失

当前脚本启动前通常会跑 `scripts/verify_refined_news_cache.py`。

这个校验器现在支持：

- 检查 chronological ordering
- 检查 identity coverage
- 允许用 `--allow-missing` 放行“部分缺失但格式没坏”的 cache

所以当前框架状态下：

- cache 缺少一部分新闻时，训练可以继续
- 这些缺失新闻会被忽略

## SignNet 在当前版本里到底做什么

SignNet 是一个可选模块，不是每次训练都必须启用。

它现在的职责是：

- 在 DELTA 之前，先学习一个更稳定的外部修正信号
- 在 `additive` 模式下，给 DELTA 提供“更像往上修还是往下修”的信号
- 在 `relative` 模式下，给 DELTA 提供“当前相对残差百分比应该放大还是缩小”的信号

### SignNet 当前吃什么输入

当前版本里，SignNet 的历史输入已经是 `history_z`，不是 `history_raw`。

它当前主要会用到：

- `history_z`
- `base_pred_z`
- `structured_feats`

当前版本里，SignNet 已经不再接新闻数量，也不会直接接新闻 token 序列；
但如果开启 `delta_temporal_text_enable`，它会通过共享 `TemporalTextTower` 间接接收到一个 pooled `text_summary`。

### SignNet 当前到底学的是什么标签

这件事要分 `additive` 和 `relative` 两种模式看：

- `additive` 模式下：
  如果 `cleaned_residual_enable=0`，SignNet 学的是 raw residual 的正负
  如果 `cleaned_residual_enable=1`，SignNet 学的是 cleaned residual 的正负

- `relative` 模式下：
  SignNet 学的不是二分类正负，而是 3 类状态标签：
  `shrink / neutral / amplify`
  如果 `cleaned_residual_enable=1`，当前只会先对 `q_target` 做 horizon 平滑，再据此构造状态标签，不再把结构化模板直接混进 relative 标签里

这里的 cleaned residual 当前不是一个额外模型输出，而是一个启发式构造的监督目标：

- 先对 raw residual 做 horizon 上的 EWMA 平滑
- 再按结构化新闻特征生成一个模板
- 最后把两者混合

所以它更像“平滑后的方向标签”，而不是新的 ground truth

### `val_valid` 的含义

SignNet 日志里会看到 `val_valid`。

它不是准确率，而是：

- 在验证集里，真实残差绝对值大于 `delta_sign_eps` 的位置占比

只有这些“符号真的有意义”的位置，才会参与符号监督和评估。

## DELTA 在当前版本里怎么工作

DELTA 的目标不是重新做一次完整预测，而是学习：

- Base 还剩下多少误差
- 误差方向是什么
- 哪些新闻值得起作用
- 修正幅度应该多大

但需要特别注意当前版本的训练事实：

- DELTA 模型前向仍然会输出 `sign / magnitude / pred`
- `additive` 模式下，当前参与反向传播的核心就是 `loss_final`
- `relative` 模式下，还会额外加 `loss_relative_mag`
- 如果这时 `delta_sign_mode=internal`，还会再加 `state_ce`
- 所以 `sign / magnitude / state` 既是模型结构，也是 relative 路径里的显式监督对象；只是旧版那套更复杂的辅损已经去掉了

### 当前 DELTA 可以接的新闻分支

当前 DELTA 的新闻侧输入默认是结构化新闻特征：

- `structured_feats`
  由 `structured_events` 数值化后得到的固定维度向量

如果开启 `delta_temporal_text_enable`，还会额外接一条时间对齐文本辅助分支：

- `temporal_text_ids / temporal_text_attn / temporal_text_step_mask`
  这是按每个 history step 对齐后的新闻文本序列
  它的真实路径是：
  当前样本已选新闻 -> 按 `delta_temporal_text_source` 取 `raw` 或 `refined` 文本 -> 每个 history step 只保留当时已发生新闻 -> `(B, L, T)` token 序列 -> `TemporalTextTower` 编码 -> patch 聚合 -> gated fusion 回 `ts_feat`

也就是说，当前版本已经不再把“整段 merged refined news / prompt token”直接送进 DELTA 主干；文本只会在 temporal-text 这条按时间步对齐的辅助支路里进入模型。

### 当前 DELTA 和 prompt 的关系

当前 tiny-news + two-stage 这条主路径里，DELTA 不是靠 prompt token 直接做预测的。

也就是说，当前真正进入 DELTA `forward()` 的，是：

- 时间序列 patch 和 patch mask
- structured features
- 可选的 temporal text auxiliary sequence

而：

- `base_pred_z` 不直接进入 DELTA `forward()`，它是在模型外用于构造残差目标和最终融合
- 外部 `SignNet` 信号也不直接进入 DELTA `forward()`，而是在 DELTA 先产出 magnitude / ratio 后再参与符号或状态组合
- prompt token 本身不会直接进入当前这条 DELTA 主路径

### 最终预测形式

当前主逻辑要分两种模式看：

- `additive`
  `final_pred = base_pred + delta_corr`

- `relative`
  先构造相对残差百分比 `q_hat`
  再做：
  `scale_raw = max(abs(base_pred_raw), floor_or_eps)`
  `final_pred_raw = base_pred_raw + q_hat * scale_raw`

也就是：

- Base 负责主预测
- DELTA 负责残差修正
- 在 `relative` 下，DELTA 修正的是“按百分比放大/缩小 Base”

## 日志应该怎么读

当前日志里最重要的标签大致如下：

- `[CONFIG] Parameters`
  当前运行的参数表

- `[CONFIG] Cache Decision`
  cache 的路径、模式、来源、是否显式指定、是否检测到 API key

- `[DATA_RANGE]`
  train/val/test 原始 split 的真实时间范围

- `[NEWS_DATA]`
  新闻总量和时间范围

- `[MECH]`
  当前有哪些机制真的开启了，比如 structured、external signnet 这些真实仍在工作的机制

- `[BASE]`
  Base 阶段训练与验证

- `[SIGNNET]`
  SignNet 预训练、验证、校准

- `[DELTA]`
  DELTA 阶段训练、验证、测试

- `[TEST][FINAL]`
  最终测试结果与输出写盘

## 主要脚本如何理解

当前仓库里比较关键的脚本例子包括：

- `scripts/nswelecLOAD_2024_tinynews.sh`
  NSW 电力负荷任务

- `scripts/nswelecPRICE_2024_tinynews.sh`
  NSW 电价任务

- `scripts/NAS_14ticker_22_23_combine.sh`
  NASDAQ 多 ticker 组合数据集任务

这些脚本的主要作用是：

- 指定数据集路径
- 指定新闻数据路径
- 指定 Base/DELTA/SignNet 超参数
- 在启动前决定 refined news cache 应该读哪个文件

## 当前版本最重要的几个注意事项

### 1. 多序列数据必须传 `id_col`

如果一个 CSV 里同时包含多条时间序列，例如多个 ticker，共享同一个 `date` 列，那么一定要传 `id_col`。

否则：

- 样本窗口会切错
- `val/test` 历史前缀会补错
- 验证集甚至可能变成空的

### 2. 输入 split 本身应该已经是按时间顺序排好的

当前 `SlidingDataset` 不会替你强制把每个 group 再排序一遍。

所以：

- train/val/test CSV 最好在进入框架前就已经按时间升序排好

### 3. 当前 cache 匹配已经非常严格

当前 cache 读取按 `title + date + url` 匹配。

因此：

- 旧 cache 如果是按旧规则生成的，可能会出现 coverage 不完整
- 这不是读取 bug，而是旧 cache 本身和当前 identity 规则不一致

### 4. read_only 模式不会再偷偷补 cache

现在 read-only 的原则就是：

- 只读
- 不补
- 缺条目就跳过

## 最容易记住的心智模型

如果只记一个最简单的理解方式，可以把当前框架想成三个协作模块：

- `Base`
  负责回答“正常情况下，只看历史会怎样”

- `SignNet`
  负责回答“这次残差方向更像往上还是往下”

- `DELTA`
  负责回答“该不该修、修多少、修正后最终值是多少”

而新闻在当前版本里的主要作用，是先被 refine 和结构化，再以 `structured_feats` 这种数值化形式去帮助 DELTA 和 SignNet 解释 Base 没看出来的那部分误差；如果开启 `delta_temporal_text_enable`，还会再生成一条按历史时间步对齐的文本辅助支路。

- 在默认 `summary_gated` 架构下，这条支路主要给 DELTA 提供 patch-level gated fusion，并给外部 SignNet 提供 pooled `text_summary`
- 在 `plan_c_mvp` 架构下，这条支路仍然先压成 `text_summary + text_strength`，再和结构化新闻、时间序列统计一起参与 DELTA 与外部 SignNet 的 regime routing / expert mixture

## 用最通俗的话总结

当前框架不是“新闻模型替代时间序列模型”，而是：

- 先让时间序列模型给出主判断
- 再让新闻去解释偏差
- 最后做一个受控的残差修正

这是当前代码真正实现的工作方式。
