# 框架工作逻辑说明

## 这份文档是给谁看的

这份文档面向想理解框架整体工作方式、但不想先扎进源码细节的人。

它描述的是当前仓库里的真实状态，而不是早期版本的旧设计。

## 一句话概括

这个框架当前做的事可以概括为：

- 先用 `Base` 模型只根据历史时间序列做一个基线预测
- 再用 `DELTA` 模型结合新闻信息去修正 Base 还没解释好的残差
- 如果开启 `SignNet`，就先额外学一个“残差方向判断器”，帮助 DELTA 更稳定地决定该往上修还是往下修

最终可以理解为：

`最终预测 = Base 预测 + DELTA 修正`

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

- `refined_news`
  当前样本合并后的 refined 文本

- `refined_news_docs`
  当前样本按文章保留的 refined 文档列表

- `structured_events`
  聚合后的结构化事件

- `structured_doc_events`
  文档级结构化事件

- `structured_feats`
  结构化事件向量

- `news_counts`
  当前样本最终使用到的新闻数量

- `signnet_text_ids/attn/mask`
  如果 SignNet 用的是带 temporal text 的变体，还会额外构造按历史步对齐的文本序列输入

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

- 在 DELTA 之前，先学习残差方向
- 给 DELTA 提供更稳定的“往上修还是往下修”的信号

### SignNet 当前吃什么输入

当前版本里，SignNet 的历史输入已经是 `history_z`，不是 `history_raw`。

它当前主要会用到：

- `history_z`
- `base_pred_z`
- `structured_feats`
- `news_counts`

如果启用 `dual_stream_tcn` 变体，还会额外使用：

- 按历史步对齐后的新闻文本序列

### SignNet 的两种主要形态

- `mlp`
  只吃数值输入和结构化输入

- `dual_stream_tcn`
  除了历史 z 序列外，还会通过一个冻结的文本编码器处理按时间对齐的新闻文本，再走一条文本 TCN 分支

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

### 当前 DELTA 可以接的新闻分支

当前 DELTA 路径主要能接三类新闻输入：

- merged refined text 分支
  把当前样本的 refined 新闻合成一段文本后编码

- doc-level refined text 分支
  把每篇 refined 文档分别编码，再做文档级聚合

- structured feature 分支
  把结构化事件转成固定维度数值特征

这些分支是否“真的生效”，不仅取决于开关是否打开，还取决于对应的 `fuse_lambda` 是否非零。

例如：

- `delta_text_direct_enable=1` 但 `delta_text_fuse_lambda=0`

这种情况下，结构还在，但文本直连分支实际上不会对当前运行产生贡献。

### 当前 DELTA 和 prompt 的关系

当前 tiny-news + two-stage 这条主路径里，DELTA 不是靠 prompt token 直接做预测的。

也就是说，当前真正进入 DELTA 模型的，是：

- refined text
- refined docs
- structured features
- Base 预测
- 可选的 SignNet 信号

而不是“整段 prompt token”本身。

### 最终预测形式

当前主逻辑仍然是：

`final_pred = base_pred + delta_corr`

也就是：

- Base 负责主预测
- DELTA 负责残差修正

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
  当前有哪些机制真的开启了，比如 structured、text direct、doc direct、gate

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

而新闻在当前版本里的主要作用，是通过 refined text、doc-level text、structured features 这些路径去帮助 DELTA 和 SignNet 解释 Base 没看出来的那部分误差。

## 用最通俗的话总结

当前框架不是“新闻模型替代时间序列模型”，而是：

- 先让时间序列模型给出主判断
- 再让新闻去解释偏差
- 最后做一个受控的残差修正

这是当前代码真正实现的工作方式。
