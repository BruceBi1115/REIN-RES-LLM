方案 2：缩短 --max_seq_len（脚本参数，无需改代码）
瓶颈原理：TemporalTextTower 每个历史步的文本经 tokenizer 截断到 max_seq_len，然后过 encoder。Transformer self-attention 复杂度是 O(n²)，序列越长越慢。

当前配置：LOAD 和 PRICE 都用 max_seq_len=196

建议：改为 96 或 128。新闻文章平均 3308 字符，tokenize 后远超 196，已经在截断了。从 196 缩到 96 只是多截掉一些尾部内容，新闻的关键信息（标题、首段）通常在前 96 token 内。

脚本改法：


# scripts/run_tinynews_experiment.sh 中
TEMPORAL_TEXT_MAX_LEN=96    # 原来 196
预期效果：attention 计算量降低约 (196/96)² ≈ 4x，但实际端到端加速约 2-4x（因为还有 linear 层等非 attention 开销）。

方案 3：增大 batch_size（脚本参数，无需改代码）
瓶颈原理：当前 batch_size=8，GPU 利用率可能未打满。增大 batch_size 可以提高 GPU 并行度，减少 Python 循环和 DataLoader 开销。

当前配置：batch_size=8, grad_accum=8（等效 effective batch = 64）

建议：先试 batch_size=16, grad_accum=4（保持 effective batch=64 不变，训练动态一致），看 GPU 显存是否够用。

脚本改法：


BATCH_SIZE=16
GRAD_ACCUM=4
预期效果：~1.5-2x 加速。如果显存不够会 OOM，此时退回 batch_size=8。

注意：保持 batch_size × grad_accum 不变，学习率和训练行为不受影响，预测效果应完全一致。

方案 4：预缓存 TemporalTextTower embedding（需改代码，加速最大）
这是效果最大的方案，但需要改代码。核心思路：

问题：当前每个 epoch 的每个 batch 都对文本重新跑 encoder forward。但文本内容是固定的——同一个 (target_time, history_step) 对应的新闻不会变。100 个 epoch 里同样的文本被编码了 100 次。

方案：在 DELTA 训练开始前，一次性遍历所有样本，把每个历史步的文本 encoding 结果（step_dim=64 的向量）缓存到内存或磁盘。训练时直接查表取向量，跳过 encoder。

预期效果：~10-15x 加速。encoder forward 从"每 batch 384 次 × 918 batch × 100 epoch"变为"一次性 7345×48 次"。

代价：

需要改 src/temporal_text.py、src/base/common.py、src/delta/stage.py
缓存内存占用：7345 样本 × 48 步 × 64 维 × 4 bytes ≈ 90 MB，完全可接受
如果设置了 unfreeze_last_n > 0（微调 encoder 最后几层），则不能缓存，因为权重在变化。但当前配置 unfreeze_last_n=0（encoder 完全冻结），所以可以安全缓存
推荐优先级
立即可试（不改代码）：

方案 2：max_seq_len 从 196 降到 96 — 最低风险，加速 2-4x
方案 3：batch_size 16 + grad_accum 4 — 零风险（effective batch 不变），加速 1.5-2x
方案 2+3 叠加：预期叠加加速 3-6x
需改代码但效果最大：
4. 方案 4：预缓存 embedding — 加速 10-15x，但需要一定代码改动量

关于 delta_stride 效果变差：这说明你的数据中相邻窗口的残差模式确实有变化，DELTA 需要密集采样才能学好。这反而验证了 TemporalTextTower 确实在学习时序相关的文本信号，只是训练效率低。方案 4（预缓存）是最理想的解决路径——保持 stride=1 的全部样本，只砍掉重复的 encoder 计算。