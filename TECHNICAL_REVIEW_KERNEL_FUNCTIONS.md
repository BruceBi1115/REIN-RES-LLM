# REIN-LLM-Forecast Kernel 函数技术复盘

## 1. 评审范围

本文聚焦 `kernel_tokens` 主路径中与 kernel 参数构造、拟合、还原和推理融合直接相关的函数逻辑：

- [`src/sft/impact_kernel.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/impact_kernel.py)
- [`src/sft/kernel_fitter.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/kernel_fitter.py)
- [`src/sft/sft_kernel_dataset.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/sft/sft_kernel_dataset.py)
- [`src/base_delta_decoouple_trainer.py`](/home/brucebi/Projects/REIN-LLM-Forecast/src/base_delta_decoouple_trainer.py)

---

## 2. 端到端调用链（kernel 主路径）

1. `build_kernel_sft_samples` 逐样本计算 `raw_residual = target_z - base_pred_z`，再调用 `fit_kernel_params_from_residual` 产出离散参数标签。
2. `format_param_tokens` 将参数转成 `<REL_...> ... <AMP_...>` token 序列，形成 SFT 监督目标。
3. 推理阶段 `evaluate_metrics_kernel_tokens` 用 LLM 生成 token，`parse_param_tokens` 解析为参数字典。
4. `params_to_delta` 将参数字典 + `amp_table` 还原为 `delta` 序列。
5. `_fuse_base_delta_z` 将 `delta` 与 `base_pred_z` 融合（`add/mul_z/mul_raw`），输出最终预测。

---

## 3. `impact_kernel.py` 逻辑拆解

### 3.1 参数清洗与边界约束

- `default_kernel_params()` 给出统一默认值（`REL=0`，`AMP=0`）。
- `sanitize_kernel_params(...)` 做集中边界裁剪：
  - `REL` 强制二值；
  - `SIGN` 非法值回退 `UP`；
  - `LAG/DUR/HALF_LIFE/AMP_BIN` 按 `horizon`、`max_*` 与 `amp_table` 自动裁剪；
  - shape 非法值回退为 `SPIKE`。

### 3.2 核函数构造

- `half_life_to_lambda(hl)` 按 `lambda = hl / ln(2)` 转换衰减常数。
- `build_unit_kernel(...)` 支持 `SPIKE/STEP_DECAY/RAMP_DECAY` 三种形状，输出长度 `H` 的单位核 `k(t)`。
- `params_to_delta(...)` 将 `sign * amp * k` 组合后执行 `clip_low/clip_high` 裁剪。

### 3.3 幅度离散化

- `build_amp_table_from_residuals(...)` 用绝对残差分位数构造 `amp_table`。
- `quantize_amp_to_bin(...)` 通过最近邻映射把连续幅度量化为 `AMP_BIN`。
- `load_amp_table(...)`/`save_amp_table(...)` 负责落盘与容错恢复。

### 3.4 token 编解码

- `format_param_tokens(...)` 固定输出顺序。
- `parse_param_tokens(...)` 使用正则容错解析：
  - 接受不完整尖括号（如 `REL_1>`）；
  - 若缺失 `REL` 但出现 `AMP_x(x>0)`，兜底设置 `REL=1`。

---

## 4. `kernel_fitter.py` 逻辑拆解

### 4.1 搜索框架

- `fit_kernel_params_from_residual(...)` 输入残差向量 `r`，输出离散参数。
- 执行顺序：
  1. 计算 `rel_norm`，低于阈值直接 `REL=0`；
  2. 确定 `SIGN` 候选；
  3. coarse 搜索 `shape × lag × hl × dur`；
  4. 在 coarse 最优点邻域 refine；
  5. 根据 `improve` 门槛决定是否保留 `REL=1`；
  6. 将最优连续幅度量化为 `AMP_BIN`。

### 4.2 关键数学点

- 幅度采用闭式投影：
  - `a* = clip((y·k)/(k·k), 0, a_max)`；
  - `err = ||y - a*k||^2`。
- 最终有效性判定：
  - `improve = ||r||^2 - best_err`；
  - 要求 `improve >= max(rel_improve_min_abs, rel_improve_min_ratio * ||r||^2)`。

### 4.3 先验约束并入

- `_sanitize_search_priors(...)` 会约束 `sign/shape/range`。
- 当前实现中 `SHAPE_VALUES=("STEP_DECAY","RAMP_DECAY")`，因此自动拟合不会搜索 `SPIKE`。

---

## 5. `sft_kernel_dataset.py` 逻辑拆解

### 5.1 样本构建

- `build_kernel_sft_samples(...)`：
  - 用 `make_loader` 逐滑窗样本跑 `base_backbone`；
  - 计算 `raw_residual`；
  - 检索新闻并打 `has_news`；
  - 用有新闻样本优先构建 `amp_table`；
  - 生成 `{prompt, label_tokens, params}`。

### 5.2 API 复标机制

- 两类 API：
  - `priors`：抽取 `causal/sign/shape_candidates/...`；
  - `relsign`：只校验 `REL/SIGN`。
- 关键行为：
  - `causal=0` 可强制 `REL=0`；
  - `relsign` 只允许下调 `REL` 或修正 `SIGN`，不把 `REL=0`上调为 `1`；
  - 调用与缓存共用预算 `kernel_api_max_calls`；
  - 固定模型常量 `KERNEL_API_MODEL_FIXED="gpt-5.1"`。

### 5.3 SFT 输入格式

- `to_kernel_sft_format(...)` 将 prompt 与 label 拼接成 Causal LM 训练样本。
- prompt 结尾强制追加 `"[Output Tokens]"`，并把 prompt 部分 label 设为 `-100`（仅监督输出 token 区域）。

---

## 6. `base_delta_decoouple_trainer.py` 中的 kernel 推理

- `evaluate_metrics_kernel_tokens(...)`：
  - 构建推理 prompt；
  - 调 `delta_model.lm.generate(...)`；
  - 解析 token -> 参数 -> `delta`；
  - 融合并计算 `zMAE/raw MSE/raw MAE`；
  - 记录 `token_hit_rate/rel1_rate/amp_nonzero_rate/empty_gen_rate/mean_abs_delta`。

- `_fuse_base_delta_z(...)`：
  - `add`: `base + delta`；
  - `mul_z`: `base * clip(1 + s*delta, c_min, c_max)`；
  - `mul_raw`: 先反标准化乘系数，再回到 z-space。

---

## 7. 评审结论（kernel 函数层）

1. 代码链路完整，参数边界与失败回退（默认 `REL=0`）设计较稳健。
2. parser 对格式噪声有容错，能降低生成文本轻微异常导致的全样本失效风险。
3. 当前自动拟合 shape 搜索空间不含 `SPIKE`，这与 token 语法可表达集合不同，属于“可表达 > 可拟合”的实现选择，文档应显式说明。
