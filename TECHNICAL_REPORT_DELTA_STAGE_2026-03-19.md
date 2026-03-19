# Technical Report: Delta Stage (REIN-LLM-Forecast)

Date: 2026-03-19  
Project root: `/home/brucebi/Projects/REIN-LLM-Forecast`

## 1. 报告范围

本报告聚焦以下问题：

1. `delta stage` 在本项目中如何工作（训练/验证/测试流程）
2. `delta stage` 的输入是什么
3. `delta stage` 的输出是什么
4. 预测器（predictor）架构是什么样子

核心依据代码：

- `src/base_delta_decoouple_trainer.py`
- `src/model2.py`
- `src/base_backbone.py`
- `scripts/nswelecPRICE_2024_tinynews.sh`

## 2. 两阶段总体流程（Base -> Delta）

入口由 `run.py` 解析参数后调用 `src/base_delta_decoouple_trainer.py:main()`。

- `stage=base`：只训练 base backbone
- `stage=delta`：加载已有 base checkpoint，训练 delta
- `stage=all`：先 base 再 delta

在你当前 NSW 电价脚本中，`STAGE="all"`，所以是完整两阶段流程。

## 3. Delta Stage 如何工作

## 3.1 初始化与依赖

`train_delta_stage()` 关键动作：

1. 加载 base 模型（冻结）作为残差参照  
2. 构建 delta 模型（`build_delta_model()` -> `TinyNewsTSRegressor`）  
3. 初始化新闻 refine/structured cache，并可 prewarm  
4. 预训练外部符号网络 `ResidualSignNet`（当前默认模式 `signnet_binary`）  
5. 按 epoch/batch 训练 delta；按配置执行 val 与 early stop；最终 test

## 3.2 每个 batch 的训练逻辑

对每个 batch（简化后）：

1. **Base 预测（无梯度）**  
   `base_pred = base_backbone(history_z)`

2. **构造 Delta 输入（有新闻）**  
   从 `build_delta_batch_inputs()` 得到：
   - 时间序列 patch
   - refined news token（单文档+多文档）
   - structured feature 向量
   - `targets_z`

3. **构造 Null 输入（无新闻）**  
   `force_no_news=True` 再构一次输入，用于可选 counterfactual/null 正则

4. **构造监督目标**  
   `delta_target = target_z - base_pred`（当前内部 gate 模式下固定 additive 残差目标）

5. **Delta 前向**  
   `out_delta = delta_model(..., head_mode="delta")`  
   取出：
   - `gate_logits/gate`
   - `sign_logits/sign_soft`
   - `magnitude`
   - `delta_pred_model`

6. **外部 SignNet 覆盖符号分支（当前配置默认开启）**  
   若 `signnet_binary`：
   - 由 `ResidualSignNet` 给出 `sign_soft`
   - 用 `gate * sign_soft * magnitude` 重组 delta  
   这一步会覆盖内部 sign 头在训练/验证/测试的使用路径。

7. **融合得到最终预测**  
   `pred_real_z = fuse(base_pred, delta_pred)`  
   当前运行配置下是 additive：
   `final_pred_z = base_pred_z + delta_pred`

8. **计算损失并反传**  
   总损失由多项加权组成（按参数可开关）：
   - `loss_final`：最终预测 vs `targets_z`
   - `loss_delta_aux`：`delta_init` vs `delta_target`
   - `loss_gate_sup`：gate BCE 监督
   - `loss_delta_sign`：sign BCE（外部 signnet 开启时置 0）
   - `loss_delta_mag`：幅度回归
   - `loss_non_degrade`：不劣化约束（相对 base）
   - `loss_struct_consistency`：结构化一致性
   - 可选 `loss_cf/loss_null/loss_gate_null`

## 3.3 验证与最佳模型保存

`delta_val_mode` 支持：

- `each_epoch`（当前常用）
- `end_only`
- `none`

验证调用 `evaluate_metrics_residual()`，同时报告：

- delta+base 的指标
- base-only 对照指标
- `delta_vs_base` 差值

最佳 delta checkpoint 保存到：

- `checkpoints/<task>/best_delta_<task>/`

并额外保存：

- `external_signnet.pt`
- `residual_pair.json`

## 3.4 测试

测试时重新加载最佳 delta checkpoint，调用 `evaluate_metrics_residual()` 输出：

- `final_loss/final_mse/final_mae`
- `base_loss/base_mse/base_mae`（对照）

并写入 `test_answers_*.json`、`true_pred_*.csv`、`test_delta_residual_debug_*.csv`。

## 4. Delta Stage 输入

## 4.1 运行级输入（NSW price 脚本）

来自 `scripts/nswelecPRICE_2024_tinynews.sh` 的核心输入：

- 时序数据：
  - `dataset/2024NSWelecprice/2024NSWelecprice_trainset.csv`
  - `dataset/2024NSWelecprice/2024NSWelecprice_valset.csv`
  - `dataset/2024NSWelecprice/2024NSWelecprice_testset.csv`
- 新闻数据：
  - `dataset/news_2024_2025_elecprice.json`
- 目标列：
  - `TIME_COL=SETTLEMENTDATE`
  - `VALUE_COL=RRP`
- 任务窗口：
  - `history_len`（默认 48）
  - `horizon`（脚本 sweep: 48/96/192/336/720）

## 4.2 batch 级输入张量

`build_delta_batch_inputs()` 产物（按 batch）：

- `ts_patches`: `(B, P, patch_len)`
- `ts_patch_mask`: `(B, P)`
- `targets_z`: `(B, H)`
- `news_counts`: `(B,)`
- `structured_feats`: `(B, D_struct)`，通常 `D_struct=12`
- `refined_news_ids/refined_news_attn`: `(B, T_text)`
- `refined_news_doc_ids/refined_news_doc_attn`: `(B, D_doc, T_doc)`
- `refined_news_doc_mask`: `(B, D_doc)`

外部 signnet 还使用：

- `history_raw`: `(B, L)`
- `base_pred_z`: `(B, H)`
- `structured_feats`: `(B, D_struct)`
- `news_counts`: `(B,)`

## 5. Delta Stage 输出

## 5.1 模型前向输出（delta head）

`TinyNewsTSRegressor(..., head_mode="delta")` 返回字典，核心字段：

- `pred`: delta correction，形状 `(B, H)`（z-space）
- `gate/gate_logits`
- `sign_logits/sign_soft`
- `magnitude/magnitude_raw`
- `delta_init`
- `news_available_mask`
- 可选结构化与文档注意力诊断字段

## 5.2 最终预测输出

Trainer 融合后得到：

- `final_pred_z`（z-space）
- 逆标准化得到 `final_pred_raw`

评估输出：

- `loss_avg`（z-space）
- `mse_avg`、`mae_avg`（raw scale）
- 对照 `base_*` 指标

## 5.3 落盘产物

- `best_base_<task>/`（base checkpoint）
- `best_delta_<task>/`（delta checkpoint + meta）
- `best_signnet_<task>.pt`
- `test_answers_<filename>.json`
- `true_pred_<filename>.csv`
- `val_delta_residual_debug_<filename>.csv`
- `test_delta_residual_debug_<filename>.csv`

## 6. 预测器架构（Predictor Architecture）

## 6.1 Base Predictor

Base 为纯时序 backbone（`src/base_backbone.py`）：

- `DLinearBackbone` 或 `MLPBackbone`
- 输入：`history_z (B, L)`
- 输出：`base_pred_z (B, H)`

你当前 NSW 脚本设置 `BASE_BACKBONES=("mlp")`，即使用 MLP base。

## 6.2 Delta Predictor: TinyNewsTSRegressor

Delta 主体在 `src/model2.py`，可概括为：

1. **TS patch encoder**  
   `patch_proj + patch_gate + pooling` 生成时序上下文 `pooled`

2. **新闻/结构化分支融合**  
   - text-direct（可选）
   - doc-direct（可选）
   - structured feature（可选）
   合并得到 `fused_news_context`

3. **残差三因子头（核心）**  
   - sign head: `sign_logits -> tanh(...) = sign_soft`
   - magnitude head: `softplus(magnitude_raw)`
   - gate head: `sigmoid(gate_logits)` 并乘 `news_strength`

4. **delta 生成**  
   `delta_init = sign_soft * magnitude`  
   `delta_pred = gate * delta_init`（再做 `tanh` clip）

5. **最终融合（trainer 侧）**  
   当前配置 additive：`final = base + delta`

## 6.3 External SignNet（当前配置下是实际符号来源）

`ResidualSignNet` 是独立预训练二分类器：

- 输入：`history_raw + base_pred_z + structured_feats + news_count`
- 输出：`horizon-wise sign logits`
- 训练后冻结，在 delta train/val/test 阶段提供 `sign_soft`
- 当前 `delta_sign_mode=signnet_binary` 时，这一路会覆盖内部 sign 分支

## 7. 结合 NSW Price 脚本的“当前生效配置”解读

对 `scripts/nswelecPRICE_2024_tinynews.sh`，关键结论如下：

1. `stage=all`：先训练 base，再训练 delta  
2. `delta_residual_mode=additive` + `delta_internal_gate_in_model=1`：走 additive 残差路径  
3. `delta_sign_mode=signnet_binary`：外部 signnet 生效（内部 sign loss 置 0）  
4. `delta_structured_enable=1`：结构化事件特征进入 delta  
5. `delta_text_direct_enable=1` 但 `delta_text_fuse_lambda=0.0`：文本直连分支配置上开启，但对当前稳定配置贡献为 0  
6. `delta_doc_direct_enable=0`：文档级分支关闭  
7. `delta_non_degrade_lambda=1.0`：启用“不劣化于 base”约束  
8. `delta_cf_lambda=0.0`、`delta_null_lambda=0.0`、`gate_null_lambda=0.0`：counterfactual/null 类正则当前关闭

---

如果你希望，我可以再补一版“图示版”报告（Mermaid），把 `base_pred -> delta -> final_pred` 和各 loss 分支画成流程图，便于直接放到论文附录或组会 PPT。
