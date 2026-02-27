# trainer.py (REGRESSION version - full file)
# [RESIDUAL BASE+DELTA, NO TRUE-vs-SHUFFLED]
# Refactor: split BASE and DELTA into separate runnable stages via args.stage in {"all","base","delta"}.
# - stage=base : train/save best_base only
# - stage=delta: load existing base checkpoint and train/save best_delta (+test)
# - stage=all  : base -> delta (original behavior)
#
# Notes:
# - This file assumes run.py (argparse) provides optional args:
#   --stage, --base_ckpt, --base_epochs, --delta_epochs
#   If not provided, getattr defaults will be used.

from __future__ import annotations

import csv
import gc
import os
import json
import math
from collections import deque
from contextlib import nullcontext
import shutil

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

from src.data_construction.DataStatistic import DataStatistic
from .data_construction.data import make_loader
from .news_rules import load_news, get_candidates, select_news, _load_keywords
from .data_construction.prompt import format_news, load_templates, build_prompt
from .RL.rl_bandit import LinTS, LinUCB, RewardNormalizer
from .ValidationState import ValidationState
from .utils.logger import setup_live_logger
from .RL.features import bandit_select, get_context_features, encode_instruction

from .model2 import load_checkpoint, load_llama_lora, save_checkpoint
from .utils.residual_utils import freeze_module, zero_regressor_head, split_two_stage_epochs

from .utils.utils import (
    set_seed,
    device_from_id,
    compute_volatility_bin,
    draw_metric_trend,
    draw_pred_true,
    print_prompt_stats,
    record_test_results_csv,
)

dataStatistic = DataStatistic()


# ----------------------------
# helpers
# ----------------------------
def _adapter_off(peft_model):
    # peft >= 0.8 common: disable_adapter()
    if hasattr(peft_model, "disable_adapter"):
        return peft_model.disable_adapter()
    if hasattr(peft_model, "disable_adapters"):
        return peft_model.disable_adapters()
    return nullcontext()


def _single_device_map(args):
    """
    Ensure the whole model loads onto ONE GPU, matching device_from_id(args.gpu).
    Important to avoid cuda:0/cuda:1 mismatch.
    """
    if torch.cuda.is_available():
        return {"": int(args.gpu)}
    return None


def _zstats(x, eps: float = 1e-6):
    x = np.asarray(x, dtype=np.float32)
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma < eps:
        sigma = eps
    return mu, sigma


def _zscore(x, mu, sigma):
    x = np.asarray(x, dtype=np.float32)
    return ((x - mu) / sigma).tolist()


def _inv_zscore(z, mu, sigma):
    z = np.asarray(z, dtype=np.float32)
    return (z * sigma + mu).tolist()


def _maybe_news_dropout(news_str: str, args) -> str:
    p = float(getattr(args, "news_dropout", 0.0) or 0.0)
    if p <= 0:
        return news_str
    if np.random.rand() < p:
        return ""
    return news_str


def _bounded_sigmoid_gate(logits: torch.Tensor, args) -> torch.Tensor:
    """
    Sample-wise gate in [floor, 1], controlled by temperature.
    """
    temperature = float(getattr(args, "news_gate_temperature", 1.0) or 1.0)
    floor = float(getattr(args, "news_gate_floor", 0.0) or 0.0)
    temperature = max(1e-6, temperature)
    floor = max(0.0, min(1.0, floor))
    gate = torch.sigmoid(logits / temperature)
    return floor + (1.0 - floor) * gate


def _residual_elementwise(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mse":
        return (pred - target) ** 2
    if mode == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none")
    return torch.abs(pred - target)  # "mae"


def _build_delta_optimizer(delta_model, args):
    base_lr = float(args.lr)
    wd = float(args.weight_decay)

    lora_scale = float(getattr(args, "delta_lora_lr_scale", 1.0))
    head_scale = float(getattr(args, "delta_head_lr_scale", 1.0))
    other_scale = float(getattr(args, "delta_other_lr_scale", 1.0))

    lora_params, head_params, other_params = [], [], []
    for name, p in delta_model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "lora_" in lname:
            lora_params.append(p)
        elif (
            lname.startswith("delta_head")
            or lname.startswith("delta_gate")
            or lname.startswith("delta_fuse")
            or lname.startswith("delta_text_ln")
            or lname.startswith("delta_log_scale")
            or lname.startswith("rel_head")
        ):
            head_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": base_lr * lora_scale, "weight_decay": wd})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr * head_scale, "weight_decay": wd})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr * other_scale, "weight_decay": wd})

    if not param_groups:
        raise ValueError("No trainable parameters found for DELTA optimizer.")

    optimizer = AdamW(param_groups)
    lr_info = {
        "base_lr": base_lr,
        "lora_lr": base_lr * lora_scale if lora_params else 0.0,
        "head_lr": base_lr * head_scale if head_params else 0.0,
        "other_lr": base_lr * other_scale if other_params else 0.0,
        "n_lora": len(lora_params),
        "n_head": len(head_params),
        "n_other": len(other_params),
    }
    return optimizer, lr_info


def _make_patches(seq: list[float], patch_len: int, stride: int):
    """
    seq: length L list
    returns: patches (P, patch_len), mask (P,)
    """
    x = np.asarray(seq, dtype=np.float32)
    L = int(x.shape[0])
    patch_len = int(patch_len)
    stride = int(stride)

    if patch_len <= 0:
        raise ValueError("patch_len must be > 0")
    if stride <= 0:
        raise ValueError("patch_stride must be > 0")

    if L < patch_len:
        p = np.zeros((1, patch_len), dtype=np.float32)
        p[0, :L] = x
        m = np.ones((1,), dtype=np.int64)
        return p, m

    idxs = list(range(0, L - patch_len + 1, stride))
    patches = np.stack([x[i : i + patch_len] for i in idxs], axis=0).astype(np.float32)  # (P, patch_len)
    mask = np.ones((patches.shape[0],), dtype=np.int64)
    return patches, mask


def history_text(history_z: list[float], mu: float, sigma: float) -> str:
    hz = np.asarray(history_z, dtype=np.float32)
    last = hz.tolist() if len(hz) >= 8 else hz.tolist()
    slope = float(hz[-1] - hz[-9]) if len(hz) >= 9 else float(hz[-1] - hz[0]) if len(hz) >= 2 else 0.0
    return (
        f"History z-scored (mu={mu:.4f}, std={sigma:.4f}). "
        f"Last {len(last)} z: {', '.join([f'{v:.3f}' for v in last])}. "
        f"Recent slope: {slope:.3f}."
    )



def _pad_2d_int(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(seqs)
    max_t = max(len(s) for s in seqs) if B > 0 else 1
    input_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_t), dtype=torch.long)
    for i, s in enumerate(seqs):
        t = len(s)
        if t == 0:
            continue
        input_ids[i, :t] = torch.tensor(s, dtype=torch.long)
        attn[i, :t] = 1
    return input_ids, attn


def _pad_patches(
    patches_list: list[np.ndarray], mask_list: list[np.ndarray], patch_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(patches_list)
    max_p = max(p.shape[0] for p in patches_list) if B > 0 else 1
    ts_patches = torch.zeros((B, max_p, patch_len), dtype=torch.float32)
    ts_patch_mask = torch.zeros((B, max_p), dtype=torch.long)
    for i, (p, pm) in enumerate(zip(patches_list, mask_list)):
        P_i = p.shape[0]
        ts_patches[i, :P_i, :] = torch.tensor(p, dtype=torch.float32)
        ts_patch_mask[i, :P_i] = torch.tensor(pm, dtype=torch.long)
    return ts_patches, ts_patch_mask


# ----------------------------
# batch build
# ----------------------------
def build_batch_inputs(
    batch,
    tokenizer,
    templates,
    tpl_id,
    args,
    news_df,
    policy_name,
    policy_kw,
    volatility_bin,
    epoch: int = -1,
    record_train_prompt: bool = False,
    testing: bool = False,
    force_no_news: bool = False,
    news_dropout: bool = False,
    prompt_path: str = None
):
    """
    Returns:
      input_ids, attn,
      ts_patches, ts_patch_mask,
      targets_z, metas,
      prompt_texts
    """
    L, H = int(args.history_len), int(args.horizon)
    news_budget = int(args.token_budget * args.token_budget_news_frac)

    patch_len = int(getattr(args, "patch_len", 4))
    patch_stride = int(getattr(args, "patch_stride", patch_len))

    tpl_text = templates[tpl_id]["text"]
    B = len(batch["history_value"])

    targets_z_list = []
    patches_list = []
    patchmask_list = []
    metas = []

    hist_strs = []
    news_str_list = []
    rel_labels_list = []

    start_dates = []
    end_dates = []
    pred_starts = []
    pred_ends = []

    len_selected_news = []

    for i in range(B):
        history = batch["history_value"][i].tolist()
        target = batch["target_value"][i].tolist()
        t_target = batch["target_time"][i]

        mu, sigma = _zstats(history, eps=float(getattr(args, "zscore_eps", 1e-6)))
        history_z = _zscore(history, mu, sigma)
        target_z = _zscore(target, mu, sigma)


        p, pm = _make_patches(history_z, patch_len=patch_len, stride=patch_stride)
        if policy_name == "no_sum":
            args.news_text_col = "no_sum"
        elif policy_name == "sum_v0":
            args.news_text_col = "sum_v0"
        # news
        avg_rate = 0.0
        if force_no_news or (news_df is None) or (len(news_df) == 0):
            selected = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])
        else:
            cand = get_candidates(news_df, args.news_time_col, t_target, args.news_window_days, args.news_topM)
            selected, avg_rate = select_news(cand, policy_name, args.news_text_col, policy_kw, args.news_topK)
            # print(len(selected))

        len_selected_news.append(len(selected))

        news_str = ""
        if (not force_no_news) and len(selected) > 0:
            news_str = format_news(
                selected,
                args.news_text_col,
                news_budget,
                tokenizer,
                summary_method=args.news_summary_method,
                max_sentences=args.news_max_sentences,
            )
            if news_dropout:
                news_str = _maybe_news_dropout(news_str, args)


        rel = float(avg_rate) if np.isfinite(avg_rate) else 0.0
        rel = max(0.0, min(1.0, rel))
        rel_labels_list.append(rel)

        # if len(selected) > 5:
        #     print(news_str)
        # time meta for prompt
        start_date = batch["history_times"][0][i]
        end_date = batch["history_times"][-1][i]
        prediction_start = batch["target_times"][0][i]
        prediction_end = batch["target_times"][-1][i]

        targets_z_list.append(np.asarray(target_z, dtype=np.float32))
        patches_list.append(p)
        patchmask_list.append(pm)
        metas.append({"mu": mu, "sigma": sigma})

        hist_strs.append(history_text(history_z, mu, sigma))
        news_str_list.append(news_str)

        start_dates.append(start_date)
        end_dates.append(end_date)
        pred_starts.append(prediction_start)
        pred_ends.append(prediction_end)

    # tokenize prompts
    ids_list = []
    prompt_texts = []

    for i in range(B):
        prompt = build_prompt(
            tpl_text,
            L,
            H,
            args.unit,
            args.description,
            hist_strs[i],
            news_str_list[i],
            start_date=start_dates[i],
            end_date=end_dates[i],
            freq=args.freq_min,
            value_col=args.value_col,
            pred_end=pred_ends[i],
            pred_start=pred_starts[i],
            region=args.region,
        )
        prompt = prompt + "\n\n[Output]\n" + f"Predict the next {H} steps (internally as z-values).\n"

        dataStatistic.news_num_stats_update(len_selected_news[i], prompt=prompt)

        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=int(args.max_seq_len),
            return_attention_mask=False,
        )
        ids_list.append(enc["input_ids"])
        prompt_texts.append(prompt)

        if record_train_prompt and epoch == 0:
            ckpt_dir = os.path.join("./checkpoints", args.taskName)
            os.makedirs(ckpt_dir, exist_ok=True)

            rec = {
                "batch_idx": i,
                "epoch_num": epoch + 1,
                "template_id": int(tpl_id),
                "policy": str(policy_name),
                "force_no_news": bool(force_no_news),
                "prompt": prompt,
                "mu": float(metas[i]["mu"]),
                "sigma": float(metas[i]["sigma"]),
            }
            with open(prompt_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    input_ids, attn = _pad_2d_int(ids_list, pad_id=tokenizer.pad_token_id)
    ts_patches, ts_patch_mask = _pad_patches(patches_list, patchmask_list, patch_len=patch_len)
    targets_z = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets_z_list], dim=0)
    # print("max = ", len_selected_news)
    rel_labels = torch.tensor(rel_labels_list, dtype=torch.float32)
    news_counts = torch.tensor(len_selected_news, dtype=torch.float32)
    return input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, prompt_texts, rel_labels, news_counts

# ----------------------------
# eval
# ----------------------------
@torch.no_grad()
def evaluate_metrics_single(
    model,
    tokenizer,
    data_loader,
    templates,
    tpl_id,
    args,
    news_df,
    policy_name,
    policy_kw,
    device,
    volatility_bin,
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    news_dropout: bool = False,
    force_no_news: bool = False,
    filename: str = None,
):
    """
    Single-model evaluation: used for BASE stage.
    Returns:
      - loss_avg: z-space MSE
      - mse_avg: raw-scale MSE
      - mae_avg: raw-scale MAE
    """
    model.eval()

    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")

    for _, batch in enumerate(data_loader):
        input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, prompt_texts, rel_labels, n_selected = build_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            epoch=-1,
            record_train_prompt=False,
            testing=testing,
            force_no_news=force_no_news,
            news_dropout=news_dropout,
        )

        input_ids = input_ids.to(device)
        attn = attn.to(device)
        ts_patches = ts_patches.to(device)
        ts_patch_mask = ts_patch_mask.to(device)
        targets_z = targets_z.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            ts_patches=ts_patches,
            ts_patch_mask=ts_patch_mask,
            targets=targets_z,
            rel_targets=rel_labels,
            rel_lambda=args.rel_lambda,
        )
        loss = out["loss_fore"]
        pred_z = out["pred"]

        bs = input_ids.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_samples += bs

        pred_z_cpu = pred_z.detach().to(torch.float32).cpu().numpy()
        targets_cpu = batch["target_value"].detach().cpu().numpy()  # raw

        for i in range(bs):
            mu = float(metas[i]["mu"])
            sigma = float(metas[i]["sigma"])

            pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu, sigma)
            true_vals = targets_cpu[i].reshape(-1).tolist()
            true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

            pred = np.asarray(pred_denorm, dtype=np.float32)
            true = np.asarray(true_vals, dtype=np.float32)

            se_sum += float(((pred - true) ** 2).sum())
            ae_sum += float(np.abs(pred - true).sum())
            n_elems += int(args.horizon)

            if true_pred_csv_path is not None:
                with open(true_pred_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_denorm, true_vals))

            if testing:
                record = {
                    "test_prompt": prompt_texts[i],
                    "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                    "pred": [float(x) for x in pred_denorm],
                    "true": [float(x) for x in true_vals],
                    "mu": mu,
                    "sigma": sigma,
                }
                with open(ans_json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    return loss_avg, mse_avg, mae_avg


@torch.no_grad()
def evaluate_metrics_residual(
    base_model,
    delta_model,
    tokenizer,
    data_loader,
    templates,
    tpl_id,
    args,
    news_df,
    policy_name,
    policy_kw,
    device,
    volatility_bin,
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    news_dropout: bool = False,
    filename: str = None,
):
    """
    Residual evaluation: pred = base(no-news, adapter_off) + delta(with-news, adapter_on)
    Returns combined metrics.
    """
    delta_model.eval()

    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")

    for _, batch in enumerate(data_loader):
        # build delta (with news)
        ids_d, attn_d, ts_p, ts_pm, targets_z, metas, prompt_texts, rel_labels_d, news_counts = build_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            epoch=-1,
            record_train_prompt=False,
            testing=testing,
            force_no_news=False,
            news_dropout=news_dropout,
        )
        # build base (no news)
        ids_b, attn_b, _, _, _, _, _, _, _ = build_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            epoch=-1,
            record_train_prompt=False,
            testing=False,
            force_no_news=True,
            news_dropout=False,
        )

        ids_d = ids_d.to(device)
        attn_d = attn_d.to(device)
        ids_b = ids_b.to(device)
        attn_b = attn_b.to(device)
        ts_p = ts_p.to(device)
        ts_pm = ts_pm.to(device)
        targets_z = targets_z.to(device)

        # base pred: adapter off + no news
        with torch.no_grad():
            with _adapter_off(delta_model.lm):
                out_base = delta_model(
                    input_ids=ids_b,
                    attention_mask=attn_b,
                    ts_patches=ts_p,
                    ts_patch_mask=ts_pm,
                    targets=None,
                    head_mode="base",   # NEW
                )
                base_pred = out_base["pred"].to(torch.float32)

        delta_targets = (targets_z.to(torch.float32) - base_pred).detach()

        # delta pred: adapter on + with news
        out_delta = delta_model(
            input_ids=ids_d,
            attention_mask=attn_d,
            ts_patches=ts_p,
            ts_patch_mask=ts_pm,
            targets=delta_targets,
            head_mode="delta",
            rel_targets=rel_labels_d,     # NEW
            rel_lambda=args.rel_lambda,   # NEW
        )
        delta_corr = out_delta["pred"].to(torch.float32)

        pred_z = base_pred + delta_corr

        loss = F.l1_loss(pred_z.to(torch.float32), targets_z.to(torch.float32), reduction="mean")

        bs = ids_d.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_samples += bs

        pred_z_cpu = pred_z.detach().to(torch.float32).cpu().numpy()
        targets_cpu = batch["target_value"].detach().cpu().numpy()  # raw

        for i in range(bs):
            mu = float(metas[i]["mu"])
            sigma = float(metas[i]["sigma"])

            pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu, sigma)
            true_vals = targets_cpu[i].reshape(-1).tolist()
            true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

            pred = np.asarray(pred_denorm, dtype=np.float32)
            true = np.asarray(true_vals, dtype=np.float32)

            se_sum += float(((pred - true) ** 2).sum())
            ae_sum += float(np.abs(pred - true).sum())
            n_elems += int(args.horizon)

            if true_pred_csv_path is not None:
                with open(true_pred_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_denorm, true_vals))

            if testing:
                record = {
                    "test_prompt": prompt_texts[i],
                    "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                    "pred": [float(x) for x in pred_denorm],
                    "true": [float(x) for x in true_vals],
                    "mu": mu,
                    "sigma": sigma,
                    "policy": str(policy_name),
                    "template_id": int(tpl_id),
                }
                with open(ans_json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    return loss_avg, mse_avg, mae_avg


# ----------------------------
# bandit utilities (unchanged)
# ----------------------------
def make_tpl_feature_fn(templates, add_one_hot=True, add_cost_proxy=False, add_cross_terms=False):
    if isinstance(templates, dict):
        tpl_by_id = templates
    else:
        tpl_by_id = {int(t["id"]): t for t in templates}

    tpl_ids = sorted(tpl_by_id.keys())
    T = len(tpl_ids)

    id2idx = {tid: i for i, tid in enumerate(tpl_ids)}
    I = np.eye(T, dtype=np.float32)

    tpl_list = [tpl_by_id[tid] for tid in tpl_ids]
    n_paths_list = [float(t.get("n_paths", 1) or 1) for t in tpl_list]
    max_n_paths = max(n_paths_list) if n_paths_list else 1.0

    raw_breath_intensity = []
    for t in tpl_list:
        hb = float(bool(t.get("has_breath", False)))
        bf = float(t.get("breath_freq", 0) or 0)
        raw_breath_intensity.append(hb * (1.0 / bf) if hb > 0 and bf > 0 else 0.0)

    bi_min = min(raw_breath_intensity) if raw_breath_intensity else 0.0
    bi_max = max(raw_breath_intensity) if raw_breath_intensity else 1.0
    bi_range = (bi_max - bi_min) if bi_max > bi_min else 1.0

    def _cost_proxy(t):
        he = float(bool(t.get("has_example", False)))
        hb = float(bool(t.get("has_breath", False)))
        hd = float(bool(t.get("has_decomp", False)))
        hsc = float(bool(t.get("has_self_consistency", False)))
        np_norm = float(t.get("n_paths", 1) or 1) / max_n_paths
        return 0.4 * he + 0.5 * hd + 1.0 * hsc + 0.6 * np_norm + 0.2 * hb

    raw_costs = [_cost_proxy(t) for t in tpl_list]
    c_min = min(raw_costs) if raw_costs else 0.0
    c_max = max(raw_costs) if raw_costs else 1.0
    c_range = (c_max - c_min) if c_max > c_min else 1.0

    def _single_tpl_vec(tid: int) -> np.ndarray:
        t = tpl_by_id[int(tid)]

        he = float(bool(t.get("has_example", False)))
        hb = float(bool(t.get("has_breath", False)))
        hd = float(bool(t.get("has_decomp", False)))
        hsc = float(bool(t.get("has_self_consistency", False)))
        np_norm = float(t.get("n_paths", 1) or 1) / max_n_paths

        bf = float(t.get("breath_freq", 0) or 0)
        bi = hb * (1.0 / bf) if hb > 0 and bf > 0 else 0.0
        bi_norm = (bi - bi_min) / bi_range

        vec = [1.0, he, hb, hd, hsc, np_norm, bi_norm]

        if add_cost_proxy:
            vec.append((_cost_proxy(t) - c_min) / c_range)

        if add_one_hot:
            vec.extend(I[id2idx[tid]].tolist())

        return np.asarray(vec, dtype=np.float32)

    def tpl_features(tid: int, context_vector) -> np.ndarray:
        arm = _single_tpl_vec(tid)
        if add_cross_terms:
            if context_vector is None:
                raise ValueError("add_cross_terms=True requires context_vector")
            cross = np.outer(context_vector.astype(np.float32), arm).ravel()
            return np.concatenate([arm, cross]).astype(np.float32)
        return arm

    def feat_dim(context_dim) -> int:
        base = len(_single_tpl_vec(tpl_ids[0]))
        if add_cross_terms:
            return base + base * context_dim
        return base

    return tpl_features, feat_dim


def bandit_round_update_residual(
    base_model,
    delta_model,
    tokenizer,
    probe_loader,
    templates,
    allowed_tpl_ids,
    news_df,
    policy_space,
    policy_kw,
    args,
    device,
    volatility_bin,
    context_vector,
    tpl_features,
    bandit_tpl,
    bandit_pol,
    normalizer,
    live_logger,
    round_id,
    bidx,
    global_step,
):
    delta_model.eval()

    cand = bandit_select(
        args,
        context_vector,
        live_logger,
        allowed_tpl_ids,
        policy_space,
        bandit_tpl,
        bandit_pol,
        tpl_features,
        pol_features=None,
        epoch=round_id,
        bidx=bidx,
        global_step=global_step,
    )
    tpl_id = cand["tpl_id"]
    policy_name = cand["policy_name"]
    pol_idx = cand["pol_idx"]

    probe_loss, probe_mse, probe_mae = evaluate_metrics_residual(
        base_model=base_model,
        delta_model=delta_model,
        tokenizer=tokenizer,
        data_loader=probe_loader,
        templates=templates,
        tpl_id=tpl_id,
        args=args,
        news_df=news_df,
        policy_name=policy_name,
        policy_kw=policy_kw,
        device=device,
        volatility_bin=volatility_bin,
        testing=False,
        true_pred_csv_path=None,
        news_dropout=False,
    )

    if args.reward_metric == "loss":
        metric_now = probe_loss
    elif args.reward_metric == "mse":
        metric_now = probe_mse
    else:
        metric_now = probe_mae

    r = -metric_now
    r_hat = normalizer.update_and_normalize(
        r, group_key=(args.region, args.horizon) if args.domain_reward_norm else None
    )

    x_tpl = np.concatenate([context_vector, tpl_features(tpl_id, context_vector)], axis=0).astype(np.float32)
    x_pol = context_vector.astype(np.float32)

    bandit_tpl.update(x_tpl, r_hat)
    bandit_pol.update(x_pol, r_hat)

    live_logger.info(
        f"BANDIT_ROUND(residual) round={round_id} tpl_id={tpl_id} policy={policy_name} "
        f"probe_loss={probe_loss:.6f} probe_mse={probe_mse:.6f} probe_mae={probe_mae:.6f} "
        f"reward_norm={r_hat:.6f}"
    )

    return tpl_id, policy_name, pol_idx


# ----------------------------
# setup (new)
# ----------------------------
def setup_env_and_data(args):
    stage = str(getattr(args, "stage", "all")).lower()
    news_path_clean = (args.news_path or "").replace("dataset/", "")
    filename = f"{args.taskName}_{args.stage}_s{args.stride}_h{args.horizon}_news_{news_path_clean}_RL_{args.rl_use}_{args.patch_dropout}_{args.head_dropout}_{args.news_dropout}_{args.delta_null_lambda}_{args.delta_margin_lambda}_{args.delta_adv_margin}_base_{args.base_epochs}_delta_{args.delta_epochs}_lr_{args.lr}_gradacc_{args.grad_accum}_sche_{args.scheduler}_lookback_{args.news_window_days}_topK_{args.news_topK}"
    log_filename = filename + ".log"

    live_logger, live_path, log_jsonl = setup_live_logger(
        save_dir=args.save_dir + "/" + args.taskName, filename=log_filename
    )
    print(f"[live log] {live_path}  (实时查看: tail -f '{live_path}')")

    ckpt_dir = os.path.join("./checkpoints", args.taskName)
    os.makedirs(ckpt_dir, exist_ok=True)

    # fixed output paths (clearing controlled by main())
    prompt_path = os.path.join(ckpt_dir, f"prompts_{filename}.json")
    ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")
    true_pred_csv_path = os.path.join(ckpt_dir, f"true_pred_{filename}.csv")

    set_seed(args.seed)
    device = device_from_id(args.gpu)

    def _read(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    train_df = _read(args.train_file)
    val_df = _read(args.val_file)
    test_df = _read(args.test_file)

    train_df[args.time_col] = pd.to_datetime(train_df[args.time_col], dayfirst=args.dayFirst)
    val_df[args.time_col] = pd.to_datetime(val_df[args.time_col], dayfirst=args.dayFirst)
    test_df[args.time_col] = pd.to_datetime(test_df[args.time_col], dayfirst=args.dayFirst)

    train_loader = make_loader(
        train_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=True,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )
    val_loader = make_loader(
        val_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )
    test_loader = make_loader(
        test_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )

    # news
    
    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])

    news_df[args.news_time_col] = pd.to_datetime(news_df[args.news_time_col], dayfirst=args.dayFirst)
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)
        print(len(news_df))
        # 去除空的总结后的新闻
        col = args.news_text_col
        news_df = news_df.loc[
            news_df[col].fillna("").astype(str).str.strip().ne("")
        ].reset_index(drop=True)
        print(len(news_df))

    policy_kw = _load_keywords(args.keyword_path)
    templates = load_templates(args.template_pool)

    patch_len = int(getattr(args, "patch_len", 4))

    volatility_bin = compute_volatility_bin(
        train_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )
    volatility_bin_val = compute_volatility_bin(
        val_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )
    volatility_bin_test = compute_volatility_bin(
        test_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )

    return {
        "stage": stage,
        "live_logger": live_logger,
        "live_path": live_path,
        "log_jsonl": log_jsonl,
        "ckpt_dir": ckpt_dir,
        "prompt_path": prompt_path,
        "ans_json_path": ans_json_path,
        "true_pred_csv_path": true_pred_csv_path,
        "device": device,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "news_df": news_df,
        "policy_kw": policy_kw,
        "templates": templates,
        "patch_len": patch_len,
        "volatility_bin": volatility_bin,
        "volatility_bin_val": volatility_bin_val,
        "volatility_bin_test": volatility_bin_test,
        "prompt_path": prompt_path,
        "test_filename": filename,
    }


# ----------------------------
# stage 1: BASE (new)
# ----------------------------
def train_base_stage(args, bundle):
    live_logger = bundle["live_logger"]
    device = bundle["device"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    news_df = bundle["news_df"]
    policy_kw = bundle["policy_kw"]
    templates = bundle["templates"]
    patch_len = bundle["patch_len"]
    volatility_bin = bundle["volatility_bin"]
    volatility_bin_val = bundle["volatility_bin_val"]

    lora_cfg = {
        "r": int(args.lora_r),
        "alpha": int(args.lora_alpha),
        "dropout": float(args.lora_dropout),
        "target_modules": args.target_modules,
    }

    # base epochs
    base_epochs_override = int(getattr(args, "base_epochs", -1))
    if base_epochs_override >= 0:
        base_epochs = base_epochs_override
    else:
        base_frac = float(getattr(args, "residual_base_frac", 0.3))
        base_epochs, _ = split_two_stage_epochs(
            total_epochs=int(args.epochs),
            base_frac=base_frac,
            min_base=int(getattr(args, "residual_min_base_epochs", 1)),
            min_delta=int(getattr(args, "residual_min_delta_epochs", 1)),
        )
        if getattr(args, "residual_base_epochs", None) is not None:
            base_epochs = int(getattr(args, "residual_base_epochs"))
            base_epochs = max(0, min(base_epochs, int(args.epochs) - 1))

    live_logger.info("-----------------------------------------------------")
    live_logger.info(f"[BASE] Training BASE only: epochs={base_epochs} (force_no_news=True)")
    live_logger.info("-----------------------------------------------------")

    tokenizer, base_train_model = load_llama_lora(
        base_model=args.base_model,
        tokenizer_id=args.tokenizer,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        max_seq_len=args.max_seq_len,
        device=device,
        horizon=args.horizon,
        patch_dim=patch_len,
        patch_dropout=args.patch_dropout,
        head_dropout=args.head_dropout,
        head_mlp=args.head_mlp,
        delta_gate_init_bias=float(getattr(args, "delta_gate_init_bias", -1.0)),
        delta_clip=float(getattr(args, "delta_clip", 3.0)),
        delta_news_tail_tokens=int(getattr(args, "delta_news_tail_tokens", 160)),
        delta_rel_floor=float(getattr(args, "delta_rel_floor", 0.05)),
    )
    base_train_model.to(device)

    optim_base = AdamW(
        filter(lambda p: p.requires_grad, base_train_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_batches = len(train_loader)
    total_opt_steps_base = math.ceil((num_batches * max(1, base_epochs)) / max(1, args.grad_accum))
    warmup_steps_base = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps_base)
    warmup_steps_base = min(warmup_steps_base, max(0, total_opt_steps_base - 1))


    if args.scheduler == 1:
        scheduler_base = get_cosine_schedule_with_warmup(
            optim_base,
            num_warmup_steps=warmup_steps_base,
            num_training_steps=total_opt_steps_base,
        )
    else:
        scheduler_base = None

    allowed_tpl_ids = sorted([t["id"] for t in templates.values()])
    tpl_id = allowed_tpl_ids[0]
    policy_name_base = "all"

    best_base_metric = float("inf")
    stale_rounds = 0
    loss_window = deque(maxlen=50)
    global_step = 0

    for epoch in range(base_epochs):
        pbar = tqdm(train_loader, desc=f"[BASE] Epoch {epoch+1}/{base_epochs}")

        for _, batch in enumerate(pbar):
            input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, _, _, _ = build_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                news_df=news_df,
                policy_name=policy_name_base,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                epoch=epoch,
                record_train_prompt=False,
                testing=False,
                force_no_news=True,
                news_dropout=False,
            )

            input_ids = input_ids.to(device)
            attn = attn.to(device)
            ts_patches = ts_patches.to(device)
            ts_patch_mask = ts_patch_mask.to(device)
            targets_z = targets_z.to(device)

            base_train_model.train()
            out = base_train_model(
                input_ids=input_ids,
                attention_mask=attn,
                ts_patches=ts_patches,
                ts_patch_mask=ts_patch_mask,
                targets=targets_z,
            )
            loss = out["loss_fore"] / args.grad_accum
            loss.backward()

            loss_window.append(float(loss.detach().cpu()))
            if global_step % 10 == 0:
                avg_train_loss = sum(loss_window) / len(loss_window)
                pbar.set_postfix(train_loss=f"{avg_train_loss:.6f}")

            if (global_step + 1) % args.grad_accum == 0:
                optim_base.step()
                if args.scheduler == 1:
                    scheduler_base.step()
                optim_base.zero_grad(set_to_none=True)

            global_step += 1

        val_loss, val_mse, val_mae = evaluate_metrics_single(
            model=base_train_model,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name_base,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            testing=False,
            true_pred_csv_path=None,
            news_dropout=False,
            force_no_news=True,
        )

        if args.reward_metric == "loss":
            metric_now = val_loss
        elif args.reward_metric == "mse":
            metric_now = val_mse
        else:
            metric_now = val_mae

        live_logger.info(
            f"[BASE][EVAL] epoch={epoch+1} tpl_id={tpl_id} "
            f"val_loss(zMSE)={val_loss:.6f} val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
        )

        best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")
        if metric_now < best_base_metric - 1e-6:
            best_base_metric = metric_now
            stale_rounds = 0
            # if os.path.isfile(best_base_path):
            #     os.remove(best_base_path)
            shutil.rmtree(best_base_path, ignore_errors=True)

            save_checkpoint(
                best_base_path,
                tokenizer,
                base_train_model,
                base_model_id=args.base_model,
                tokenizer_id=args.tokenizer or args.base_model,
                lora_cfg=lora_cfg,
                optimizer=optim_base,
                scheduler=scheduler_base,
                epoch=epoch,
                global_step=global_step,
            )
            live_logger.info(f"[BASE] New best saved to {best_base_path} ({args.reward_metric}={best_base_metric:.6f})")
        else:
            stale_rounds += 1
            live_logger.info(f"[BASE] stale_rounds={stale_rounds}/{args.early_stop_patience} best={best_base_metric:.6f}")

        if stale_rounds >= args.early_stop_patience:
            live_logger.info(f"[BASE] Early stopping triggered at epoch {epoch+1}.")
            break

    # cleanup
    del base_train_model
    gc.collect()
    torch.cuda.empty_cache()

    best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")
    if not os.path.exists(best_base_path):
        raise FileNotFoundError(f"Base checkpoint not found: {best_base_path}")

    return {
        "best_base_path":best_base_path,
        "device": device,
        "live_logger": live_logger,
        "tpl_id":tpl_id,
        "templates":templates,
        "news_df":news_df,
        "policy_kw":policy_kw,
        "best_base_metric": best_base_metric,
    }



# ----------------------------
# stage 2: DELTA (new)
# ----------------------------
def train_delta_stage(args, bundle, best_base_path: str, best_base_metric):
    live_logger = bundle["live_logger"]
    device = bundle["device"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    news_df = bundle["news_df"]
    policy_kw = bundle["policy_kw"]
    templates = bundle["templates"]
    volatility_bin = bundle["volatility_bin"]
    volatility_bin_val = bundle["volatility_bin_val"]
    volatility_bin_test = bundle["volatility_bin_test"]
    true_pred_csv_path = bundle["true_pred_csv_path"]
    

    lora_cfg = {
        "r": int(args.lora_r),
        "alpha": int(args.lora_alpha),
        "dropout": float(args.lora_dropout),
        "target_modules": args.target_modules,
    }

    delta_epochs_override = int(getattr(args, "delta_epochs", -1))
    if delta_epochs_override >= 0:
        delta_epochs = delta_epochs_override
    else:
        base_frac = float(getattr(args, "residual_base_frac", 0.3))
        base_epochs, delta_epochs = split_two_stage_epochs(
            total_epochs=int(args.epochs),
            base_frac=base_frac,
            min_base=int(getattr(args, "residual_min_base_epochs", 1)),
            min_delta=int(getattr(args, "residual_min_delta_epochs", 1)),
        )
        if (args.delta_epochs > 0):
            delta_epochs = args.delta_epochs
        else:
            delta_epochs = args.epochs - base_epochs

    live_logger.info("-----------------------------------------------------")
    live_logger.info(f"[DELTA] Training DELTA: epochs={delta_epochs}, base_ckpt={best_base_path}")
    live_logger.info("-----------------------------------------------------")

    # delta model init from base checkpoint but trainable
    tokenizer, delta_model = load_checkpoint(
        best_base_path,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        device_map=_single_device_map(args),
        is_trainable=True,
        head_mlp=args.head_mlp,
        hd=args.head_dropout,
        pd=args.patch_dropout
    )
    delta_model.to(device)
    # zero_regressor_head(delta_model)

    # freeze base head
    for p in delta_model.base_head.parameters():
        p.requires_grad = False

    # keep backbone patch/text pooling stable so delta focuses on residual adaptation
    freeze_modules = [
        "patch_proj",
        "patch_gate",
        "patch_pos",
        "pool_attn",
        "pool_ln",
        "text_ctx_ln",
        "text2q",
    ]
    for name in freeze_modules:
        if hasattr(delta_model, name):
            m = getattr(delta_model, name)
            if hasattr(m, "parameters"):
                for p in m.parameters():
                    p.requires_grad = False
    if hasattr(delta_model, "pool_q"):
        delta_model.pool_q.requires_grad = False
    if hasattr(delta_model, "layer_w"):
        delta_model.layer_w.requires_grad = False

    # ensure delta-specific heads remain trainable
    train_modules = ["delta_head", "delta_gate", "delta_fuse", "delta_text_ln", "rel_head"]
    for name in train_modules:
        if hasattr(delta_model, name):
            m = getattr(delta_model, name)
            if hasattr(m, "parameters"):
                for p in m.parameters():
                    p.requires_grad = True
    if hasattr(delta_model, "delta_log_scale"):
        delta_model.delta_log_scale.requires_grad = True

    # base teacher (frozen) - provides base_pred
    base_teacher = None
    # _, base_teacher = load_checkpoint(
    #     best_base_path,
    #     load_in_4bit=args.load_in_4bit,
    #     gradient_checkpointing=args.gradient_checkpointing,
    #     device_map=_single_device_map(args),
    #     is_trainable=False,
    #     head_mlp=args.head_mlp,
    #     hd=args.head_dropout,
    #     pd=args.patch_dropout
    # )
    # base_teacher.to(device)
    # base_teacher.eval()
    # for p in base_teacher.parameters():
    #     p.requires_grad = False


    optim_delta, lr_info = _build_delta_optimizer(delta_model, args)
    live_logger.info(
        "[DELTA] optimizer groups: "
        f"base_lr={lr_info['base_lr']:.3e}, "
        f"lora_lr={lr_info['lora_lr']:.3e} (n={lr_info['n_lora']}), "
        f"head_lr={lr_info['head_lr']:.3e} (n={lr_info['n_head']}), "
        f"other_lr={lr_info['other_lr']:.3e} (n={lr_info['n_other']})"
    )

    total_opt_steps_delta = math.ceil((len(train_loader) * max(1, delta_epochs)) / max(1, args.grad_accum))
    warmup_steps_delta = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps_delta)
    warmup_steps_delta = min(warmup_steps_delta, max(0, total_opt_steps_delta - 1))

    if args.scheduler == 1:
        scheduler_delta = get_cosine_schedule_with_warmup(
            optim_delta,
            num_warmup_steps=warmup_steps_delta,
            num_training_steps=total_opt_steps_delta,
        )
    else:
        scheduler_delta = None

    # RL setup for DELTA stage (kept as-is)
    normalizer = RewardNormalizer(ema=args.reward_ema, use_group_norm=args.domain_reward_norm)
    val_state = ValidationState(ema_alpha=args.val_ema_alpha)
    context_vector = encode_instruction(args, ctx={}, volatility_bin=volatility_bin)

    tpl_features, feat_dim = make_tpl_feature_fn(
        templates=templates,
        add_one_hot=True,
        add_cost_proxy=False,
        add_cross_terms=True,
    )
    allowed_tpl_ids = sorted([t["id"] for t in templates.values()])
    tpl_id = allowed_tpl_ids[0]
    policy_space = args.policy_space
    policy_name = args.default_policy
    best_tpl_id = tpl_id
    best_policy_name = policy_name

    d_tpl = len(context_vector) + len(tpl_features(allowed_tpl_ids[0], context_vector=context_vector))
    d_pol = len(context_vector)
    bandit_tpl = LinTS(d_tpl, v=args.ts_v) if args.rl_algo == "lints" else LinUCB(d_tpl, alpha=args.ucb_alpha)
    bandit_pol = LinTS(d_pol, v=args.ts_v) if args.rl_algo == "lints" else LinUCB(d_pol, alpha=args.ucb_alpha)
    if math.isfinite(float(best_base_metric)):
        best_metric = float(best_base_metric)
    else:
        base_ref_loss, base_ref_mse, base_ref_mae = evaluate_metrics_single(
            model=delta_model,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=allowed_tpl_ids[0],
            args=args,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            testing=False,
            true_pred_csv_path=None,
            news_dropout=False,
            force_no_news=True,
        )
        if args.reward_metric == "loss":
            best_metric = float(base_ref_loss)
        elif args.reward_metric == "mse":
            best_metric = float(base_ref_mse)
        else:
            best_metric = float(base_ref_mae)
        live_logger.info(
            f"[DELTA] base reference on val: loss={base_ref_loss:.6f} mse={base_ref_mse:.6f} "
            f"mae={base_ref_mae:.6f}; threshold={best_metric:.6f}"
        )
    stale_rounds = 0
    loss_window = deque(maxlen=50)

    val_loss_per_epoch, mse_loss_per_epoch, mae_loss_per_epoch = [], [], []
    global_step = 0

    for epoch in range(delta_epochs):
        pbar = tqdm(train_loader, desc=f"[DELTA] Epoch {epoch+1}/{delta_epochs}")

        # epoch-level bandit selection (optional)
        if (args.select_policy_by == "epoch") and args.rl_use == 1:
            context_vector = get_context_features(
                None,
                news_df,
                args,
                prev_model_loss_n=None,
                prev_model_loss_ema_n=None,
                val_state=val_state,
                train_loader=train_loader,
                volatility_bin=volatility_bin,
            )

            tpl_id, policy_name, pol_idx = bandit_round_update_residual(
                base_model=None,
                delta_model=delta_model,
                tokenizer=tokenizer,
                probe_loader=val_loader,
                templates=templates,
                allowed_tpl_ids=allowed_tpl_ids,
                news_df=news_df,
                policy_space=policy_space,
                policy_kw=policy_kw,
                args=args,
                device=device,
                volatility_bin=volatility_bin_val,
                context_vector=context_vector,
                tpl_features=tpl_features,
                bandit_tpl=bandit_tpl,
                bandit_pol=bandit_pol,
                normalizer=normalizer,
                live_logger=live_logger,
                round_id=epoch + 1,
                bidx=None,
                global_step=global_step,
            )

            live_logger.info(f"[DELTA] EPOCH_BEGIN epoch={epoch+1}, tpl_id={tpl_id}, policy={policy_name}")

        for bidx, batch in enumerate(pbar):
            # batch-level bandit selection (optional)
            if (args.select_policy_by == "batch") and args.rl_use == 1 and global_step % 50 == 0:
                context_vector = get_context_features(
                    batch,
                    news_df,
                    args,
                    prev_model_loss_n=None,
                    prev_model_loss_ema_n=None,
                    val_state=val_state,
                    train_loader=train_loader,
                    volatility_bin=volatility_bin,
                )

                tpl_id, policy_name, pol_idx = bandit_round_update_residual(
                    base_model=None,
                    delta_model=delta_model,
                    tokenizer=tokenizer,
                    probe_loader=val_loader,
                    templates=templates,
                    allowed_tpl_ids=allowed_tpl_ids,
                    news_df=news_df,
                    policy_space=policy_space,
                    policy_kw=policy_kw,
                    args=args,
                    device=device,
                    volatility_bin=volatility_bin_val,
                    context_vector=context_vector,
                    tpl_features=tpl_features,
                    bandit_tpl=bandit_tpl,
                    bandit_pol=bandit_pol,
                    normalizer=normalizer,
                    live_logger=live_logger,
                    round_id=epoch * len(pbar) + bidx,
                    bidx=bidx,
                    global_step=global_step,
                )

            # build delta inputs (with news)
            ids_d, attn_d, ts_p, ts_pm, targets_z, metas, _, rel_labels_d, news_counts_d = build_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                epoch=epoch,
                record_train_prompt=True,
                testing=False,
                force_no_news=False,
                news_dropout=True,
                prompt_path=bundle["prompt_path"]
            )
            # build base text inputs (no news)
            ids_b, attn_b, _, _, _, _, _, _, _ = build_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                epoch=epoch,
                record_train_prompt=False,
                testing=False,
                force_no_news=True,
                news_dropout=False,
            )

            ids_d = ids_d.to(device)
            attn_d = attn_d.to(device)
            ids_b = ids_b.to(device)
            attn_b = attn_b.to(device)

            ts_p = ts_p.to(device)
            ts_pm = ts_pm.to(device)
            targets_z = targets_z.to(device)
            rel_labels_d = rel_labels_d.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            news_counts_d = news_counts_d.to(device=device, dtype=torch.float32)
            has_news = (news_counts_d > 0).to(dtype=torch.float32)

            delta_model.train()

            # -----------------------------
            # (1) BASE prediction (no-news, adapter_off, base head)  -> base_pred
            # -----------------------------
            was_training = delta_model.training
            delta_model.eval()
            with torch.no_grad():
                with _adapter_off(delta_model.lm):
                    out_base = delta_model(
                        input_ids=ids_b,
                        attention_mask=attn_b,
                        ts_patches=ts_p,
                        ts_patch_mask=ts_pm,
                        targets=None,
                        head_mode="base",
                    )
                    base_pred = out_base["pred"].to(torch.float32)

            if was_training:
                delta_model.train()

            # residual target for delta head
            delta_targets = (targets_z.to(torch.float32) - base_pred).detach()

            # -----------------------------
            # (2) REAL delta forward (with-news, adapter_on, delta head)
            # -----------------------------
            out_delta = delta_model(
                input_ids=ids_d,
                attention_mask=attn_d,
                ts_patches=ts_p,
                ts_patch_mask=ts_pm,
                targets=delta_targets,
                head_mode="delta",
                rel_targets=None,
                rel_lambda=0.0,
            )
            delta_pred_real = out_delta["pred"].to(torch.float32)  # (B,H)
            rel_logits_real = out_delta["rel_logits"].to(torch.float32)  # (B,)

            # -----------------------------
            # (3) NULL delta forward (no-news, adapter_on, delta head) -> delta_pred_null
            #     Counterfactual: same history/patch, but zero news content.
            # -----------------------------
            out_null = delta_model(
                input_ids=ids_b,
                attention_mask=attn_b,
                ts_patches=ts_p,
                ts_patch_mask=ts_pm,
                targets=None,            # IMPORTANT: do NOT regress to delta_targets here
                head_mode="delta",
                rel_targets=None,
                rel_lambda=0.0,
            )
            delta_pred_null = out_null["pred"].to(torch.float32)    # (B,H)
            rel_logits_null = out_null["rel_logits"].to(torch.float32)

            # -----------------------------
            # (4) Counterfactual Regularization
            #   4.1 Null-shrink: force delta(no-news) -> 0
            #   4.2 Margin: pred(with-news) should beat pred(no-news)
            #   4.3 Non-degradation: pred(with-news) should not be worse than base
            #   4.4 Gate pseudo-labeling: learn to suppress noisy news
            # -----------------------------
            delta_null_lambda = float(getattr(args, "delta_null_lambda", 0.5))
            delta_margin_lambda = float(getattr(args, "delta_margin_lambda", 1.0))
            delta_adv_margin = float(getattr(args, "delta_adv_margin", 0.02))
            delta_non_degrade_lambda = float(getattr(args, "delta_non_degrade_lambda", 0.0))
            delta_non_degrade_margin = float(getattr(args, "delta_non_degrade_margin", 0.0))
            gate_lambda = float(getattr(args, "gate_lambda", 0.0))
            gate_null_lambda = float(getattr(args, "gate_null_lambda", 0.0))
            cf_min_weight = float(getattr(args, "cf_min_weight", 0.05))
            cf_min_weight = max(0.0, min(1.0, cf_min_weight))
            cf_pseudo_margin = float(getattr(args, "cf_pseudo_margin", 0.01))
            cf_pseudo_temp = float(getattr(args, "cf_pseudo_temp", 0.02))
            cf_pseudo_temp = max(1e-6, cf_pseudo_temp)
            cf_pseudo_hard = int(getattr(args, "cf_pseudo_hard", 0))
            violation_cap = float(getattr(args, "delta_violation_cap", 0.0))

            gate_enable = int(getattr(args, "news_gate_enable", 1)) == 1
            if gate_enable:
                gate_real = _bounded_sigmoid_gate(rel_logits_real, args)
                gate_null = _bounded_sigmoid_gate(rel_logits_null, args)
            else:
                gate_real = torch.ones_like(rel_logits_real)
                gate_null = torch.ones_like(rel_logits_null)

            # 4.1 shrink delta output when news is absent
            loss_null = (delta_pred_null ** 2).mean()

            # final predictions in z-space
            pred_real_z = base_pred + delta_pred_real
            pred_null_z = base_pred + delta_pred_null
            pred_base_z = base_pred

            # per-sample errors in z-space
            err_real = torch.abs(pred_real_z - targets_z.to(torch.float32)).mean(dim=1)  # (B,)
            err_null = torch.abs(pred_null_z - targets_z.to(torch.float32)).mean(dim=1)  # (B,)
            err_base = torch.abs(pred_base_z - targets_z.to(torch.float32)).mean(dim=1)  # (B,)

            # pseudo label from counterfactual gain; no-news samples are always 0.
            gain = (err_null - err_real) - cf_pseudo_margin
            pseudo_soft = torch.sigmoid(gain / cf_pseudo_temp) * has_news
            if cf_pseudo_hard == 1:
                pseudo_gate = (pseudo_soft >= 0.5).to(torch.float32)
            else:
                pseudo_gate = pseudo_soft
            pseudo_gate = pseudo_gate.detach()
            pseudo_target = (0.8 * pseudo_gate + 0.2 * rel_labels_d * has_news).detach()

            # weighted residual target fitting
            residual_mode = str(getattr(args, "residual_loss", "mae")).lower()
            per_elem_res = _residual_elementwise(delta_pred_real, delta_targets, residual_mode)
            per_sample_res = per_elem_res.mean(dim=1)
            sample_w = cf_min_weight + (1.0 - cf_min_weight) * pseudo_soft.detach()
            sample_w = torch.where(has_news > 0.5, sample_w, torch.ones_like(sample_w))
            loss_res = (sample_w * per_sample_res).sum() / sample_w.sum().clamp_min(1e-6)

            # 4.2 hinge margin: err_null >= err_real + margin
            hinge_margin = torch.relu(delta_adv_margin + err_real - err_null)
            if violation_cap > 0:
                hinge_margin = hinge_margin.clamp(max=violation_cap)
            margin_w = pseudo_soft.detach().clamp_min(1e-3)
            loss_margin = (margin_w * hinge_margin).sum() / margin_w.sum().clamp_min(1e-6)

            # 4.3 non-degradation vs base
            hinge_non_degrade = torch.relu(delta_non_degrade_margin + err_real - err_base)
            if violation_cap > 0:
                hinge_non_degrade = hinge_non_degrade.clamp(max=violation_cap)
            loss_non_degrade = hinge_non_degrade.mean()

            # 4.4 gate training: useful-news samples -> high gate, no-news -> low gate
            if gate_enable:
                loss_gate = F.binary_cross_entropy_with_logits(rel_logits_real, pseudo_target)
                loss_gate_null = F.binary_cross_entropy_with_logits(rel_logits_null, torch.zeros_like(rel_logits_null))
            else:
                loss_gate = torch.zeros((), device=device, dtype=torch.float32)
                loss_gate_null = torch.zeros((), device=device, dtype=torch.float32)

            warm = int(getattr(args, "delta_curriculum_epochs", 0))
            if warm > 0:
                curriculum = min(1.0, float(epoch + 1) / float(warm))
            else:
                curriculum = 1.0

            loss_total = (
                loss_res
                + curriculum * delta_null_lambda * loss_null
                + curriculum * delta_margin_lambda * loss_margin
                + curriculum * delta_non_degrade_lambda * loss_non_degrade
                + curriculum * gate_lambda * loss_gate
                + curriculum * gate_null_lambda * loss_gate_null
            )

            # backward with grad accumulation
            loss = loss_total / args.grad_accum
            loss.backward()

            loss_window.append(float(loss.detach().cpu()))
            if global_step % 10 == 0:
                avg_train_loss = sum(loss_window) / len(loss_window)
                pbar.set_postfix(
                    train_loss=f"{avg_train_loss:.6f}",
                    gate=float(gate_real.mean().detach().cpu()),
                    useful=float(pseudo_soft.mean().detach().cpu()),
                )

            if (global_step + 1) % args.grad_accum == 0:
                grad_clip = float(getattr(args, "delta_grad_clip", 0.0))
                if grad_clip > 0:
                    clip_grad_norm_((p for p in delta_model.parameters() if p.requires_grad), grad_clip)
                optim_delta.step()
                if args.scheduler == 1:
                    scheduler_delta.step()
                optim_delta.zero_grad(set_to_none=True)

            global_step += 1

        # end-of-epoch eval (combined)
        val_loss, val_mse, val_mae = evaluate_metrics_residual(
            base_model=base_teacher,
            delta_model=delta_model,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            testing=False,
            true_pred_csv_path=None,
            news_dropout=False,
        )
        val_loss_per_epoch.append(val_loss)
        mse_loss_per_epoch.append(val_mse)
        mae_loss_per_epoch.append(val_mae)

        live_logger.info(
            f"[DELTA][EVAL] epoch={epoch+1} tpl_id={tpl_id} policy={policy_name} "
            f"val_loss(zMSE)={val_loss:.6f} val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
        )

        # update val_state for context
        if args.reward_metric == "loss":
            val_state.update(val_loss)
            metric_now = val_loss
        elif args.reward_metric == "mse":
            val_state.update(val_mse)
            metric_now = val_mse
        else:
            val_state.update(val_mae)
            metric_now = val_mae

        if metric_now < best_metric - 1e-6:
            best_metric = metric_now
            stale_rounds = 0
            best_tpl_id = tpl_id
            best_policy_name = policy_name

            best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")
            shutil.rmtree(best_delta_path, ignore_errors=True)

            save_checkpoint(
                best_delta_path,
                tokenizer,
                delta_model,
                base_model_id=args.base_model,
                tokenizer_id=args.tokenizer or args.base_model,
                lora_cfg=lora_cfg,
                optimizer=optim_delta,
                scheduler=scheduler_delta,
                epoch=epoch,
                global_step=global_step,
            )

            with open(os.path.join(f"./checkpoints/{args.taskName}", "residual_pair.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_base": os.path.basename(best_base_path),
                        "best_delta": f"best_delta_{args.taskName}",
                        "best_tpl_id": int(best_tpl_id),
                        "best_policy_name": str(best_policy_name),
                        "reward_metric": str(args.reward_metric),
                        "best_metric": float(best_metric),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            live_logger.info(f"[DELTA] New best delta saved: {best_delta_path} ({args.reward_metric}={best_metric:.6f})")
        else:
            stale_rounds += 1
            live_logger.info(f"[DELTA] stale_rounds={stale_rounds}/{args.early_stop_patience} best={best_metric:.6f}")

        if stale_rounds >= args.early_stop_patience:
            live_logger.info(f"[DELTA] Early stopping triggered at epoch {epoch+1}.")
            break

        if epoch == 0:
            live_logger.info("---------------------trainset and valset prompt statistics--------------------------------")
            print_prompt_stats(live_logger, dataStatistic)
            live_logger.info("-----------------------------------------------------")

    # draw_metric_trend(args, live_logger, val_loss_per_epoch, mse_loss_per_epoch, mae_loss_per_epoch)
    dataStatistic.clear()

    # TEST (combined with best delta; base computed via adapter_off)
    if test_loader is not None:
        del delta_model
         # ---- free training-time GPU objects ----
        # del delta_model
        # del optim_delta
        # if scheduler_delta is not None:
        #     del scheduler_delta

        gc.collect()
        torch.cuda.empty_cache()

        best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")

        tok_d, model_best = load_checkpoint(
            best_delta_path,
            args.load_in_4bit,
            args.gradient_checkpointing,
            _single_device_map(args),
            False,
            head_mlp=args.head_mlp,
            hd=args.head_dropout,
            pd=args.patch_dropout


        )

        tokenizer = tok_d
        model_best.to(device)
        model_best.eval()

        live_logger.info(
            "Loaded best DELTA model for testing (final = base(adapter_off,no-news) + delta(adapter_on,news))."
        )

        tpl_for_test = best_tpl_id if args.rl_use == 1 else tpl_id
        pol_for_test = best_policy_name if args.rl_use == 1 else policy_name

        test_loss, test_mse, test_mae = evaluate_metrics_residual(
            base_model=base_teacher,
            delta_model=model_best,
            tokenizer=tokenizer,
            data_loader=test_loader,
            templates=templates,
            tpl_id=tpl_for_test,
            args=args,
            news_df=news_df,
            policy_name=pol_for_test,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_test,
            testing=True,
            true_pred_csv_path=true_pred_csv_path,
            news_dropout=False,
            filename=bundle["test_filename"]
        )

        live_logger.info("---------------------testset prompt statistics--------------------------------")
        print_prompt_stats(live_logger, dataStatistic)
        live_logger.info("-----------------------------------------------------")

        tqdm.write(f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
        live_logger.info(f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")

        record_test_results_csv(args, live_logger, test_mse, test_mae)
        draw_pred_true(live_logger, args, true_pred_csv_path)

    return {
        "test_loader":test_loader,
        "device": device,
        "live_logger": live_logger,
        "best_tpl_id":best_tpl_id,
        "best_policy_name":best_policy_name,
        "tpl_id":tpl_id,
        "policy_name":policy_name,
        "templates":templates,
        "news_df":news_df,
        "policy_kw":policy_kw,
        "volatility_bin_test": volatility_bin_test,
        "true_pred_csv_path": true_pred_csv_path
    }


# ----------------------------
# main entry (refactored)
# ----------------------------
def testing_base(test_loader, args, device, live_logger, templates, volatility_bin_test,true_pred_csv_path):
    if test_loader is not None:
        # del base_model
        gc.collect()
        torch.cuda.empty_cache()

        best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")

        tok_d, model_best = load_checkpoint(
            best_base_path,
            args.load_in_4bit,
            args.gradient_checkpointing,
            _single_device_map(args),
            False,
            head_mlp=args.head_mlp,
            hd=args.head_dropout,
            pd=args.patch_dropout
        )

        tokenizer = tok_d
        model_best.to(device)
        model_best.eval()

        live_logger.info(
            "Loaded best BASE model for testing (final = base(no-news))."
        )

        tpl_for_test = 1

        test_loss, test_mse, test_mae = evaluate_metrics_single(
            model=model_best,
            tokenizer=tokenizer,
            data_loader=test_loader,
            templates=templates,
            tpl_id=tpl_for_test,
            args=args,
            news_df=None,
            policy_name=None,
            policy_kw=None,
            device=device,
            volatility_bin=volatility_bin_test,
            testing=True,
        )

        live_logger.info("---------------------testset prompt statistics--------------------------------")
        print_prompt_stats(live_logger, dataStatistic)
        live_logger.info("-----------------------------------------------------")

        tqdm.write(f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
        live_logger.info(f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")

        record_test_results_csv(args, live_logger, test_mse, test_mae)
        draw_pred_true(live_logger, args, true_pred_csv_path)

def testing_delta(test_loader, args, device, live_logger, best_tpl_id, best_policy_name, tpl_id, policy_name, templates,news_df,policy_kw,volatility_bin_test,true_pred_csv_path):
    if test_loader is not None:
        # del delta_model
        gc.collect()
        torch.cuda.empty_cache()

        best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")

        tok_d, model_best = load_checkpoint(
            best_delta_path,
            args.load_in_4bit,
            args.gradient_checkpointing,
            _single_device_map(args),
            False,
            head_mlp=args.head_mlp,
            hd=args.head_dropout,
            pd=args.patch_dropout
        )

        tokenizer = tok_d
        model_best.to(device)
        model_best.eval()

        live_logger.info(
            "Loaded best DELTA model for testing (final = base(adapter_off,no-news) + delta(adapter_on,news))."
        )

        tpl_for_test = best_tpl_id if args.rl_use == 1 else tpl_id
        pol_for_test = best_policy_name if args.rl_use == 1 else policy_name

        test_loss, test_mse, test_mae = evaluate_metrics_residual(
            base_model=None,
            delta_model=model_best,
            tokenizer=tokenizer,
            data_loader=test_loader,
            templates=templates,
            tpl_id=tpl_for_test,
            args=args,
            news_df=news_df,
            policy_name=pol_for_test,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_test,
            testing=True,
            true_pred_csv_path=true_pred_csv_path,
            news_dropout=False,
        )

        live_logger.info("---------------------testset prompt statistics--------------------------------")
        print_prompt_stats(live_logger, dataStatistic)
        live_logger.info("-----------------------------------------------------")

        tqdm.write(f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
        live_logger.info(f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")

        record_test_results_csv(args, live_logger, test_mse, test_mae)
        draw_pred_true(live_logger, args, true_pred_csv_path)

def main(args):
    bundle = setup_env_and_data(args)
    stage = bundle["stage"]

    ckpt_dir = bundle["ckpt_dir"]
    prompt_path = bundle["prompt_path"]
    ans_json_path = bundle["ans_json_path"]
    true_pred_csv_path = bundle["true_pred_csv_path"]

    with open(prompt_path, "w", encoding="utf-8"):
        pass
    with open(ans_json_path, "w", encoding="utf-8"):
        pass
    with open(true_pred_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pred", "true"])

    # resolve base ckpt path for delta-only / all
    def _resolve_base_ckpt():
        return os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")

    if stage == "base":
        cfg = train_base_stage(args, bundle)

        testing_base(
            bundle["test_loader"],
            args,
            bundle["device"],
            bundle["live_logger"],
            cfg["templates"],
            bundle["volatility_bin_test"],
            bundle["true_pred_csv_path"],
        )
        return

    if stage == "delta":
        best_base_path = _resolve_base_ckpt()
        if not os.path.exists(best_base_path):
            raise FileNotFoundError(f"Base checkpoint not found: {best_base_path}")
        cfg = train_delta_stage(args, bundle, best_base_path=best_base_path, best_base_metric=float("inf"))

        # testing_delta(
        #     cfg["test_loader"],
        #     args,
        #     cfg["device"],
        #     cfg["live_logger"],
        #     cfg["best_tpl_id"],
        #     cfg["best_policy_name"],
        #     cfg["tpl_id"],
        #     cfg["policy_name"],
        #     cfg["templates"],
        #     cfg["news_df"],                    # 对应 news_df 参数
        #     cfg["policy_kw"],
        #     cfg["volatility_bin_test"],
        #     cfg["true_pred_csv_path"],
        # )
        return

    # stage == "all"
    if stage == "all":
        cfg_base = train_base_stage(args, bundle)
        cfg = train_delta_stage(
            args,
            bundle,
            best_base_path=cfg_base["best_base_path"],
            best_base_metric=float(cfg_base["best_base_metric"]),
        )
        # testing_delta(
        #     cfg["test_loader"],
        #     args,
        #     cfg["device"],
        #     cfg["live_logger"],
        #     cfg["best_tpl_id"],
        #     cfg["best_policy_name"],
        #     cfg["tpl_id"],
        #     cfg["policy_name"],
        #     cfg["templates"],
        #     cfg["news_df"],                    # 对应 news_df 参数
        #     cfg["policy_kw"],
        #     cfg["volatility_bin_test"],
        #     cfg["true_pred_csv_path"],
        # )
        return
