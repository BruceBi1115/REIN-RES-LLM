from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoModel

from ..base.common import _eval_iter, _z_batch_tensors, build_delta_batch_inputs
from ..delta.core import (
    _build_delta_residual_position_weights,
    _build_news_usefulness_weights,
    _match_gate_shape,
    _masked_binary_accuracy_from_logits,
    _masked_binary_balanced_accuracy_from_logits,
    _masked_binary_pos_weight,
    _masked_weighted_mean,
    _use_external_signnet,
)
from ..model2 import _resolve_text_spec
from .model import DualStreamTCNSignNet, ResidualSignNet
from .utils import _normalize_external_signnet_variant, _normalize_signnet_select_metric

def _run_external_signnet(
    signnet_model: nn.Module,
    history_z: torch.Tensor,
    base_pred_z: torch.Tensor,
    structured_feats: torch.Tensor,
    news_counts: torch.Tensor,
    tau: float,
    signnet_text_ids: torch.Tensor | None = None,
    signnet_text_attn: torch.Tensor | None = None,
    signnet_text_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = signnet_model(
        history_z=history_z,
        base_pred_z=base_pred_z.to(torch.float32),
        structured_feats=structured_feats.to(torch.float32),
        news_counts=news_counts.to(torch.float32),
        signnet_text_ids=signnet_text_ids,
        signnet_text_attn=signnet_text_attn,
        signnet_text_mask=signnet_text_mask,
    ).to(torch.float32)
    bias = getattr(signnet_model, "decision_bias", None)
    if torch.is_tensor(bias):
        logits = logits + bias.to(device=logits.device, dtype=logits.dtype)
    elif bias is not None:
        logits = logits + float(bias)
    t = max(1e-6, float(tau))
    sign_soft = torch.tanh(logits / t)
    return logits, sign_soft

def _compose_delta_with_external_sign(
    *,
    gate: torch.Tensor,
    magnitude: torch.Tensor,
    sign_soft: torch.Tensor,
    delta_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mag = magnitude.to(torch.float32)
    gate_h = _match_gate_shape(gate.to(torch.float32), mag)
    sign_h = _match_gate_shape(sign_soft.to(torch.float32), mag)
    delta = gate_h * sign_h * mag
    clip_v = float(delta_clip)
    if clip_v > 0.0:
        c = torch.tensor(clip_v, device=delta.device, dtype=delta.dtype)
        delta = c * torch.tanh(delta / c)
    return delta, gate_h

def _evaluate_external_signnet(
    *,
    signnet_model: nn.Module,
    base_backbone,
    tokenizer,
    data_loader,
    templates,
    tpl_id: int,
    args,
    global_zstats,
    news_df,
    policy_name: str,
    policy_kw,
    device,
    volatility_bin,
    api_adapter=None,
    eval_desc: str = "[SIGNNET][VAL]",
    return_tensors: bool = False,
):
    if data_loader is None:
        if return_tensors:
            empty = torch.zeros(0, dtype=torch.float32)
            return 0.0, 0.0, 0.0, 0.0, {"logits": empty, "targets": empty, "mask": empty}
        return 0.0, 0.0, 0.0, 0.0
    signnet_model.eval()
    base_backbone.eval()
    use_news_weighting = int(getattr(args, "delta_sign_external_use_news_weighting", 0)) == 1
    use_residual_weighting = int(getattr(args, "delta_sign_external_use_residual_weighting", 0)) == 1
    use_pos_weight = int(getattr(args, "delta_sign_external_use_pos_weight", 1)) == 1
    pos_weight_floor = float(max(0.0, getattr(args, "delta_sign_external_pos_weight_floor", 0.5)))
    pos_weight_clip = float(max(pos_weight_floor + 1e-6, getattr(args, "delta_sign_external_pos_weight_clip", 3.0)))

    loss_sum = 0.0
    acc_sum = 0.0
    bacc_sum = 0.0
    n_samples = 0
    valid_sum = 0.0
    total_positions = 0.0
    all_logits = []
    all_targets = []
    all_masks = []
    loader, use_pbar = _eval_iter(data_loader, args, desc=eval_desc)

    for _, batch in enumerate(loader):
        history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
        history_z = history_z.to(device)
        base_pred = base_backbone(history_z).to(torch.float32)

        delta_inputs = build_delta_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            testing=False,
            force_no_news=False,
            news_dropout=False,
            api_adapter=api_adapter,
        )
        targets_z = delta_inputs["targets_z"].to(device=device, dtype=torch.float32)
        structured_feats = delta_inputs["structured_feats"].to(device=device, dtype=torch.float32)
        news_counts = delta_inputs["news_counts"].to(device=device, dtype=torch.float32)
        signnet_text_ids = delta_inputs["signnet_text_ids"].to(device=device, dtype=torch.long)
        signnet_text_attn = delta_inputs["signnet_text_attn"].to(device=device, dtype=torch.long)
        signnet_text_mask = delta_inputs["signnet_text_mask"].to(device=device, dtype=torch.float32)
        sign_logits, _ = _run_external_signnet(
            signnet_model=signnet_model,
            history_z=history_z.to(torch.float32),
            base_pred_z=base_pred,
            structured_feats=structured_feats,
            news_counts=news_counts,
            tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
            signnet_text_ids=signnet_text_ids,
            signnet_text_attn=signnet_text_attn,
            signnet_text_mask=signnet_text_mask,
        )

        true_residual_z = targets_z - base_pred
        abs_residual = true_residual_z.abs()
        sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
        valid_mask = (abs_residual > sign_eps).to(torch.float32)
        sign_target_bin = (true_residual_z > 0).to(torch.float32)

        has_news = (news_counts > 0).to(torch.float32)
        if use_news_weighting:
            sample_weight = _build_news_usefulness_weights(
                has_news=has_news,
                news_counts=news_counts,
                structured_feats=structured_feats,
                enabled=True,
            )
        else:
            sample_weight = torch.ones_like(news_counts, dtype=torch.float32)
        if use_residual_weighting:
            position_weight = _build_delta_residual_position_weights(abs_residual, args)
        else:
            position_weight = torch.ones_like(abs_residual, dtype=torch.float32)
        sample_pos_weight = sample_weight.unsqueeze(1) * position_weight

        bce_kwargs = {"reduction": "none"}
        if use_pos_weight:
            bce_kwargs["pos_weight"] = _masked_binary_pos_weight(
                sign_target_bin,
                valid_mask,
                weight_floor=pos_weight_floor,
                weight_clip=pos_weight_clip,
            )
        sign_bce = F.binary_cross_entropy_with_logits(
            sign_logits,
            sign_target_bin,
            **bce_kwargs,
        )
        loss = _masked_weighted_mean(sign_bce, sample_pos_weight, mask=valid_mask)
        acc = _masked_binary_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)
        bacc = _masked_binary_balanced_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)

        bs = int(targets_z.size(0))
        loss_sum += float(loss.detach().cpu()) * bs
        acc_sum += float(acc.detach().cpu()) * bs
        bacc_sum += float(bacc.detach().cpu()) * bs
        n_samples += bs
        valid_sum += float(valid_mask.sum().detach().cpu())
        total_positions += float(valid_mask.numel())
        if return_tensors:
            all_logits.append(sign_logits.detach().cpu())
            all_targets.append(sign_target_bin.detach().cpu())
            all_masks.append(valid_mask.detach().cpu())
        if use_pbar:
            loader.set_postfix(loss=f"{loss_sum / max(1, n_samples):.6f}")

    eval_tuple = (
        loss_sum / max(1, n_samples),
        acc_sum / max(1, n_samples),
        valid_sum / max(1.0, total_positions),
        bacc_sum / max(1, n_samples),
    )
    if not return_tensors:
        return eval_tuple
    if all_logits:
        pack = {
            "logits": torch.cat(all_logits, dim=0).to(torch.float32),
            "targets": torch.cat(all_targets, dim=0).to(torch.float32),
            "mask": torch.cat(all_masks, dim=0).to(torch.float32),
        }
    else:
        empty = torch.zeros(0, dtype=torch.float32)
        pack = {"logits": empty, "targets": empty, "mask": empty}
    return eval_tuple[0], eval_tuple[1], eval_tuple[2], eval_tuple[3], pack

def _calibrate_external_signnet_bias(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    objective: str = "acc",
    max_abs_bias: float = 2.0,
) -> tuple[float, float, int]:
    if logits.numel() <= 0 or targets.numel() <= 0 or mask.numel() <= 0:
        return 0.0, 0.0, 0
    x = logits.reshape(-1).to(torch.float32)
    y = targets.reshape(-1).to(torch.float32)
    m = mask.reshape(-1).to(torch.float32) > 0.5
    if int(m.sum().item()) <= 0:
        return 0.0, 0.0, 0
    x = x[m]
    y = (y[m] > 0.5).to(torch.bool)
    if x.numel() <= 0:
        return 0.0, 0.0, 0

    q = torch.linspace(0.02, 0.98, 97, dtype=torch.float32, device=x.device)
    cand = torch.quantile(x, q)
    zero = torch.zeros(1, dtype=torch.float32, device=x.device)
    cand = torch.unique(torch.cat([cand, zero], dim=0))
    pred_mat = x.unsqueeze(0) > cand.unsqueeze(1)
    obj = _normalize_signnet_select_metric(objective)
    use_bacc = obj == "balanced_acc"
    if use_bacc:
        n_pos = int(y.to(torch.float32).sum().item())
        n_neg = int((~y).to(torch.float32).sum().item())
        if n_pos <= 0 or n_neg <= 0:
            use_bacc = False
    if use_bacc:
        y_row = y.unsqueeze(0)
        tp = (pred_mat & y_row).to(torch.float32).sum(dim=1)
        fn = ((~pred_mat) & y_row).to(torch.float32).sum(dim=1)
        tn = ((~pred_mat) & (~y_row)).to(torch.float32).sum(dim=1)
        fp = (pred_mat & (~y_row)).to(torch.float32).sum(dim=1)
        score_vec = 0.5 * (tp / (tp + fn).clamp_min(1.0) + tn / (tn + fp).clamp_min(1.0))
    else:
        score_vec = (pred_mat == y.unsqueeze(0)).to(torch.float32).mean(dim=1)

    best_idx = int(torch.argmax(score_vec).item())
    best_thr = float(cand[best_idx].item())
    bias = float(-best_thr)
    clip_v = float(max(0.0, max_abs_bias))
    if clip_v > 0.0:
        bias = float(max(-clip_v, min(clip_v, bias)))
    pred_final = (x + bias) > 0
    if use_bacc:
        tp = ((pred_final & y)).to(torch.float32).sum()
        fn = (((~pred_final) & y)).to(torch.float32).sum()
        tn = (((~pred_final) & (~y))).to(torch.float32).sum()
        fp = ((pred_final & (~y))).to(torch.float32).sum()
        final_score = float((0.5 * (tp / (tp + fn).clamp_min(1.0) + tn / (tn + fp).clamp_min(1.0))).item())
    else:
        final_score = float((pred_final == y).to(torch.float32).mean().item())
    return bias, final_score, int(x.numel())

def _train_external_signnet(
    *,
    args,
    base_backbone,
    tokenizer,
    templates,
    tpl_id: int,
    policy_name: str,
    policy_kw,
    train_loader,
    val_loader,
    test_loader,
    news_df,
    volatility_bin,
    volatility_bin_val,
    volatility_bin_test,
    global_zstats,
    device,
    live_logger,
    api_adapter=None,
) -> nn.Module | None:
    if not _use_external_signnet(args):
        return None
    if val_loader is None:
        raise ValueError("delta_sign_mode=signnet_binary requires a non-empty val_loader for signnet selection.")

    epochs = int(max(0, getattr(args, "delta_sign_external_epochs", 0)))
    if epochs <= 0:
        raise ValueError("delta_sign_mode=signnet_binary requires delta_sign_external_epochs > 0.")

    structured_dim = int(getattr(args, "delta_structured_feature_dim", 12))
    signnet_variant = _normalize_external_signnet_variant(getattr(args, "delta_sign_external_variant", "mlp"))
    signnet_text_model_id = ""
    signnet_text_dim = int(max(32, getattr(args, "delta_sign_external_text_dim", 64)))
    if signnet_variant == "dual_stream_tcn":
        model_id, _tok_id = _resolve_text_spec(
            getattr(args, "tiny_news_model_preset", "custom"),
            getattr(args, "tiny_news_model", ""),
            getattr(args, "tiny_news_tokenizer", ""),
        )
        signnet_text_model_id = str(model_id)
        try:
            text_encoder = AutoModel.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dual_stream_tcn signnet text encoder from {model_id!r}. "
                "Please use a cached encoder-style tiny_news preset/model such as distilbert or bert_base."
            ) from exc
        text_hidden = getattr(text_encoder.config, "hidden_size", None)
        if text_hidden is None:
            text_hidden = getattr(text_encoder.config, "n_embd", None)
        if text_hidden is None:
            raise ValueError(f"Cannot infer hidden size for signnet text encoder {model_id!r}.")
        signnet = DualStreamTCNSignNet(
            history_len=int(max(1, getattr(args, "history_len", 1))),
            horizon=int(max(1, getattr(args, "horizon", 1))),
            structured_dim=max(0, structured_dim),
            text_encoder=text_encoder,
            text_encoder_hidden_size=int(text_hidden),
            hidden_size=int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
            text_low_dim=signnet_text_dim,
            dropout=float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
        ).to(device)
    else:
        signnet = ResidualSignNet(
            history_len=int(max(1, getattr(args, "history_len", 1))),
            horizon=int(max(1, getattr(args, "horizon", 1))),
            structured_dim=max(0, structured_dim),
            hidden_size=int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
            dropout=float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
        ).to(device)

    lr = float(getattr(args, "delta_sign_external_lr", args.lr))
    wd = float(getattr(args, "delta_sign_external_weight_decay", args.weight_decay))
    grad_clip = float(getattr(args, "delta_sign_external_grad_clip", 1.0))
    patience = int(max(0, getattr(args, "delta_sign_external_patience", 0)))
    min_delta = float(max(0.0, getattr(args, "delta_sign_external_min_delta", 1e-4)))
    select_metric = _normalize_signnet_select_metric(getattr(args, "delta_sign_external_select_metric", "acc"))
    lr_factor = float(getattr(args, "delta_sign_external_lr_factor", 0.5))
    lr_patience = int(max(0, getattr(args, "delta_sign_external_lr_patience", 1)))
    min_lr = float(max(0.0, getattr(args, "delta_sign_external_min_lr", 1e-5)))
    calibrate_bias = int(getattr(args, "delta_sign_external_calibrate_bias", 1)) == 1
    bias_clip = float(max(0.0, getattr(args, "delta_sign_external_bias_clip", 2.0)))
    signnet_news_dropout = int(getattr(args, "delta_sign_external_news_dropout", 0)) == 1
    use_news_weighting = int(getattr(args, "delta_sign_external_use_news_weighting", 0)) == 1
    use_residual_weighting = int(getattr(args, "delta_sign_external_use_residual_weighting", 0)) == 1
    use_pos_weight = int(getattr(args, "delta_sign_external_use_pos_weight", 1)) == 1
    pos_weight_floor = float(max(0.0, getattr(args, "delta_sign_external_pos_weight_floor", 0.5)))
    pos_weight_clip = float(max(pos_weight_floor + 1e-6, getattr(args, "delta_sign_external_pos_weight_clip", 3.0)))
    if hasattr(signnet, "decision_bias"):
        signnet.decision_bias.data.zero_()
    trainable_params = [p for p in signnet.parameters() if p.requires_grad]
    if len(trainable_params) <= 0:
        raise ValueError("External signnet has no trainable parameters.")
    opt = AdamW(trainable_params, lr=lr, weight_decay=wd)
    scheduler = None
    if 0.0 < lr_factor < 1.0:
        scheduler_mode = "min" if select_metric == "loss" else "max"
        scheduler = ReduceLROnPlateau(
            opt,
            mode=scheduler_mode,
            factor=lr_factor,
            patience=max(1, lr_patience),
            min_lr=min_lr,
        )

    best_val = float("inf")
    best_acc = float("-inf")
    best_bacc = float("-inf")
    best_state = None
    stale = 0
    if live_logger is not None:
        live_logger.info(
            "[SIGNNET] pretrain start: "
            f"variant={signnet_variant} "
            f"epochs={epochs} lr={lr:.3e} wd={wd:.3e} hidden={int(max(32, getattr(args, 'delta_sign_external_hidden', 256)))} "
            f"dropout={float(max(0.0, getattr(args, 'delta_sign_external_dropout', 0.1))):.3f} "
            f"patience={patience} select_metric={select_metric} min_delta={min_delta:.1e} "
            f"lr_sched_factor={lr_factor:.3f} lr_sched_patience={max(1, lr_patience)} min_lr={min_lr:.3e} "
            f"news_dropout={int(signnet_news_dropout)} calibrate_bias={int(calibrate_bias)} bias_clip={bias_clip:.3f} "
            f"use_news_weighting={int(use_news_weighting)} use_residual_weighting={int(use_residual_weighting)} "
            f"use_pos_weight={int(use_pos_weight)} pos_weight_floor={pos_weight_floor:.3f} pos_weight_clip={pos_weight_clip:.3f} "
            f"text_dim={signnet_text_dim if signnet_variant == 'dual_stream_tcn' else 0} "
            f"text_model={signnet_text_model_id or 'n/a'}"
        )

    for epoch in range(epochs):
        signnet.train()
        base_backbone.eval()
        pbar = tqdm(train_loader, desc=f"[SIGNNET] Epoch {epoch+1}/{epochs}")
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_bacc_sum = 0.0
        n_samples = 0

        for _, batch in enumerate(pbar):
            with torch.no_grad():
                history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
                history_z = history_z.to(device)
                base_pred = base_backbone(history_z).to(torch.float32)

            delta_inputs = build_delta_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                testing=False,
                force_no_news=False,
                news_dropout=signnet_news_dropout,
                api_adapter=api_adapter,
            )
            targets_z = delta_inputs["targets_z"].to(device=device, dtype=torch.float32)
            structured_feats = delta_inputs["structured_feats"].to(device=device, dtype=torch.float32)
            news_counts = delta_inputs["news_counts"].to(device=device, dtype=torch.float32)
            signnet_text_ids = delta_inputs["signnet_text_ids"].to(device=device, dtype=torch.long)
            signnet_text_attn = delta_inputs["signnet_text_attn"].to(device=device, dtype=torch.long)
            signnet_text_mask = delta_inputs["signnet_text_mask"].to(device=device, dtype=torch.float32)
            sign_logits, _ = _run_external_signnet(
                signnet_model=signnet,
                history_z=history_z.to(torch.float32),
                base_pred_z=base_pred,
                structured_feats=structured_feats,
                news_counts=news_counts,
                tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
                signnet_text_ids=signnet_text_ids,
                signnet_text_attn=signnet_text_attn,
                signnet_text_mask=signnet_text_mask,
            )

            true_residual_z = targets_z - base_pred
            abs_residual = true_residual_z.abs()
            sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
            valid_mask = (abs_residual > sign_eps).to(torch.float32)
            sign_target_bin = (true_residual_z > 0).to(torch.float32)

            has_news = (news_counts > 0).to(torch.float32)
            if use_news_weighting:
                sample_weight = _build_news_usefulness_weights(
                    has_news=has_news,
                    news_counts=news_counts,
                    structured_feats=structured_feats,
                    enabled=True,
                )
            else:
                sample_weight = torch.ones_like(news_counts, dtype=torch.float32)
            if use_residual_weighting:
                position_weight = _build_delta_residual_position_weights(abs_residual, args)
            else:
                position_weight = torch.ones_like(abs_residual, dtype=torch.float32)
            sample_pos_weight = sample_weight.unsqueeze(1) * position_weight

            bce_kwargs = {"reduction": "none"}
            if use_pos_weight:
                bce_kwargs["pos_weight"] = _masked_binary_pos_weight(
                    sign_target_bin,
                    valid_mask,
                    weight_floor=pos_weight_floor,
                    weight_clip=pos_weight_clip,
                )
            sign_bce = F.binary_cross_entropy_with_logits(
                sign_logits,
                sign_target_bin,
                **bce_kwargs,
            )
            loss = _masked_weighted_mean(sign_bce, sample_pos_weight, mask=valid_mask)
            acc = _masked_binary_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)
            bacc = _masked_binary_balanced_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                clip_grad_norm_(signnet.parameters(), grad_clip)
            opt.step()

            bs = int(targets_z.size(0))
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_acc_sum += float(acc.detach().cpu()) * bs
            train_bacc_sum += float(bacc.detach().cpu()) * bs
            n_samples += bs
            pbar.set_postfix(
                loss=f"{train_loss_sum / max(1, n_samples):.6f}",
                acc=f"{train_acc_sum / max(1, n_samples):.4f}",
                bacc=f"{train_bacc_sum / max(1, n_samples):.4f}",
            )

        val_loss, val_acc, val_valid, val_bacc = _evaluate_external_signnet(
            signnet_model=signnet,
            base_backbone=base_backbone,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            api_adapter=api_adapter,
            eval_desc=f"[SIGNNET][VAL] Epoch {epoch+1}/{epochs}",
        )
        if live_logger is not None:
            curr_lr = float(opt.param_groups[0]["lr"])
            live_logger.info(
                f"[SIGNNET][EVAL] epoch={epoch+1} "
                f"train_loss={train_loss_sum / max(1, n_samples):.6f} "
                f"train_acc={train_acc_sum / max(1, n_samples):.4f} "
                f"train_bacc={train_bacc_sum / max(1, n_samples):.4f} "
                f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} val_bacc={val_bacc:.4f} val_valid={val_valid:.4f} lr={curr_lr:.3e}"
            )
        if scheduler is not None:
            if select_metric == "loss":
                scheduler.step(float(val_loss))
            elif select_metric == "balanced_acc":
                scheduler.step(float(val_bacc))
            else:
                scheduler.step(float(val_acc))

        improved = False
        if select_metric == "acc":
            if val_acc > best_acc + min_delta:
                improved = True
            elif abs(val_acc - best_acc) <= min_delta and val_loss < best_val - 1e-6:
                improved = True
        elif select_metric == "balanced_acc":
            if val_bacc > best_bacc + min_delta:
                improved = True
            elif abs(val_bacc - best_bacc) <= min_delta and val_loss < best_val - 1e-6:
                improved = True
        else:
            if val_loss < best_val - min_delta:
                improved = True

        if improved:
            best_val = float(val_loss)
            best_acc = float(val_acc)
            best_bacc = float(val_bacc)
            best_state = {k: v.detach().cpu().clone() for k, v in signnet.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                if live_logger is not None:
                    live_logger.info(f"[SIGNNET] early stop at epoch={epoch+1}")
                break

    if best_state is not None:
        signnet.load_state_dict(best_state, strict=False)
    signnet.eval()
    for p in signnet.parameters():
        p.requires_grad = False

    if calibrate_bias:
        _, _, _, _, val_pack = _evaluate_external_signnet(
            signnet_model=signnet,
            base_backbone=base_backbone,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            api_adapter=api_adapter,
            eval_desc="[SIGNNET][VAL][CALIBRATE]",
            return_tensors=True,
        )
        calibrate_objective = "balanced_acc" if select_metric == "balanced_acc" else "acc"
        bias, cal_score, n_valid = _calibrate_external_signnet_bias(
            logits=val_pack["logits"],
            targets=val_pack["targets"],
            mask=val_pack["mask"],
            objective=calibrate_objective,
            max_abs_bias=bias_clip,
        )
        if hasattr(signnet, "decision_bias"):
            signnet.decision_bias.data.fill_(float(bias))
        if live_logger is not None:
            live_logger.info(
                f"[SIGNNET][CALIBRATE] decision_bias={bias:.6f} objective={calibrate_objective} "
                f"val_score_cal={cal_score:.4f} n_valid={n_valid}"
            )

    if test_loader is not None:
        test_loss, test_acc, test_valid, test_bacc = _evaluate_external_signnet(
            signnet_model=signnet,
            base_backbone=base_backbone,
            tokenizer=tokenizer,
            data_loader=test_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_test,
            api_adapter=api_adapter,
            eval_desc="[SIGNNET][TEST]",
        )
    else:
        test_loss, test_acc, test_valid, test_bacc = float("nan"), float("nan"), float("nan"), float("nan")
    signnet_ckpt_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_signnet_{args.taskName}.pt")
    os.makedirs(os.path.dirname(signnet_ckpt_path), exist_ok=True)
    torch.save(
        {
            "state_dict": signnet.state_dict(),
            "variant": signnet_variant,
            "history_len": int(max(1, getattr(args, "history_len", 1))),
            "horizon": int(max(1, getattr(args, "horizon", 1))),
            "structured_dim": int(max(0, structured_dim)),
            "hidden_size": int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
            "dropout": float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
            "text_dim": int(signnet_text_dim if signnet_variant == "dual_stream_tcn" else 0),
            "text_model_id": str(signnet_text_model_id or ""),
            "best_val_loss": float(best_val),
            "best_val_acc": float(best_acc),
            "best_val_bacc": float(best_bacc),
            "decision_bias": float(getattr(signnet, "decision_bias", torch.zeros((), dtype=torch.float32)).detach().cpu().item())
            if torch.is_tensor(getattr(signnet, "decision_bias", None))
            else float(getattr(signnet, "decision_bias", 0.0)),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_bacc": float(test_bacc),
            "test_valid": float(test_valid),
        },
        signnet_ckpt_path,
    )
    if live_logger is not None:
        bias_now = 0.0
        if hasattr(signnet, "decision_bias"):
            try:
                bias_now = float(signnet.decision_bias.detach().cpu().item())
            except Exception:
                bias_now = 0.0
        live_logger.info(
            f"[SIGNNET][TEST] loss={test_loss:.6f} acc={test_acc:.4f} bacc={test_bacc:.4f} valid={test_valid:.4f} "
            f"bias={bias_now:.6f} "
            f"ckpt={signnet_ckpt_path}"
        )
    return signnet
