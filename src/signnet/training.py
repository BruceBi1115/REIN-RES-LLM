from __future__ import annotations
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ..base.common import _eval_iter, _z_batch_tensors, build_delta_batch_inputs
from ..delta.core import (
    _build_cleaned_residual_targets,
    _build_delta_targets,
    _build_delta_residual_position_weights,
    _build_news_usefulness_weights,
    _build_relative_state_targets,
    _build_soft_sign_targets,
    _build_windowed_sign_targets,
    _match_horizon_shape,
    _masked_binary_accuracy_from_logits,
    _masked_binary_balanced_accuracy_from_logits,
    _masked_multiclass_inverse_freq_weights,
    _masked_binary_pos_weight,
    _masked_multiclass_accuracy_from_logits,
    _masked_multiclass_macro_balanced_accuracy_from_logits,
    _masked_weighted_mean,
    _relative_state_score_from_logits,
    _resolve_residual_arch,
    _resolve_delta_residual_mode,
    _use_external_signnet,
)
from .model import ResidualSignNet
from .utils import _normalize_signnet_select_metric


def _derive_has_news_from_structured_feats(structured_feats: torch.Tensor) -> torch.Tensor:
    sf = structured_feats.to(torch.float32)
    if sf.ndim == 1:
        sf = sf.unsqueeze(0)
    if sf.ndim != 2:
        sf = sf.reshape(sf.size(0), -1)
    return (sf.abs().sum(dim=1) > 0.0).to(torch.float32)


def _signnet_task_type(args) -> str:
    return "relative_state" if _resolve_delta_residual_mode(args) == "relative" else "binary_sign"


def _build_sign_label_pack(delta_target: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cleaned = delta_target.to(torch.float32)
    sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
    mode = str(getattr(args, "sign_label_mode", "hard") or "hard").lower().strip()
    if mode == "soft":
        loss_target = _build_soft_sign_targets(
            cleaned,
            temperature=float(getattr(args, "sign_soft_temperature", 1.0)),
        )
        hard_target = (cleaned > 0).to(torch.float32)
        valid_mask = (cleaned.abs() > sign_eps).to(torch.float32)
        return loss_target, hard_target, valid_mask
    if mode == "windowed":
        smoothed = _build_windowed_sign_targets(
            cleaned,
            window_size=int(getattr(args, "sign_window_size", 8)),
        )
        valid_mask = torch.ones_like(smoothed, dtype=torch.float32)
        return smoothed, smoothed, valid_mask
    hard_target = (cleaned > 0).to(torch.float32)
    valid_mask = (cleaned.abs() > sign_eps).to(torch.float32)
    return hard_target, hard_target, valid_mask


def _encode_text_pack_from_tower(
    *,
    temporal_text_tower,
    ts_patches: torch.Tensor | None,
    temporal_text_ids: torch.Tensor | None,
    temporal_text_attn: torch.Tensor | None,
    temporal_text_step_mask: torch.Tensor | None,
    device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | None]:
    if (
        temporal_text_tower is None
        or ts_patches is None
        or temporal_text_ids is None
        or temporal_text_attn is None
        or ts_patches.ndim < 2
    ):
        return {
            "text_summary": None,
            "text_strength": None,
        }
    pack = temporal_text_tower(
        temporal_text_ids=temporal_text_ids,
        temporal_text_attn=temporal_text_attn,
        temporal_text_step_mask=temporal_text_step_mask,
        target_patch_count=int(max(1, ts_patches.size(1))),
        device=device,
        dtype=dtype,
    )
    return {
        "text_summary": pack.get("text_summary"),
        "text_strength": pack.get("text_strength"),
    }


def _run_external_signnet(
    signnet_model: nn.Module,
    history_z: torch.Tensor,
    base_pred_z: torch.Tensor,
    structured_feats: torch.Tensor,
    text_summary: torch.Tensor | None,
    text_strength: torch.Tensor | None,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = signnet_model(
        history_z=history_z,
        base_pred_z=base_pred_z.to(torch.float32),
        structured_feats=structured_feats.to(torch.float32),
        text_summary=text_summary,
        text_strength=text_strength,
    ).to(torch.float32)
    if str(getattr(signnet_model, "task_type", "binary_sign")) == "relative_state":
        return logits, _relative_state_score_from_logits(logits)
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
    magnitude: torch.Tensor,
    sign_soft: torch.Tensor,
    delta_clip: float,
    residual_mode: str = "additive",
    relative_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = str(residual_mode or "additive").lower().strip()
    mag = magnitude.to(torch.float32)
    sign_h = _match_horizon_shape(sign_soft.to(torch.float32), mag)
    if mode == "relative":
        if relative_delta is not None:
            mag = relative_delta.to(torch.float32).abs()
        delta = mag * sign_h.clamp(-1.0, 1.0)
    else:
        delta = sign_h * mag
    clip_v = float(delta_clip)
    if clip_v > 0.0:
        c = torch.tensor(clip_v, device=delta.device, dtype=delta.dtype)
        delta = c * torch.tanh(delta / c)
    return delta

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
    temporal_text_tokenizer=None,
    temporal_text_tower=None,
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
    if temporal_text_tower is not None:
        temporal_text_tower.eval()
    relative_mode = _resolve_delta_residual_mode(args) == "relative"
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
            temporal_text_tokenizer=temporal_text_tokenizer,
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
        ts_patches = delta_inputs["ts_patches"].to(device=device, dtype=torch.float32)
        targets_z = delta_inputs["targets_z"].to(device=device, dtype=torch.float32)
        structured_feats = delta_inputs["structured_feats"].to(device=device, dtype=torch.float32)
        temporal_text_ids = delta_inputs.get("temporal_text_ids")
        temporal_text_attn = delta_inputs.get("temporal_text_attn")
        temporal_text_step_mask = delta_inputs.get("temporal_text_step_mask")
        if temporal_text_ids is not None:
            temporal_text_ids = temporal_text_ids.to(device=device, dtype=torch.long)
        if temporal_text_attn is not None:
            temporal_text_attn = temporal_text_attn.to(device=device, dtype=torch.long)
        if temporal_text_step_mask is not None:
            temporal_text_step_mask = temporal_text_step_mask.to(device=device, dtype=torch.long)
        text_pack = _encode_text_pack_from_tower(
            temporal_text_tower=temporal_text_tower,
            ts_patches=ts_patches,
            temporal_text_ids=temporal_text_ids,
            temporal_text_attn=temporal_text_attn,
            temporal_text_step_mask=temporal_text_step_mask,
            device=device,
            dtype=targets_z.dtype,
        )
        sign_logits, _ = _run_external_signnet(
            signnet_model=signnet_model,
            history_z=history_z.to(torch.float32),
            base_pred_z=base_pred,
            structured_feats=structured_feats,
            text_summary=text_pack.get("text_summary"),
            text_strength=text_pack.get("text_strength"),
            tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
        )

        raw_residual_target = _build_delta_targets(
            targets_z=targets_z,
            base_pred=base_pred,
            mu_global=float(global_zstats["mu_global"]),
            sigma_global=float(global_zstats["sigma_global"]),
            args=args,
        )
        if relative_mode:
            abs_residual = raw_residual_target.abs()
            sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
            non_neutral_mask = (abs_residual > sign_eps).to(torch.float32)
            valid_mask = non_neutral_mask
            state_targets = _build_relative_state_targets(
                raw_residual_target,
                args,
                structured_feats=structured_feats,
            )
        else:
            cleaned_residual_target = _build_cleaned_residual_targets(
                raw_residual=raw_residual_target,
                structured_feats=structured_feats,
                args=args,
            )
            abs_residual = cleaned_residual_target.abs()
            sign_target_loss, sign_target_bin, valid_mask = _build_sign_label_pack(cleaned_residual_target, args)

        has_news = _derive_has_news_from_structured_feats(structured_feats)
        if use_news_weighting:
            sample_weight = _build_news_usefulness_weights(
                has_news=has_news,
                structured_feats=structured_feats,
                enabled=True,
            )
        else:
            sample_weight = torch.ones_like(has_news, dtype=torch.float32)
        if use_residual_weighting:
            position_weight = _build_delta_residual_position_weights(abs_residual, args)
        else:
            position_weight = torch.ones_like(abs_residual, dtype=torch.float32)
        sample_pos_weight = sample_weight.unsqueeze(1) * position_weight

        if relative_mode:
            ce = F.cross_entropy(
                sign_logits.reshape(-1, sign_logits.size(-1)),
                state_targets.reshape(-1),
                reduction="none",
            ).reshape_as(state_targets)
            class_weight = _masked_multiclass_inverse_freq_weights(
                state_targets,
                valid_mask,
                num_classes=int(sign_logits.size(-1)),
            )
            loss = _masked_weighted_mean(ce, sample_pos_weight * class_weight, mask=valid_mask)
            acc = _masked_multiclass_accuracy_from_logits(sign_logits, state_targets, valid_mask)
            bacc = _masked_multiclass_macro_balanced_accuracy_from_logits(sign_logits, state_targets, valid_mask)
        else:
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
                sign_target_loss,
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
        if relative_mode:
            valid_sum += float(non_neutral_mask.sum().detach().cpu())
        else:
            valid_sum += float(valid_mask.sum().detach().cpu())
        total_positions += float(valid_mask.numel())
        if return_tensors:
            all_logits.append(sign_logits.detach().cpu())
            all_targets.append((state_targets if relative_mode else sign_target_bin).detach().cpu())
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
    temporal_text_tokenizer=None,
    temporal_text_tower=None,
) -> nn.Module | None:
    if not _use_external_signnet(args):
        return None
    if val_loader is None:
        raise ValueError("delta_sign_mode=signnet_binary requires a non-empty val_loader for signnet selection.")

    epochs = int(max(0, getattr(args, "delta_sign_external_epochs", 0)))
    if epochs <= 0:
        raise ValueError("delta_sign_mode=signnet_binary requires delta_sign_external_epochs > 0.")

    signnet_task_type = _signnet_task_type(args)
    relative_mode = signnet_task_type == "relative_state"
    structured_dim = int(getattr(args, "delta_structured_feature_dim", 12))
    text_summary_dim = int(getattr(temporal_text_tower, "hidden_size", 0)) if temporal_text_tower is not None else 0
    signnet = ResidualSignNet(
        history_len=int(max(1, getattr(args, "history_len", 1))),
        horizon=int(max(1, getattr(args, "horizon", 1))),
        structured_dim=max(0, structured_dim),
        text_summary_dim=max(0, text_summary_dim),
        hidden_size=int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
        dropout=float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
        task_type=signnet_task_type,
        arch="plan_c_mvp" if str(getattr(args, "delta_multimodal_arch", "summary_gated")).lower().strip() == "plan_c_mvp" else "mlp",
        multimodal_fuse_lambda=float(max(0.0, getattr(args, "delta_multimodal_fuse_lambda", 1.0))),
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
    signnet_news_dropout_enabled = int(getattr(args, "delta_sign_external_news_dropout", 0)) == 1
    signnet_news_dropout = float(getattr(args, "news_dropout", 0.0) or 0.0) if signnet_news_dropout_enabled else 0.0
    use_news_weighting = int(getattr(args, "delta_sign_external_use_news_weighting", 0)) == 1
    use_residual_weighting = int(getattr(args, "delta_sign_external_use_residual_weighting", 0)) == 1
    use_pos_weight = int(getattr(args, "delta_sign_external_use_pos_weight", 1)) == 1
    pos_weight_floor = float(max(0.0, getattr(args, "delta_sign_external_pos_weight_floor", 0.5)))
    pos_weight_clip = float(max(pos_weight_floor + 1e-6, getattr(args, "delta_sign_external_pos_weight_clip", 3.0)))
    if hasattr(signnet, "decision_bias"):
        signnet.decision_bias.data.zero_()
    trainable_params = [p for p in signnet.parameters() if p.requires_grad]
    tower_trainable_params = []
    if temporal_text_tower is not None:
        tower_trainable_params = [p for p in temporal_text_tower.parameters() if p.requires_grad]
        trainable_params.extend(tower_trainable_params)
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
    best_tower_state = None
    stale = 0
    if live_logger is not None:
        live_logger.info(
            "[SIGNNET] pretrain start: "
            f"variant={str(getattr(signnet, 'model_variant', 'mlp'))} task_type={signnet_task_type} "
            f"epochs={epochs} lr={lr:.3e} wd={wd:.3e} hidden={int(max(32, getattr(args, 'delta_sign_external_hidden', 256)))} "
            f"dropout={float(max(0.0, getattr(args, 'delta_sign_external_dropout', 0.1))):.3f} "
            f"patience={patience} select_metric={select_metric} min_delta={min_delta:.1e} "
            f"lr_sched_factor={lr_factor:.3f} lr_sched_patience={max(1, lr_patience)} min_lr={min_lr:.3e} "
            f"news_dropout={signnet_news_dropout:.3f} calibrate_bias={int(calibrate_bias)} bias_clip={bias_clip:.3f} "
            f"use_news_weighting={int(use_news_weighting)} use_residual_weighting={int(use_residual_weighting)} "
            f"use_pos_weight={int(use_pos_weight)} pos_weight_floor={pos_weight_floor:.3f} pos_weight_clip={pos_weight_clip:.3f} "
            f"text_summary_dim={text_summary_dim} "
            f"shared_tower_trainable={int(len(tower_trainable_params) > 0)}"
        )
        live_logger.info(
            "[SIGNNET] cleaned residual supervision: "
            f"enable={int(getattr(args, 'cleaned_residual_enable', 1))} "
            f"smooth_alpha={float(getattr(args, 'cleaned_residual_smooth_alpha', 0.6) or 0.0):.3f} "
            f"structured_mix={float(getattr(args, 'cleaned_residual_structured_mix', 0.35) or 0.0):.3f}"
        )

    for epoch in range(epochs):
        signnet.train()
        base_backbone.eval()
        if temporal_text_tower is not None:
            temporal_text_tower.train()
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
                temporal_text_tokenizer=temporal_text_tokenizer,
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
            ts_patches = delta_inputs["ts_patches"].to(device=device, dtype=torch.float32)
            targets_z = delta_inputs["targets_z"].to(device=device, dtype=torch.float32)
            structured_feats = delta_inputs["structured_feats"].to(device=device, dtype=torch.float32)
            temporal_text_ids = delta_inputs.get("temporal_text_ids")
            temporal_text_attn = delta_inputs.get("temporal_text_attn")
            temporal_text_step_mask = delta_inputs.get("temporal_text_step_mask")
            if temporal_text_ids is not None:
                temporal_text_ids = temporal_text_ids.to(device=device, dtype=torch.long)
            if temporal_text_attn is not None:
                temporal_text_attn = temporal_text_attn.to(device=device, dtype=torch.long)
            if temporal_text_step_mask is not None:
                temporal_text_step_mask = temporal_text_step_mask.to(device=device, dtype=torch.long)
            text_pack = _encode_text_pack_from_tower(
                temporal_text_tower=temporal_text_tower,
                ts_patches=ts_patches,
                temporal_text_ids=temporal_text_ids,
                temporal_text_attn=temporal_text_attn,
                temporal_text_step_mask=temporal_text_step_mask,
                device=device,
                dtype=targets_z.dtype,
            )
            sign_logits, _ = _run_external_signnet(
                signnet_model=signnet,
                history_z=history_z.to(torch.float32),
                base_pred_z=base_pred,
                structured_feats=structured_feats,
                text_summary=text_pack.get("text_summary"),
                text_strength=text_pack.get("text_strength"),
                tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
            )

            raw_residual_target = _build_delta_targets(
                targets_z=targets_z,
                base_pred=base_pred,
                mu_global=float(global_zstats["mu_global"]),
                sigma_global=float(global_zstats["sigma_global"]),
                args=args,
            )
            if relative_mode:
                abs_residual = raw_residual_target.abs()
                sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
                valid_mask = (abs_residual > sign_eps).to(torch.float32)
                state_targets = _build_relative_state_targets(
                    raw_residual_target,
                    args,
                    structured_feats=structured_feats,
                )
            else:
                cleaned_residual_target = _build_cleaned_residual_targets(
                    raw_residual=raw_residual_target,
                    structured_feats=structured_feats,
                    args=args,
                )
                abs_residual = cleaned_residual_target.abs()
                sign_target_loss, sign_target_bin, valid_mask = _build_sign_label_pack(cleaned_residual_target, args)

            has_news = _derive_has_news_from_structured_feats(structured_feats)
            if use_news_weighting:
                sample_weight = _build_news_usefulness_weights(
                    has_news=has_news,
                    structured_feats=structured_feats,
                    enabled=True,
                )
            else:
                sample_weight = torch.ones_like(has_news, dtype=torch.float32)
            if use_residual_weighting:
                position_weight = _build_delta_residual_position_weights(abs_residual, args)
            else:
                position_weight = torch.ones_like(abs_residual, dtype=torch.float32)
            sample_pos_weight = sample_weight.unsqueeze(1) * position_weight

            if relative_mode:
                ce = F.cross_entropy(
                    sign_logits.reshape(-1, sign_logits.size(-1)),
                    state_targets.reshape(-1),
                    reduction="none",
                ).reshape_as(state_targets)
                class_weight = _masked_multiclass_inverse_freq_weights(
                    state_targets,
                    valid_mask,
                    num_classes=int(sign_logits.size(-1)),
                )
                loss = _masked_weighted_mean(ce, sample_pos_weight * class_weight, mask=valid_mask)
                acc = _masked_multiclass_accuracy_from_logits(sign_logits, state_targets, valid_mask)
                bacc = _masked_multiclass_macro_balanced_accuracy_from_logits(sign_logits, state_targets, valid_mask)
            else:
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
                    sign_target_loss,
                    **bce_kwargs,
                )
                loss = _masked_weighted_mean(sign_bce, sample_pos_weight, mask=valid_mask)
                acc = _masked_binary_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)
                bacc = _masked_binary_balanced_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                clip_grad_norm_(trainable_params, grad_clip)
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
            temporal_text_tokenizer=temporal_text_tokenizer,
            temporal_text_tower=temporal_text_tower,
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
                f"train_{'state_acc' if relative_mode else 'acc'}={train_acc_sum / max(1, n_samples):.4f} "
                f"train_{'state_bacc' if relative_mode else 'bacc'}={train_bacc_sum / max(1, n_samples):.4f} "
                f"val_loss={val_loss:.6f} val_{'state_acc' if relative_mode else 'acc'}={val_acc:.4f} "
                f"val_{'state_bacc' if relative_mode else 'bacc'}={val_bacc:.4f} val_valid={val_valid:.4f} lr={curr_lr:.3e}"
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
            if temporal_text_tower is not None:
                best_tower_state = {k: v.detach().cpu().clone() for k, v in temporal_text_tower.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                if live_logger is not None:
                    live_logger.info(f"[SIGNNET] early stop at epoch={epoch+1}")
                break

    if best_state is not None:
        signnet.load_state_dict(best_state, strict=False)
    if best_tower_state is not None and temporal_text_tower is not None:
        temporal_text_tower.load_state_dict(best_tower_state, strict=False)
    signnet.eval()
    if temporal_text_tower is not None:
        temporal_text_tower.eval()
    for p in signnet.parameters():
        p.requires_grad = False

    if calibrate_bias and not relative_mode:
        _, _, _, _, val_pack = _evaluate_external_signnet(
            signnet_model=signnet,
            base_backbone=base_backbone,
            tokenizer=tokenizer,
            temporal_text_tokenizer=temporal_text_tokenizer,
            temporal_text_tower=temporal_text_tower,
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
    elif calibrate_bias and relative_mode and live_logger is not None:
        live_logger.info("[SIGNNET][CALIBRATE] skipped for relative-state 3-class supervision.")

    if test_loader is not None:
        test_loss, test_acc, test_valid, test_bacc = _evaluate_external_signnet(
            signnet_model=signnet,
            base_backbone=base_backbone,
            tokenizer=tokenizer,
            temporal_text_tokenizer=temporal_text_tokenizer,
            temporal_text_tower=temporal_text_tower,
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
            "variant": str(getattr(signnet, "model_variant", "mlp")),
            "task_type": signnet_task_type,
            "history_len": int(max(1, getattr(args, "history_len", 1))),
            "horizon": int(max(1, getattr(args, "horizon", 1))),
            "structured_dim": int(max(0, structured_dim)),
            "text_summary_dim": int(max(0, text_summary_dim)),
            "hidden_size": int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
            "dropout": float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
            "multimodal_fuse_lambda": float(max(0.0, getattr(args, "delta_multimodal_fuse_lambda", 1.0))),
            "best_val_loss": float(best_val),
            "best_val_acc": float(best_acc),
            "best_val_bacc": float(best_bacc),
            "decision_bias": float(getattr(signnet, "decision_bias", torch.zeros((), dtype=torch.float32)).detach().cpu().item())
            if torch.is_tensor(getattr(signnet, "decision_bias", None)) and getattr(signnet, "decision_bias", None).ndim == 0
            else float(getattr(signnet, "decision_bias", 0.0)) if not torch.is_tensor(getattr(signnet, "decision_bias", None)) else 0.0,
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
            f"[SIGNNET][TEST] loss={test_loss:.6f} "
            f"{'state_acc' if relative_mode else 'acc'}={test_acc:.4f} "
            f"{'state_bacc' if relative_mode else 'bacc'}={test_bacc:.4f} valid={test_valid:.4f} "
                f"bias={bias_now:.6f} "
                f"ckpt={signnet_ckpt_path}"
        )
    setattr(
        args,
        "_last_signnet_metrics",
        {
            "test_acc": float(test_acc),
            "test_bacc": float(test_bacc),
            "test_valid": float(test_valid),
            "task_type": str(signnet_task_type),
            "residual_arch": _resolve_residual_arch(args),
        },
    )
    return signnet
