from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.optim import AdamW

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

def _match_gate_shape(gate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    g = gate.to(torch.float32)
    if reference.ndim != 2:
        return g
    horizon = int(reference.size(1))
    if g.ndim == 1:
        return g.unsqueeze(1).expand(-1, horizon)
    if g.ndim == 2:
        if g.size(1) == horizon:
            return g
        if g.size(1) == 1:
            return g.expand(-1, horizon)
        if g.size(1) > horizon:
            return g[:, :horizon]
        pad = g[:, -1:].expand(-1, horizon - g.size(1))
        return torch.cat([g, pad], dim=1)
    return g.reshape(g.size(0), -1)

def _build_horizon_gate_targets(
    *,
    pred_real_z: torch.Tensor,
    pred_null_z: torch.Tensor,
    targets_z: torch.Tensor,
    args,
) -> torch.Tensor:
    err_real_h = torch.abs(pred_real_z.to(torch.float32) - targets_z.to(torch.float32))
    err_null_h = torch.abs(pred_null_z.to(torch.float32) - targets_z.to(torch.float32))
    margin = float(getattr(args, "cf_pseudo_margin", 0.01) or 0.0)
    temp = max(1e-6, float(getattr(args, "cf_pseudo_temp", 0.2) or 0.2))
    gain = err_null_h - err_real_h - margin
    if int(getattr(args, "cf_pseudo_hard", 0)) == 1:
        return (gain > 0).to(dtype=torch.float32)
    return torch.sigmoid(gain / temp).to(dtype=torch.float32)

def _weighted_sample_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    v = values.to(torch.float32)
    w = weights.to(torch.float32)
    if v.ndim > 1:
        v = v.reshape(v.size(0), -1).mean(dim=1)
    else:
        v = v.reshape(-1)
    if w.ndim > 1:
        w = w.reshape(w.size(0), -1).mean(dim=1)
    else:
        w = w.reshape(-1)
    denom = w.sum().clamp_min(1e-6)
    return (v * w).sum() / denom

def _build_news_usefulness_weights(
    *,
    has_news: torch.Tensor,
    news_counts: torch.Tensor | None,
    structured_feats: torch.Tensor | None,
    enabled: bool,
) -> torch.Tensor:
    base = torch.ones_like(has_news, dtype=torch.float32)
    if not enabled:
        return base

    w = base.clone()
    has_news_f = has_news.to(torch.float32)
    w = w - 0.25 * (1.0 - has_news_f)

    if news_counts is not None:
        counts = news_counts.to(torch.float32).clamp_min(0.0)
        w = w + 0.10 * torch.clamp(counts, 0.0, 4.0) / 4.0

    if structured_feats is not None and structured_feats.ndim == 2 and structured_feats.size(1) >= 5:
        sf = structured_feats.to(torch.float32)
        relevance = sf[:, 0].clamp(0.0, 1.0)
        confidence = sf[:, 4].clamp(0.0, 1.0)
        w = w + 0.25 * relevance * confidence

    return w.clamp(0.5, 1.5)

def _structured_consistency_losses(
    *,
    out_delta: dict,
    structured_feats: torch.Tensor,
    sample_weight: torch.Tensor,
    gate_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    device = structured_feats.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if structured_feats.ndim != 2 or structured_feats.size(1) < 5:
        return zero, {
            "sign": zero,
            "scale": zero,
            "decay": zero,
            "mask": zero,
        }

    sign_logit = out_delta.get("struct_sign_logit", None)
    struct_scale = out_delta.get("struct_scale", None)
    struct_decay = out_delta.get("struct_decay", None)
    struct_mask_logits = out_delta.get("struct_mask_logits", None)
    if sign_logit is None and struct_scale is None and struct_decay is None and struct_mask_logits is None:
        return zero, {
            "sign": zero,
            "scale": zero,
            "decay": zero,
            "mask": zero,
        }

    sf = structured_feats.to(torch.float32)
    relevance = sf[:, 0].clamp(0.0, 1.0)
    direction = sf[:, 1].clamp(-1.0, 1.0)
    strength = sf[:, 2].clamp(0.0, 1.0)
    persistence = sf[:, 3].clamp(0.0, 1.0)
    active_mask = (relevance > 0.0).to(torch.float32)
    sample_w = sample_weight.to(torch.float32) * active_mask

    loss_sign = zero
    if sign_logit is not None:
        pred_sign = torch.tanh(sign_logit.to(torch.float32))
        pred_sign = pred_sign.reshape(pred_sign.size(0), -1).mean(dim=1)
        sign_mask = (active_mask > 0.0) * (direction.abs() > 0.0).to(torch.float32)
        if float(sign_mask.sum().detach().cpu()) > 0.0:
            loss_sign = _weighted_sample_mean(
                torch.abs(pred_sign - direction),
                sample_weight.to(torch.float32) * sign_mask,
            )

    loss_scale = zero
    if struct_scale is not None:
        pred_scale = 1.0 - torch.exp(-struct_scale.to(torch.float32).reshape(struct_scale.size(0), -1).mean(dim=1).clamp_min(0.0))
        if float(active_mask.sum().detach().cpu()) > 0.0:
            loss_scale = _weighted_sample_mean(torch.abs(pred_scale - strength), sample_w)

    loss_decay = zero
    if struct_decay is not None:
        pred_persistence = torch.exp(-struct_decay.to(torch.float32).reshape(struct_decay.size(0), -1).mean(dim=1).clamp_min(0.0))
        if float(active_mask.sum().detach().cpu()) > 0.0:
            loss_decay = _weighted_sample_mean(torch.abs(pred_persistence - persistence), sample_w)

    loss_mask = zero
    if struct_mask_logits is not None:
        logits = struct_mask_logits.to(torch.float32)
        if gate_targets is not None:
            target = gate_targets.to(torch.float32)
            if target.shape != logits.shape:
                target = _match_gate_shape(target, logits)
        else:
            target = relevance.unsqueeze(1).expand_as(logits)
        mask_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weight_h = sample_weight.to(torch.float32).unsqueeze(1) * active_mask.unsqueeze(1).expand_as(mask_loss)
        denom = weight_h.sum().clamp_min(1.0)
        loss_mask = (mask_loss * weight_h).sum() / denom

    total = loss_sign + loss_scale + loss_decay + loss_mask
    return total, {
        "sign": loss_sign,
        "scale": loss_scale,
        "decay": loss_decay,
        "mask": loss_mask,
    }

def _all_gates_disabled(args) -> bool:
    return int(getattr(args, "disable_all_gates", 0)) == 1

def _epoch_ramp_scale(epoch_idx: int, warmup_epochs: int, curriculum_epochs: int) -> float:
    """
    Piecewise ramp:
      - 0 before warmup end
      - linearly to 1 during curriculum window
      - 1 afterwards
    """
    e = int(max(0, epoch_idx))
    w = int(max(0, warmup_epochs))
    c = int(max(1, curriculum_epochs))
    if e < w:
        return 0.0
    progress = float(e - w + 1) / float(c)
    return float(max(0.0, min(1.0, progress)))

def _step_ramp_scale(step_idx: int, warmup_steps: int, ramp_steps: int) -> float:
    """
    Step-wise ramp used for null regularization:
      - 0 before warmup steps
      - linearly to 1 during ramp window
      - 1 afterwards
    """
    s = int(max(0, step_idx))
    w = int(max(0, warmup_steps))
    r = int(max(0, ramp_steps))
    if s < w:
        return 0.0
    if r <= 0:
        return 1.0
    progress = float(s - w + 1) / float(r)
    return float(max(0.0, min(1.0, progress)))

def _delta_gate_is_modeled_internally(args) -> bool:
    return int(getattr(args, "delta_internal_gate_in_model", getattr(args, "delta_internal_gate", 1))) == 1

def _resolve_delta_residual_mode(args) -> str:
    if _delta_gate_is_modeled_internally(args):
        return "additive"
    mode = str(getattr(args, "delta_residual_mode", "additive")).lower().strip()
    if mode not in {"additive", "relative"}:
        mode = "additive"
    return mode

def _resolve_delta_sign_mode(args) -> str:
    mode = str(getattr(args, "delta_sign_mode", "signnet_binary") or "signnet_binary").lower().strip()
    if mode != "signnet_binary":
        raise ValueError(
            f"Unsupported delta_sign_mode={mode!r}. This framework now supports only 'signnet_binary'."
        )
    return mode

def _use_external_signnet(args) -> bool:
    return _resolve_delta_sign_mode(args) == "signnet_binary"

def _z_to_raw_tensor(x_z: torch.Tensor, mu_global: float, sigma_global: float) -> torch.Tensor:
    return x_z.to(torch.float32) * float(sigma_global) + float(mu_global)

def _raw_to_z_tensor(x_raw: torch.Tensor, mu_global: float, sigma_global: float) -> torch.Tensor:
    sigma = max(float(sigma_global), 1e-6)
    return (x_raw.to(torch.float32) - float(mu_global)) / sigma

def _safe_signed_denom_tensor(base_raw: torch.Tensor, args) -> torch.Tensor:
    floor = float(getattr(args, "delta_relative_denom_floor", 1.0) or 1.0)
    floor = max(1e-6, floor)
    sign = torch.where(base_raw >= 0, torch.ones_like(base_raw), -torch.ones_like(base_raw))
    return torch.where(base_raw.abs() >= floor, base_raw, sign * floor)

def _fuse_base_and_delta(
    *,
    base_pred_z: torch.Tensor,
    delta_pred: torch.Tensor,
    gate_h: torch.Tensor,
    args,
    mu_global: float,
    sigma_global: float,
) -> torch.Tensor:
    mode = _resolve_delta_residual_mode(args)
    base_z = base_pred_z.to(torch.float32)
    delta = delta_pred.to(torch.float32)
    gate = gate_h.to(torch.float32)
    if mode == "additive":
        if _delta_gate_is_modeled_internally(args):
            return base_z + delta
        return base_z + gate * delta

    base_raw = _z_to_raw_tensor(base_z, mu_global=mu_global, sigma_global=sigma_global)
    ratio = gate * delta
    ratio_clip = float(getattr(args, "delta_relative_ratio_clip", 0.0) or 0.0)
    if ratio_clip > 0.0:
        ratio = ratio.clamp(min=-ratio_clip, max=ratio_clip)
    pred_raw = base_raw * (1.0 + ratio)
    return _raw_to_z_tensor(pred_raw, mu_global=mu_global, sigma_global=sigma_global)

def _build_delta_targets(
    targets_z: torch.Tensor,
    base_pred: torch.Tensor,
    mu_global: float,
    sigma_global: float,
    args,
) -> torch.Tensor:
    if _delta_gate_is_modeled_internally(args):
        delta_target = (targets_z.to(torch.float32) - base_pred.to(torch.float32)).detach()
        target_clip = float(getattr(args, "delta_target_clip", 0.0) or 0.0)
        if target_clip > 0.0:
            delta_target = delta_target.clamp(min=-target_clip, max=target_clip)
        return delta_target

    mode = _resolve_delta_residual_mode(args)
    if mode == "relative":
        target_raw = _z_to_raw_tensor(targets_z.to(torch.float32), mu_global=mu_global, sigma_global=sigma_global)
        base_raw = _z_to_raw_tensor(base_pred.to(torch.float32), mu_global=mu_global, sigma_global=sigma_global)
        denom = _safe_signed_denom_tensor(base_raw, args)
        delta_target = ((target_raw - base_raw) / denom).detach()
    else:
        delta_target = (targets_z.to(torch.float32) - base_pred.to(torch.float32)).detach()
    target_clip = float(getattr(args, "delta_target_clip", 0.0) or 0.0)
    if target_clip > 0.0:
        delta_target = delta_target.clamp(min=-target_clip, max=target_clip)
    return delta_target

def _resolve_delta_mag_target(args) -> str:
    mode = str(getattr(args, "delta_mag_target", "log1p") or "log1p").lower().strip()
    return mode if mode in {"raw", "log1p"} else "log1p"

def _transform_delta_magnitude_target(x: torch.Tensor, args) -> torch.Tensor:
    x_pos = x.to(torch.float32).clamp_min(0.0)
    if _resolve_delta_mag_target(args) == "log1p":
        return torch.log1p(x_pos)
    return x_pos

def _build_delta_residual_position_weights(abs_target: torch.Tensor, args) -> torch.Tensor:
    scale = float(getattr(args, "delta_residual_weight_scale", 1.0) or 0.0)
    if scale <= 0.0:
        return torch.ones_like(abs_target, dtype=torch.float32)
    return 1.0 + scale * abs_target.to(torch.float32).clamp(0.0, 1.0)

def _masked_weighted_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    v = values.to(torch.float32)
    w = weights.to(torch.float32)
    if mask is not None:
        w = w * mask.to(torch.float32)
    denom = w.sum()
    if float(denom.detach().cpu()) <= 0.0:
        # Preserve the computation graph for all-masked batches so callers can
        # safely run backward() and simply receive zero gradients.
        return v.sum() * 0.0
    return (v * w).sum() / denom.clamp_min(1.0)

def _masked_binary_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = (logits.to(torch.float32) > 0).to(torch.float32)
    tgt = targets.to(torch.float32)
    correct = (pred == tgt).to(torch.float32)
    if mask is None:
        return correct.mean()
    m = mask.to(torch.float32)
    denom = m.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return torch.zeros((), device=correct.device, dtype=torch.float32)
    return (correct * m).sum() / denom.clamp_min(1.0)

def _masked_binary_balanced_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = logits.to(torch.float32) > 0.0
    tgt = targets.to(torch.float32) > 0.5
    if mask is None:
        m = torch.ones_like(targets, dtype=torch.float32)
    else:
        m = mask.to(torch.float32)
    valid = m > 0.5
    if int(valid.sum().detach().cpu().item()) <= 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    tp = ((pred & tgt) & valid).to(torch.float32).sum()
    fn = (((~pred) & tgt) & valid).to(torch.float32).sum()
    tn = (((~pred) & (~tgt)) & valid).to(torch.float32).sum()
    fp = ((pred & (~tgt)) & valid).to(torch.float32).sum()
    tpr = tp / (tp + fn).clamp_min(1.0)
    tnr = tn / (tn + fp).clamp_min(1.0)
    return 0.5 * (tpr + tnr)

def _masked_binary_pos_weight(
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight_floor: float = 0.5,
    weight_clip: float = 3.0,
) -> torch.Tensor:
    t = targets.to(torch.float32)
    m = mask.to(torch.float32)
    pos = (t * m).sum()
    neg = ((1.0 - t) * m).sum()
    if float(pos.detach().cpu()) <= 0.0 or float(neg.detach().cpu()) <= 0.0:
        return torch.ones((), device=t.device, dtype=torch.float32)
    w = neg / pos.clamp_min(1.0)
    lo = float(max(0.0, weight_floor))
    hi = float(max(lo + 1e-6, weight_clip))
    return w.clamp(min=lo, max=hi).to(torch.float32)

def _select_metric(loss_v: float, mse_v: float, mae_v: float, select_metric: str) -> float:
    rm = str(select_metric).lower()
    if rm == "loss":
        return float(loss_v)
    if rm == "mse":
        return float(mse_v)
    return float(mae_v)

def _build_delta_optimizer(delta_model, args):
    base_lr = float(args.lr)
    wd = float(args.weight_decay)

    head_scale = float(getattr(args, "delta_head_lr_scale", 1.0))
    other_scale = float(getattr(args, "delta_other_lr_scale", 1.0))

    head_params, other_params = [], []
    for name, p in delta_model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if (
            lname.startswith("delta_head")
            or lname.startswith("delta_gate")
            or lname.startswith("delta_fuse")
            or lname.startswith("delta_mag_head")
            or lname.startswith("delta_text_ln")
            or lname.startswith("delta_log_scale")
            or lname.startswith("delta_rel_head")
            or lname.startswith("text_")
            or lname.startswith("doc_")
            or lname.startswith("rel_head")
        ):
            head_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr * head_scale, "weight_decay": wd})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr * other_scale, "weight_decay": wd})

    if not param_groups:
        raise ValueError("No trainable parameters found for DELTA optimizer.")

    optimizer = AdamW(param_groups)
    lr_info = {
        "base_lr": base_lr,
        "head_lr": base_lr * head_scale if head_params else 0.0,
        "other_lr": base_lr * other_scale if other_params else 0.0,
        "n_head": len(head_params),
        "n_other": len(other_params),
    }
    return optimizer, lr_info

def _log_last_residual_eval_diag(args, live_logger, tag: str):
    if live_logger is None:
        return
    diag = getattr(args, "_last_residual_eval_diag", None)
    if not isinstance(diag, dict) or not diag:
        return
    live_logger.info(
        f"{tag} gate_mean={float(diag.get('gate_mean', 0.0)):.4f} "
        f"gate>0.5={float(diag.get('gate_frac_gt_0_5', 0.0)):.4f} "
        f"sign_acc={float(diag.get('sign_acc', 0.0)):.4f} "
        f"pred_mag={float(diag.get('pred_mag_mean', 0.0)):.4f} "
        f"true_|res|={float(diag.get('true_abs_residual_mean', 0.0)):.4f} "
        f"|delta|={float(diag.get('delta_abs_mean', 0.0)):.4f}"
    )
