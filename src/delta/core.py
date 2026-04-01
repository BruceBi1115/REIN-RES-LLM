from __future__ import annotations

import math

import torch
from torch.optim import AdamW

def _match_horizon_shape(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    g = values.to(torch.float32)
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

def _build_news_usefulness_weights(
    *,
    has_news: torch.Tensor,
    news_counts: torch.Tensor | None = None,
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
def _resolve_delta_residual_mode(args) -> str:
    mode = str(getattr(args, "delta_residual_mode", "additive")).lower().strip()
    if mode not in {"additive", "relative"}:
        mode = "additive"
    return mode

def _resolve_delta_sign_mode(args) -> str:
    mode = str(getattr(args, "delta_sign_mode", "signnet_binary") or "signnet_binary").lower().strip()
    if mode not in {"signnet_binary", "internal"}:
        raise ValueError(
            f"Unsupported delta_sign_mode={mode!r}. This framework now supports only 'signnet_binary' and 'internal'."
        )
    return mode

def _use_external_signnet(args) -> bool:
    return _resolve_delta_sign_mode(args) == "signnet_binary"

def _z_to_raw_tensor(x_z: torch.Tensor, mu_global: float, sigma_global: float) -> torch.Tensor:
    return x_z.to(torch.float32) * float(sigma_global) + float(mu_global)

def _raw_to_z_tensor(x_raw: torch.Tensor, mu_global: float, sigma_global: float) -> torch.Tensor:
    sigma = max(float(sigma_global), 1e-6)
    return (x_raw.to(torch.float32) - float(mu_global)) / sigma

RELATIVE_STATE_SHRINK = 0
RELATIVE_STATE_NEUTRAL = 1
RELATIVE_STATE_AMPLIFY = 2


def _relative_denom_floor_value(args) -> float:
    raw_floor = getattr(args, "delta_relative_denom_floor", 1.0)
    if raw_floor is None:
        raw_floor = 1.0
    return max(0.0, float(raw_floor))


def _relative_scale_tensor(base_raw: torch.Tensor, args) -> torch.Tensor:
    floor = _relative_denom_floor_value(args)
    eps = 1e-6
    min_scale = max(eps, floor)
    return base_raw.to(torch.float32).abs().clamp_min(min_scale)

def _fuse_base_and_delta(
    *,
    base_pred_z: torch.Tensor,
    delta_pred: torch.Tensor,
    args,
    mu_global: float,
    sigma_global: float,
) -> torch.Tensor:
    mode = _resolve_delta_residual_mode(args)
    base_z = base_pred_z.to(torch.float32)
    delta = delta_pred.to(torch.float32)
    if mode == "additive":
        return base_z + delta

    base_raw = _z_to_raw_tensor(base_z, mu_global=mu_global, sigma_global=sigma_global)
    ratio = delta
    ratio_clip = float(getattr(args, "delta_relative_ratio_clip", 0.0) or 0.0)
    if ratio_clip > 0.0:
        ratio = ratio.clamp(min=-ratio_clip, max=ratio_clip)
    scale_raw = _relative_scale_tensor(base_raw, args)
    pred_raw = base_raw + ratio * scale_raw
    return _raw_to_z_tensor(pred_raw, mu_global=mu_global, sigma_global=sigma_global)

def _build_delta_targets(
    targets_z: torch.Tensor,
    base_pred: torch.Tensor,
    mu_global: float,
    sigma_global: float,
    args,
) -> torch.Tensor:
    mode = _resolve_delta_residual_mode(args)
    if mode == "relative":
        target_raw = _z_to_raw_tensor(targets_z.to(torch.float32), mu_global=mu_global, sigma_global=sigma_global)
        base_raw = _z_to_raw_tensor(base_pred.to(torch.float32), mu_global=mu_global, sigma_global=sigma_global)
        scale_raw = _relative_scale_tensor(base_raw, args)
        delta_target = ((target_raw - base_raw) / scale_raw).detach()
    else:
        delta_target = (targets_z.to(torch.float32) - base_pred.to(torch.float32)).detach()
    target_clip = float(getattr(args, "delta_target_clip", 0.0) or 0.0)
    if target_clip > 0.0:
        delta_target = delta_target.clamp(min=-target_clip, max=target_clip)
    return delta_target


def _build_relative_state_targets(
    q_target: torch.Tensor,
    args,
    *,
    structured_feats: torch.Tensor | None = None,
) -> torch.Tensor:
    q = q_target.to(torch.float32)
    if _resolve_delta_residual_mode(args) == "relative" and _cleaned_residual_enabled(args):
        q = _build_cleaned_residual_targets(
            raw_residual=q,
            structured_feats=structured_feats,
            args=args,
        )
    eps = float(getattr(args, "delta_sign_eps", 0.03))
    labels = torch.full_like(q, RELATIVE_STATE_NEUTRAL, dtype=torch.long)
    labels = torch.where(q > eps, torch.full_like(labels, RELATIVE_STATE_AMPLIFY), labels)
    labels = torch.where(q < -eps, torch.full_like(labels, RELATIVE_STATE_SHRINK), labels)
    return labels


def _relative_state_score_from_logits(state_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(state_logits.to(torch.float32), dim=-1)
    return probs[..., RELATIVE_STATE_AMPLIFY] - probs[..., RELATIVE_STATE_SHRINK]

def _resolve_delta_mag_target(args) -> str:
    mode = str(getattr(args, "delta_mag_target", "log1p") or "log1p").lower().strip()
    return mode if mode in {"raw", "log1p"} else "log1p"

def _cleaned_residual_enabled(args) -> bool:
    return int(getattr(args, "cleaned_residual_enable", 1)) == 1

def _ewma_smooth_horizon(values: torch.Tensor, alpha: float) -> torch.Tensor:
    v = values.to(torch.float32)
    if v.ndim != 2 or v.size(1) <= 1:
        return v
    a = float(max(0.0, min(1.0, alpha)))
    if a <= 0.0:
        return v
    prev = v[:, :1]
    cols = [prev]
    for t in range(1, int(v.size(1))):
        curr = v[:, t : t + 1]
        prev = a * prev + (1.0 - a) * curr
        cols.append(prev)
    return torch.cat(cols, dim=1)

def _build_cleaned_residual_targets(
    *,
    raw_residual: torch.Tensor,
    structured_feats: torch.Tensor | None,
    args,
) -> torch.Tensor:
    target = raw_residual.to(torch.float32)
    if not _cleaned_residual_enabled(args):
        return target

    smooth_alpha = float(getattr(args, "cleaned_residual_smooth_alpha", 0.6) or 0.0)
    cleaned = _ewma_smooth_horizon(target, smooth_alpha)

    # In relative mode, cleaned residual is only used to smooth q_target before
    # relative-state label construction. Do not inject structured templates here.
    if _resolve_delta_residual_mode(args) == "relative":
        return cleaned

    mix = float(getattr(args, "cleaned_residual_structured_mix", 0.35) or 0.0)
    mix = max(0.0, min(1.0, mix))
    if mix <= 0.0 or structured_feats is None:
        return cleaned

    sf = structured_feats.to(torch.float32)
    if sf.ndim == 1:
        sf = sf.unsqueeze(0)
    if sf.ndim != 2:
        sf = sf.reshape(sf.size(0), -1)
    if sf.size(0) != cleaned.size(0) or sf.size(1) < 5:
        return cleaned

    relevance = sf[:, 0:1].clamp(0.0, 1.0)
    direction = sf[:, 1:2].clamp(-1.0, 1.0)
    strength = sf[:, 2:3].clamp(0.0, 1.0)
    persistence = sf[:, 3:4].clamp(0.0, 1.0)
    confidence = sf[:, 4:5].clamp(0.0, 1.0)

    logic_weight = (mix * relevance * confidence * direction.abs()).clamp(0.0, 1.0)
    if float(logic_weight.sum().detach().cpu()) <= 0.0:
        return cleaned

    horizon = int(max(1, cleaned.size(1)))
    steps = torch.arange(horizon, device=cleaned.device, dtype=cleaned.dtype).unsqueeze(0)
    tau = 1.0 + persistence * float(max(1, horizon - 1))
    decay = torch.exp(-steps / tau.clamp_min(1e-6))

    base_mag = cleaned.abs().mean(dim=1, keepdim=True)
    template_mag = base_mag * (0.25 + 0.75 * strength)
    direction_sign = torch.where(direction >= 0.0, torch.ones_like(direction), -torch.ones_like(direction))
    template = direction_sign * template_mag * decay
    return (1.0 - logic_weight) * cleaned + logic_weight * template

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


def _masked_multiclass_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = logits.to(torch.float32).argmax(dim=-1)
    tgt = targets.to(torch.long)
    correct = (pred == tgt).to(torch.float32)
    if mask is None:
        return correct.mean()
    m = mask.to(torch.float32)
    denom = m.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return torch.zeros((), device=correct.device, dtype=torch.float32)
    return (correct * m).sum() / denom.clamp_min(1.0)


def _masked_multiclass_macro_balanced_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    num_classes: int = 3,
) -> torch.Tensor:
    pred = logits.to(torch.float32).argmax(dim=-1)
    tgt = targets.to(torch.long)
    if mask is None:
        valid = torch.ones_like(tgt, dtype=torch.bool)
    else:
        valid = mask.to(torch.float32) > 0.5
    if int(valid.sum().detach().cpu().item()) <= 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    recalls = []
    for cls_idx in range(int(max(1, num_classes))):
        cls_mask = valid & (tgt == cls_idx)
        denom = cls_mask.to(torch.float32).sum()
        if float(denom.detach().cpu()) <= 0.0:
            continue
        tp = ((pred == cls_idx) & cls_mask).to(torch.float32).sum()
        recalls.append(tp / denom.clamp_min(1.0))
    if not recalls:
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    return torch.stack(recalls, dim=0).mean()

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
            or lname.startswith("delta_fuse")
            or lname.startswith("delta_mag_head")
            or lname.startswith("delta_log_scale")
            or lname.startswith("delta_rel_head")
            or lname.startswith("rel_head")
            or lname.startswith("text_mag_head")
            or lname.startswith("text_summary_ln")
            or lname.startswith("temporal_text_gate")
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
    acc_key = "state_acc" if "state_acc" in diag else "sign_acc"
    live_logger.info(
        f"{tag} {acc_key}={float(diag.get(acc_key, 0.0)):.4f} "
        f"pred_mag={float(diag.get('pred_mag_mean', 0.0)):.4f} "
        f"true_|res|={float(diag.get('true_abs_residual_mean', 0.0)):.4f} "
        f"|delta|={float(diag.get('delta_abs_mean', 0.0)):.4f}"
    )
