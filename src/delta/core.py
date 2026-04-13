from __future__ import annotations

import math

import torch
from torch.optim import AdamW


def _match_horizon_shape(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if values.shape == reference.shape:
        return values
    if values.ndim == reference.ndim - 1:
        return values.unsqueeze(-1).expand_as(reference)
    if values.ndim == reference.ndim and values.size(-1) == 1 and reference.size(-1) > 1:
        return values.expand_as(reference)
    raise ValueError(f"Cannot match shape {tuple(values.shape)} to {tuple(reference.shape)}")


def _masked_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    v = values.to(torch.float32)
    w = weights.to(torch.float32)
    if mask is not None:
        m = mask.to(torch.float32)
        v = v * m
        w = w * m
    denom = w.sum().clamp_min(1e-6)
    return (v * w).sum() / denom


def _select_metric(loss_v: float, mse_v: float, mae_v: float, select_metric: str) -> float:
    metric = str(select_metric or "mae").lower().strip()
    if metric == "loss":
        return float(loss_v)
    if metric == "mse":
        return float(mse_v)
    return float(mae_v)


def _build_residual_importance_weights(
    residual_target: torch.Tensor,
    *,
    top_pct: float = 0.10,
    top_weight: float = 3.0,
) -> torch.Tensor:
    residual = residual_target.to(torch.float32)
    if residual.ndim > 1:
        magnitude = residual.abs().mean(dim=-1)
    else:
        magnitude = residual.abs()

    weights = torch.ones_like(magnitude, dtype=torch.float32)
    pct = float(max(0.0, min(1.0, top_pct)))
    boost = float(max(1.0, top_weight))
    if magnitude.numel() == 0 or pct <= 0.0 or boost <= 1.0:
        return weights

    threshold = torch.quantile(magnitude.detach(), max(0.0, 1.0 - pct))
    weights = torch.where(magnitude >= threshold, torch.full_like(weights, boost), weights)
    return weights / weights.mean().clamp_min(1e-6)


def _build_delta_optimizer(model: torch.nn.Module, args) -> AdamW:
    lr = float(getattr(args, "lr", 1e-4))
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def skill_score(metric_value: float, base_value: float) -> float:
    base = float(base_value)
    if not math.isfinite(base) or abs(base) < 1e-12:
        return float("nan")
    return 1.0 - float(metric_value) / base
