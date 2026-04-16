from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..delta.core import _build_delta_scheduler


def _iter_batches(payload: dict[str, torch.Tensor], batch_size: int, *, shuffle: bool):
    if not payload:
        return
    num_rows = int(next(iter(payload.values())).shape[0])
    if num_rows <= 0:
        return
    indices = torch.arange(num_rows)
    if shuffle:
        indices = indices[torch.randperm(num_rows)]
    for start in range(0, num_rows, max(1, int(batch_size))):
        batch_idx = indices[start : start + max(1, int(batch_size))]
        yield {key: value[batch_idx] for key, value in payload.items()}


def run_regime_self_supervised_pretrain(
    model,
    helper_head,
    train_payload: dict[str, torch.Tensor],
    val_payload: dict[str, torch.Tensor],
    *,
    epochs: int,
    lr: float,
    device,
    batch_size: int = 64,
    scheduler_name: str = "warmup_cosine",
    warmup_pct: float = 0.10,
    min_lr_ratio: float = 0.05,
    live_logger=None,
):
    if int(max(0, epochs)) <= 0:
        return {"best_loss": float("nan")}
    if not train_payload or int(next(iter(train_payload.values())).shape[0]) <= 0:
        return {"best_loss": float("nan")}

    for param in model.parameters():
        param.requires_grad = False
    for param in model.regime_projector.parameters():
        param.requires_grad = True

    helper_head = helper_head.to(device)
    optimizer = torch.optim.AdamW(
        list(model.regime_projector.parameters()) + list(helper_head.parameters()),
        lr=float(lr),
        weight_decay=1e-5,
    )
    steps_per_epoch = math.ceil(int(next(iter(train_payload.values())).shape[0]) / max(1, int(batch_size)))
    total_steps = int(max(1, steps_per_epoch) * int(epochs))
    scheduler = _build_delta_scheduler(
        optimizer,
        total_steps=total_steps,
        scheduler_name=scheduler_name,
        warmup_pct=warmup_pct,
        min_lr_ratio=min_lr_ratio,
    )

    best_loss = math.inf
    best_state = None
    for epoch in range(int(epochs)):
        model.regime_projector.train()
        helper_head.train()
        train_loss_sum = 0.0
        train_steps = 0
        for batch in _iter_batches(train_payload, batch_size=batch_size, shuffle=True):
            regime_pack = {k: v.to(device) for k, v in batch.items() if k in {"regime_vec", "topic_tag_mass", "text_emb", "relevance_mass"}}
            regime_repr = model.encode_regime(regime_pack)
            vol_pred, spike_logit, shape_pred = helper_head(regime_repr)
            vol_target = batch["vol_target"].to(device)
            spike_target = batch["spike_target"].to(device)
            shape_target = batch["shape_target"].to(device)
            loss = (
                F.mse_loss(vol_pred, vol_target)
                + F.binary_cross_entropy_with_logits(spike_logit, spike_target)
                + F.mse_loss(shape_pred, shape_target)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_loss_sum += float(loss.detach().cpu())
            train_steps += 1

        model.regime_projector.eval()
        helper_head.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in _iter_batches(val_payload, batch_size=batch_size, shuffle=False):
                regime_pack = {k: v.to(device) for k, v in batch.items() if k in {"regime_vec", "topic_tag_mass", "text_emb", "relevance_mass"}}
                regime_repr = model.encode_regime(regime_pack)
                vol_pred, spike_logit, shape_pred = helper_head(regime_repr)
                vol_target = batch["vol_target"].to(device)
                spike_target = batch["spike_target"].to(device)
                shape_target = batch["shape_target"].to(device)
                loss = (
                    F.mse_loss(vol_pred, vol_target)
                    + F.binary_cross_entropy_with_logits(spike_logit, spike_target)
                    + F.mse_loss(shape_pred, shape_target)
                )
                val_loss_sum += float(loss.detach().cpu())
                val_steps += 1

        val_loss = val_loss_sum / max(1, val_steps)
        if live_logger is not None:
            live_logger.info(
                f"[DELTA_V3][PRETRAIN_V2] epoch={epoch + 1} "
                f"train_loss={train_loss_sum / max(1, train_steps):.4f} val_loss={val_loss:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                "regime_projector": model.regime_projector.state_dict(),
                "helper_head": helper_head.state_dict(),
            }

    if best_state is not None:
        model.regime_projector.load_state_dict(best_state["regime_projector"])
        helper_head.load_state_dict(best_state["helper_head"])
    for param in model.parameters():
        param.requires_grad = True
    return {"best_loss": best_loss}
