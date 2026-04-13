from __future__ import annotations

import json
import os

import torch
import torch.nn as nn


class ForecastPatchRegressor(nn.Module):
    def __init__(
        self,
        horizon: int,
        patch_dim: int,
        hidden_size: int,
        patch_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.patch_dim = int(patch_dim)
        self.hidden_size = int(hidden_size)

        mid = max(32, self.hidden_size * 2)
        self.patch_proj = nn.Sequential(
            nn.Linear(self.patch_dim, mid),
            nn.GELU(),
            nn.Linear(mid, self.hidden_size),
        )
        self.patch_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
        )
        self.patch_drop = nn.Dropout(float(patch_dropout))
        self.ts_ln = nn.LayerNorm(self.hidden_size)
        self.head_drop = nn.Dropout(float(head_dropout))
        self.base_head = nn.Linear(self.hidden_size, self.horizon)

    def forward(self, ts_patches: torch.Tensor, ts_patch_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.patch_proj(ts_patches)
        hidden = hidden * self.patch_gate(hidden)
        hidden = self.patch_drop(hidden)
        hidden = self.ts_ln(hidden)

        if ts_patch_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = ts_patch_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        pred = self.base_head(self.head_drop(pooled))
        return {
            "base_pred_z": pred,
            "hidden": pooled,
        }


def build_delta_model(
    history_len: int,
    horizon: int,
    patch_len: int,
    hidden_size: int,
    dropout: float = 0.0,
):
    _ = int(history_len)
    return ForecastPatchRegressor(
        horizon=int(horizon),
        patch_dim=int(patch_len),
        hidden_size=int(hidden_size),
        patch_dropout=float(dropout),
        head_dropout=float(dropout),
    )


def save_checkpoint(ckpt_dir: str, model: nn.Module, *, meta: dict | None = None, optimizer=None, scheduler=None, epoch: int | None = None):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, os.path.join(ckpt_dir, "model.pt"))
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta or {}, handle, ensure_ascii=False, indent=2)
    if optimizer is not None:
        state = {"optimizer": optimizer.state_dict(), "epoch": epoch}
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))


def _load_state_dict_compatible(model: nn.Module, state_dict: dict) -> None:
    model.load_state_dict(state_dict, strict=True)


def load_checkpoint(ckpt_dir: str, device=None):
    with open(os.path.join(ckpt_dir, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    model = build_delta_model(
        history_len=int(meta.get("history_len", 48)),
        horizon=int(meta.get("horizon", 1)),
        patch_len=int(meta.get("patch_len", 4)),
        hidden_size=int(meta.get("hidden_size", 256)),
        dropout=float(meta.get("dropout", 0.0)),
    )
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
    _load_state_dict_compatible(model, state["state_dict"])
    if device is not None:
        model = model.to(device)
    return model, meta


def load_trainer_state(ckpt_dir: str, optimizer, scheduler=None):
    state = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"), map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state
