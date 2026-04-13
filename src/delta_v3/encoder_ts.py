from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn


def _patchify(history_z: torch.Tensor, patch_len: int, patch_stride: int) -> torch.Tensor:
    batch_size, seq_len = history_z.shape
    if seq_len < patch_len:
        pad = torch.zeros((batch_size, patch_len - seq_len), dtype=history_z.dtype, device=history_z.device)
        history_z = torch.cat([history_z, pad], dim=-1)
        seq_len = patch_len
    num_patches = 1 + max(0, (seq_len - patch_len) // patch_stride)
    return history_z.unfold(dimension=-1, size=patch_len, step=patch_stride)[:, :num_patches, :]


def build_patch_calendar_indices(history_times, patch_len: int, patch_stride: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(history_times, (list, tuple)) or len(history_times) == 0:
        return (
            torch.zeros((0, 0), dtype=torch.long),
            torch.zeros((0, 0), dtype=torch.long),
            torch.zeros((0, 0), dtype=torch.long),
        )

    batch_size = len(history_times)
    seq_len = max((len(seq) for seq in history_times), default=0)
    if seq_len <= 0:
        return (
            torch.zeros((batch_size, 1), dtype=torch.long),
            torch.zeros((batch_size, 1), dtype=torch.long),
            torch.zeros((batch_size, 1), dtype=torch.long),
        )

    num_patches = 1 if seq_len < patch_len else 1 + max(0, (seq_len - patch_len) // patch_stride)
    dow_idx = torch.zeros((batch_size, num_patches), dtype=torch.long)
    hod_idx = torch.zeros((batch_size, num_patches), dtype=torch.long)
    holiday_idx = torch.zeros((batch_size, num_patches), dtype=torch.long)

    for i, seq in enumerate(history_times):
        parsed = pd.to_datetime(list(seq), errors="coerce")
        for patch_id in range(num_patches):
            end_idx = min(len(parsed) - 1, patch_id * patch_stride + patch_len - 1)
            if end_idx < 0:
                continue
            ts = parsed[end_idx]
            if pd.isna(ts):
                continue
            dow_idx[i, patch_id] = int(ts.dayofweek)
            hod_idx[i, patch_id] = int(ts.hour * 2 + int(ts.minute >= 30))
            holiday_idx[i, patch_id] = 0
    return dow_idx, hod_idx, holiday_idx


class PatchTSTTSEncoder(nn.Module):
    def __init__(
        self,
        patch_len: int,
        patch_stride: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        use_base_hidden: bool = True,
    ):
        super().__init__()
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.hidden_size = int(hidden_size)
        self.use_base_hidden = bool(use_base_hidden)

        self.patch_proj = nn.Linear(self.patch_len, self.hidden_size)
        self.dow_embed = nn.Embedding(7, self.hidden_size)
        self.hod_embed = nn.Embedding(48, self.hidden_size)
        self.holiday_embed = nn.Embedding(2, self.hidden_size)
        self.base_hidden_proj = nn.LazyLinear(self.hidden_size) if self.use_base_hidden else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        history_z: torch.Tensor,
        history_times,
        base_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = _patchify(history_z, self.patch_len, self.patch_stride)
        tokens = self.patch_proj(tokens)

        dow_idx, hod_idx, holiday_idx = build_patch_calendar_indices(history_times, self.patch_len, self.patch_stride)
        if dow_idx.numel() > 0:
            device = history_z.device
            tokens = tokens + self.dow_embed(dow_idx.to(device)) + self.hod_embed(hod_idx.to(device)) + self.holiday_embed(holiday_idx.to(device))

        if self.use_base_hidden and base_hidden is not None:
            base_token = self.base_hidden_proj(base_hidden).unsqueeze(1)
            tokens = torch.cat([base_token, tokens], dim=1)

        tokens = self.norm(self.encoder(tokens))
        summary = tokens.mean(dim=1)
        return tokens, summary
