from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedResidualTrunk(nn.Module):
    """
    Shared residual reasoning trunk for unified direction/magnitude/confidence prediction.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        horizon: int,
        history_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(max(1, hidden_size))
        self.horizon = int(max(1, horizon))
        self.history_len = int(max(1, history_len))
        drop = float(max(0.0, dropout))

        self.history_proj = nn.Sequential(
            nn.Linear(self.history_len, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )
        self.base_proj = nn.Sequential(
            nn.Linear(self.horizon, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )
        self.input_fuse = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )

        self.direction_head = nn.Linear(self.hidden_size, self.horizon)
        self.magnitude_head = nn.Linear(self.hidden_size, self.horizon)
        self.confidence_head = nn.Linear(self.hidden_size, self.horizon)

        nn.init.normal_(self.direction_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.direction_head.bias)
        nn.init.normal_(self.magnitude_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.magnitude_head.bias, -1.0)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

    def _normalize_history(self, history_z: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if history_z is None:
            return ref.new_zeros(ref.size(0), self.history_len)
        hz = history_z.to(device=ref.device, dtype=ref.dtype)
        if hz.ndim != 2:
            hz = hz.reshape(hz.size(0), -1)
        if hz.size(1) < self.history_len:
            pad = hz.new_zeros(hz.size(0), self.history_len - hz.size(1))
            hz = torch.cat([pad, hz], dim=1)
        elif hz.size(1) > self.history_len:
            hz = hz[:, -self.history_len :]
        return hz

    def _normalize_base_pred(self, base_pred_z: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if base_pred_z is None:
            return ref.new_zeros(ref.size(0), self.horizon)
        bz = base_pred_z.to(device=ref.device, dtype=ref.dtype)
        if bz.ndim != 2:
            bz = bz.reshape(bz.size(0), -1)
        if bz.size(1) < self.horizon:
            pad = bz.new_zeros(bz.size(0), self.horizon - bz.size(1))
            bz = torch.cat([bz, pad], dim=1)
        elif bz.size(1) > self.horizon:
            bz = bz[:, : self.horizon]
        return bz

    def forward(
        self,
        *,
        residual_context: torch.Tensor,
        history_z: torch.Tensor | None,
        base_pred_z: torch.Tensor | None,
        text_summary: torch.Tensor | None,
        text_strength: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        ctx = residual_context.to(torch.float32)
        if ctx.ndim != 2:
            ctx = ctx.reshape(ctx.size(0), -1)

        history_pack = self.history_proj(self._normalize_history(history_z, ctx))
        base_pack = self.base_proj(self._normalize_base_pred(base_pred_z, ctx))

        txt = text_summary
        if txt is None:
            txt = ctx.new_zeros(ctx.size(0), self.hidden_size)
        txt = txt.to(device=ctx.device, dtype=ctx.dtype)
        if txt.ndim != 2:
            txt = txt.reshape(txt.size(0), -1)
        if txt.size(1) < self.hidden_size:
            pad = txt.new_zeros(txt.size(0), self.hidden_size - txt.size(1))
            txt = torch.cat([txt, pad], dim=1)
        elif txt.size(1) > self.hidden_size:
            txt = txt[:, : self.hidden_size]
        strength = text_strength
        if strength is None:
            strength = ctx.new_zeros(ctx.size(0), 1)
        strength = strength.to(device=ctx.device, dtype=ctx.dtype)
        if strength.ndim != 2:
            strength = strength.reshape(strength.size(0), -1)
        if strength.size(1) < 1:
            strength = torch.cat([strength, strength.new_zeros(strength.size(0), 1 - strength.size(1))], dim=1)
        strength = strength[:, :1].clamp(0.0, 1.0)
        text_pack = self.text_proj(torch.cat([txt, strength], dim=-1))

        fused = self.input_fuse(torch.cat([ctx, history_pack, base_pack, text_pack], dim=-1))
        hidden = self.trunk(fused + ctx)
        direction_logits = self.direction_head(hidden)
        magnitude_raw = self.magnitude_head(hidden)
        confidence_logits = self.confidence_head(hidden)
        direction_score = torch.tanh(direction_logits)
        magnitude = F.softplus(magnitude_raw)
        confidence = torch.sigmoid(confidence_logits)
        return {
            "hidden": hidden,
            "direction_logits": direction_logits,
            "direction_score": direction_score,
            "magnitude_raw": magnitude_raw,
            "magnitude": magnitude,
            "confidence_logits": confidence_logits,
            "confidence": confidence,
        }
