from __future__ import annotations

import torch
import torch.nn as nn


class ResidualSignNet(nn.Module):
    """
    External residual-state classifier trained before DELTA.
    Inputs:
      - z-scored history values
      - base prediction in z-space
      - structured news feature vector
    Output:
      - additive mode: horizon-wise binary sign logits
      - relative mode: horizon-wise 3-class state logits
    """

    def __init__(
        self,
        history_len: int,
        horizon: int,
        structured_dim: int,
        text_summary_dim: int = 0,
        hidden_size: int = 256,
        dropout: float = 0.1,
        task_type: str = "binary_sign",
    ):
        super().__init__()
        self.model_variant = "mlp"
        self.history_len = int(max(1, history_len))
        self.horizon = int(max(1, horizon))
        self.structured_dim = int(max(0, structured_dim))
        self.text_summary_dim = int(max(0, text_summary_dim))
        task_norm = str(task_type or "binary_sign").lower().strip()
        if task_norm not in {"binary_sign", "relative_state"}:
            task_norm = "binary_sign"
        self.task_type = task_norm
        self.num_classes = 3 if self.task_type == "relative_state" else 2
        hidden = int(max(32, hidden_size))
        drop = float(max(0.0, dropout))

        self.hist_norm = nn.LayerNorm(self.history_len)
        self.hist_proj = nn.Sequential(
            nn.Linear(self.history_len, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(hidden),
        )
        self.base_proj = nn.Sequential(
            nn.Linear(self.horizon, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(hidden),
        )
        if self.structured_dim > 0:
            self.structured_proj = nn.Sequential(
                nn.Linear(self.structured_dim, hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.LayerNorm(hidden),
            )
        else:
            self.structured_proj = None

        if self.text_summary_dim > 0:
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_summary_dim + 1, hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.LayerNorm(hidden),
            )
        else:
            self.text_proj = None

        fuse_in = (
            hidden * 2
            + (hidden if self.structured_proj is not None else 0)
            + (hidden if self.text_proj is not None else 0)
        )
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(hidden),
        )
        out_dim = self.horizon * 3 if self.task_type == "relative_state" else self.horizon
        self.out_head = nn.Linear(hidden, out_dim)
        self.register_buffer("decision_bias", torch.zeros((), dtype=torch.float32))

    def forward(
        self,
        history_z: torch.Tensor,
        base_pred_z: torch.Tensor,
        structured_feats: torch.Tensor | None,
        text_summary: torch.Tensor | None = None,
        text_strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_z = history_z.to(torch.float32)
        if h_z.ndim != 2:
            h_z = h_z.reshape(h_z.size(0), -1)
        if h_z.size(1) < self.history_len:
            pad = h_z.new_zeros(h_z.size(0), self.history_len - h_z.size(1))
            h_z = torch.cat([pad, h_z], dim=1)
        elif h_z.size(1) > self.history_len:
            h_z = h_z[:, -self.history_len :]

        base = base_pred_z.to(torch.float32)
        if base.ndim != 2:
            base = base.reshape(base.size(0), -1)
        if base.size(1) < self.horizon:
            pad = base.new_zeros(base.size(0), self.horizon - base.size(1))
            base = torch.cat([base, pad], dim=1)
        elif base.size(1) > self.horizon:
            base = base[:, : self.horizon]

        parts = [
            self.hist_proj(self.hist_norm(h_z)),
            self.base_proj(base),
        ]
        if self.structured_proj is not None:
            sf = structured_feats
            if sf is None:
                sf = h_z.new_zeros(h_z.size(0), self.structured_dim)
            sf = sf.to(torch.float32)
            if sf.ndim != 2:
                sf = sf.reshape(sf.size(0), -1)
            if sf.size(1) < self.structured_dim:
                pad = sf.new_zeros(sf.size(0), self.structured_dim - sf.size(1))
                sf = torch.cat([sf, pad], dim=1)
            elif sf.size(1) > self.structured_dim:
                sf = sf[:, : self.structured_dim]
            parts.append(self.structured_proj(sf))

        if self.text_proj is not None:
            txt = text_summary
            if txt is None:
                txt = h_z.new_zeros(h_z.size(0), self.text_summary_dim)
            txt = txt.to(torch.float32)
            if txt.ndim != 2:
                txt = txt.reshape(txt.size(0), -1)
            if txt.size(1) < self.text_summary_dim:
                pad = txt.new_zeros(txt.size(0), self.text_summary_dim - txt.size(1))
                txt = torch.cat([txt, pad], dim=1)
            elif txt.size(1) > self.text_summary_dim:
                txt = txt[:, : self.text_summary_dim]

            strength = text_strength
            if strength is None:
                fill_v = 0.0 if text_summary is None else 1.0
                strength = txt.new_full((txt.size(0), 1), fill_v)
            else:
                strength = strength.to(torch.float32)
                if strength.ndim == 1:
                    strength = strength.unsqueeze(1)
                elif strength.ndim != 2:
                    strength = strength.reshape(strength.size(0), -1)
                if strength.size(1) <= 0:
                    strength = txt.new_zeros(txt.size(0), 1)
                elif strength.size(1) > 1:
                    strength = strength[:, :1]
            parts.append(self.text_proj(torch.cat([txt, strength], dim=-1)))

        fused = self.fuse(torch.cat(parts, dim=-1))
        logits = self.out_head(fused)
        if self.task_type == "relative_state":
            return logits.view(logits.size(0), self.horizon, 3)
        return logits
