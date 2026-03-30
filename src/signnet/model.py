from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(float(max(0.0, dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.drop(y)
        return F.gelu(x + y)

class ResidualSignNet(nn.Module):
    """
    Independent sign classifier trained before DELTA.
    Inputs:
      - z-scored history values
      - base prediction in z-space
      - structured news feature vector
      - news count scalar
    Output:
      - horizon-wise binary sign logits (positive vs negative residual)
    """

    def __init__(
        self,
        history_len: int,
        horizon: int,
        structured_dim: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_variant = "mlp"
        self.expects_temporal_text = False
        self.history_len = int(max(1, history_len))
        self.horizon = int(max(1, horizon))
        self.structured_dim = int(max(0, structured_dim))
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
        self.news_count_proj = nn.Sequential(
            nn.Linear(1, hidden // 2),
            nn.GELU(),
            nn.Dropout(drop),
        )

        fuse_in = hidden * 2 + (hidden if self.structured_proj is not None else 0) + hidden // 2
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(hidden),
        )
        self.out_head = nn.Linear(hidden, self.horizon)
        self.register_buffer("decision_bias", torch.zeros((), dtype=torch.float32))

    def forward(
        self,
        history_z: torch.Tensor,
        base_pred_z: torch.Tensor,
        structured_feats: torch.Tensor | None,
        news_counts: torch.Tensor | None,
        signnet_text_ids: torch.Tensor | None = None,
        signnet_text_attn: torch.Tensor | None = None,
        signnet_text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del signnet_text_ids, signnet_text_attn, signnet_text_mask
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

        nc = news_counts
        if nc is None:
            nc = h_z.new_zeros(h_z.size(0))
        nc = nc.to(torch.float32).reshape(-1, 1)
        parts.append(self.news_count_proj(nc))

        fused = self.fuse(torch.cat(parts, dim=-1))
        return self.out_head(fused)

class DualStreamTCNSignNet(nn.Module):
    """
    Lightweight external sign classifier with:
      - history z-value TCN
      - aligned refined-news text TCN
      - base_pred_z MLP branch
      - gated late fusion
    """

    def __init__(
        self,
        history_len: int,
        horizon: int,
        structured_dim: int,
        text_encoder: nn.Module,
        text_encoder_hidden_size: int,
        hidden_size: int = 256,
        text_low_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_variant = "dual_stream_tcn"
        self.expects_temporal_text = True
        self.history_len = int(max(1, history_len))
        self.horizon = int(max(1, horizon))
        self.structured_dim = int(max(0, structured_dim))
        self.hidden_size = int(max(32, hidden_size))
        self.text_low_dim = int(max(32, text_low_dim))
        self.text_encoder_hidden_size = int(max(1, text_encoder_hidden_size))
        self.text_encoder = text_encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        drop = float(max(0.0, dropout))
        tcn_width = self.hidden_size

        self.hist_norm = nn.LayerNorm(self.history_len)
        self.hist_diff_norm = nn.LayerNorm(self.history_len)
        self.hist_in = nn.Conv1d(2, tcn_width, kernel_size=1)
        self.hist_tcn = nn.ModuleList(
            [_ResidualTCNBlock(tcn_width, dilation=d, dropout=drop) for d in (1, 2, 4)]
        )
        self.hist_out = nn.Sequential(
            nn.Linear(tcn_width, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )

        self.text_step_proj = nn.Sequential(
            nn.Linear(self.text_encoder_hidden_size, self.text_low_dim),
            nn.GELU(),
            nn.LayerNorm(self.text_low_dim),
        )
        self.text_in = nn.Conv1d(self.text_low_dim, tcn_width, kernel_size=1)
        self.text_tcn = nn.ModuleList(
            [_ResidualTCNBlock(tcn_width, dilation=d, dropout=drop) for d in (1, 2, 4)]
        )
        self.text_out = nn.Sequential(
            nn.Linear(tcn_width, self.hidden_size),
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
        if self.structured_dim > 0:
            self.structured_proj = nn.Sequential(
                nn.Linear(self.structured_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(drop),
                nn.LayerNorm(self.hidden_size),
            )
        else:
            self.structured_proj = None
        self.news_count_proj = nn.Sequential(
            nn.Linear(1, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(drop),
        )

        numeric_in = self.hidden_size * 2 + (self.hidden_size if self.structured_proj is not None else 0) + self.hidden_size // 2
        self.numeric_fuse = nn.Sequential(
            nn.Linear(numeric_in, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )
        self.text_aux = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + 1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
        )
        self.out_ln = nn.LayerNorm(self.hidden_size)
        self.out_drop = nn.Dropout(drop)
        self.out_head = nn.Linear(self.hidden_size, self.horizon)
        self.register_buffer("decision_bias", torch.zeros((), dtype=torch.float32))

    def _run_text_encoder(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor | None:
        if self.text_encoder is None:
            return None
        was_training = bool(self.text_encoder.training)
        self.text_encoder.eval()
        with torch.no_grad():
            enc = self.text_encoder(input_ids=ids, attention_mask=attn)
        if was_training:
            self.text_encoder.train()
        hidden = getattr(enc, "last_hidden_state", None)
        if hidden is None and isinstance(enc, (tuple, list)) and len(enc) > 0:
            hidden = enc[0]
        return hidden

    def _pool_temporal_features(self, seq_feat: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is None:
            pooled = seq_feat.mean(dim=-1)
        else:
            weights = valid_mask.to(device=seq_feat.device, dtype=seq_feat.dtype).unsqueeze(1)
            denom = weights.sum(dim=-1).clamp_min(1.0)
            pooled = (seq_feat * weights).sum(dim=-1) / denom
        return pooled

    def _encode_history(self, history_z: torch.Tensor) -> torch.Tensor:
        h_z = history_z.to(torch.float32)
        if h_z.ndim != 2:
            h_z = h_z.reshape(h_z.size(0), -1)
        if h_z.size(1) < self.history_len:
            pad = h_z.new_zeros(h_z.size(0), self.history_len - h_z.size(1))
            h_z = torch.cat([pad, h_z], dim=1)
        elif h_z.size(1) > self.history_len:
            h_z = h_z[:, -self.history_len :]

        h_diff = torch.diff(h_z, dim=1, prepend=h_z[:, :1])
        hist_seq = torch.stack([self.hist_norm(h_z), self.hist_diff_norm(h_diff)], dim=1)
        hist_feat = self.hist_in(hist_seq)
        for block in self.hist_tcn:
            hist_feat = block(hist_feat)
        return self.hist_out(self._pool_temporal_features(hist_feat, valid_mask=None))

    def _encode_text_sequence(
        self,
        signnet_text_ids: torch.Tensor | None,
        signnet_text_attn: torch.Tensor | None,
        signnet_text_mask: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        valid = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        if (
            signnet_text_ids is None
            or signnet_text_attn is None
            or signnet_text_ids.ndim != 3
            or signnet_text_attn.ndim != 3
            or signnet_text_ids.size(0) != batch_size
        ):
            return zeros, valid

        ids = signnet_text_ids.to(device=device, dtype=torch.long)
        attn = signnet_text_attn.to(device=device, dtype=torch.long)
        B, L, T = ids.shape
        flat_ids = ids.reshape(B * L, T)
        flat_attn = attn.reshape(B * L, T)
        hidden = self._run_text_encoder(flat_ids, flat_attn)
        if hidden is None:
            return zeros, valid

        hidden = hidden.to(device=device, dtype=dtype)
        tok_mask = flat_attn.to(device=device, dtype=dtype).unsqueeze(-1)
        flat_valid = (tok_mask.sum(dim=1) > 0).to(dtype=dtype)
        if float(flat_valid.sum().item()) <= 0.0:
            return zeros, valid

        denom = tok_mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * tok_mask).sum(dim=1) / denom
        step_feat = self.text_step_proj(pooled).reshape(B, L, self.text_low_dim)
        if signnet_text_mask is not None and signnet_text_mask.ndim == 2:
            step_valid = signnet_text_mask.to(device=device, dtype=dtype).unsqueeze(-1)
        else:
            step_valid = flat_valid.reshape(B, L, 1)
        step_feat = step_feat * step_valid
        text_seq = self.text_in(step_feat.transpose(1, 2))
        for block in self.text_tcn:
            text_seq = block(text_seq)
        pooled_text = self._pool_temporal_features(text_seq, valid_mask=step_valid.squeeze(-1))
        valid = (step_valid.sum(dim=1) > 0).to(dtype=dtype)
        return self.text_out(pooled_text) * valid, valid

    def forward(
        self,
        history_z: torch.Tensor,
        base_pred_z: torch.Tensor,
        structured_feats: torch.Tensor | None,
        news_counts: torch.Tensor | None,
        signnet_text_ids: torch.Tensor | None = None,
        signnet_text_attn: torch.Tensor | None = None,
        signnet_text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hist_feat = self._encode_history(history_z)

        base = base_pred_z.to(torch.float32)
        if base.ndim != 2:
            base = base.reshape(base.size(0), -1)
        if base.size(1) < self.horizon:
            pad = base.new_zeros(base.size(0), self.horizon - base.size(1))
            base = torch.cat([base, pad], dim=1)
        elif base.size(1) > self.horizon:
            base = base[:, : self.horizon]
        base_feat = self.base_proj(base)

        parts = [hist_feat, base_feat]
        if self.structured_proj is not None:
            sf = structured_feats
            if sf is None:
                sf = hist_feat.new_zeros(hist_feat.size(0), self.structured_dim)
            sf = sf.to(torch.float32)
            if sf.ndim != 2:
                sf = sf.reshape(sf.size(0), -1)
            if sf.size(1) < self.structured_dim:
                pad = sf.new_zeros(sf.size(0), self.structured_dim - sf.size(1))
                sf = torch.cat([sf, pad], dim=1)
            elif sf.size(1) > self.structured_dim:
                sf = sf[:, : self.structured_dim]
            parts.append(self.structured_proj(sf))

        nc = news_counts
        if nc is None:
            nc = hist_feat.new_zeros(hist_feat.size(0))
        nc = nc.to(torch.float32).reshape(-1, 1)
        parts.append(self.news_count_proj(nc))

        fused = self.numeric_fuse(torch.cat(parts, dim=-1))
        text_feat, text_valid = self._encode_text_sequence(
            signnet_text_ids=signnet_text_ids,
            signnet_text_attn=signnet_text_attn,
            signnet_text_mask=signnet_text_mask,
            dtype=fused.dtype,
            device=fused.device,
            batch_size=fused.size(0),
        )
        if float(text_valid.sum().item()) > 0.0:
            text_gate = self.text_gate(torch.cat([fused, text_feat, nc], dim=-1))
            fused = fused + text_gate * self.text_aux(text_feat) * text_valid

        fused = self.out_ln(fused)
        return self.out_head(self.out_drop(fused))
