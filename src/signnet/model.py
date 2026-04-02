from __future__ import annotations

import torch
import torch.nn as nn

from ..regime_blocks import RegimeRouter, ResidualExpertMixture


class ResidualSignNet(nn.Module):
    """
    External residual-state classifier trained before DELTA.
    Inputs:
      - z-scored history values
      - base prediction in z-space
      - structured news feature vector
      - optional temporal-text summary
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
        arch: str = "mlp",
        multimodal_fuse_lambda: float = 1.0,
    ):
        super().__init__()
        arch_norm = str(arch or "mlp").lower().strip()
        if arch_norm not in {"mlp", "plan_c_mvp"}:
            arch_norm = "mlp"
        self.model_variant = arch_norm
        self.history_len = int(max(1, history_len))
        self.horizon = int(max(1, horizon))
        self.structured_dim = int(max(0, structured_dim))
        self.text_summary_dim = int(max(0, text_summary_dim))
        self.multimodal_fuse_lambda = float(max(0.0, multimodal_fuse_lambda))
        self.regime_route_names = ("none", "trend", "event", "reversal", "sparse")
        task_norm = str(task_type or "binary_sign").lower().strip()
        if task_norm not in {"binary_sign", "relative_state"}:
            task_norm = "binary_sign"
        self.task_type = task_norm
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

        if self.model_variant == "plan_c_mvp":
            self.regime_router = RegimeRouter(
                hidden_size=hidden,
                scalar_dim=6,
                num_routes=len(self.regime_route_names),
                dropout=drop,
            )
            self.regime_experts = ResidualExpertMixture(
                hidden_size=hidden,
                num_experts=max(1, len(self.regime_route_names) - 1),
                dropout=drop,
            )
            self.route_summary_ln = nn.LayerNorm(hidden)
        else:
            self.regime_router = None
            self.regime_experts = None
            self.route_summary_ln = None

        out_dim = self.horizon * 3 if self.task_type == "relative_state" else self.horizon
        self.out_head = nn.Linear(hidden, out_dim)
        self.register_buffer("decision_bias", torch.zeros((), dtype=torch.float32))

    def _normalize_structured_feats(self, structured_feats: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
        if self.structured_dim <= 0:
            return None
        sf = structured_feats
        if sf is None:
            return ref.new_zeros(ref.size(0), self.structured_dim)
        sf = sf.to(torch.float32)
        if sf.ndim != 2:
            sf = sf.reshape(sf.size(0), -1)
        if sf.size(1) < self.structured_dim:
            pad = sf.new_zeros(sf.size(0), self.structured_dim - sf.size(1))
            sf = torch.cat([sf, pad], dim=1)
        elif sf.size(1) > self.structured_dim:
            sf = sf[:, : self.structured_dim]
        return sf

    def _build_route_scalars(
        self,
        *,
        history_z: torch.Tensor,
        base_pred_z: torch.Tensor,
        structured_feats: torch.Tensor | None,
        text_strength: torch.Tensor | None,
    ) -> torch.Tensor:
        device = history_z.device
        dtype = history_z.dtype
        hist_diff = history_z[:, 1:] - history_z[:, :-1] if history_z.size(1) > 1 else history_z
        hist_vol = hist_diff.std(dim=1, unbiased=False).to(device=device, dtype=dtype).unsqueeze(1)
        base_std = base_pred_z.std(dim=1, unbiased=False).to(device=device, dtype=dtype).unsqueeze(1)
        base_span = (
            (base_pred_z.max(dim=1).values - base_pred_z.min(dim=1).values)
            .to(device=device, dtype=dtype)
            .unsqueeze(1)
        )
        txt_strength = (
            text_strength.to(device=device, dtype=dtype)
            if text_strength is not None
            else torch.zeros(history_z.size(0), 1, device=device, dtype=dtype)
        )
        if structured_feats is not None and structured_feats.size(1) > 0:
            sf = structured_feats.to(device=device, dtype=dtype)
            relevance = sf[:, :1].clamp(0.0, 1.0)
            confidence = sf[:, 4:5].clamp(0.0, 1.0) if sf.size(1) >= 5 else torch.ones_like(relevance)
            struct_strength = (relevance * (0.5 + 0.5 * confidence)).clamp(0.0, 1.0)
        else:
            struct_strength = torch.zeros(history_z.size(0), 1, device=device, dtype=dtype)
        news_strength = torch.maximum(txt_strength, struct_strength)
        return torch.cat([hist_vol, base_std, base_span, txt_strength, struct_strength, news_strength], dim=-1)

    def _build_expert_inputs(
        self,
        *,
        fused_base: torch.Tensor,
        hist_part: torch.Tensor,
        base_part: torch.Tensor,
        structured_part: torch.Tensor | None,
        text_part: torch.Tensor | None,
        news_strength: torch.Tensor,
    ) -> list[torch.Tensor]:
        zero = torch.zeros_like(fused_base)
        struct_or_zero = structured_part if structured_part is not None else zero
        text_or_zero = text_part if text_part is not None else zero
        sparse_scale = 1.0 - news_strength.clamp(0.0, 1.0)
        return [
            self.route_summary_ln(fused_base + hist_part),
            self.route_summary_ln(fused_base + struct_or_zero + text_or_zero),
            self.route_summary_ln(fused_base + base_part - hist_part),
            self.route_summary_ln(fused_base + (sparse_scale * base_part)),
        ]

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

        hist_part = self.hist_proj(self.hist_norm(h_z))
        base_part = self.base_proj(base)
        parts = [hist_part, base_part]

        sf = self._normalize_structured_feats(structured_feats, h_z)
        structured_part = None
        if self.structured_proj is not None and sf is not None:
            structured_part = self.structured_proj(sf)
            parts.append(structured_part)

        text_part = None
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
            text_part = self.text_proj(torch.cat([txt, strength], dim=-1))
            parts.append(text_part)
        else:
            strength = None

        fused = self.fuse(torch.cat(parts, dim=-1))

        if self.model_variant == "plan_c_mvp" and self.regime_router is not None and self.regime_experts is not None:
            route_scalars = self._build_route_scalars(
                history_z=h_z,
                base_pred_z=base,
                structured_feats=sf,
                text_strength=strength,
            )
            route_pack = self.regime_router(fused, route_scalars)
            expert_inputs = self._build_expert_inputs(
                fused_base=fused,
                hist_part=hist_part,
                base_part=base_part,
                structured_part=structured_part,
                text_part=text_part,
                news_strength=route_scalars[:, -1:].to(device=fused.device, dtype=fused.dtype),
            )
            fused, _ = self.regime_experts(
                expert_inputs,
                route_pack["expert_probs"],
                mix_scale=self.multimodal_fuse_lambda,
            )
            fused = self.route_summary_ln(fused + route_pack["scalar_hidden"])

        logits = self.out_head(fused)
        if self.task_type == "relative_state":
            return logits.view(logits.size(0), self.horizon, 3)
        return logits
