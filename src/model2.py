import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from .regime_blocks import RegimeRouter, ResidualExpertMixture
from .temporal_text import TemporalTextTower


def _load_hf_tokenizer(tokenizer_id: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
    return tok


class TinyNewsTSRegressor(nn.Module):
    """
    Lightweight DELTA regressor used by this project:
    - TS patches -> MLP encoder
    - pooled TS feature -> base/delta heads
    - structured-news features can bias the residual correction path
    """

    def __init__(
        self,
        horizon: int,
        patch_dim: int,
        patch_stride: int,
        hidden_size: int,
        patch_dropout: float,
        head_dropout: float,
        delta_head_init_std: float = 0.01,
        delta_clip: float = 3.0,
        structured_feat_dim: int = 0,
        huber_beta: float = 0.5,
        use_horizon_weight: bool = True,
        horizon_weight_end: float = 0.5,
        delta_alpha_scale: float = 0.75,
        delta_patch_prototypes: int = 0,
        delta_patch_proto_temp: float = 1.0,
        delta_sign_tau: float = 1.0,
        delta_residual_mode: str = "additive",
        delta_sign_mode: str = "signnet_binary",
        delta_mag_max: float = 0.0,
        doc_candidate_mode: str = "beta_only",
        #news text encode
        temporal_text_enable: bool = False,
        temporal_text_model_id: str = "",
        temporal_text_dim: int = 8,
        temporal_text_fuse_lambda: float = 0.5,
        temporal_text_freeze_encoder: bool = True,
        multimodal_arch: str = "summary_gated",
        multimodal_fuse_lambda: float = 1.0,
        route_conf_floor: float = 0.25,
    ):
        super().__init__()
        self.model_variant = "tiny_news_ts"
        self.horizon = int(horizon)
        self.patch_dim = int(patch_dim)
        self.patch_stride = int(max(1, patch_stride))
        self.hidden_size = int(hidden_size)

        mid = max(32, self.hidden_size * 2)
        self.patch_proj = nn.Sequential(
            nn.Linear(self.patch_dim, mid),
            nn.GELU(),
            nn.Linear(mid, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.patch_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
        )
        self.patch_drop = nn.Dropout(float(patch_dropout))
        self.ts_ln = nn.LayerNorm(self.hidden_size)
        self.delta_patch_prototypes = int(max(0, delta_patch_prototypes))
        self.delta_patch_proto_temp = float(max(1e-6, delta_patch_proto_temp))
        if self.delta_patch_prototypes > 0:
            self.patch_proto_router = nn.Linear(self.hidden_size, self.delta_patch_prototypes)
            self.patch_prototypes = nn.Parameter(torch.randn(self.delta_patch_prototypes, self.hidden_size) * 0.02)
            nn.init.zeros_(self.patch_proto_router.weight)
            nn.init.zeros_(self.patch_proto_router.bias)
        else:
            self.patch_proto_router = None
            self.patch_prototypes = None

        self.head_drop = nn.Dropout(float(head_dropout))
        self.base_head = nn.Linear(self.hidden_size, self.horizon)

        self.delta_head_drop = nn.Dropout(float(head_dropout))
        # Kept under the existing name for checkpoint compatibility; used as the sign head.
        self.delta_head = nn.Linear(self.hidden_size, self.horizon)
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=float(delta_head_init_std))
        nn.init.zeros_(self.delta_head.bias)
        sign_mode = str(delta_sign_mode or "signnet_binary").lower().strip()
        if sign_mode not in {"signnet_binary", "internal"}:
            sign_mode = "signnet_binary"
        self.delta_sign_mode = sign_mode
        multimodal_arch_norm = str(multimodal_arch or "summary_gated").lower().strip()
        if multimodal_arch_norm not in {"summary_gated", "plan_c_mvp"}:
            multimodal_arch_norm = "summary_gated"
        self.multimodal_arch = multimodal_arch_norm
        self.multimodal_fuse_lambda = float(max(0.0, multimodal_fuse_lambda))
        self.route_conf_floor = float(max(0.0, min(1.0, route_conf_floor)))
        self.regime_route_names = ("none", "trend", "event", "reversal", "sparse")
        residual_mode = str(delta_residual_mode or "additive").lower().strip()
        if residual_mode not in {"additive", "relative"}:
            residual_mode = "additive"
        self.delta_residual_mode = residual_mode

        self.rel_head = nn.Linear(self.hidden_size, 1)
        self.delta_rel_head = nn.Linear(self.hidden_size, self.horizon)
        nn.init.zeros_(self.delta_rel_head.weight)
        nn.init.zeros_(self.delta_rel_head.bias)
        self.delta_log_scale = nn.Parameter(torch.zeros(1))
        self.delta_clip = float(delta_clip)
        self.delta_alpha_scale = float(max(0.0, delta_alpha_scale))
        self.delta_sign_tau = float(max(1e-6, delta_sign_tau))
        self.delta_mag_max = float(max(0.0, delta_mag_max))
        self.doc_candidate_mode = str(doc_candidate_mode or "beta_only")
        delta_fuse_in = self.hidden_size * 3
        self.delta_fuse = nn.Sequential(
            nn.Linear(delta_fuse_in, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        self.text_summary_ln = nn.LayerNorm(self.hidden_size)
        self.route_summary_ln = nn.LayerNorm(self.hidden_size)
        self.delta_state_head = nn.Linear(self.hidden_size, self.horizon * 3)
        nn.init.zeros_(self.delta_state_head.weight)
        nn.init.zeros_(self.delta_state_head.bias)
        self.delta_mag_head = nn.Linear(self.hidden_size, self.horizon)
        nn.init.normal_(self.delta_mag_head.weight, mean=0.0, std=float(delta_head_init_std))
        nn.init.constant_(self.delta_mag_head.bias, -2.0)
        self.text_mag_head = nn.Linear(self.hidden_size, self.horizon)
        nn.init.zeros_(self.text_mag_head.weight)
        nn.init.zeros_(self.text_mag_head.bias)
        self.route_mag_head = nn.Linear(self.hidden_size, self.horizon)
        nn.init.zeros_(self.route_mag_head.weight)
        nn.init.zeros_(self.route_mag_head.bias)
        self.confidence_head = nn.Linear(self.hidden_size, self.horizon)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

        self.structured_feat_dim = int(max(0, structured_feat_dim))
        if self.structured_feat_dim > 0:
            self.structured_proj = nn.Linear(self.structured_feat_dim, self.hidden_size)
            self.structured_gate_bias = nn.Linear(self.structured_feat_dim, self.horizon)
            self.structured_rel_bias = nn.Linear(self.structured_feat_dim, self.horizon)
            self.structured_sign_head = nn.Linear(self.structured_feat_dim, self.horizon)
            self.structured_scale_head = nn.Linear(self.structured_feat_dim, self.horizon)
            self.structured_decay_head = nn.Linear(self.structured_feat_dim, 1)
            self.structured_mask_head = nn.Linear(self.structured_feat_dim, self.horizon)
            nn.init.zeros_(self.structured_proj.weight)
            nn.init.zeros_(self.structured_proj.bias)
            nn.init.zeros_(self.structured_gate_bias.weight)
            nn.init.zeros_(self.structured_gate_bias.bias)
            nn.init.zeros_(self.structured_rel_bias.weight)
            nn.init.zeros_(self.structured_rel_bias.bias)
            nn.init.zeros_(self.structured_sign_head.weight)
            nn.init.zeros_(self.structured_sign_head.bias)
            nn.init.zeros_(self.structured_scale_head.weight)
            nn.init.zeros_(self.structured_scale_head.bias)
            nn.init.zeros_(self.structured_decay_head.weight)
            nn.init.zeros_(self.structured_decay_head.bias)
            nn.init.zeros_(self.structured_mask_head.weight)
            nn.init.zeros_(self.structured_mask_head.bias)
        else:
            self.structured_proj = None
            self.structured_gate_bias = None
            self.structured_rel_bias = None
            self.structured_sign_head = None
            self.structured_scale_head = None
            self.structured_decay_head = None
            self.structured_mask_head = None

        if self.multimodal_arch == "plan_c_mvp":
            self.regime_router = RegimeRouter(
                hidden_size=self.hidden_size,
                scalar_dim=6,
                num_routes=len(self.regime_route_names),
                dropout=float(head_dropout),
            )
            self.regime_experts = ResidualExpertMixture(
                hidden_size=self.hidden_size,
                num_experts=max(1, len(self.regime_route_names) - 1),
                dropout=float(head_dropout),
            )
        else:
            self.regime_router = None
            self.regime_experts = None

        self.news_fuse = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.news_fuse_ln = nn.LayerNorm(self.hidden_size)
        self.alpha_head = nn.Linear(self.hidden_size, self.horizon)
        self.beta_head = nn.Linear(self.hidden_size, self.horizon)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

        self.huber_beta = float(huber_beta)
        self.use_horizon_weight = bool(use_horizon_weight)
        self.horizon_weight_end = float(horizon_weight_end)
        self.temporal_text_enable = bool(temporal_text_enable)
        self.temporal_text_dim = int(max(1, temporal_text_dim))
        self.temporal_text_fuse_lambda = float(max(0.0, temporal_text_fuse_lambda))
        self.temporal_text_freeze_encoder = bool(temporal_text_freeze_encoder)
        self.temporal_text_model_id = str(temporal_text_model_id or "").strip()
        if self.temporal_text_enable:
            if not self.temporal_text_model_id:
                raise ValueError("temporal_text_model_id must be provided when temporal text auxiliary input is enabled.")
            self.temporal_text_tower = TemporalTextTower(
                model_id=self.temporal_text_model_id,
                step_dim=self.temporal_text_dim,
                hidden_size=self.hidden_size,
                patch_dim=self.patch_dim,
                patch_stride=self.patch_stride,
                freeze_encoder=self.temporal_text_freeze_encoder,
            )
            self.temporal_text_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            nn.init.zeros_(self.temporal_text_gate[-1].weight)
            nn.init.constant_(self.temporal_text_gate[-1].bias, -2.0)
        else:
            self.temporal_text_tower = None
            self.temporal_text_gate = None

        # trainer may override this field dynamically
        self.patch_mask_p = 0.0

    def _pool_ts(self, ts_feat: torch.Tensor, ts_patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        if ts_patch_mask is None:
            w = torch.ones(ts_feat.size(0), ts_feat.size(1), device=ts_feat.device, dtype=ts_feat.dtype)
        else:
            w = ts_patch_mask.to(device=ts_feat.device, dtype=ts_feat.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (ts_feat * w.unsqueeze(-1)).sum(dim=1) / denom
        return self.ts_ln(pooled)

    def _clip_delta_tensor(self, x: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.delta_clip > 0:
            c = torch.tensor(self.delta_clip, device=device, dtype=dtype)
            return c * torch.tanh(x / c)
        return x

    def _encode_temporal_text_features(
        self,
        temporal_text_ids: torch.Tensor | None,
        temporal_text_attn: torch.Tensor | None,
        temporal_text_step_mask: torch.Tensor | None,
        *,
        target_patch_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | None]:
        if (
            (not self.temporal_text_enable)
            or self.temporal_text_tower is None
            or temporal_text_ids is None
            or temporal_text_attn is None
            or temporal_text_ids.ndim != 3
            or temporal_text_attn.ndim != 3
        ):
            return {
                "patch_context": None,
                "patch_mask": None,
                "text_summary": None,
                "text_strength": None,
            }
        return self.temporal_text_tower(
            temporal_text_ids=temporal_text_ids,
            temporal_text_attn=temporal_text_attn,
            temporal_text_step_mask=temporal_text_step_mask,
            target_patch_count=int(max(1, target_patch_count)),
            device=device,
            dtype=dtype,
        )

    def _build_structured_pack(
        self,
        structured_feats: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | None]:
        if self.structured_feat_dim <= 0 or structured_feats is None:
            return {
                "feats": None,
                "weight": None,
                "summary": None,
            }

        sf = structured_feats.to(device=device, dtype=dtype)
        if sf.dim() == 1:
            sf = sf.unsqueeze(0)
        if sf.size(-1) < self.structured_feat_dim:
            pad = sf.new_zeros(sf.size(0), self.structured_feat_dim - sf.size(-1))
            sf = torch.cat([sf, pad], dim=-1)
        elif sf.size(-1) > self.structured_feat_dim:
            sf = sf[:, : self.structured_feat_dim]

        relevance = sf[:, :1].clamp(0.0, 1.0) if sf.size(-1) >= 1 else torch.ones(
            (sf.size(0), 1), device=sf.device, dtype=sf.dtype
        )
        confidence = sf[:, 4:5].clamp(0.0, 1.0) if sf.size(-1) >= 5 else torch.ones(
            (sf.size(0), 1), device=sf.device, dtype=sf.dtype
        )
        structured_weight = (relevance * (0.5 + 0.5 * confidence)).clamp(0.0, 1.0)
        structured_summary = self.structured_proj(sf) * structured_weight if self.structured_proj is not None else None

        return {
            "feats": sf,
            "weight": structured_weight,
            "summary": structured_summary,
        }

    def _build_route_scalars(
        self,
        *,
        ts_patches: torch.Tensor,
        ts_patch_mask: torch.Tensor | None,
        text_strength: torch.Tensor | None,
        structured_weight: torch.Tensor | None,
        text_summary: torch.Tensor | None,
    ) -> torch.Tensor:
        device = ts_patches.device
        dtype = ts_patches.dtype
        ts_vol = ts_patches.to(dtype=torch.float32).std(dim=(1, 2), unbiased=False).to(device=device, dtype=dtype).unsqueeze(1)
        if ts_patch_mask is None:
            patch_density = torch.ones(ts_patches.size(0), 1, device=device, dtype=dtype)
        else:
            patch_density = ts_patch_mask.to(device=device, dtype=dtype).mean(dim=1, keepdim=True)
        txt_strength = (
            text_strength.to(device=device, dtype=dtype)
            if text_strength is not None
            else torch.zeros(ts_patches.size(0), 1, device=device, dtype=dtype)
        )
        struct_strength = (
            structured_weight.to(device=device, dtype=dtype)
            if structured_weight is not None
            else torch.zeros(ts_patches.size(0), 1, device=device, dtype=dtype)
        )
        news_strength = torch.maximum(txt_strength, struct_strength)
        text_norm = (
            text_summary.norm(dim=1, keepdim=True).to(device=device, dtype=dtype) / float(max(1, self.hidden_size))
            if text_summary is not None
            else torch.zeros(ts_patches.size(0), 1, device=device, dtype=dtype)
        )
        return torch.cat([ts_vol, patch_density, txt_strength, struct_strength, news_strength, text_norm], dim=-1)

    def _build_regime_expert_inputs(
        self,
        *,
        residual_base: torch.Tensor,
        pooled_ts: torch.Tensor,
        fused_news_context: torch.Tensor,
        text_summary: torch.Tensor,
        text_strength: torch.Tensor,
        news_strength: torch.Tensor,
    ) -> list[torch.Tensor]:
        txt_ctx = text_summary * text_strength
        sparse_scale = 1.0 - news_strength.clamp(0.0, 1.0)
        return [
            self.route_summary_ln(residual_base + pooled_ts),
            self.route_summary_ln(residual_base + fused_news_context + txt_ctx),
            self.route_summary_ln(residual_base + fused_news_context - pooled_ts),
            self.route_summary_ln(residual_base + (sparse_scale * pooled_ts)),
        ]

    def forward(
        self,
        ts_patches: torch.Tensor,
        ts_patch_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
        head_mode: str = "base",
        rel_targets: torch.Tensor | None = None,
        rel_lambda: float = 0.0,
        structured_feats: torch.Tensor | None = None,
        temporal_text_ids: torch.Tensor | None = None,
        temporal_text_attn: torch.Tensor | None = None,
        temporal_text_step_mask: torch.Tensor | None = None,
    ):
        proj_dtype = next(self.patch_proj.parameters()).dtype
        if ts_patches.dtype != proj_dtype:
            ts_patches = ts_patches.to(dtype=proj_dtype)
        ts_feat = self.patch_proj(ts_patches)
        ts_feat = ts_feat * self.patch_gate(ts_feat)
        ts_feat = self.patch_drop(ts_feat)

        if self.training and float(getattr(self, "patch_mask_p", 0.0)) > 0:
            keep = (torch.rand(ts_feat.size(0), ts_feat.size(1), 1, device=ts_feat.device) > self.patch_mask_p).to(
                ts_feat.dtype
            )
            ts_feat = ts_feat * keep

        if self.delta_patch_prototypes > 0 and self.patch_proto_router is not None and self.patch_prototypes is not None:
            proto_logits = self.patch_proto_router(ts_feat) / self.delta_patch_proto_temp
            proto_w = torch.softmax(proto_logits, dim=-1)
            ts_feat = torch.einsum("bpk,kh->bph", proto_w, self.patch_prototypes.to(device=ts_feat.device, dtype=ts_feat.dtype))

        structured_weight = None
        structured_context = None
        struct_sign_logit = None
        struct_scale = None
        struct_decay = None
        struct_mask_logits = None
        temporal_text_patch_context = None
        temporal_text_patch_mask = None
        temporal_text_gate = None
        temporal_text_summary = None
        temporal_text_strength = None
        route_summary = None
        route_logits = None
        route_probs = None
        route_top_idx = None
        route_abstain = None
        expert_outputs = None
        confidence = None
        confidence_logits = None

        if head_mode == "base":
            pooled = self._pool_ts(ts_feat, ts_patch_mask=ts_patch_mask)
            rel_input = pooled
            pred = self.base_head(self.head_drop(pooled))
        elif head_mode == "delta":
            temporal_text_features = self._encode_temporal_text_features(
                temporal_text_ids=temporal_text_ids,
                temporal_text_attn=temporal_text_attn,
                temporal_text_step_mask=temporal_text_step_mask,
                target_patch_count=ts_feat.size(1),
                device=ts_feat.device,
                dtype=ts_feat.dtype,
            )
            temporal_text_patch_context = temporal_text_features.get("patch_context")
            temporal_text_patch_mask = temporal_text_features.get("patch_mask")
            temporal_text_summary = temporal_text_features.get("text_summary")
            temporal_text_strength = temporal_text_features.get("text_strength")
            if (
                self.multimodal_arch != "plan_c_mvp"
                and temporal_text_patch_context is not None
                and self.temporal_text_fuse_lambda > 0.0
            ):
                if self.temporal_text_gate is not None:
                    temporal_text_gate = torch.sigmoid(
                        self.temporal_text_gate(torch.cat([ts_feat, temporal_text_patch_context], dim=-1))
                    )
                    if temporal_text_patch_mask is not None:
                        gate_mask = temporal_text_patch_mask.to(device=ts_feat.device, dtype=ts_feat.dtype).unsqueeze(-1)
                        temporal_text_gate = temporal_text_gate * gate_mask
                    gated_temporal_text = temporal_text_gate * temporal_text_patch_context
                else:
                    gated_temporal_text = temporal_text_patch_context
                ts_feat = ts_feat + (self.temporal_text_fuse_lambda * gated_temporal_text)

            structured_pack = self._build_structured_pack(
                structured_feats=structured_feats,
                device=ts_feat.device,
                dtype=ts_feat.dtype,
            )
            sf = structured_pack.get("feats")
            structured_weight = structured_pack.get("weight")
            structured_context = structured_pack.get("summary")

            if sf is not None:
                struct_sign_logit = self.structured_sign_head(sf)
                struct_scale = F.softplus(self.structured_scale_head(sf))
                struct_decay = F.softplus(self.structured_decay_head(sf))
                struct_mask_logits = self.structured_mask_head(sf)
            else:
                struct_sign_logit = None
                struct_scale = None
                struct_decay = None
                struct_mask_logits = None

            pooled = self._pool_ts(ts_feat, ts_patch_mask=ts_patch_mask)
            if temporal_text_summary is None:
                temporal_text_summary = torch.zeros_like(pooled)
                temporal_text_strength = torch.zeros(pooled.size(0), 1, device=pooled.device, dtype=pooled.dtype)
            else:
                temporal_text_summary = self.text_summary_ln(temporal_text_summary.to(device=pooled.device, dtype=pooled.dtype))
                if temporal_text_strength is None:
                    temporal_text_strength = torch.ones(pooled.size(0), 1, device=pooled.device, dtype=pooled.dtype)
                else:
                    temporal_text_strength = temporal_text_strength.to(device=pooled.device, dtype=pooled.dtype)

            news_context_sum = torch.zeros_like(pooled)
            news_context_weight = torch.zeros(pooled.size(0), 1, device=pooled.device, dtype=pooled.dtype)
            if structured_context is not None and structured_weight is not None:
                news_context_sum = news_context_sum + structured_context
                news_context_weight = news_context_weight + structured_weight
            if float((news_context_weight > 0).to(dtype=pooled.dtype).sum().item()) > 0.0:
                fused_news_context = news_context_sum / news_context_weight.clamp_min(1e-6)
                fused_news_context = self.news_fuse_ln(fused_news_context + self.news_fuse(fused_news_context))
            else:
                fused_news_context = torch.zeros_like(pooled)
            news_strength = torch.maximum(
                news_context_weight.clamp(min=0.0, max=1.0),
                temporal_text_strength.clamp(min=0.0, max=1.0),
            )
            residual_base = self.delta_fuse(
                torch.cat([pooled, fused_news_context, temporal_text_summary * temporal_text_strength], dim=-1)
            )
            residual_context = residual_base

            if self.multimodal_arch == "plan_c_mvp" and self.regime_router is not None and self.regime_experts is not None:
                route_scalars = self._build_route_scalars(
                    ts_patches=ts_patches.to(device=pooled.device, dtype=pooled.dtype),
                    ts_patch_mask=ts_patch_mask,
                    text_strength=temporal_text_strength,
                    structured_weight=structured_weight,
                    text_summary=temporal_text_summary,
                )
                route_pack = self.regime_router(residual_base, route_scalars)
                route_logits = route_pack["route_logits"]
                route_probs = route_pack["route_probs"]
                route_abstain = route_pack["abstain_prob"]
                route_top_idx = torch.argmax(route_probs, dim=-1)
                expert_inputs = self._build_regime_expert_inputs(
                    residual_base=residual_base,
                    pooled_ts=pooled,
                    fused_news_context=fused_news_context,
                    text_summary=temporal_text_summary,
                    text_strength=temporal_text_strength,
                    news_strength=news_strength,
                )
                residual_context, expert_outputs = self.regime_experts(
                    expert_inputs,
                    route_pack["expert_probs"],
                    mix_scale=self.multimodal_fuse_lambda,
                )
                route_summary = self.route_summary_ln(residual_context + route_pack["scalar_hidden"])

            magnitude_raw = self.delta_mag_head(self.delta_head_drop(residual_context))
            magnitude_raw = magnitude_raw + (
                self.text_mag_head(self.delta_head_drop(temporal_text_summary)) * temporal_text_strength
            )
            if self.multimodal_arch == "plan_c_mvp" and route_summary is not None:
                magnitude_raw = magnitude_raw + self.route_mag_head(self.delta_head_drop(route_summary))
            confidence_logits = self.confidence_head(self.delta_head_drop(residual_context))
            confidence = torch.sigmoid(confidence_logits)
            if route_abstain is not None:
                abstain_gate = 1.0 - route_abstain.to(device=confidence.device, dtype=confidence.dtype)
                if self.route_conf_floor > 0.0:
                    abstain_gate = self.route_conf_floor + (1.0 - self.route_conf_floor) * abstain_gate
                confidence = confidence * abstain_gate
            magnitude = F.softplus(magnitude_raw)
            magnitude = magnitude * (0.5 + confidence)
            if self.delta_mag_max > 0.0:
                magnitude = magnitude.clamp(max=self.delta_mag_max)
            state_logits = None
            state_probs = None
            state_score = None
            sign_logits = None
            sign_soft = None

            if self.delta_residual_mode == "relative":
                delta_init = magnitude
                state_logits = self.delta_state_head(self.delta_head_drop(residual_context)).view(
                    residual_context.size(0), self.horizon, 3
                )
                state_probs = torch.softmax(state_logits, dim=-1)
                state_score = state_probs[..., 2] - state_probs[..., 0]
                if self.delta_sign_mode == "internal":
                    pred = delta_init * state_score
                else:
                    pred = delta_init
            else:
                sign_logits = self.delta_head(self.delta_head_drop(residual_context))
                sign_soft = torch.tanh(sign_logits / self.delta_sign_tau)
                delta_init = sign_soft * magnitude
                pred = delta_init
            pred = self._clip_delta_tensor(pred, dtype=pred.dtype, device=pred.device)
            rel_input = residual_context
        else:
            raise ValueError(f"Unknown head_mode={head_mode}")

        pred = pred.to(torch.float32)
        rel_logit = self.rel_head(rel_input).squeeze(-1).to(torch.float32)

        out = {"pred": pred, "rel_logits": rel_logit}
        if head_mode == "delta":
            out["delta_scale"] = torch.ones((), device=pred.device, dtype=pred.dtype)
            out["delta_init"] = delta_init.to(torch.float32)
            if sign_logits is not None:
                out["sign_logits"] = sign_logits.to(torch.float32)
            if sign_soft is not None:
                out["sign_soft"] = sign_soft.to(torch.float32)
            if state_logits is not None:
                out["state_logits"] = state_logits.to(torch.float32)
            if state_probs is not None:
                out["state_probs"] = state_probs.to(torch.float32)
            if state_score is not None:
                out["state_score"] = state_score.to(torch.float32)
            out["magnitude"] = magnitude.to(torch.float32)
            out["magnitude_raw"] = magnitude_raw.to(torch.float32)
            out["delta_sign_mode"] = str(self.delta_sign_mode)
            out["delta_residual_mode"] = str(self.delta_residual_mode)
            out["alpha_news"] = torch.ones_like(pred, dtype=torch.float32)
            out["beta_news"] = torch.zeros_like(pred, dtype=torch.float32)
            out["doc_impact"] = torch.zeros_like(pred, dtype=torch.float32)
            out["struct_impact"] = torch.zeros_like(pred, dtype=torch.float32)
            out["news_available_mask"] = news_strength.to(torch.float32)
            out["fused_news_context_norm"] = fused_news_context.norm(dim=1).mean().detach()
            if temporal_text_summary is not None:
                out["text_summary"] = temporal_text_summary.to(torch.float32)
                out["text_summary_norm"] = temporal_text_summary.norm(dim=1).mean().detach()
            if temporal_text_strength is not None:
                out["text_strength"] = temporal_text_strength.to(torch.float32)
                out["text_strength_mean"] = temporal_text_strength.mean().detach()
            if temporal_text_gate is not None:
                out["temporal_text_gate_mean"] = temporal_text_gate.mean().detach()
            if route_summary is not None:
                out["route_summary"] = route_summary.to(torch.float32)
                out["route_summary_norm"] = route_summary.norm(dim=1).mean().detach()
            if route_logits is not None:
                out["route_logits"] = route_logits.to(torch.float32)
            if route_probs is not None:
                out["route_probs"] = route_probs.to(torch.float32)
                out["route_abstain_mean"] = route_probs[:, :1].mean().detach()
            if route_top_idx is not None:
                out["route_top_idx"] = route_top_idx.to(torch.int64)
            if expert_outputs is not None:
                out["expert_output_norm"] = expert_outputs.norm(dim=-1).mean().detach()
            if confidence is not None:
                out["confidence"] = confidence.to(torch.float32)
                out["confidence_mean"] = confidence.mean().detach()
            if confidence_logits is not None:
                out["confidence_logits"] = confidence_logits.to(torch.float32)
            if structured_weight is not None:
                out["structured_weight_mean"] = structured_weight.mean().detach()
                out["structured_weight"] = structured_weight.to(torch.float32)
            if struct_sign_logit is not None:
                out["struct_sign_logit"] = struct_sign_logit.to(torch.float32)
            if struct_scale is not None:
                out["struct_scale"] = struct_scale.to(torch.float32)
            if struct_decay is not None:
                out["struct_decay"] = struct_decay.to(torch.float32)
            if struct_mask_logits is not None:
                out["struct_mask_logits"] = struct_mask_logits.to(torch.float32)
                out["struct_mask"] = torch.sigmoid(struct_mask_logits).to(torch.float32)
            out["doc_impact_mean_abs"] = torch.zeros((), device=pred.device, dtype=pred.dtype)
            out["struct_impact_mean_abs"] = torch.zeros((), device=pred.device, dtype=pred.dtype)

        loss = None
        loss_fore = None
        loss_rel = None
        if targets is not None:
            if targets.dtype != pred.dtype:
                targets = targets.to(dtype=pred.dtype)
            per = F.smooth_l1_loss(pred, targets, beta=self.huber_beta, reduction="none")
            if self.use_horizon_weight and self.horizon > 1:
                w = torch.linspace(
                    1.0, float(self.horizon_weight_end), steps=self.horizon, device=per.device, dtype=per.dtype
                )[None, :]
                loss_fore = (per * w).mean()
            else:
                loss_fore = per.mean()
            loss = loss_fore
        if rel_targets is not None:
            rel_targets = rel_targets.to(device=rel_logit.device, dtype=rel_logit.dtype)
            if rel_logit.ndim == 2:
                if rel_targets.ndim == 1:
                    rel_targets = rel_targets.unsqueeze(1)
                if rel_targets.ndim == 2 and rel_targets.size(1) == 1 and rel_logit.size(1) > 1:
                    rel_targets = rel_targets.expand(-1, rel_logit.size(1))
                elif rel_targets.ndim == 2 and rel_targets.size(1) != rel_logit.size(1):
                    if rel_targets.size(1) > rel_logit.size(1):
                        rel_targets = rel_targets[:, : rel_logit.size(1)]
                    else:
                        pad = rel_targets[:, -1:].expand(-1, rel_logit.size(1) - rel_targets.size(1))
                        rel_targets = torch.cat([rel_targets, pad], dim=1)
            loss_rel = F.binary_cross_entropy_with_logits(rel_logit, rel_targets)
            loss = rel_lambda * loss_rel if loss is None else (loss + rel_lambda * loss_rel)
        if loss is not None:
            out["loss"] = loss
            out["loss_fore"] = loss_fore
            out["loss_rel"] = loss_rel
        return out

def build_delta_model(
    base_model: str,
    tokenizer_id: str,
    horizon: int = 48,
    patch_dim: int = 4,
    patch_stride: int = 4,
    patch_dropout: float = 0.0,
    head_dropout: float = 0.0,
    head_mlp: bool = False,
    huber_beta: float = 0.5,
    use_horizon_weight: bool = True,
    horizon_weight_end: float = 0.5,
    delta_head_init_std: float = 0.01,
    delta_clip: float = 3.0,
    delta_news_tail_tokens: int = 160,
    delta_structured_feature_dim: int = 0,
    delta_model_variant: str = "tiny_news_ts",
    tiny_news_hidden_size: int = 256,
    delta_alpha_scale: float = 0.75,
    delta_patch_prototypes: int = 0,
    delta_patch_proto_temp: float = 1.0,
    delta_sign_tau: float = 1.0,
    delta_residual_mode: str = "additive",
    delta_sign_mode: str = "signnet_binary",
    delta_mag_max: float = 0.0,
    doc_candidate_mode: str = "beta_only",
    delta_temporal_text_enable: int = 0,
    delta_temporal_text_model_id: str = "",
    delta_temporal_text_dim: int = 8,
    delta_temporal_text_fuse_lambda: float = 0.5,
    delta_temporal_text_freeze_encoder: int = 1,
    delta_multimodal_arch: str = "summary_gated",
    delta_multimodal_fuse_lambda: float = 1.0,
    delta_route_conf_floor: float = 0.25,
):
    del (
        head_mlp,
        delta_news_tail_tokens,
        delta_model_variant,
    )

    tok_id = str(tokenizer_id or "").strip() or str(base_model or "").strip() or "distilbert-base-uncased"
    temporal_text_model_id = str(delta_temporal_text_model_id or "").strip() or str(tok_id)
    tok = _load_hf_tokenizer(tok_id)
    temporal_text_tok = _load_hf_tokenizer(temporal_text_model_id) if bool(int(delta_temporal_text_enable)) else tok

    hidden_size = int(tiny_news_hidden_size) if int(tiny_news_hidden_size) > 0 else 256

    model = TinyNewsTSRegressor(
        horizon=int(horizon),
        patch_dim=int(patch_dim),
        patch_stride=int(max(1, patch_stride)),
        hidden_size=int(hidden_size),
        patch_dropout=float(patch_dropout),
        head_dropout=float(head_dropout),
        delta_head_init_std=float(delta_head_init_std),
        delta_clip=float(delta_clip),
        structured_feat_dim=int(delta_structured_feature_dim),
        huber_beta=float(huber_beta),
        use_horizon_weight=bool(use_horizon_weight),
        horizon_weight_end=float(horizon_weight_end),
        delta_alpha_scale=float(delta_alpha_scale),
        delta_patch_prototypes=int(delta_patch_prototypes),
        delta_patch_proto_temp=float(delta_patch_proto_temp),
        delta_sign_tau=float(delta_sign_tau),
        delta_residual_mode=str(delta_residual_mode or "additive"),
        delta_sign_mode=str(delta_sign_mode or "signnet_binary"),
        delta_mag_max=float(delta_mag_max),
        doc_candidate_mode=str(doc_candidate_mode or "beta_only"),
        temporal_text_enable=bool(int(delta_temporal_text_enable)),
        temporal_text_model_id=temporal_text_model_id,
        temporal_text_dim=int(max(1, delta_temporal_text_dim)),
        temporal_text_fuse_lambda=float(max(0.0, delta_temporal_text_fuse_lambda)),
        temporal_text_freeze_encoder=bool(int(delta_temporal_text_freeze_encoder)),
        multimodal_arch=str(delta_multimodal_arch or "summary_gated"),
        multimodal_fuse_lambda=float(max(0.0, delta_multimodal_fuse_lambda)),
        route_conf_floor=float(max(0.0, min(1.0, delta_route_conf_floor))),
    )
    model.delta_tokenizer_id = str(tok_id)
    model.patch_stride = int(max(1, patch_stride))
    model.delta_alpha_scale = float(delta_alpha_scale)
    model.delta_patch_prototypes = int(max(0, delta_patch_prototypes))
    model.delta_patch_proto_temp = float(max(1e-6, delta_patch_proto_temp))
    model.delta_sign_tau = float(max(1e-6, delta_sign_tau))
    delta_residual_mode_norm = str(delta_residual_mode or "additive").lower().strip()
    if delta_residual_mode_norm not in {"additive", "relative"}:
        delta_residual_mode_norm = "additive"
    model.delta_residual_mode = delta_residual_mode_norm
    delta_sign_mode_norm = str(delta_sign_mode or "signnet_binary").lower().strip()
    if delta_sign_mode_norm not in {"signnet_binary", "internal"}:
        delta_sign_mode_norm = "signnet_binary"
    model.delta_sign_mode = delta_sign_mode_norm
    model.delta_mag_max = float(max(0.0, delta_mag_max))
    model.doc_candidate_mode = str(doc_candidate_mode or "beta_only")
    model.temporal_text_enable = bool(int(delta_temporal_text_enable))
    model.temporal_text_dim = int(max(1, delta_temporal_text_dim))
    model.temporal_text_fuse_lambda = float(max(0.0, delta_temporal_text_fuse_lambda))
    model.temporal_text_freeze_encoder = bool(int(delta_temporal_text_freeze_encoder))
    model.temporal_text_model_id = temporal_text_model_id
    model.multimodal_arch = str(delta_multimodal_arch or "summary_gated")
    model.multimodal_fuse_lambda = float(max(0.0, delta_multimodal_fuse_lambda))
    model.route_conf_floor = float(max(0.0, min(1.0, delta_route_conf_floor)))
    return tok, temporal_text_tok, model


def save_checkpoint(
    ckpt_dir: str,
    tok,
    model,
    base_model_id: str,
    tokenizer_id: str | None,
    train_cfg: dict,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    global_step: int | None = None,
    extra_meta: dict | None = None,
):
    os.makedirs(ckpt_dir, exist_ok=True)

    tok_dir = os.path.join(ckpt_dir, "tokenizer")
    tok.save_pretrained(tok_dir)

    reg_path = os.path.join(ckpt_dir, "regressor.pt")
    torch.save(
        {
            "model_variant": "tiny_news_ts",
            "model_state": model.state_dict(),
            "horizon": int(model.horizon),
            "patch_dim": int(model.patch_dim),
            "patch_stride": int(getattr(model, "patch_stride", model.patch_dim)),
            "hidden_size": int(model.hidden_size),
            "structured_feat_dim": int(getattr(model, "structured_feat_dim", 0)),
            "delta_clip": float(getattr(model, "delta_clip", 3.0)),
            "huber_beta": float(getattr(model, "huber_beta", 0.5)),
            "use_horizon_weight": bool(getattr(model, "use_horizon_weight", True)),
            "horizon_weight_end": float(getattr(model, "horizon_weight_end", 0.5)),
            "delta_tokenizer_id": str(getattr(model, "delta_tokenizer_id", "")),
            "delta_alpha_scale": float(getattr(model, "delta_alpha_scale", 0.75)),
            "delta_patch_prototypes": int(getattr(model, "delta_patch_prototypes", 0)),
            "delta_patch_proto_temp": float(getattr(model, "delta_patch_proto_temp", 1.0)),
            "delta_sign_tau": float(getattr(model, "delta_sign_tau", 1.0)),
            "delta_residual_mode": str(getattr(model, "delta_residual_mode", "additive")),
            "delta_sign_mode": str(getattr(model, "delta_sign_mode", "signnet_binary")),
            "delta_mag_max": float(getattr(model, "delta_mag_max", 0.0)),
            "doc_candidate_mode": str(getattr(model, "doc_candidate_mode", "beta_only")),
            "temporal_text_enable": int(bool(getattr(model, "temporal_text_enable", False))),
            "temporal_text_dim": int(getattr(model, "temporal_text_dim", 8)),
            "temporal_text_fuse_lambda": float(getattr(model, "temporal_text_fuse_lambda", 0.5)),
            "temporal_text_freeze_encoder": int(bool(getattr(model, "temporal_text_freeze_encoder", True))),
            "temporal_text_model_id": str(getattr(model, "temporal_text_model_id", "")),
            "multimodal_arch": str(getattr(model, "multimodal_arch", "summary_gated")),
            "multimodal_fuse_lambda": float(getattr(model, "multimodal_fuse_lambda", 1.0)),
            "route_conf_floor": float(getattr(model, "route_conf_floor", 0.25)),
        },
        reg_path,
    )

    meta = {
        "model_variant": "tiny_news_ts",
        "base_model_id": base_model_id,
        "tokenizer_id": tokenizer_id or base_model_id,
        "train_cfg": train_cfg,
        "horizon": int(model.horizon),
        "patch_dim": int(model.patch_dim),
        "patch_stride": int(getattr(model, "patch_stride", model.patch_dim)),
        "hidden_size": int(model.hidden_size),
        "huber_beta": float(getattr(model, "huber_beta", 0.5)),
        "use_horizon_weight": bool(getattr(model, "use_horizon_weight", True)),
        "horizon_weight_end": float(getattr(model, "horizon_weight_end", 0.5)),
        "delta_clip": float(getattr(model, "delta_clip", 3.0)),
        "structured_feat_dim": int(getattr(model, "structured_feat_dim", 0)),
        "delta_tokenizer_id": str(getattr(model, "delta_tokenizer_id", "")),
        "delta_alpha_scale": float(getattr(model, "delta_alpha_scale", 0.75)),
        "delta_patch_prototypes": int(getattr(model, "delta_patch_prototypes", 0)),
        "delta_patch_proto_temp": float(getattr(model, "delta_patch_proto_temp", 1.0)),
        "delta_sign_tau": float(getattr(model, "delta_sign_tau", 1.0)),
        "delta_residual_mode": str(getattr(model, "delta_residual_mode", "additive")),
        "delta_sign_mode": str(getattr(model, "delta_sign_mode", "signnet_binary")),
        "delta_mag_max": float(getattr(model, "delta_mag_max", 0.0)),
        "doc_candidate_mode": str(getattr(model, "doc_candidate_mode", "beta_only")),
        "temporal_text_enable": int(bool(getattr(model, "temporal_text_enable", False))),
        "temporal_text_dim": int(getattr(model, "temporal_text_dim", 8)),
        "temporal_text_fuse_lambda": float(getattr(model, "temporal_text_fuse_lambda", 0.5)),
        "temporal_text_freeze_encoder": int(bool(getattr(model, "temporal_text_freeze_encoder", True))),
        "temporal_text_model_id": str(getattr(model, "temporal_text_model_id", "")),
        "multimodal_arch": str(getattr(model, "multimodal_arch", "summary_gated")),
        "multimodal_fuse_lambda": float(getattr(model, "multimodal_fuse_lambda", 1.0)),
        "route_conf_floor": float(getattr(model, "route_conf_floor", 0.25)),
    }
    if isinstance(extra_meta, dict):
        for k, v in extra_meta.items():
            meta[str(k)] = v
    with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if optimizer is not None:
        state = {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))


def _load_state_dict_compatible(model: nn.Module, state_dict: dict) -> None:
    current = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if key not in current:
            continue
        if current[key].shape != value.shape:
            continue
        filtered[key] = value
    model.load_state_dict(filtered, strict=False)


def load_checkpoint(
    ckpt_dir: str,
    device_map=None,
    is_trainable: bool = False,
    head_mlp: bool = False,
    hd: float = 0.0,
    pd: float = 0.0,
):
    del device_map, is_trainable, head_mlp

    with open(os.path.join(ckpt_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    tok = _load_hf_tokenizer(os.path.join(ckpt_dir, "tokenizer"))
    temporal_text_model_id = str(
        meta.get("temporal_text_model_id", meta.get("delta_tokenizer_id", meta.get("tiny_text_tokenizer_id", "")))
    ).strip()
    temporal_text_enable = bool(int(meta.get("temporal_text_enable", 0)))
    temporal_text_tok = _load_hf_tokenizer(temporal_text_model_id) if temporal_text_enable and temporal_text_model_id else tok

    model = TinyNewsTSRegressor(
        horizon=int(meta.get("horizon", 48)),
        patch_dim=int(meta.get("patch_dim", 4)),
        patch_stride=int(meta.get("patch_stride", meta.get("patch_dim", 4))),
        hidden_size=int(meta.get("hidden_size", 256)),
        patch_dropout=float(pd),
        head_dropout=float(hd),
        delta_clip=float(meta.get("delta_clip", 3.0)),
        structured_feat_dim=int(meta.get("structured_feat_dim", 0)),
        huber_beta=float(meta.get("huber_beta", 0.5)),
        use_horizon_weight=bool(meta.get("use_horizon_weight", True)),
        horizon_weight_end=float(meta.get("horizon_weight_end", 0.5)),
        delta_alpha_scale=float(meta.get("delta_alpha_scale", 0.75)),
        delta_patch_prototypes=int(meta.get("delta_patch_prototypes", 0) or 0),
        delta_patch_proto_temp=float(meta.get("delta_patch_proto_temp", 1.0)),
        delta_sign_tau=float(meta.get("delta_sign_tau", 1.0)),
        delta_residual_mode=str(meta.get("delta_residual_mode", "additive")),
        delta_sign_mode=str(meta.get("delta_sign_mode", "signnet_binary")),
        delta_mag_max=float(meta.get("delta_mag_max", 0.0)),
        doc_candidate_mode=str(meta.get("doc_candidate_mode", "beta_only")),
        temporal_text_enable=temporal_text_enable,
        temporal_text_model_id=temporal_text_model_id,
        temporal_text_dim=int(meta.get("temporal_text_dim", 8)),
        temporal_text_fuse_lambda=float(meta.get("temporal_text_fuse_lambda", 0.5)),
        temporal_text_freeze_encoder=bool(int(meta.get("temporal_text_freeze_encoder", 1))),
        multimodal_arch=str(meta.get("multimodal_arch", "summary_gated")),
        multimodal_fuse_lambda=float(meta.get("multimodal_fuse_lambda", 1.0)),
        route_conf_floor=float(meta.get("route_conf_floor", 0.25)),
    )
    model.delta_tokenizer_id = str(
        meta.get("delta_tokenizer_id", meta.get("tiny_text_tokenizer_id", ""))
    )
    model.patch_stride = int(meta.get("patch_stride", meta.get("patch_dim", 4)))
    model.delta_alpha_scale = float(meta.get("delta_alpha_scale", 0.75))
    model.delta_patch_prototypes = int(max(0, meta.get("delta_patch_prototypes", 0) or 0))
    model.delta_patch_proto_temp = float(max(1e-6, meta.get("delta_patch_proto_temp", 1.0) or 1.0))
    model.delta_sign_tau = float(max(1e-6, meta.get("delta_sign_tau", 1.0) or 1.0))
    delta_residual_mode = str(meta.get("delta_residual_mode", "additive")).lower().strip()
    if delta_residual_mode not in {"additive", "relative"}:
        delta_residual_mode = "additive"
    model.delta_residual_mode = delta_residual_mode
    delta_sign_mode = str(meta.get("delta_sign_mode", "signnet_binary")).lower().strip()
    if delta_sign_mode not in {"signnet_binary", "internal"}:
        delta_sign_mode = "signnet_binary"
    model.delta_sign_mode = delta_sign_mode
    model.delta_mag_max = float(max(0.0, meta.get("delta_mag_max", 0.0) or 0.0))
    model.doc_candidate_mode = str(meta.get("doc_candidate_mode", "beta_only"))
    model.temporal_text_enable = bool(int(meta.get("temporal_text_enable", 0)))
    model.temporal_text_dim = int(meta.get("temporal_text_dim", 8))
    model.temporal_text_fuse_lambda = float(meta.get("temporal_text_fuse_lambda", 0.5))
    model.temporal_text_freeze_encoder = bool(int(meta.get("temporal_text_freeze_encoder", 1)))
    model.temporal_text_model_id = str(meta.get("temporal_text_model_id", meta.get("delta_tokenizer_id", "")))
    model.multimodal_arch = str(meta.get("multimodal_arch", "summary_gated"))
    model.multimodal_fuse_lambda = float(meta.get("multimodal_fuse_lambda", 1.0))
    model.route_conf_floor = float(meta.get("route_conf_floor", 0.25))

    reg = torch.load(os.path.join(ckpt_dir, "regressor.pt"), map_location="cpu")
    state = reg.get("model_state", reg)
    _load_state_dict_compatible(model, state)
    return tok, temporal_text_tok, model


def load_trainer_state(ckpt_dir: str, optimizer, scheduler=None):
    path = os.path.join(ckpt_dir, "trainer_state.pt")
    if not os.path.exists(path):
        return None

    st = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(st["optimizer"])
    if scheduler is not None and "scheduler" in st:
        scheduler.load_state_dict(st["scheduler"])

    torch.set_rng_state(st["rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in st:
        torch.cuda.set_rng_state_all(st["cuda_rng_state_all"])

    return st
