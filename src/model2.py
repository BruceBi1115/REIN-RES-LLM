# model.py (UPDATED)

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

END_TOKEN = "<END>"


class NewsConvEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        channels: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 5),
        dropout: float = 0.1,
        meta_dim: int = 16,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.channels = int(max(1, channels))
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes if int(k) > 0)
        if len(self.kernel_sizes) == 0:
            self.kernel_sizes = (2, 3, 5)

        self.convs = nn.ModuleList(
            [nn.Conv1d(self.embed_dim, self.channels, kernel_size=k, bias=True) for k in self.kernel_sizes]
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.age_proj = nn.Linear(1, int(max(1, meta_dim)))
        self.rate_proj = nn.Linear(1, int(max(1, meta_dim)))
        text_dim = self.channels * len(self.kernel_sizes)
        self.item_proj = nn.Linear(text_dim + int(max(1, meta_dim)) * 2, text_dim)
        self.gate_head = nn.Linear(text_dim, 1)
        self.out_dim = int(text_dim)

    def forward(
        self,
        news_emb: torch.Tensor,
        news_attention_mask: torch.Tensor | None = None,
        news_item_mask: torch.Tensor | None = None,
        news_age_hours: torch.Tensor | None = None,
        news_rate_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # news_emb: (B,N,T,H)
        B, N, T, H = news_emb.shape
        flat = news_emb.reshape(B * N, T, H).transpose(1, 2)  # (B*N,H,T)

        if news_attention_mask is None:
            tok_mask = torch.ones((B * N, T), device=news_emb.device, dtype=news_emb.dtype)
        else:
            tok_mask = news_attention_mask.reshape(B * N, T).to(device=news_emb.device, dtype=news_emb.dtype)

        conv_pooled = []
        mask_for_pool = tok_mask.unsqueeze(1)
        for conv, k in zip(self.convs, self.kernel_sizes):
            if T >= k:
                y = self.act(conv(flat))  # (B*N,C,Lk)
                valid = F.max_pool1d(mask_for_pool, kernel_size=k, stride=1) > 0
            else:
                y = flat.new_zeros((flat.size(0), self.channels, 1))
                valid = torch.zeros((flat.size(0), 1, 1), device=flat.device, dtype=torch.bool)

            y = y.masked_fill(~valid, -1e4)
            pooled = y.max(dim=-1).values
            has_valid = valid.squeeze(1).any(dim=-1, keepdim=True)
            pooled = torch.where(has_valid, pooled, torch.zeros_like(pooled))
            conv_pooled.append(pooled)

        text_feat = torch.cat(conv_pooled, dim=-1)  # (B*N, C*len(K))

        dtype = text_feat.dtype
        if news_age_hours is None:
            age = torch.zeros((B * N, 1), device=news_emb.device, dtype=dtype)
        else:
            age = news_age_hours.reshape(B * N, 1).to(device=news_emb.device, dtype=dtype)
        if news_rate_values is None:
            rate = torch.zeros((B * N, 1), device=news_emb.device, dtype=dtype)
        else:
            rate = news_rate_values.reshape(B * N, 1).to(device=news_emb.device, dtype=dtype)

        age_feat = self.act(self.age_proj(age))
        rate_feat = self.act(self.rate_proj(rate))
        item_feat = torch.cat([text_feat, age_feat, rate_feat], dim=-1)
        item_feat = self.dropout(self.act(self.item_proj(item_feat)))  # (B*N,Dn)
        item_gate = self.gate_head(item_feat).squeeze(-1)  # (B*N,)

        if news_item_mask is None:
            item_mask = (tok_mask.sum(dim=-1) > 0).to(dtype=dtype)
        else:
            item_mask = news_item_mask.reshape(B * N).to(device=news_emb.device, dtype=dtype)

        item_feat = item_feat.view(B, N, -1)
        item_gate = item_gate.view(B, N)
        item_mask = item_mask.view(B, N)

        denom = item_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        news_global_feat = (item_feat * item_mask.unsqueeze(-1)).sum(dim=1) / denom
        news_gate_logit = (item_gate * item_mask).sum(dim=1) / denom.squeeze(-1)
        return news_global_feat, news_gate_logit


class TinyNewsTSRegressor(nn.Module):
    """
    Small DELTA branch:
    - TS patches -> lightweight MLP encoder
    - News tokens -> small text encoder -> NewsConvEncoder
    - Fused feature -> delta head / rel head
    """

    def __init__(
        self,
        text_encoder,
        text_model_id: str,
        text_tokenizer_id: str,
        horizon: int,
        patch_dim: int,
        hidden_size: int,
        patch_dropout: float,
        head_dropout: float,
        delta_gate_init_bias: float = 0.0,
        delta_head_init_std: float = 0.01,
        delta_internal_gate: bool = True,
        delta_clip: float = 3.0,
        retrieval_feat_dim: int = 12,
        news_conv_enable: bool = True,
        news_conv_max_items: int = 8,
        news_conv_text_max_tokens: int = 96,
        news_conv_channels: int = 64,
        news_conv_dropout: float = 0.1,
        news_conv_gate_scale: float = 1.0,
        huber_beta: float = 0.5,
        use_horizon_weight: bool = True,
        horizon_weight_end: float = 0.5,
        text_trainable: bool = False,
    ):
        super().__init__()
        self.model_variant = "tiny_news_ts"
        self.text_encoder = text_encoder
        self.text_model_id = str(text_model_id or "")
        self.text_tokenizer_id = str(text_tokenizer_id or "")
        self.text_trainable = bool(text_trainable)

        if not self.text_trainable:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.horizon = int(horizon)
        self.patch_dim = int(patch_dim)
        self.hidden_size = int(hidden_size)
        self.text_hidden_size = int(getattr(self.text_encoder.config, "hidden_size", self.hidden_size))

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

        self.head_drop = nn.Dropout(float(head_dropout))
        self.base_head = nn.Linear(self.hidden_size, self.horizon)

        self.delta_head_drop = nn.Dropout(float(head_dropout))
        self.delta_head = nn.Linear(self.hidden_size, self.horizon)
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=float(delta_head_init_std))
        nn.init.zeros_(self.delta_head.bias)

        self.delta_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.horizon),
        )
        nn.init.zeros_(self.delta_gate[-1].weight)
        nn.init.constant_(self.delta_gate[-1].bias, float(delta_gate_init_bias))

        self.rel_head = nn.Linear(self.hidden_size, 1)
        self.delta_log_scale = nn.Parameter(torch.zeros(1))
        self.delta_internal_gate = bool(delta_internal_gate)
        self.delta_clip = float(delta_clip)

        self.retrieval_feat_dim = int(max(0, retrieval_feat_dim))
        if self.retrieval_feat_dim > 0:
            self.retrieval_proj = nn.Linear(self.retrieval_feat_dim, self.hidden_size)
            self.retrieval_gate_bias = nn.Linear(self.retrieval_feat_dim, self.horizon)
            self.retrieval_rel_bias = nn.Linear(self.retrieval_feat_dim, 1)
            nn.init.zeros_(self.retrieval_proj.weight)
            nn.init.zeros_(self.retrieval_proj.bias)
            nn.init.zeros_(self.retrieval_gate_bias.weight)
            nn.init.zeros_(self.retrieval_gate_bias.bias)
            nn.init.zeros_(self.retrieval_rel_bias.weight)
            nn.init.zeros_(self.retrieval_rel_bias.bias)
        else:
            self.retrieval_proj = None
            self.retrieval_gate_bias = None
            self.retrieval_rel_bias = None

        self.news_conv_enable = bool(news_conv_enable)
        self.news_conv_max_items = int(max(1, news_conv_max_items))
        self.news_conv_text_max_tokens = int(max(1, news_conv_text_max_tokens))
        self.news_conv_channels = int(max(1, news_conv_channels))
        self.news_conv_dropout = float(news_conv_dropout)
        self.news_conv_gate_scale = float(news_conv_gate_scale)
        if self.news_conv_enable:
            self.news_conv_encoder = NewsConvEncoder(
                embed_dim=self.text_hidden_size,
                channels=self.news_conv_channels,
                kernel_sizes=(2, 3, 5),
                dropout=self.news_conv_dropout,
            )
            self.news_conv_feat_proj = nn.Sequential(
                nn.Linear(self.news_conv_encoder.out_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.news_conv_dropout),
            )
            self.news_conv_delta_fuse = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.news_conv_dropout),
                nn.LayerNorm(self.hidden_size),
            )
        else:
            self.news_conv_encoder = None
            self.news_conv_feat_proj = None
            self.news_conv_delta_fuse = None

        self.huber_beta = float(huber_beta)
        self.use_horizon_weight = bool(use_horizon_weight)
        self.horizon_weight_end = float(horizon_weight_end)
        self.patch_mask_p = 0.0

    def _pool_ts(self, ts_feat: torch.Tensor, ts_patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        if ts_patch_mask is None:
            w = torch.ones(ts_feat.size(0), ts_feat.size(1), device=ts_feat.device, dtype=ts_feat.dtype)
        else:
            w = ts_patch_mask.to(device=ts_feat.device, dtype=ts_feat.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (ts_feat * w.unsqueeze(-1)).sum(dim=1) / denom
        return self.ts_ln(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ts_patches: torch.Tensor,
        ts_patch_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
        head_mode: str = "base",
        rel_targets: torch.Tensor | None = None,
        rel_lambda: float = 0.0,
        retrieval_feats: torch.Tensor | None = None,
        retrieval_gate_only: bool = False,
        news_input_ids: torch.Tensor | None = None,
        news_attention_mask: torch.Tensor | None = None,
        news_item_mask: torch.Tensor | None = None,
        news_age_hours: torch.Tensor | None = None,
        news_rate_values: torch.Tensor | None = None,
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

        pooled = self._pool_ts(ts_feat, ts_patch_mask=ts_patch_mask)

        retrieval_weight = None
        retrieval_gate_bias = None
        retrieval_rel_bias = None
        news_gate_logit = None

        if head_mode == "base":
            rel_input = pooled
            pred = self.base_head(self.head_drop(pooled))
        elif head_mode == "delta":
            delta_feat = pooled
            rel_input = delta_feat

            if self.retrieval_feat_dim > 0 and retrieval_feats is not None:
                rf = retrieval_feats.to(device=delta_feat.device, dtype=delta_feat.dtype)
                if rf.dim() == 1:
                    rf = rf.unsqueeze(0)
                if rf.size(-1) < self.retrieval_feat_dim:
                    pad = rf.new_zeros(rf.size(0), self.retrieval_feat_dim - rf.size(-1))
                    rf = torch.cat([rf, pad], dim=-1)
                elif rf.size(-1) > self.retrieval_feat_dim:
                    rf = rf[:, : self.retrieval_feat_dim]

                if rf.size(-1) >= 2:
                    retrieval_weight = (rf[:, :1] * rf[:, 1:2]).clamp(0.0, 1.0)
                elif rf.size(-1) == 1:
                    retrieval_weight = rf[:, :1].clamp(0.0, 1.0)
                else:
                    retrieval_weight = rf.new_zeros((rf.size(0), 1))

                retrieval_gate_bias = self.retrieval_gate_bias(rf) * retrieval_weight
                retrieval_rel_bias = self.retrieval_rel_bias(rf).squeeze(-1) * retrieval_weight.squeeze(-1)
                if not retrieval_gate_only:
                    delta_feat = delta_feat + self.retrieval_proj(rf) * retrieval_weight

            if self.news_conv_enable and news_input_ids is not None and self.news_conv_encoder is not None:
                n_ids = news_input_ids.to(device=delta_feat.device, dtype=torch.long)
                if n_ids.dim() == 2:
                    n_ids = n_ids.unsqueeze(1)
                if n_ids.dim() == 3:
                    bsz, n_items, n_tok = n_ids.shape
                    flat_ids = n_ids.reshape(bsz * n_items, n_tok)
                    if news_attention_mask is None:
                        flat_attn = torch.ones_like(flat_ids, dtype=torch.long, device=delta_feat.device)
                        n_attn = flat_attn.view(bsz, n_items, n_tok)
                    else:
                        n_attn = news_attention_mask.to(device=delta_feat.device, dtype=torch.long)
                        flat_attn = n_attn.reshape(bsz * n_items, n_tok)

                    with torch.set_grad_enabled(self.text_trainable and self.training):
                        txt_out = self.text_encoder(
                            input_ids=flat_ids,
                            attention_mask=flat_attn,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        if hasattr(txt_out, "last_hidden_state") and txt_out.last_hidden_state is not None:
                            txt_hidden = txt_out.last_hidden_state
                        elif hasattr(txt_out, "hidden_states") and txt_out.hidden_states is not None:
                            txt_hidden = txt_out.hidden_states[-1]
                        else:
                            raise RuntimeError("Tiny news text encoder output has no hidden states.")
                        n_emb = txt_hidden.reshape(bsz, n_items, n_tok, -1)
                    conv_dtype = next(self.news_conv_encoder.parameters()).dtype
                    if n_emb.dtype != conv_dtype:
                        n_emb = n_emb.to(dtype=conv_dtype)

                    if news_item_mask is None:
                        n_item_mask = (n_attn.sum(dim=-1) > 0).to(dtype=delta_feat.dtype, device=delta_feat.device)
                    else:
                        n_item_mask = news_item_mask.to(device=delta_feat.device, dtype=delta_feat.dtype)
                    if news_age_hours is None:
                        n_age = torch.zeros((bsz, n_items), device=delta_feat.device, dtype=delta_feat.dtype)
                    else:
                        n_age = news_age_hours.to(device=delta_feat.device, dtype=delta_feat.dtype)
                    if news_rate_values is None:
                        n_rate = torch.zeros((bsz, n_items), device=delta_feat.device, dtype=delta_feat.dtype)
                    else:
                        n_rate = news_rate_values.to(device=delta_feat.device, dtype=delta_feat.dtype)

                    news_global_feat, news_gate_logit = self.news_conv_encoder(
                        news_emb=n_emb,
                        news_attention_mask=n_attn,
                        news_item_mask=n_item_mask,
                        news_age_hours=n_age,
                        news_rate_values=n_rate,
                    )
                    news_aux = self.news_conv_feat_proj(news_global_feat.to(dtype=delta_feat.dtype))
                    delta_feat = self.news_conv_delta_fuse(torch.cat([delta_feat, news_aux], dim=-1))
                    rel_input = delta_feat

            raw_delta = self.delta_head(self.delta_head_drop(delta_feat))
            delta_gate_logits = self.delta_gate(delta_feat)
            if retrieval_gate_bias is not None:
                delta_gate_logits = delta_gate_logits + retrieval_gate_bias
            delta_gate = torch.sigmoid(delta_gate_logits)
            delta_scale = torch.exp(self.delta_log_scale).to(device=raw_delta.device, dtype=raw_delta.dtype).clamp(max=5.0)
            if self.delta_internal_gate:
                pred = raw_delta * delta_gate * delta_scale
                if self.delta_clip > 0:
                    c = torch.tensor(self.delta_clip, device=pred.device, dtype=pred.dtype)
                    pred = c * torch.tanh(pred / c)
            else:
                pred = raw_delta * delta_scale
        else:
            raise ValueError(f"Unknown head_mode={head_mode}")

        pred = pred.to(torch.float32)
        rel_logit = self.rel_head(rel_input).squeeze(-1).to(torch.float32)
        if head_mode == "delta" and retrieval_rel_bias is not None:
            rel_logit = rel_logit + retrieval_rel_bias
        if head_mode == "delta" and news_gate_logit is not None:
            rel_logit = rel_logit + float(self.news_conv_gate_scale) * news_gate_logit.to(torch.float32)

        out = {"pred": pred, "rel_logits": rel_logit}
        if head_mode == "delta":
            rel_gate = torch.ones_like(delta_gate)
            out["delta_gate_mean"] = delta_gate.mean().detach()
            out["delta_scale"] = delta_scale.detach()
            out["delta_rel_gate_mean"] = rel_gate.mean().detach()
            out["delta_internal_gate"] = int(self.delta_internal_gate)
            if retrieval_weight is not None:
                out["retrieval_weight_mean"] = retrieval_weight.mean().detach()

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
            loss_rel = F.binary_cross_entropy_with_logits(rel_logit, rel_targets)
            loss = rel_lambda * loss_rel if loss is None else (loss + rel_lambda * loss_rel)
        if loss is not None:
            out["loss"] = loss
            out["loss_fore"] = loss_fore
            out["loss_rel"] = loss_rel
        return out


class TSForecastRegressor(nn.Module):
    """
    Text + (time-series patches as soft tokens) -> LLM -> (layer-mix) -> attentive pooling over patch tokens -> head -> H values

    Key upgrades vs your original:
    - Patch encoder: MLP + LayerNorm + gating
    - Patch position embedding (by patch index)
    - Layer fusion: learned weighted mix of last K layers (default K=4)
    - Pooling: text-conditioned query cross-attn over patch hidden states (instead of mean pooling)
    - Loss: SmoothL1 (Huber) + optional horizon weighting
    """

    def __init__(
        self,
        lm,
        horizon: int,
        patch_dim: int,
        hidden_size: int,
        patch_dropout: float,
        head_dropout: float,
        head_mlp: bool = False,
        # ---- new knobs (safe defaults) ----
        max_patches: int = 2048,
        pool_queries: int = 4,
        pool_heads: int = 8,
        layer_mix_k: int = 4,
        huber_beta: float = 0.5,
        use_horizon_weight: bool = True,
        horizon_weight_end: float = 0.5,
        delta_gate_init_bias: float = 0.0,
        delta_head_init_std: float = 0.01,
        delta_internal_gate: bool = True,
        delta_clip: float = 3.0,
        delta_news_tail_tokens: int = 160,
        delta_rel_floor: float = 0.05,
        retrieval_feat_dim: int = 12,
        news_conv_enable: bool = False,
        news_conv_max_items: int = 8,
        news_conv_text_max_tokens: int = 96,
        news_conv_channels: int = 64,
        news_conv_dropout: float = 0.1,
        news_conv_gate_scale: float = 1.0,
    ):
        super().__init__()
        self.lm = lm
        self.horizon = int(horizon)
        self.patch_dim = int(patch_dim)
        self.hidden_size = int(hidden_size)

        # ---- patch encoder (MLP + LN + gate) ----
        mid = self.hidden_size * 2
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

        # ---- patch position embedding ----
        self.max_patches = int(max_patches)
        self.patch_pos = nn.Embedding(self.max_patches, self.hidden_size)

        # ---- layer fusion over last K layers ----
        self.layer_mix_k = int(layer_mix_k)
        self.layer_w = nn.Parameter(torch.zeros(self.layer_mix_k))

        # ---- attentive pooling over patch tokens (Q queries attend patch_hid) ----
        self.pool_queries = int(pool_queries)
        self.pool_heads = int(pool_heads)
        self.pool_q = nn.Parameter(torch.randn(1, self.pool_queries, self.hidden_size) * 0.02)
        self.pool_attn = nn.MultiheadAttention(self.hidden_size, num_heads=self.pool_heads, batch_first=True)
        self.pool_ln = nn.LayerNorm(self.hidden_size)
        self.text_ctx_ln = nn.LayerNorm(self.hidden_size)
        self.text2q = nn.Linear(self.hidden_size, self.pool_queries * self.hidden_size)

        # relevance head (kept)
        self.rel_head = nn.Linear(self.hidden_size, 1)

        # ---- heads (base + delta), keep your interface ----
        if head_mlp:
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(float(head_dropout)),
                nn.Linear(self.hidden_size, self.horizon),
            )
            self.base_head = self.head
            self.delta_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(float(head_dropout)),
                nn.Linear(self.hidden_size, self.horizon),
            )
            nn.init.normal_(self.delta_head[-1].weight, mean=0.0, std=float(delta_head_init_std))
            nn.init.zeros_(self.delta_head[-1].bias)
            self.delta_gate = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.horizon),
            )
            nn.init.zeros_(self.delta_gate[-1].weight)
            nn.init.constant_(self.delta_gate[-1].bias, float(delta_gate_init_bias))
            self.head_drop = None
            self.delta_head_drop = None
        else:
            self.head_drop = nn.Dropout(float(head_dropout))
            self.head = nn.Linear(self.hidden_size, self.horizon)
            self.base_head = self.head

            self.delta_head_drop = nn.Dropout(float(head_dropout))
            self.delta_head = nn.Linear(self.hidden_size, self.horizon)
            nn.init.normal_(self.delta_head.weight, mean=0.0, std=float(delta_head_init_std))
            nn.init.zeros_(self.delta_head.bias)
            self.delta_gate = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.horizon),
            )
            nn.init.zeros_(self.delta_gate[-1].weight)
            nn.init.constant_(self.delta_gate[-1].bias, float(delta_gate_init_bias))

        self.delta_log_scale = nn.Parameter(torch.zeros(1))
        self.delta_internal_gate = bool(delta_internal_gate)
        self.delta_clip = float(delta_clip)
        self.delta_news_tail_tokens = int(delta_news_tail_tokens)
        self.delta_rel_floor = float(max(0.0, min(1.0, delta_rel_floor)))
        self.delta_text_ln = nn.LayerNorm(self.hidden_size)
        self.delta_fuse = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(head_dropout)),
            nn.LayerNorm(self.hidden_size),
        )
        self.retrieval_feat_dim = int(max(0, retrieval_feat_dim))
        if self.retrieval_feat_dim > 0:
            self.retrieval_proj = nn.Linear(self.retrieval_feat_dim, self.hidden_size)
            self.retrieval_gate_bias = nn.Linear(self.retrieval_feat_dim, self.horizon)
            self.retrieval_rel_bias = nn.Linear(self.retrieval_feat_dim, 1)
            nn.init.zeros_(self.retrieval_proj.weight)
            nn.init.zeros_(self.retrieval_proj.bias)
            nn.init.zeros_(self.retrieval_gate_bias.weight)
            nn.init.zeros_(self.retrieval_gate_bias.bias)
            nn.init.zeros_(self.retrieval_rel_bias.weight)
            nn.init.zeros_(self.retrieval_rel_bias.bias)
        else:
            self.retrieval_proj = None
            self.retrieval_gate_bias = None
            self.retrieval_rel_bias = None
        self.news_conv_enable = bool(news_conv_enable)
        self.news_conv_max_items = int(max(1, news_conv_max_items))
        self.news_conv_text_max_tokens = int(max(1, news_conv_text_max_tokens))
        self.news_conv_channels = int(max(1, news_conv_channels))
        self.news_conv_dropout = float(news_conv_dropout)
        self.news_conv_gate_scale = float(news_conv_gate_scale)
        if self.news_conv_enable:
            self.news_conv_encoder = NewsConvEncoder(
                embed_dim=self.hidden_size,
                channels=self.news_conv_channels,
                kernel_sizes=(2, 3, 5),
                dropout=self.news_conv_dropout,
            )
            self.news_conv_feat_proj = nn.Sequential(
                nn.Linear(self.news_conv_encoder.out_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.news_conv_dropout),
            )
            self.news_conv_delta_fuse = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.news_conv_dropout),
                nn.LayerNorm(self.hidden_size),
            )
        else:
            self.news_conv_encoder = None
            self.news_conv_feat_proj = None
            self.news_conv_delta_fuse = None

        # ---- loss knobs ----
        self.huber_beta = float(huber_beta)
        self.use_horizon_weight = bool(use_horizon_weight)
        self.horizon_weight_end = float(horizon_weight_end)

        # optional patch masking prob (trainer may set this attribute)
        self.patch_mask_p = 0.0

        # dtype align to LM
        lm_dtype = next(self.lm.parameters()).dtype
        self.to(dtype=lm_dtype)

    def _mix_last_layers(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        hidden_states: tuple of (B,T,H), includes embeddings + each layer.
        returns: (B,T,H) mixed from last K layers
        """
        k = min(self.layer_mix_k, len(hidden_states) - 1)  # exclude embeddings if you want; but HS includes them
        lastk = torch.stack([hidden_states[-i] for i in range(1, k + 1)], dim=0)  # (k,B,T,H)
        w = torch.softmax(self.layer_w[:k], dim=0).to(dtype=lastk.dtype, device=lastk.device)  # (k,)
        mix = (w[:, None, None, None] * lastk).sum(dim=0)  # (B,T,H)
        return mix

    def _attn_pool_patches(
        self,
        mixed_hidden: torch.Tensor,
        text_len: int,
        ts_patch_mask: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        include_text_in_kv: bool = False,
    ) -> torch.Tensor:
        """
        mixed_hidden: (B, T_text + P, H)
        text_len: T_text
        ts_patch_mask: (B, P) 0/1
        returns pooled: (B, H)
        """
        P = ts_patch_mask.size(1)
        patch_hid = mixed_hidden[:, text_len : text_len + P, :]  # (B,P,H)
        patch_key_padding = (ts_patch_mask == 0)  # (B,P) True indicates pad

        text_hid = mixed_hidden[:, :text_len, :]  # (B,T,H)
        if text_hid.size(1) == 0:
            text_ctx = patch_hid.mean(dim=1)
        elif text_mask is None:
            text_ctx = text_hid.mean(dim=1)
        else:
            tm = text_mask[:, :text_len].to(dtype=text_hid.dtype, device=text_hid.device).unsqueeze(-1)  # (B,T,1)
            denom = tm.sum(dim=1).clamp_min(1.0)
            text_ctx = (text_hid * tm).sum(dim=1) / denom

        text_ctx = self.text_ctx_ln(text_ctx)
        q_delta = self.text2q(text_ctx).view(text_ctx.size(0), self.pool_queries, self.hidden_size)
        q = self.pool_q.expand(patch_hid.size(0), -1, -1) + q_delta  # (B,Q,H)
        if include_text_in_kv:
            kv_hid = torch.cat([text_hid, patch_hid], dim=1)  # (B,T+P,H)
            if text_mask is None:
                text_key_padding = torch.zeros(
                    (patch_hid.size(0), text_len), device=patch_hid.device, dtype=torch.bool
                )
            else:
                text_key_padding = (text_mask[:, :text_len] == 0)
            key_padding = torch.cat([text_key_padding, patch_key_padding], dim=1)
        else:
            kv_hid = patch_hid
            key_padding = patch_key_padding

        attn_out, _ = self.pool_attn(q, kv_hid, kv_hid, key_padding_mask=key_padding)  # (B,Q,H)
        pooled = attn_out.mean(dim=1)  # (B,H)
        return self.pool_ln(pooled)

    def _tail_text_pool(
        self,
        mixed_hidden: torch.Tensor,
        text_len: int,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pool the tail part of text tokens where news usually appears.
        returns: (B,H)
        """
        text_hid = mixed_hidden[:, :text_len, :]  # (B,T,H)
        B, T, H = text_hid.shape
        if T == 0:
            return text_hid.new_zeros((B, H))

        if text_mask is None:
            vm = torch.ones(B, T, device=text_hid.device, dtype=text_hid.dtype)
        else:
            vm = text_mask[:, :text_len].to(device=text_hid.device, dtype=text_hid.dtype)

        tail_k = max(1, min(int(self.delta_news_tail_tokens), T))
        idx = torch.arange(T, device=text_hid.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        valid_len = vm.sum(dim=1).to(dtype=torch.long)
        start = (valid_len - tail_k).clamp_min(0).unsqueeze(1)
        tail_mask = (idx >= start).to(dtype=vm.dtype) * vm

        w = tail_mask.unsqueeze(-1)  # (B,T,1)
        denom = w.sum(dim=1).clamp_min(1.0)
        pooled = (text_hid * w).sum(dim=1) / denom
        return self.delta_text_ln(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ts_patches: torch.Tensor,
        ts_patch_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
        head_mode: str = "base",
        rel_targets: torch.Tensor | None = None,
        rel_lambda: float = 0.0,
        retrieval_feats: torch.Tensor | None = None,
        retrieval_gate_only: bool = False,
        news_input_ids: torch.Tensor | None = None,
        news_attention_mask: torch.Tensor | None = None,
        news_item_mask: torch.Tensor | None = None,
        news_age_hours: torch.Tensor | None = None,
        news_rate_values: torch.Tensor | None = None,
    ):
        # --- text token embeddings ---
        tok_emb = self.lm.get_input_embeddings()(input_ids)  # (B, T, H)
        tok_dtype = tok_emb.dtype

        # --- patch encoder (dtype align) ---
        proj_dtype = next(self.patch_proj.parameters()).dtype
        if ts_patches.dtype != proj_dtype:
            ts_patches = ts_patches.to(dtype=proj_dtype)

        patch_emb = self.patch_proj(ts_patches)  # (B,P,H)
        patch_emb = patch_emb * self.patch_gate(patch_emb)
        patch_emb = self.patch_drop(patch_emb)

        # patch position embedding
        P = patch_emb.size(1)
        pos_ids = torch.arange(P, device=patch_emb.device).clamp_max(self.max_patches - 1)
        patch_emb = patch_emb + self.patch_pos(pos_ids)[None, :, :].to(dtype=patch_emb.dtype)

        # align to token embedding dtype for concat
        if patch_emb.dtype != tok_dtype:
            patch_emb = patch_emb.to(dtype=tok_dtype)

        # optional random patch masking (trainer can set self.patch_mask_p)
        if self.training and float(getattr(self, "patch_mask_p", 0.0)) > 0:
            keep = (torch.rand(patch_emb.size(0), patch_emb.size(1), 1, device=patch_emb.device) > self.patch_mask_p).to(
                patch_emb.dtype
            )
            patch_emb = patch_emb * keep

        # concat as soft tokens
        inputs_embeds = torch.cat([tok_emb, patch_emb], dim=1)  # (B, T+P, H)
        attn = torch.cat([attention_mask, ts_patch_mask], dim=1)  # (B, T+P)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        # layer-mix + attentive pooling on patch part
        mixed = self._mix_last_layers(outputs.hidden_states)  # (B,T+P,H)
        text_len = attention_mask.size(1)

        pooled = None  # 延迟计算
        retrieval_weight = None
        retrieval_gate_bias = None
        retrieval_rel_bias = None
        news_gate_logit = None

        if head_mode == "base":
            pooled = self._attn_pool_patches(mixed, text_len=text_len, 
                                            ts_patch_mask=ts_patch_mask, 
                                            text_mask=attention_mask)
            rel_input = pooled
            pred = self.base_head(self.head_drop(pooled)) if isinstance(self.base_head, nn.Linear) \
                else self.base_head(pooled)

        elif head_mode == "delta":
            pooled = self._attn_pool_patches(mixed, text_len=text_len,
                                            ts_patch_mask=ts_patch_mask,
                                            text_mask=attention_mask,
                                            include_text_in_kv=True)
            text_tail = self._tail_text_pool(mixed, text_len=text_len, text_mask=attention_mask)
            delta_feat = torch.cat([pooled, text_tail, pooled * text_tail, pooled - text_tail], dim=-1)
            delta_feat = self.delta_fuse(delta_feat)
            rel_input = delta_feat

            if self.retrieval_feat_dim > 0 and retrieval_feats is not None:
                rf = retrieval_feats.to(device=delta_feat.device, dtype=delta_feat.dtype)
                if rf.dim() == 1:
                    rf = rf.unsqueeze(0)
                if rf.size(-1) < self.retrieval_feat_dim:
                    pad = rf.new_zeros(rf.size(0), self.retrieval_feat_dim - rf.size(-1))
                    rf = torch.cat([rf, pad], dim=-1)
                elif rf.size(-1) > self.retrieval_feat_dim:
                    rf = rf[:, : self.retrieval_feat_dim]

                if rf.size(-1) >= 2:
                    retrieval_weight = (rf[:, :1] * rf[:, 1:2]).clamp(0.0, 1.0)
                elif rf.size(-1) == 1:
                    retrieval_weight = rf[:, :1].clamp(0.0, 1.0)
                else:
                    retrieval_weight = rf.new_zeros((rf.size(0), 1))

                retrieval_gate_bias = self.retrieval_gate_bias(rf) * retrieval_weight
                retrieval_rel_bias = self.retrieval_rel_bias(rf).squeeze(-1) * retrieval_weight.squeeze(-1)
                if not retrieval_gate_only:
                    delta_feat = delta_feat + self.retrieval_proj(rf) * retrieval_weight

            if self.news_conv_enable and news_input_ids is not None and self.news_conv_encoder is not None:
                n_ids = news_input_ids.to(device=delta_feat.device, dtype=torch.long)
                if n_ids.dim() == 2:
                    n_ids = n_ids.unsqueeze(1)
                if n_ids.dim() == 3:
                    Bn, Nn, Tn = n_ids.shape
                    emb_layer = self.lm.get_input_embeddings()
                    n_emb = emb_layer(n_ids.reshape(Bn * Nn, Tn)).reshape(Bn, Nn, Tn, -1)

                    if news_attention_mask is None:
                        n_attn = torch.ones_like(n_ids, dtype=attention_mask.dtype, device=delta_feat.device)
                    else:
                        n_attn = news_attention_mask.to(device=delta_feat.device)
                    if news_item_mask is None:
                        n_item_mask = (n_attn.sum(dim=-1) > 0).to(dtype=delta_feat.dtype, device=delta_feat.device)
                    else:
                        n_item_mask = news_item_mask.to(device=delta_feat.device, dtype=delta_feat.dtype)
                    if news_age_hours is None:
                        n_age = torch.zeros((Bn, Nn), device=delta_feat.device, dtype=delta_feat.dtype)
                    else:
                        n_age = news_age_hours.to(device=delta_feat.device, dtype=delta_feat.dtype)
                    if news_rate_values is None:
                        n_rate = torch.zeros((Bn, Nn), device=delta_feat.device, dtype=delta_feat.dtype)
                    else:
                        n_rate = news_rate_values.to(device=delta_feat.device, dtype=delta_feat.dtype)

                    news_global_feat, news_gate_logit = self.news_conv_encoder(
                        news_emb=n_emb,
                        news_attention_mask=n_attn,
                        news_item_mask=n_item_mask,
                        news_age_hours=n_age,
                        news_rate_values=n_rate,
                    )
                    news_aux = self.news_conv_feat_proj(news_global_feat.to(dtype=delta_feat.dtype))
                    delta_feat = self.news_conv_delta_fuse(torch.cat([delta_feat, news_aux], dim=-1))

            if isinstance(self.delta_head, nn.Linear):
                raw_delta = self.delta_head(self.delta_head_drop(delta_feat))
            else:
                raw_delta = self.delta_head(delta_feat)

            delta_gate_logits = self.delta_gate(delta_feat)
            if retrieval_gate_bias is not None:
                delta_gate_logits = delta_gate_logits + retrieval_gate_bias
            delta_gate = torch.sigmoid(delta_gate_logits)
            delta_scale = torch.exp(self.delta_log_scale).to(device=raw_delta.device, dtype=raw_delta.dtype).clamp(max=5.0)
            if self.delta_internal_gate:
                pred = raw_delta * delta_gate * delta_scale
                if self.delta_clip > 0:
                    c = torch.tensor(self.delta_clip, device=pred.device, dtype=pred.dtype)
                    pred = c * torch.tanh(pred / c)
            else:
                # quick ablation: bypass internal gate/rel-gate/clip, keep only optional global scale
                pred = raw_delta * delta_scale
        else:
            raise ValueError(f"Unknown head_mode={head_mode}")

        # relevance
        pred = pred.to(torch.float32)
        rel_logit = self.rel_head(rel_input).squeeze(-1)  # (B,)
        if head_mode == "delta" and retrieval_rel_bias is not None:
            rel_logit = rel_logit + retrieval_rel_bias
        if head_mode == "delta" and news_gate_logit is not None:
            rel_logit = rel_logit + float(self.news_conv_gate_scale) * news_gate_logit
        pred = pred.to(torch.float32)
        rel_logit = rel_logit.to(torch.float32)

        out = {"pred": pred, "rel_logits": rel_logit}
        if head_mode == "delta":
            rel_gate = torch.ones_like(delta_gate)
            out["delta_gate_mean"] = delta_gate.mean().detach()
            out["delta_scale"] = delta_scale.detach()
            out["delta_rel_gate_mean"] = rel_gate.mean().detach()
            out["delta_internal_gate"] = int(self.delta_internal_gate)
            if retrieval_weight is not None:
                out["retrieval_weight_mean"] = retrieval_weight.mean().detach()

        # losses
        loss = None
        loss_fore = None
        loss_rel = None

        if targets is not None:
            if targets.dtype != pred.dtype:
                targets = targets.to(dtype=pred.dtype)

            # SmoothL1 (Huber) with optional horizon weighting
            per = F.smooth_l1_loss(pred, targets, beta=self.huber_beta, reduction="none")  # (B,H)
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
            loss_rel = F.binary_cross_entropy_with_logits(rel_logit, rel_targets)
            loss = rel_lambda * loss_rel if loss is None else (loss + rel_lambda * loss_rel)

        if loss is not None:
            out["loss"] = loss
            out["loss_fore"] = loss_fore
            out["loss_rel"] = loss_rel

        return out


def load_llama_lora(
    base_model: str,
    tokenizer_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = False,
    max_seq_len: int = 1536,
    device=None,
    horizon: int = 48,
    patch_dim: int = 4,
    patch_dropout: float = 0.0,
    head_dropout: float = 0.0,
    head_mlp: bool = False,
    # new knobs
    max_patches: int = 2048,
    pool_queries: int = 4,
    pool_heads: int = 8,
    layer_mix_k: int = 4,
    huber_beta: float = 0.5,
    use_horizon_weight: bool = True,
    horizon_weight_end: float = 0.5,
    delta_gate_init_bias: float = 0.0,
    delta_head_init_std: float = 0.01,
    delta_internal_gate: bool = True,
    delta_clip: float = 3.0,
    delta_news_tail_tokens: int = 160,
    delta_rel_floor: float = 0.05,
    retrieval_feat_dim: int = 12,
    news_conv_enable: bool = False,
    news_conv_max_items: int = 8,
    news_conv_text_max_tokens: int = 96,
    news_conv_channels: int = 64,
    news_conv_dropout: float = 0.1,
    news_conv_gate_scale: float = 1.0,
    delta_model_variant: str = "llama",
    tiny_news_model_preset: str = "custom",
    tiny_news_model: str = "",
    tiny_news_tokenizer: str = "",
    tiny_news_hidden_size: int = 256,
    tiny_news_text_trainable: bool = False,
    tiny_news_loader: str = "auto",
):
    def _resolve_tiny_news_spec(preset_raw: str, model_raw: str, tok_raw: str):
        preset = str(preset_raw or "custom").lower().strip()
        preset_map = {
            "custom": ("", ""),
            "distilbert": ("distilbert-base-uncased", "distilbert-base-uncased"),
            "gpt2": ("gpt2", "gpt2"),
            "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        }
        m_p, t_p = preset_map.get(preset, ("", ""))
        model_id = str(model_raw or "").strip() or m_p
        tok_id = str(tok_raw or "").strip() or t_p or model_id
        if not model_id:
            model_id = "distilbert-base-uncased"
            tok_id = tok_id or model_id
        return model_id, tok_id

    def _load_tiny_text_encoder(model_id: str, loader_raw: str):
        loader = str(loader_raw or "auto").lower().strip()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if loader == "encoder":
            return AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        if loader == "causal_lm":
            return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        try:
            return AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        except Exception:
            return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

    variant = str(delta_model_variant or "llama").lower().strip()
    if variant == "tiny_news_ts":
        text_model_id, tok_id = _resolve_tiny_news_spec(
            tiny_news_model_preset,
            tiny_news_model or tokenizer_id or base_model,
            tiny_news_tokenizer,
        )
        tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token_id = tok.eos_token_id

        text_encoder = _load_tiny_text_encoder(text_model_id, tiny_news_loader)
        if gradient_checkpointing and hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()

        hidden_cfg = int(getattr(text_encoder.config, "hidden_size", 0) or 0)
        hidden_size = int(tiny_news_hidden_size) if int(tiny_news_hidden_size) > 0 else hidden_cfg
        if hidden_size <= 0:
            hidden_size = max(128, hidden_cfg if hidden_cfg > 0 else 256)

        model = TinyNewsTSRegressor(
            text_encoder=text_encoder,
            text_model_id=text_model_id,
            text_tokenizer_id=tok_id,
            horizon=int(horizon),
            patch_dim=int(patch_dim),
            hidden_size=hidden_size,
            patch_dropout=float(patch_dropout),
            head_dropout=float(head_dropout),
            delta_gate_init_bias=float(delta_gate_init_bias),
            delta_head_init_std=float(delta_head_init_std),
            delta_internal_gate=bool(delta_internal_gate),
            delta_clip=float(delta_clip),
            retrieval_feat_dim=int(retrieval_feat_dim),
            news_conv_enable=bool(news_conv_enable),
            news_conv_max_items=int(news_conv_max_items),
            news_conv_text_max_tokens=int(news_conv_text_max_tokens),
            news_conv_channels=int(news_conv_channels),
            news_conv_dropout=float(news_conv_dropout),
            news_conv_gate_scale=float(news_conv_gate_scale),
            huber_beta=float(huber_beta),
            use_horizon_weight=bool(use_horizon_weight),
            horizon_weight_end=float(horizon_weight_end),
            text_trainable=bool(tiny_news_text_trainable),
        )
        return tok, model

    tok = AutoTokenizer.from_pretrained(tokenizer_id or base_model, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    special = tok.special_tokens_map.get("additional_special_tokens", [])
    if END_TOKEN not in special:
        tok.add_special_tokens({"additional_special_tokens": [END_TOKEN]})

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device is None else None,
        load_in_4bit=load_in_4bit,
    )

    base.resize_token_embeddings(len(tok))

    if gradient_checkpointing:
        base.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lm = get_peft_model(base, peft_cfg)
    lm.print_trainable_parameters()

    hidden_size = lm.config.hidden_size

    model = TSForecastRegressor(
        lm=lm,
        horizon=int(horizon),
        patch_dim=int(patch_dim),
        hidden_size=int(hidden_size),
        patch_dropout=float(patch_dropout),
        head_dropout=float(head_dropout),
        head_mlp=bool(head_mlp),
        max_patches=int(max_patches),
        pool_queries=int(pool_queries),
        pool_heads=int(pool_heads),
        layer_mix_k=int(layer_mix_k),
        huber_beta=float(huber_beta),
        use_horizon_weight=bool(use_horizon_weight),
        horizon_weight_end=float(horizon_weight_end),
        delta_gate_init_bias=float(delta_gate_init_bias),
        delta_head_init_std=float(delta_head_init_std),
        delta_internal_gate=bool(delta_internal_gate),
        delta_clip=float(delta_clip),
        delta_news_tail_tokens=int(delta_news_tail_tokens),
        delta_rel_floor=float(delta_rel_floor),
        retrieval_feat_dim=int(retrieval_feat_dim),
        news_conv_enable=bool(news_conv_enable),
        news_conv_max_items=int(news_conv_max_items),
        news_conv_text_max_tokens=int(news_conv_text_max_tokens),
        news_conv_channels=int(news_conv_channels),
        news_conv_dropout=float(news_conv_dropout),
        news_conv_gate_scale=float(news_conv_gate_scale),
    )

    return tok, model


def save_checkpoint(
    ckpt_dir: str,
    tok,
    model,  # TSForecastRegressor
    base_model_id: str,
    tokenizer_id: str | None,
    lora_cfg: dict,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    global_step: int | None = None,
    extra_meta: dict | None = None,
):
    os.makedirs(ckpt_dir, exist_ok=True)

    variant = str(getattr(model, "model_variant", "llama")).lower().strip()

    tok_dir = os.path.join(ckpt_dir, "tokenizer")
    tok.save_pretrained(tok_dir)

    if variant == "tiny_news_ts":
        reg_path = os.path.join(ckpt_dir, "regressor.pt")
        torch.save(
            {
                "model_variant": variant,
                "model_state": model.state_dict(),
                "horizon": int(model.horizon),
                "patch_dim": int(model.patch_dim),
                "hidden_size": int(model.hidden_size),
                "retrieval_feat_dim": int(getattr(model, "retrieval_feat_dim", 0)),
                "news_conv_enable": int(bool(getattr(model, "news_conv_enable", False))),
                "news_conv_max_items": int(getattr(model, "news_conv_max_items", 8)),
                "news_conv_text_max_tokens": int(getattr(model, "news_conv_text_max_tokens", 96)),
                "news_conv_channels": int(getattr(model, "news_conv_channels", 64)),
                "news_conv_dropout": float(getattr(model, "news_conv_dropout", 0.1)),
                "news_conv_gate_scale": float(getattr(model, "news_conv_gate_scale", 1.0)),
                "delta_internal_gate": int(bool(getattr(model, "delta_internal_gate", True))),
                "delta_clip": float(getattr(model, "delta_clip", 3.0)),
                "huber_beta": float(getattr(model, "huber_beta", 0.5)),
                "use_horizon_weight": bool(getattr(model, "use_horizon_weight", True)),
                "horizon_weight_end": float(getattr(model, "horizon_weight_end", 0.5)),
                "text_model_id": str(getattr(model, "text_model_id", "")),
                "text_tokenizer_id": str(getattr(model, "text_tokenizer_id", "")),
                "text_trainable": int(bool(getattr(model, "text_trainable", False))),
            },
            reg_path,
        )

        meta = {
            "model_variant": variant,
            "base_model_id": base_model_id,
            "tokenizer_id": tokenizer_id or base_model_id,
            "lora_cfg": lora_cfg,
            "horizon": int(model.horizon),
            "patch_dim": int(model.patch_dim),
            "hidden_size": int(model.hidden_size),
            "huber_beta": float(getattr(model, "huber_beta", 0.5)),
            "use_horizon_weight": bool(getattr(model, "use_horizon_weight", True)),
            "horizon_weight_end": float(getattr(model, "horizon_weight_end", 0.5)),
            "delta_internal_gate": int(bool(getattr(model, "delta_internal_gate", True))),
            "delta_clip": float(getattr(model, "delta_clip", 3.0)),
            "retrieval_feat_dim": int(getattr(model, "retrieval_feat_dim", 0)),
            "news_conv_enable": int(bool(getattr(model, "news_conv_enable", False))),
            "news_conv_max_items": int(getattr(model, "news_conv_max_items", 8)),
            "news_conv_text_max_tokens": int(getattr(model, "news_conv_text_max_tokens", 96)),
            "news_conv_channels": int(getattr(model, "news_conv_channels", 64)),
            "news_conv_dropout": float(getattr(model, "news_conv_dropout", 0.1)),
            "news_conv_gate_scale": float(getattr(model, "news_conv_gate_scale", 1.0)),
            "tiny_news_model_id": str(getattr(model, "text_model_id", "")),
            "tiny_news_tokenizer_id": str(getattr(model, "text_tokenizer_id", "")),
            "tiny_news_text_trainable": int(bool(getattr(model, "text_trainable", False))),
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
        return

    adapter_dir = os.path.join(ckpt_dir, "adapter")
    model.lm.save_pretrained(adapter_dir, safe_serialization=True)

    reg_path = os.path.join(ckpt_dir, "regressor.pt")
    torch.save(
        {
            # patch side
            "patch_proj": model.patch_proj.state_dict(),
            "patch_gate": model.patch_gate.state_dict(),
            "patch_pos": model.patch_pos.state_dict(),
            # pooling + layer mix
            "pool_q": model.pool_q.detach().cpu(),
            "pool_attn": model.pool_attn.state_dict(),
            "pool_ln": model.pool_ln.state_dict(),
            "text_ctx_ln": model.text_ctx_ln.state_dict(),
            "text2q": model.text2q.state_dict(),
            "layer_w": model.layer_w.detach().cpu(),
            "layer_mix_k": model.layer_mix_k,
            "max_patches": model.max_patches,
            "pool_queries": model.pool_queries,
            "pool_heads": model.pool_heads,
            # heads
            "head": model.base_head.state_dict(),
            "delta_head": model.delta_head.state_dict(),
            "delta_gate": model.delta_gate.state_dict(),
            "delta_log_scale": model.delta_log_scale.detach().cpu(),
            "delta_fuse": model.delta_fuse.state_dict(),
            "delta_text_ln": model.delta_text_ln.state_dict(),
            "rel_head": model.rel_head.state_dict(),
            "retrieval_feat_dim": int(getattr(model, "retrieval_feat_dim", 0)),
            "retrieval_proj": model.retrieval_proj.state_dict() if model.retrieval_proj is not None else None,
            "retrieval_gate_bias": model.retrieval_gate_bias.state_dict() if model.retrieval_gate_bias is not None else None,
            "retrieval_rel_bias": model.retrieval_rel_bias.state_dict() if model.retrieval_rel_bias is not None else None,
            "news_conv_enable": int(bool(getattr(model, "news_conv_enable", False))),
            "news_conv_max_items": int(getattr(model, "news_conv_max_items", 8)),
            "news_conv_text_max_tokens": int(getattr(model, "news_conv_text_max_tokens", 96)),
            "news_conv_channels": int(getattr(model, "news_conv_channels", 64)),
            "news_conv_dropout": float(getattr(model, "news_conv_dropout", 0.1)),
            "news_conv_gate_scale": float(getattr(model, "news_conv_gate_scale", 1.0)),
            "news_conv_encoder": model.news_conv_encoder.state_dict() if model.news_conv_encoder is not None else None,
            "news_conv_feat_proj": model.news_conv_feat_proj.state_dict() if model.news_conv_feat_proj is not None else None,
            "news_conv_delta_fuse": model.news_conv_delta_fuse.state_dict() if model.news_conv_delta_fuse is not None else None,
            # meta
            "horizon": model.horizon,
            "patch_dim": model.patch_dim,
            "hidden_size": model.hidden_size,
            "huber_beta": getattr(model, "huber_beta", 0.5),
            "use_horizon_weight": getattr(model, "use_horizon_weight", True),
            "horizon_weight_end": getattr(model, "horizon_weight_end", 0.5),
            "delta_internal_gate": int(bool(getattr(model, "delta_internal_gate", True))),
            "delta_clip": getattr(model, "delta_clip", 3.0),
            "delta_news_tail_tokens": getattr(model, "delta_news_tail_tokens", 160),
            "delta_rel_floor": getattr(model, "delta_rel_floor", 0.05),
        },
        reg_path,
    )

    meta = {
        "model_variant": "llama",
        "base_model_id": base_model_id,
        "tokenizer_id": tokenizer_id or base_model_id,
        "lora_cfg": lora_cfg,
        "horizon": int(model.horizon),
        "patch_dim": int(model.patch_dim),
        "hidden_size": int(model.hidden_size),
        "max_patches": int(getattr(model, "max_patches", 2048)),
        "pool_queries": int(getattr(model, "pool_queries", 4)),
        "pool_heads": int(getattr(model, "pool_heads", 8)),
        "layer_mix_k": int(getattr(model, "layer_mix_k", 4)),
        "huber_beta": float(getattr(model, "huber_beta", 0.5)),
        "use_horizon_weight": bool(getattr(model, "use_horizon_weight", True)),
        "horizon_weight_end": float(getattr(model, "horizon_weight_end", 0.5)),
        "delta_internal_gate": int(bool(getattr(model, "delta_internal_gate", True))),
        "delta_clip": float(getattr(model, "delta_clip", 3.0)),
        "delta_news_tail_tokens": int(getattr(model, "delta_news_tail_tokens", 160)),
        "delta_rel_floor": float(getattr(model, "delta_rel_floor", 0.05)),
        "retrieval_feat_dim": int(getattr(model, "retrieval_feat_dim", 0)),
        "news_conv_enable": int(bool(getattr(model, "news_conv_enable", False))),
        "news_conv_max_items": int(getattr(model, "news_conv_max_items", 8)),
        "news_conv_text_max_tokens": int(getattr(model, "news_conv_text_max_tokens", 96)),
        "news_conv_channels": int(getattr(model, "news_conv_channels", 64)),
        "news_conv_dropout": float(getattr(model, "news_conv_dropout", 0.1)),
        "news_conv_gate_scale": float(getattr(model, "news_conv_gate_scale", 1.0)),
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


def load_checkpoint(
    ckpt_dir: str,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = False,
    device_map=None,
    is_trainable: bool = False,
    head_mlp: bool = False,
    hd: float = 0.0,
    pd: float = 0.0,
):
    with open(os.path.join(ckpt_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_variant = str(meta.get("model_variant", "llama")).lower().strip()
    if model_variant == "tiny_news_ts":
        tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "tokenizer"), use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token_id = tok.eos_token_id

        tiny_model_id = str(meta.get("tiny_news_model_id", meta.get("base_model_id", ""))).strip()
        tiny_tok_id = str(meta.get("tiny_news_tokenizer_id", tiny_model_id)).strip()
        tiny_text_trainable = bool(int(meta.get("tiny_news_text_trainable", 0)))
        horizon = int(meta["horizon"])
        patch_dim = int(meta["patch_dim"])
        hidden_size = int(meta.get("hidden_size", 256))
        retrieval_feat_dim = int(meta.get("retrieval_feat_dim", 12))
        news_conv_enable = bool(int(meta.get("news_conv_enable", 0)))
        news_conv_max_items = int(meta.get("news_conv_max_items", 8))
        news_conv_text_max_tokens = int(meta.get("news_conv_text_max_tokens", 96))
        news_conv_channels = int(meta.get("news_conv_channels", 64))
        news_conv_dropout = float(meta.get("news_conv_dropout", 0.1))
        news_conv_gate_scale = float(meta.get("news_conv_gate_scale", 1.0))
        huber_beta = float(meta.get("huber_beta", 0.5))
        use_horizon_weight = bool(meta.get("use_horizon_weight", True))
        horizon_weight_end = float(meta.get("horizon_weight_end", 0.5))
        delta_internal_gate = bool(int(meta.get("delta_internal_gate", 1)))
        delta_clip = float(meta.get("delta_clip", 3.0))

        text_encoder = AutoModel.from_pretrained(
            tiny_model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        if gradient_checkpointing and hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()

        model = TinyNewsTSRegressor(
            text_encoder=text_encoder,
            text_model_id=tiny_model_id,
            text_tokenizer_id=tiny_tok_id,
            horizon=horizon,
            patch_dim=patch_dim,
            hidden_size=hidden_size,
            patch_dropout=pd,
            head_dropout=hd,
            delta_internal_gate=delta_internal_gate,
            delta_clip=delta_clip,
            retrieval_feat_dim=retrieval_feat_dim,
            news_conv_enable=news_conv_enable,
            news_conv_max_items=news_conv_max_items,
            news_conv_text_max_tokens=news_conv_text_max_tokens,
            news_conv_channels=news_conv_channels,
            news_conv_dropout=news_conv_dropout,
            news_conv_gate_scale=news_conv_gate_scale,
            huber_beta=huber_beta,
            use_horizon_weight=use_horizon_weight,
            horizon_weight_end=horizon_weight_end,
            text_trainable=tiny_text_trainable and is_trainable,
        )
        reg = torch.load(os.path.join(ckpt_dir, "regressor.pt"), map_location="cpu")
        state = reg.get("model_state", reg)
        model.load_state_dict(state, strict=False)
        return tok, model

    base_model_id = meta["base_model_id"]
    horizon = int(meta["horizon"])
    patch_dim = int(meta["patch_dim"])
    max_patches = int(meta.get("max_patches", 2048))
    pool_queries = int(meta.get("pool_queries", 4))
    pool_heads = int(meta.get("pool_heads", 8))
    layer_mix_k = int(meta.get("layer_mix_k", 4))
    huber_beta = float(meta.get("huber_beta", 0.5))
    use_horizon_weight = bool(meta.get("use_horizon_weight", True))
    horizon_weight_end = float(meta.get("horizon_weight_end", 0.5))
    delta_internal_gate = bool(int(meta.get("delta_internal_gate", 1)))
    delta_clip = float(meta.get("delta_clip", 3.0))
    delta_news_tail_tokens = int(meta.get("delta_news_tail_tokens", 160))
    delta_rel_floor = float(meta.get("delta_rel_floor", 0.05))
    retrieval_feat_dim = int(meta.get("retrieval_feat_dim", 12))
    news_conv_enable = bool(int(meta.get("news_conv_enable", 0)))
    news_conv_max_items = int(meta.get("news_conv_max_items", 8))
    news_conv_text_max_tokens = int(meta.get("news_conv_text_max_tokens", 96))
    news_conv_channels = int(meta.get("news_conv_channels", 64))
    news_conv_dropout = float(meta.get("news_conv_dropout", 0.1))
    news_conv_gate_scale = float(meta.get("news_conv_gate_scale", 1.0))

    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "tokenizer"), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device_map is None else device_map,
        load_in_4bit=load_in_4bit,
    )
    base.resize_token_embeddings(len(tok))

    if gradient_checkpointing:
        base.gradient_checkpointing_enable()

    adapter_dir = os.path.join(ckpt_dir, "adapter")
    lm = PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)

    hidden_size = lm.config.hidden_size
    model = TSForecastRegressor(
        lm=lm,
        horizon=horizon,
        patch_dim=patch_dim,
        hidden_size=hidden_size,
        patch_dropout=pd,
        head_dropout=hd,
        head_mlp=head_mlp,
        max_patches=max_patches,
        pool_queries=pool_queries,
        pool_heads=pool_heads,
        layer_mix_k=layer_mix_k,
        huber_beta=huber_beta,
        use_horizon_weight=use_horizon_weight,
        horizon_weight_end=horizon_weight_end,
        delta_internal_gate=delta_internal_gate,
        delta_clip=delta_clip,
        delta_news_tail_tokens=delta_news_tail_tokens,
        delta_rel_floor=delta_rel_floor,
        retrieval_feat_dim=retrieval_feat_dim,
        news_conv_enable=news_conv_enable,
        news_conv_max_items=news_conv_max_items,
        news_conv_text_max_tokens=news_conv_text_max_tokens,
        news_conv_channels=news_conv_channels,
        news_conv_dropout=news_conv_dropout,
        news_conv_gate_scale=news_conv_gate_scale,
    )

    reg = torch.load(os.path.join(ckpt_dir, "regressor.pt"), map_location="cpu")

    # patch encoder (backward compatible)
    if "patch_proj" in reg:
        model.patch_proj.load_state_dict(reg["patch_proj"], strict=True)
    if "patch_gate" in reg:
        model.patch_gate.load_state_dict(reg["patch_gate"], strict=True)
    if "patch_pos" in reg:
        model.patch_pos.load_state_dict(reg["patch_pos"], strict=False)

    # pooling + layer mix (backward compatible)
    if "pool_attn" in reg:
        model.pool_attn.load_state_dict(reg["pool_attn"], strict=True)
    if "pool_ln" in reg:
        model.pool_ln.load_state_dict(reg["pool_ln"], strict=True)
    if "text_ctx_ln" in reg:
        model.text_ctx_ln.load_state_dict(reg["text_ctx_ln"], strict=True)
    if "text2q" in reg:
        model.text2q.load_state_dict(reg["text2q"], strict=True)
    if "pool_q" in reg:
        with torch.no_grad():
            pq = reg["pool_q"]
            if pq.shape == model.pool_q.shape:
                model.pool_q.copy_(pq)
    if "layer_w" in reg:
        with torch.no_grad():
            lw = reg["layer_w"]
            if lw.numel() >= model.layer_w.numel():
                model.layer_w.copy_(lw[: model.layer_w.numel()])

    # heads (backward compatible)
    if "head" in reg:
        model.head.load_state_dict(reg["head"], strict=True)

    if "delta_head" in reg:
        model.delta_head.load_state_dict(reg["delta_head"], strict=True)
    else:
        for p in model.delta_head.parameters():
            nn.init.zeros_(p)

    if "delta_gate" in reg:
        model.delta_gate.load_state_dict(reg["delta_gate"], strict=True)
    if "delta_log_scale" in reg:
        with torch.no_grad():
            dls = reg["delta_log_scale"]
            if dls.shape == model.delta_log_scale.shape:
                model.delta_log_scale.copy_(dls)
    if "delta_fuse" in reg:
        model.delta_fuse.load_state_dict(reg["delta_fuse"], strict=True)
    if "delta_text_ln" in reg:
        model.delta_text_ln.load_state_dict(reg["delta_text_ln"], strict=True)

    if "rel_head" in reg:
        model.rel_head.load_state_dict(reg["rel_head"], strict=True)
    if model.retrieval_proj is not None:
        if reg.get("retrieval_proj", None) is not None:
            model.retrieval_proj.load_state_dict(reg["retrieval_proj"], strict=True)
        if reg.get("retrieval_gate_bias", None) is not None:
            model.retrieval_gate_bias.load_state_dict(reg["retrieval_gate_bias"], strict=True)
        if reg.get("retrieval_rel_bias", None) is not None:
            model.retrieval_rel_bias.load_state_dict(reg["retrieval_rel_bias"], strict=True)
    if model.news_conv_encoder is not None:
        if reg.get("news_conv_encoder", None) is not None:
            model.news_conv_encoder.load_state_dict(reg["news_conv_encoder"], strict=True)
        if reg.get("news_conv_feat_proj", None) is not None:
            model.news_conv_feat_proj.load_state_dict(reg["news_conv_feat_proj"], strict=True)
        if reg.get("news_conv_delta_fuse", None) is not None:
            model.news_conv_delta_fuse.load_state_dict(reg["news_conv_delta_fuse"], strict=True)

    # move custom parts to LM embed device/dtype
    emb = model.lm.get_input_embeddings().weight
    model = model.to(device=emb.device, dtype=emb.dtype)

    return tok, model


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
