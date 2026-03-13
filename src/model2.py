import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TinyNewsTSRegressor(nn.Module):
    """
    Lightweight DELTA regressor used by this project:
    - TS patches -> MLP encoder
    - pooled TS feature -> base/delta heads
    - optional retrieval feature fusion for DELTA gate/bias
    - optional refined-news text encoder branch directly predicting delta correction
    """

    def __init__(
        self,
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
        huber_beta: float = 0.5,
        use_horizon_weight: bool = True,
        horizon_weight_end: float = 0.5,
        text_encoder: nn.Module | None = None,
        text_encoder_hidden_size: int = 0,
        text_direct_enable: bool = False,
        text_fuse_lambda: float = 0.5,
        text_gate_init_bias: float = -2.0,
        text_clip: float = 1.5,
        text_trainable: bool = False,
    ):
        super().__init__()
        self.model_variant = "tiny_news_ts"
        self.horizon = int(horizon)
        self.patch_dim = int(patch_dim)
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

        self.text_direct_enable = bool(text_direct_enable)
        self.text_fuse_lambda = float(text_fuse_lambda)
        self.text_clip = float(text_clip)
        self.text_trainable = bool(text_trainable)
        self.delta_text_gate_init_bias = float(text_gate_init_bias)
        self.text_encoder_hidden_size = int(max(0, text_encoder_hidden_size))
        self.text_encoder = text_encoder if self.text_direct_enable else None
        if self.text_encoder is not None and self.text_encoder_hidden_size > 0:
            if not self.text_trainable:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_encoder_hidden_size, self.hidden_size),
                nn.GELU(),
                nn.LayerNorm(self.hidden_size),
            )
            self.text_delta_head = nn.Linear(self.hidden_size, self.horizon)
            nn.init.normal_(self.text_delta_head.weight, mean=0.0, std=float(delta_head_init_std))
            nn.init.zeros_(self.text_delta_head.bias)
            self.text_gate = nn.Linear(self.hidden_size, self.horizon)
            nn.init.zeros_(self.text_gate.weight)
            nn.init.constant_(self.text_gate.bias, float(text_gate_init_bias))
            self.text_log_scale = nn.Parameter(torch.zeros(1))
        else:
            self.text_encoder = None
            self.text_proj = None
            self.text_delta_head = None
            self.text_gate = None
            self.text_log_scale = None
            self.text_direct_enable = False

        self.huber_beta = float(huber_beta)
        self.use_horizon_weight = bool(use_horizon_weight)
        self.horizon_weight_end = float(horizon_weight_end)

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

    def _encode_refined_text(
        self,
        refined_ids: torch.Tensor | None,
        refined_attn: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            (not self.text_direct_enable)
            or self.text_encoder is None
            or self.text_proj is None
            or self.text_delta_head is None
            or self.text_gate is None
            or refined_ids is None
            or refined_attn is None
            or refined_ids.ndim != 2
            or refined_attn.ndim != 2
            or refined_ids.size(1) <= 0
        ):
            return None, None

        ids = refined_ids.to(device=device, dtype=torch.long)
        attn = refined_attn.to(device=device, dtype=torch.long)
        valid = (attn.sum(dim=1, keepdim=True) > 0).to(dtype=dtype)
        if float(valid.sum().item()) <= 0.0:
            return None, None

        if self.text_trainable:
            enc = self.text_encoder(input_ids=ids, attention_mask=attn)
        else:
            was_training = bool(self.text_encoder.training)
            self.text_encoder.eval()
            with torch.no_grad():
                enc = self.text_encoder(input_ids=ids, attention_mask=attn)
            if was_training:
                self.text_encoder.train()

        hidden = getattr(enc, "last_hidden_state", None)
        if hidden is None and isinstance(enc, (tuple, list)) and len(enc) > 0:
            hidden = enc[0]
        if hidden is None:
            return None, None
        hidden = hidden.to(device=device, dtype=dtype)
        mask = attn.to(device=device, dtype=dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * mask).sum(dim=1) / denom
        text_feat = self.text_proj(pooled)
        text_raw = self.text_delta_head(self.delta_head_drop(text_feat))
        text_gate = torch.sigmoid(self.text_gate(text_feat))
        text_scale = torch.exp(self.text_log_scale).to(device=device, dtype=dtype).clamp(max=5.0)
        text_delta = text_raw * text_gate * text_scale
        text_delta = text_delta * valid
        if self.text_clip > 0:
            c = torch.tensor(self.text_clip, device=device, dtype=dtype)
            text_delta = c * torch.tanh(text_delta / c)
        return text_delta, text_gate * valid

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ts_patches: torch.Tensor,
        ts_patch_mask: torch.Tensor,
        refined_news_input_ids: torch.Tensor | None = None,
        refined_news_attention_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        head_mode: str = "base",
        rel_targets: torch.Tensor | None = None,
        rel_lambda: float = 0.0,
        retrieval_feats: torch.Tensor | None = None,
        retrieval_gate_only: bool = False,
    ):
        del input_ids, attention_mask

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

            text_gate = None
            text_delta = None
            if self.text_direct_enable and self.text_fuse_lambda != 0.0:
                text_delta, text_gate = self._encode_refined_text(
                    refined_ids=refined_news_input_ids,
                    refined_attn=refined_news_attention_mask,
                    dtype=pred.dtype,
                    device=pred.device,
                )
                if text_delta is not None:
                    pred = pred + float(self.text_fuse_lambda) * text_delta
        else:
            raise ValueError(f"Unknown head_mode={head_mode}")

        pred = pred.to(torch.float32)
        rel_logit = self.rel_head(rel_input).squeeze(-1).to(torch.float32)
        if head_mode == "delta" and retrieval_rel_bias is not None:
            rel_logit = rel_logit + retrieval_rel_bias

        out = {"pred": pred, "rel_logits": rel_logit}
        if head_mode == "delta":
            rel_gate = torch.ones_like(delta_gate)
            out["delta_gate_mean"] = delta_gate.mean().detach()
            out["delta_scale"] = delta_scale.detach()
            out["delta_rel_gate_mean"] = rel_gate.mean().detach()
            out["delta_internal_gate"] = int(self.delta_internal_gate)
            if retrieval_weight is not None:
                out["retrieval_weight_mean"] = retrieval_weight.mean().detach()
            if text_delta is not None:
                out["text_delta_mean_abs"] = text_delta.abs().mean().detach()
            if text_gate is not None:
                out["text_gate_mean"] = text_gate.mean().detach()

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


def _resolve_text_spec(preset_raw: str, model_raw: str, tok_raw: str):
    preset = str(preset_raw or "custom").lower().strip()
    preset_map = {
        "custom": ("", ""),
        "distilbert": ("distilbert-base-uncased", "distilbert-base-uncased"),
        "gpt2": ("gpt2", "gpt2"),
        "bert_base": ("bert-base-uncased", "bert-base-uncased"),
        "roberta_base": ("roberta-base", "roberta-base"),
        "deberta_v3_base": ("microsoft/deberta-v3-base", "microsoft/deberta-v3-base"),
    }
    m_p, t_p = preset_map.get(preset, ("", ""))
    model_id = str(model_raw or "").strip() or m_p
    tok_id = str(tok_raw or "").strip() or t_p or model_id
    if not model_id:
        model_id = "distilbert-base-uncased"
        tok_id = tok_id or model_id
    return model_id, tok_id


def build_delta_model(
    base_model: str,
    tokenizer_id: str,
    horizon: int = 48,
    patch_dim: int = 4,
    patch_dropout: float = 0.0,
    head_dropout: float = 0.0,
    head_mlp: bool = False,
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
    delta_model_variant: str = "tiny_news_ts",
    tiny_news_model_preset: str = "custom",
    tiny_news_model: str = "",
    tiny_news_tokenizer: str = "",
    tiny_news_hidden_size: int = 256,
    tiny_news_text_trainable: bool = False,
    tiny_news_loader: str = "auto",
    delta_text_direct_enable: bool = False,
    delta_text_fuse_lambda: float = 0.5,
    delta_text_gate_init_bias: float = -2.0,
    delta_text_clip: float = 1.5,
    delta_text_max_len: int = 160,
):
    del (
        head_mlp,
        delta_news_tail_tokens,
        delta_rel_floor,
        delta_model_variant,
        tiny_news_loader,
    )

    model_id, tok_id = _resolve_text_spec(
        tiny_news_model_preset,
        tiny_news_model or tokenizer_id or base_model,
        tiny_news_tokenizer,
    )
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

    hidden_size = int(tiny_news_hidden_size) if int(tiny_news_hidden_size) > 0 else 256
    text_direct_enable = bool(delta_text_direct_enable)
    text_encoder = None
    text_encoder_hidden_size = 0
    if text_direct_enable:
        try:
            text_encoder = AutoModel.from_pretrained(model_id)
            hidden_guess = getattr(text_encoder.config, "hidden_size", None)
            if hidden_guess is None:
                hidden_guess = getattr(text_encoder.config, "n_embd", None)
            text_encoder_hidden_size = int(hidden_guess) if hidden_guess is not None else 0
        except Exception:
            text_encoder = None
            text_encoder_hidden_size = 0
            text_direct_enable = False

    model = TinyNewsTSRegressor(
        horizon=int(horizon),
        patch_dim=int(patch_dim),
        hidden_size=int(hidden_size),
        patch_dropout=float(patch_dropout),
        head_dropout=float(head_dropout),
        delta_gate_init_bias=float(delta_gate_init_bias),
        delta_head_init_std=float(delta_head_init_std),
        delta_internal_gate=bool(delta_internal_gate),
        delta_clip=float(delta_clip),
        retrieval_feat_dim=int(retrieval_feat_dim),
        huber_beta=float(huber_beta),
        use_horizon_weight=bool(use_horizon_weight),
        horizon_weight_end=float(horizon_weight_end),
        text_encoder=text_encoder,
        text_encoder_hidden_size=int(max(0, text_encoder_hidden_size)),
        text_direct_enable=bool(text_direct_enable),
        text_fuse_lambda=float(delta_text_fuse_lambda),
        text_gate_init_bias=float(delta_text_gate_init_bias),
        text_clip=float(delta_text_clip),
        text_trainable=bool(tiny_news_text_trainable),
    )
    model.tiny_text_model_id = str(model_id)
    model.tiny_text_tokenizer_id = str(tok_id)
    model.delta_text_max_len = int(max(1, delta_text_max_len))
    model.delta_text_gate_init_bias = float(delta_text_gate_init_bias)
    return tok, model


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
            "hidden_size": int(model.hidden_size),
            "retrieval_feat_dim": int(getattr(model, "retrieval_feat_dim", 0)),
            "delta_internal_gate": int(bool(getattr(model, "delta_internal_gate", True))),
            "delta_clip": float(getattr(model, "delta_clip", 3.0)),
            "huber_beta": float(getattr(model, "huber_beta", 0.5)),
            "use_horizon_weight": bool(getattr(model, "use_horizon_weight", True)),
            "horizon_weight_end": float(getattr(model, "horizon_weight_end", 0.5)),
            "text_direct_enable": int(bool(getattr(model, "text_direct_enable", False))),
            "text_fuse_lambda": float(getattr(model, "text_fuse_lambda", 0.0)),
            "text_clip": float(getattr(model, "text_clip", 0.0)),
            "text_trainable": int(bool(getattr(model, "text_trainable", False))),
            "text_encoder_hidden_size": int(getattr(model, "text_encoder_hidden_size", 0)),
            "tiny_text_model_id": str(getattr(model, "tiny_text_model_id", "")),
            "tiny_text_tokenizer_id": str(getattr(model, "tiny_text_tokenizer_id", "")),
            "delta_text_max_len": int(getattr(model, "delta_text_max_len", 160)),
            "delta_text_gate_init_bias": float(getattr(model, "delta_text_gate_init_bias", -2.0)),
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
        "hidden_size": int(model.hidden_size),
        "huber_beta": float(getattr(model, "huber_beta", 0.5)),
        "use_horizon_weight": bool(getattr(model, "use_horizon_weight", True)),
        "horizon_weight_end": float(getattr(model, "horizon_weight_end", 0.5)),
        "delta_internal_gate": int(bool(getattr(model, "delta_internal_gate", True))),
        "delta_clip": float(getattr(model, "delta_clip", 3.0)),
        "retrieval_feat_dim": int(getattr(model, "retrieval_feat_dim", 0)),
        "text_direct_enable": int(bool(getattr(model, "text_direct_enable", False))),
        "text_fuse_lambda": float(getattr(model, "text_fuse_lambda", 0.0)),
        "text_clip": float(getattr(model, "text_clip", 0.0)),
        "text_trainable": int(bool(getattr(model, "text_trainable", False))),
        "text_encoder_hidden_size": int(getattr(model, "text_encoder_hidden_size", 0)),
        "tiny_text_model_id": str(getattr(model, "tiny_text_model_id", "")),
        "tiny_text_tokenizer_id": str(getattr(model, "tiny_text_tokenizer_id", "")),
        "delta_text_max_len": int(getattr(model, "delta_text_max_len", 160)),
        "delta_text_gate_init_bias": float(getattr(model, "delta_text_gate_init_bias", -2.0)),
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
    device_map=None,
    is_trainable: bool = False,
    head_mlp: bool = False,
    hd: float = 0.0,
    pd: float = 0.0,
):
    del device_map, is_trainable, head_mlp

    with open(os.path.join(ckpt_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "tokenizer"), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

    text_direct_enable = bool(int(meta.get("text_direct_enable", 0)))
    text_encoder = None
    text_hidden_size = int(max(0, meta.get("text_encoder_hidden_size", 0) or 0))
    if text_direct_enable:
        text_model_id = str(meta.get("tiny_text_model_id", "")).strip()
        if text_model_id:
            try:
                text_encoder = AutoModel.from_pretrained(text_model_id)
                if text_hidden_size <= 0:
                    hidden_guess = getattr(text_encoder.config, "hidden_size", None)
                    if hidden_guess is None:
                        hidden_guess = getattr(text_encoder.config, "n_embd", None)
                    text_hidden_size = int(hidden_guess) if hidden_guess is not None else 0
            except Exception:
                text_encoder = None
                text_hidden_size = 0
                text_direct_enable = False

    model = TinyNewsTSRegressor(
        horizon=int(meta.get("horizon", 48)),
        patch_dim=int(meta.get("patch_dim", 4)),
        hidden_size=int(meta.get("hidden_size", 256)),
        patch_dropout=float(pd),
        head_dropout=float(hd),
        delta_internal_gate=bool(int(meta.get("delta_internal_gate", 1))),
        delta_clip=float(meta.get("delta_clip", 3.0)),
        retrieval_feat_dim=int(meta.get("retrieval_feat_dim", 12)),
        huber_beta=float(meta.get("huber_beta", 0.5)),
        use_horizon_weight=bool(meta.get("use_horizon_weight", True)),
        horizon_weight_end=float(meta.get("horizon_weight_end", 0.5)),
        text_encoder=text_encoder,
        text_encoder_hidden_size=int(max(0, text_hidden_size)),
        text_direct_enable=bool(text_direct_enable),
        text_fuse_lambda=float(meta.get("text_fuse_lambda", 0.0)),
        text_gate_init_bias=float(meta.get("delta_text_gate_init_bias", -2.0)),
        text_clip=float(meta.get("text_clip", 1.5)),
        text_trainable=bool(int(meta.get("text_trainable", 0))),
    )
    model.tiny_text_model_id = str(meta.get("tiny_text_model_id", ""))
    model.tiny_text_tokenizer_id = str(meta.get("tiny_text_tokenizer_id", ""))
    model.delta_text_max_len = int(max(1, meta.get("delta_text_max_len", 160) or 160))

    reg = torch.load(os.path.join(ckpt_dir, "regressor.pt"), map_location="cpu")
    state = reg.get("model_state", reg)
    model.load_state_dict(state, strict=False)
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
