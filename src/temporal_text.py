from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TemporalTextTower(nn.Module):
    def __init__(
        self,
        *,
        model_id: str,
        step_dim: int,
        hidden_size: int,
        patch_dim: int,
        patch_stride: int,
        freeze_encoder: bool = True,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        model_id = str(model_id or "").strip()
        if not model_id:
            raise ValueError("TemporalTextTower requires a non-empty model_id.")
        self.model_id = model_id
        self.step_dim = int(max(1, step_dim))
        self.hidden_size = int(max(1, hidden_size))
        self.patch_dim = int(max(1, patch_dim))
        self.patch_stride = int(max(1, patch_stride))
        self.freeze_encoder = bool(freeze_encoder)
        self.unfreeze_last_n = int(max(0, unfreeze_last_n))

        self.encoder = AutoModel.from_pretrained(self.model_id)
        text_hidden = int(
            getattr(self.encoder.config, "hidden_size", 0)
            or getattr(self.encoder.config, "n_embd", 0)
            or getattr(self.encoder.config, "d_model", 0)
            or self.hidden_size
        )
        intermediate = max(self.step_dim * 2, 64)
        self.step_proj = nn.Sequential(
            nn.Linear(text_hidden, intermediate),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate, self.step_dim),
            nn.GELU(),
            nn.LayerNorm(self.step_dim),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(self.step_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.unfreeze_last_n > 0:
                layers = list(self.encoder.encoder.layer) if hasattr(self.encoder, "encoder") else []
                if len(layers) > 0:
                    for layer in layers[-self.unfreeze_last_n :]:
                        for p in layer.parameters():
                            p.requires_grad = True

    def encode_step_series(
        self,
        temporal_text_ids: torch.Tensor | None,
        temporal_text_attn: torch.Tensor | None,
        temporal_text_step_mask: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            temporal_text_ids is None
            or temporal_text_attn is None
            or temporal_text_ids.ndim != 3
            or temporal_text_attn.ndim != 3
        ):
            return None, None

        ids = temporal_text_ids.to(device=device, dtype=torch.long)
        attn = temporal_text_attn.to(device=device, dtype=torch.long)
        bsz, hist_len, tok_len = ids.shape
        if bsz <= 0 or hist_len <= 0 or tok_len <= 0:
            return None, None

        flat_ids = ids.reshape(bsz * hist_len, tok_len)
        flat_attn = attn.reshape(bsz * hist_len, tok_len)
        valid_flat = (flat_attn.sum(dim=1, keepdim=True) > 0).to(dtype=dtype)
        if float(valid_flat.sum().item()) <= 0.0:
            return None, None

        if temporal_text_step_mask is None:
            step_mask = valid_flat.view(bsz, hist_len)
        else:
            step_mask = temporal_text_step_mask.to(device=device, dtype=dtype)
            if step_mask.ndim != 2:
                step_mask = step_mask.reshape(bsz, hist_len)

        encoder_trainable = any(p.requires_grad for p in self.encoder.parameters())
        if self.freeze_encoder and not encoder_trainable:
            self.encoder.eval()
            with torch.no_grad():
                hidden = self.encoder(
                    input_ids=flat_ids,
                    attention_mask=flat_attn,
                ).last_hidden_state
        else:
            hidden = self.encoder(
                input_ids=flat_ids,
                attention_mask=flat_attn,
            ).last_hidden_state
        hidden = hidden.to(device=device, dtype=dtype)
        token_mask = flat_attn.to(device=device, dtype=dtype).unsqueeze(-1)
        pooled = (hidden * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
        step_feat = self.step_proj(pooled).view(bsz, hist_len, self.step_dim)
        step_feat = step_feat * step_mask.unsqueeze(-1)
        return step_feat, step_mask

    def build_patch_context(
        self,
        step_feat: torch.Tensor | None,
        step_mask: torch.Tensor | None,
        *,
        target_patch_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            step_feat is None
            or step_mask is None
            or step_feat.ndim != 3
            or step_mask.ndim != 2
        ):
            return None, None

        feat = step_feat.to(device=device, dtype=dtype)
        mask = step_mask.to(device=device, dtype=dtype)
        bsz, hist_len, feat_dim = feat.shape
        patch_len = int(max(1, self.patch_dim))
        stride = int(max(1, self.patch_stride))
        if hist_len < patch_len:
            pad_len = patch_len - hist_len
            feat = torch.cat([feat.new_zeros(bsz, pad_len, feat_dim), feat], dim=1)
            mask = torch.cat([mask.new_zeros(bsz, pad_len), mask], dim=1)
            hist_len = patch_len

        patch_items = []
        patch_mask_items = []
        for start in range(0, hist_len - patch_len + 1, stride):
            seg = feat[:, start : start + patch_len, :]
            seg_mask = mask[:, start : start + patch_len].unsqueeze(-1)
            pooled = (seg * seg_mask).sum(dim=1) / seg_mask.sum(dim=1).clamp_min(1.0)
            patch_valid = (seg_mask.squeeze(-1).sum(dim=1) > 0).to(dtype=dtype)
            patch_items.append(pooled)
            patch_mask_items.append(patch_valid)

        if len(patch_items) == 0:
            return None, None

        patch_feat = torch.stack(patch_items, dim=1)
        patch_mask = torch.stack(patch_mask_items, dim=1)
        if patch_feat.size(1) > target_patch_count:
            patch_feat = patch_feat[:, :target_patch_count, :]
            patch_mask = patch_mask[:, :target_patch_count]
        elif patch_feat.size(1) < target_patch_count:
            pad_p = target_patch_count - patch_feat.size(1)
            patch_feat = torch.cat([patch_feat, patch_feat.new_zeros(bsz, pad_p, feat_dim)], dim=1)
            patch_mask = torch.cat([patch_mask, patch_mask.new_zeros(bsz, pad_p)], dim=1)

        patch_context = self.patch_proj(patch_feat) * patch_mask.unsqueeze(-1)
        return patch_context, patch_mask

    def summarize_patch_context(
        self,
        patch_context: torch.Tensor | None,
        patch_mask: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            patch_context is None
            or patch_mask is None
            or patch_context.ndim != 3
            or patch_mask.ndim != 2
        ):
            return None, None

        ctx = patch_context.to(device=device, dtype=dtype)
        mask = patch_mask.to(device=device, dtype=dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        summary = (ctx * mask.unsqueeze(-1)).sum(dim=1) / denom
        strength = (mask.sum(dim=1, keepdim=True) / float(max(1, mask.size(1)))).clamp(0.0, 1.0)
        return summary, strength

    def forward(
        self,
        *,
        temporal_text_ids: torch.Tensor | None,
        temporal_text_attn: torch.Tensor | None,
        temporal_text_step_mask: torch.Tensor | None,
        target_patch_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | None]:
        step_feat, step_mask = self.encode_step_series(
            temporal_text_ids,
            temporal_text_attn,
            temporal_text_step_mask,
            device=device,
            dtype=dtype,
        )
        patch_context, patch_mask = self.build_patch_context(
            step_feat,
            step_mask,
            target_patch_count=int(max(1, target_patch_count)),
            device=device,
            dtype=dtype,
        )
        text_summary, text_strength = self.summarize_patch_context(
            patch_context,
            patch_mask,
            device=device,
            dtype=dtype,
        )
        return {
            "step_feat": step_feat,
            "step_mask": step_mask,
            "patch_context": patch_context,
            "patch_mask": patch_mask,
            "text_summary": text_summary,
            "text_strength": text_strength,
        }
