from __future__ import annotations

import torch
import torch.nn as nn

from .config import DeltaV3Config
from .encoder_ts import PatchTSTTSEncoder
from .heads import ShapeHead, SlowHead, SpikeHead
from .modulation_heads import RegimeModulationHeads, RegimeProjector


class DeltaV3Regressor(nn.Module):
    def __init__(self, cfg: DeltaV3Config):
        super().__init__()
        self.cfg = cfg
        self.ts_encoder = PatchTSTTSEncoder(
            patch_len=cfg.patch_len,
            patch_stride=cfg.patch_stride,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            use_base_hidden=cfg.use_base_hidden,
        )
        self.regime_projector = RegimeProjector(cfg.hidden_size)
        self.slow_head = SlowHead(cfg.hidden_size)
        self.shape_head = ShapeHead(cfg.hidden_size, cfg.horizon)
        self.spike_head = SpikeHead(cfg.hidden_size, cfg.horizon, cfg.spike_gate_threshold)
        self.modulation_heads = RegimeModulationHeads(
            cfg.hidden_size,
            active_mass_threshold=cfg.active_mass_threshold,
            lambda_min=cfg.lambda_min,
            lambda_ts_cap=cfg.lambda_ts_cap,
            lambda_news_cap=cfg.lambda_news_cap,
            lambda_max=cfg.lambda_max,
            shape_gain_cap=cfg.shape_gain_cap,
            spike_bias_cap=cfg.spike_bias_cap,
        )

    def encode_regime(self, regime_pack: dict[str, torch.Tensor]) -> torch.Tensor:
        relevance_mass = regime_pack["relevance_mass"].to(torch.float32)
        if relevance_mass.ndim == 1:
            relevance_mass = relevance_mass.unsqueeze(-1)
        features = torch.cat(
            [
                regime_pack["regime_vec"].to(torch.float32),
                regime_pack["topic_tag_mass"].to(torch.float32),
                regime_pack["text_emb"].to(torch.float32),
            ],
            dim=-1,
        )
        regime_repr = self.regime_projector(features)
        active = (relevance_mass > float(self.cfg.active_mass_threshold)).to(regime_repr.dtype)
        return regime_repr * active

    def initialize_from_pretrain(self, helper_head) -> None:
        if helper_head is None:
            return
        self.modulation_heads.initialize_from_pretrain(helper_head)

    def forward(
        self,
        history_z: torch.Tensor,
        history_times,
        base_pred_z: torch.Tensor,
        base_hidden: torch.Tensor | None,
        regime_pack: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ts_tokens, ts_summary = self.ts_encoder(history_z=history_z, history_times=history_times, base_hidden=base_hidden)
        regime_repr = self.encode_regime(regime_pack)

        slow_ts = self.slow_head(ts_summary)
        shape_ts = self.shape_head(ts_tokens)
        spike_ts, spike_gate_logits_ts, _, _ = self.spike_head(ts_tokens)

        modulation = self.modulation_heads(ts_summary, regime_repr, regime_pack["relevance_mass"])
        spike_gate_logits = spike_gate_logits_ts + modulation["spike_bias"].unsqueeze(-1)
        spike_gate = torch.sigmoid(spike_gate_logits)
        spike_gate_hard = (spike_gate >= self.cfg.spike_gate_threshold).to(spike_gate.dtype)

        shape_z = shape_ts * modulation["shape_gain"].unsqueeze(-1)
        residual_z = slow_ts.unsqueeze(-1) + shape_z + spike_gate * spike_ts
        pred_z = base_pred_z + modulation["lambda_base"].unsqueeze(-1) * residual_z

        return {
            "pred_z": pred_z,
            "residual_hat": pred_z - base_pred_z,
            "residual_ts_z": residual_z,
            "slow_ts": slow_ts,
            "shape_ts": shape_ts,
            "shape_z": shape_z,
            "spike_ts": spike_ts,
            "spike_gate_logits_ts": spike_gate_logits_ts,
            "spike_gate_logits": spike_gate_logits,
            "spike_gate": spike_gate,
            "spike_gate_hard": spike_gate_hard,
            "lambda_base": modulation["lambda_base"],
            "lambda_ts": modulation["lambda_ts"],
            "lambda_news_delta": modulation["lambda_news_delta"],
            "shape_gain": modulation["shape_gain"],
            "shape_gain_raw": modulation["shape_gain_raw"],
            "spike_bias": modulation["spike_bias"],
            "trust_logit": modulation["trust_delta_raw"],
            "active_mask": modulation["active"],
            "relevance_mass": regime_pack["relevance_mass"].to(torch.float32).view(-1),
        }


def build_delta_v3_model(cfg: DeltaV3Config) -> DeltaV3Regressor:
    return DeltaV3Regressor(cfg=cfg)
