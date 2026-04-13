from __future__ import annotations

import torch
import torch.nn as nn


class _ScalarMLP(nn.Module):
    def __init__(self, hidden_size: int, *, final_bias: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), 1),
        )
        nn.init.constant_(self.net[-1].bias, float(final_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RegimeProjector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(int(hidden_size)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegimeModulationHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        active_mass_threshold: float = 0.7,
        lambda_min: float = 0.05,
        lambda_ts_cap: float = 0.30,
        lambda_news_cap: float = 0.12,
        lambda_max: float = 0.45,
        shape_gain_cap: float = 0.20,
        spike_bias_cap: float = 0.75,
    ):
        super().__init__()
        self.active_mass_threshold = float(active_mass_threshold)
        # Soft-gate sharpness for lambda_base shrinkage when relevance is below threshold.
        # Plan Priority-1: inactive samples should see lambda_base → lambda_min, so their
        # delta contribution vanishes without relying on a hard 0/1 mask during training.
        self.active_soft_tau = 0.05
        self.lambda_min = float(lambda_min)
        self.lambda_ts_cap = float(lambda_ts_cap)
        self.lambda_news_cap = float(lambda_news_cap)
        self.lambda_max = float(lambda_max)
        self.shape_gain_cap = float(shape_gain_cap)
        self.spike_bias_cap = float(spike_bias_cap)
        self.ts_trust_mlp = _ScalarMLP(hidden_size, final_bias=-1.0)
        self.trust_delta_mlp = _ScalarMLP(hidden_size)
        self.shape_gain_mlp = _ScalarMLP(hidden_size)
        self.spike_prior_mlp = _ScalarMLP(hidden_size)

    def forward(
        self,
        ts_summary: torch.Tensor,
        regime_repr: torch.Tensor,
        relevance_mass: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if relevance_mass.ndim == 2 and relevance_mass.size(-1) == 1:
            relevance = relevance_mass.squeeze(-1)
        else:
            relevance = relevance_mass.to(torch.float32).view(-1)
        active = (relevance > self.active_mass_threshold).to(regime_repr.dtype)

        lambda_ts_logit = self.ts_trust_mlp(ts_summary)
        lambda_ts = self.lambda_min + self.lambda_ts_cap * torch.sigmoid(lambda_ts_logit)
        trust_delta_raw = self.trust_delta_mlp(regime_repr) * active
        lambda_news_delta = self.lambda_news_cap * torch.tanh(trust_delta_raw)
        lambda_base_raw = (lambda_ts + lambda_news_delta).clamp(min=self.lambda_min, max=self.lambda_max)
        # Soft gate: shrink lambda_base toward lambda_min when relevance is below threshold,
        # so inactive samples see a ~5% delta contribution instead of 5-45% TS-only noise.
        soft_gate = torch.sigmoid((relevance - self.active_mass_threshold) / self.active_soft_tau)
        lambda_base = self.lambda_min + (lambda_base_raw - self.lambda_min) * soft_gate
        shape_gain_raw = self.shape_gain_mlp(regime_repr) * active
        spike_bias = self.spike_bias_cap * torch.tanh(self.spike_prior_mlp(regime_repr) * active)
        shape_gain = 1.0 + self.shape_gain_cap * torch.tanh(shape_gain_raw)
        return {
            "active": active,
            "lambda_ts_logit": lambda_ts_logit,
            "lambda_ts": lambda_ts,
            "trust_delta_raw": trust_delta_raw,
            "lambda_news_delta": lambda_news_delta,
            "shape_gain_raw": shape_gain_raw,
            "shape_gain": shape_gain,
            "spike_bias": spike_bias,
            "lambda_base": lambda_base,
        }

    def initialize_from_pretrain(self, pretrain_heads: "RegimePretrainHeads") -> None:
        _copy_first_layer(pretrain_heads.vol_head, self.trust_delta_mlp)
        _copy_first_layer(pretrain_heads.shape_head, self.shape_gain_mlp)
        _copy_first_layer(pretrain_heads.spike_head, self.spike_prior_mlp)


class RegimePretrainHeads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.vol_head = _ScalarMLP(hidden_size)
        self.spike_head = _ScalarMLP(hidden_size)
        self.shape_head = _ScalarMLP(hidden_size)

    def forward(self, regime_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vol_ratio = self.vol_head(regime_repr)
        spike_flag_logit = self.spike_head(regime_repr)
        shape_dev = self.shape_head(regime_repr)
        return vol_ratio, spike_flag_logit, shape_dev


def _copy_first_layer(source: _ScalarMLP, target: _ScalarMLP) -> None:
    src_linear = source.net[0]
    dst_linear = target.net[0]
    dst_linear.weight.data.copy_(src_linear.weight.data)
    dst_linear.bias.data.copy_(src_linear.bias.data)
