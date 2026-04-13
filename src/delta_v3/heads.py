from __future__ import annotations

import torch
import torch.nn as nn


class SlowHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, ts_summary: torch.Tensor) -> torch.Tensor:
        return self.net(ts_summary).squeeze(-1)


class ShapeHead(nn.Module):
    def __init__(self, hidden_size: int, horizon: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), int(horizon)),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        shape = self.proj(fused_tokens)
        return shape - shape.mean(dim=-1, keepdim=True)


class SpikeHead(nn.Module):
    def __init__(self, hidden_size: int, horizon: int, threshold: float):
        super().__init__()
        self.threshold = float(threshold)
        self.delta_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), int(horizon)),
        )
        self.gate_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LazyLinear(int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), int(horizon)),
        )

    def forward(self, fused_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spike_delta = self.delta_proj(fused_tokens)
        spike_gate_logits = self.gate_proj(fused_tokens)
        spike_gate = torch.sigmoid(spike_gate_logits)
        spike_gate_hard = (spike_gate >= self.threshold).to(spike_gate.dtype)
        return spike_delta, spike_gate_logits, spike_gate, spike_gate_hard
