from __future__ import annotations

import torch
import torch.nn as nn


class RegimeRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        scalar_dim: int,
        num_routes: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(max(1, hidden_size))
        self.scalar_dim = int(max(1, scalar_dim))
        self.num_routes = int(max(2, num_routes))
        drop = float(max(0.0, dropout))

        self.scalar_proj = nn.Sequential(
            nn.Linear(self.scalar_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
        )
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.num_routes),
        )

    def forward(
        self,
        context: torch.Tensor,
        scalars: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        ctx = context
        if scalars is None:
            scalars = ctx.new_zeros(ctx.size(0), self.scalar_dim)
        else:
            scalars = scalars.to(device=ctx.device, dtype=ctx.dtype)
            if scalars.ndim != 2:
                scalars = scalars.reshape(scalars.size(0), -1)
            if scalars.size(1) < self.scalar_dim:
                pad = scalars.new_zeros(scalars.size(0), self.scalar_dim - scalars.size(1))
                scalars = torch.cat([scalars, pad], dim=-1)
            elif scalars.size(1) > self.scalar_dim:
                scalars = scalars[:, : self.scalar_dim]

        scalar_hidden = self.scalar_proj(scalars)
        logits = self.router(torch.cat([ctx, scalar_hidden], dim=-1))
        probs = torch.softmax(logits, dim=-1)
        return {
            "scalar_hidden": scalar_hidden,
            "route_logits": logits,
            "route_probs": probs,
            "abstain_prob": probs[:, :1],
            "expert_probs": probs[:, 1:],
        }


class ResidualExpertMixture(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(max(1, hidden_size))
        self.num_experts = int(max(1, num_experts))
        drop = float(max(0.0, dropout))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.out_ln = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        expert_inputs: list[torch.Tensor],
        expert_probs: torch.Tensor,
        *,
        mix_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(expert_inputs) != self.num_experts:
            raise ValueError(
                f"ResidualExpertMixture expected {self.num_experts} expert inputs, got {len(expert_inputs)}."
            )
        probs = expert_probs.to(device=expert_inputs[0].device, dtype=expert_inputs[0].dtype)
        if probs.ndim != 2:
            probs = probs.reshape(probs.size(0), -1)
        if probs.size(1) < self.num_experts:
            pad = probs.new_zeros(probs.size(0), self.num_experts - probs.size(1))
            probs = torch.cat([probs, pad], dim=-1)
        elif probs.size(1) > self.num_experts:
            probs = probs[:, : self.num_experts]

        expert_outputs = []
        for expert, expert_input in zip(self.experts, expert_inputs):
            expert_outputs.append(expert(expert_input))
        stacked = torch.stack(expert_outputs, dim=1)
        mixed = (stacked * probs.unsqueeze(-1)).sum(dim=1)
        base = expert_inputs[0]
        fused = self.out_ln(base + float(max(0.0, mix_scale)) * mixed)
        return fused, stacked
