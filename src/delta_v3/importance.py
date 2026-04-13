from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class HardResidualSampler:
    def __init__(self, residual_magnitudes, top_pct: float, hard_frac: float):
        magnitudes = np.asarray(residual_magnitudes, dtype=np.float32).reshape(-1)
        self.residual_magnitudes = magnitudes
        self.top_pct = float(max(0.0, min(1.0, top_pct)))
        self.hard_frac = float(max(0.0, min(1.0, hard_frac)))

        if magnitudes.size == 0:
            self.hard_mask = np.zeros((0,), dtype=bool)
            self.weights = np.zeros((0,), dtype=np.float32)
            return

        threshold = np.quantile(magnitudes, max(0.0, 1.0 - self.top_pct)) if self.top_pct > 0 else np.inf
        self.hard_mask = magnitudes >= threshold
        hard_count = int(self.hard_mask.sum())
        easy_count = int((~self.hard_mask).sum())

        weights = np.zeros_like(magnitudes, dtype=np.float32)
        if hard_count > 0:
            weights[self.hard_mask] = self.hard_frac / max(1, hard_count)
        if easy_count > 0:
            weights[~self.hard_mask] = (1.0 - self.hard_frac) / max(1, easy_count)
        if weights.sum() <= 0:
            weights[:] = 1.0 / max(1, len(weights))
        self.weights = weights / max(1e-6, float(weights.sum()))

    def build(self, num_samples: int | None = None) -> WeightedRandomSampler:
        total = int(num_samples) if num_samples is not None else int(len(self.weights))
        return WeightedRandomSampler(
            weights=torch.tensor(self.weights, dtype=torch.double),
            num_samples=max(1, total),
            replacement=True,
        )

    @property
    def hard_hit_rate(self) -> float:
        if self.weights.size == 0:
            return 0.0
        return float(self.weights[self.hard_mask].sum())
