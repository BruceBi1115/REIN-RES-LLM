from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch

from ..base.common import _z_batch_tensors
from ..utils.batch_utils import _batch_time_seq_for_sample


def _calendar_lookup_centered(parsed_times, dow_hod_mean: np.ndarray) -> np.ndarray:
    out = np.zeros((len(parsed_times),), dtype=np.float32)
    valid_idx: list[int] = []
    values: list[float] = []
    for idx, ts in enumerate(parsed_times):
        if pd.isna(ts):
            continue
        dow = int(ts.dayofweek)
        hod = int(ts.hour * 2 + int(ts.minute >= 30))
        val = float(dow_hod_mean[dow, hod])
        out[idx] = val
        valid_idx.append(idx)
        values.append(val)
    if valid_idx:
        mean_val = float(np.mean(values))
        for idx in valid_idx:
            out[idx] -= mean_val
    return out


def compute_residual_calendar_baseline(
    train_eval_loader,
    base_backbone,
    args,
    global_zstats,
    device,
    *,
    spike_target_pct: float = 0.10,
):
    sums = np.zeros((7, 48), dtype=np.float64)
    counts = np.zeros((7, 48), dtype=np.float64)
    detrended_values: list[float] = []

    base_backbone.eval()
    with torch.no_grad():
        for batch in train_eval_loader:
            history_z, targets_z, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
            history_z = history_z.to(device)
            pred_z = base_backbone(history_z).to(torch.float32).cpu()
            residual = targets_z - pred_z

            batch_size = residual.size(0)
            for sample_idx in range(batch_size):
                target_times = _batch_time_seq_for_sample(batch.get("target_times"), sample_idx)
                parsed = pd.to_datetime(target_times, errors="coerce")
                values = residual[sample_idx].detach().cpu().numpy()
                for step_idx, ts in enumerate(parsed[: len(values)]):
                    if pd.isna(ts):
                        continue
                    dow = int(ts.dayofweek)
                    hod = int(ts.hour * 2 + int(ts.minute >= 30))
                    sums[dow, hod] += float(values[step_idx])
                    counts[dow, hod] += 1.0

    baseline = np.zeros((7, 48), dtype=np.float32)
    mask = counts > 0
    baseline[mask] = (sums[mask] / counts[mask]).astype(np.float32)

    with torch.no_grad():
        for batch in train_eval_loader:
            history_z, targets_z, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
            history_z = history_z.to(device)
            pred_z = base_backbone(history_z).to(torch.float32).cpu()
            residual = targets_z - pred_z

            batch_size = residual.size(0)
            for sample_idx in range(batch_size):
                target_times = _batch_time_seq_for_sample(batch.get("target_times"), sample_idx)
                parsed = pd.to_datetime(target_times, errors="coerce")
                values = residual[sample_idx].detach().cpu().numpy().astype(np.float32)
                centered_calendar = _calendar_lookup_centered(parsed, baseline)
                detrended = values - float(values.mean()) - centered_calendar[: len(values)]
                detrended_values.extend(detrended.tolist())

    detrended_arr = np.asarray(detrended_values, dtype=np.float32)
    sigma = float(np.std(detrended_arr)) if detrended_arr.size > 0 else 1.0
    sigma = max(1e-6, sigma)
    spike_target_pct = float(np.clip(spike_target_pct, 0.0, 1.0))
    abs_threshold = 0.0
    if detrended_arr.size > 0 and 0.0 < spike_target_pct < 1.0:
        abs_threshold = float(np.quantile(np.abs(detrended_arr), 1.0 - spike_target_pct))
    return baseline, sigma, abs_threshold


@dataclass
class ResidualTargetDecomposer:
    dow_hod_mean: np.ndarray
    spike_sigma: float
    spike_threshold_abs: float = 0.0
    spike_k: float = 3.0
    spike_target_pct: float = 0.10

    def _calendar_bias(self, target_time_sequences) -> torch.Tensor:
        batch_size = len(target_time_sequences)
        horizon = max((len(seq) for seq in target_time_sequences), default=0)
        out = torch.zeros((batch_size, max(1, horizon)), dtype=torch.float32)
        for i, seq in enumerate(target_time_sequences):
            parsed = pd.to_datetime(seq, errors="coerce")
            centered_calendar = _calendar_lookup_centered(parsed, self.dow_hod_mean)
            if centered_calendar.size > 0:
                out[i, : centered_calendar.size] = torch.from_numpy(centered_calendar)
        return out

    def decompose(self, residual: torch.Tensor, target_time_sequences) -> dict[str, torch.Tensor]:
        residual = residual.to(torch.float32)
        slow = residual.mean(dim=-1)
        calendar_bias = self._calendar_bias(target_time_sequences).to(residual.device)
        if calendar_bias.size(-1) != residual.size(-1):
            calendar_bias = calendar_bias[..., : residual.size(-1)]
            if calendar_bias.size(-1) < residual.size(-1):
                pad = torch.zeros(
                    (calendar_bias.size(0), residual.size(-1) - calendar_bias.size(-1)),
                    dtype=calendar_bias.dtype,
                    device=calendar_bias.device,
                )
                calendar_bias = torch.cat([calendar_bias, pad], dim=-1)

        detrended = residual - slow.unsqueeze(-1) - calendar_bias
        sigma_threshold = float(self.spike_k) * float(self.spike_sigma)
        quantile_threshold = float(self.spike_threshold_abs)
        # Quantile-derived thresholds are more stable on heavy-tailed price residuals:
        # they preserve a non-empty sparse spike target instead of letting global sigma
        # wash the target out entirely.
        spike_threshold = quantile_threshold if quantile_threshold > 0.0 else max(0.0, sigma_threshold)
        spike_mask = detrended.abs() > spike_threshold

        spike_target_pct = float(np.clip(self.spike_target_pct, 0.0, 1.0))
        if 0.0 < spike_target_pct < 1.0 and residual.size(-1) > 0:
            max_spikes = max(1, min(residual.size(-1), int(math.ceil(residual.size(-1) * spike_target_pct))))
            top_indices = detrended.abs().topk(k=max_spikes, dim=-1).indices
            top_mask = torch.zeros_like(spike_mask, dtype=torch.bool)
            top_mask.scatter_(1, top_indices, True)
            spike_mask = spike_mask & top_mask

        spike = torch.where(spike_mask, detrended, torch.zeros_like(detrended))
        shape = residual - slow.unsqueeze(-1) - spike
        shape_mean = shape.mean(dim=-1, keepdim=True)
        shape = shape - shape_mean
        slow = slow + shape_mean.squeeze(-1)
        return {
            "slow": slow,
            "shape": shape,
            "spike": spike,
            "spike_mask": spike_mask.to(torch.float32),
        }
