#!/usr/bin/env python
"""Build REIN placebo banks: uninformative news content with the true relevance mass.

Timestamps, relevance_mass and in_force_doc_count are copied unchanged from the real bank, so the
activity flag and the soft gate match REIN exactly. Bins with no news stay empty. Only the content
fields (regime_vec, topic_tag_mass, text_emb) change.

  random  Within each data split, whole weeks are permuted so that every week takes the content of
          another week of the same split at least two weeks away. Weekday and time of day are
          preserved, and the three content fields move together. A trailing partial week takes the
          leading bins of a week at least two weeks away.
  noise   Each content dimension is a stationary Gaussian AR(1) process whose mean, std, range and
          one-day autocorrelation are estimated on the training split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("_shared_refine_cache/v4")
OUT_DIR = CACHE / "ablations" / "rein_placebo"
CONTENT_KEYS = ("regime_vec", "topic_tag_mass", "text_emb")
MIN_WEEK_GAP = 2
DATASETS = {
    "gas_demand": ("regime_bank_gdelt_gas_nl_2025.npz", "dataset/gas_demand/entsog_gas_demand_nl_2025_hour_extracted", "date", False),
    "nsw_load": ("regime_bank_load_freeNews_reneweconomy_pvmag_watt.npz", "dataset/2024NSWelecLOAD/2024NSWelecLOAD", "date", True),
    "nsw_price": ("regime_bank_price_freeNews_reneweconomy_pvmag_watt.npz", "dataset/2024NSWelecPRICE/2024NSWelecPRICE", "date", True),
    "traffic_count": ("regime_bank_gdelt_2024.npz", "dataset/traffic_count/road_traffic_counts_hourly_permanent3_2024_hourly", "time", False),
}


def split_ids(dates: pd.DatetimeIndex, prefix: str, time_col: str, dayfirst: bool) -> np.ndarray:
    ends = []
    for name in ("train", "val"):
        ends.append(pd.to_datetime(pd.read_csv(f"{prefix}_{name}set.csv")[time_col], dayfirst=dayfirst).max())
    return np.where(dates <= ends[0], 0, np.where(dates <= ends[1], 1, 2))


def week_donors(n_bins: int, week: int, rng: np.random.Generator) -> np.ndarray:
    """Donor position (relative to the split start) for every bin of a contiguous split."""
    weeks, rest = divmod(n_bins, week)
    if weeks < MIN_WEEK_GAP * 2 + 1:
        raise ValueError(f"split too short: {weeks} whole weeks")
    while True:
        perm = rng.permutation(weeks)
        if np.all(np.abs(perm - np.arange(weeks)) >= MIN_WEEK_GAP):
            break
    donors = np.concatenate([perm[k] * week + np.arange(week) for k in range(weeks)])
    if rest:
        donor_week = int(rng.integers(0, weeks - MIN_WEEK_GAP + 1))
        donors = np.concatenate([donors, donor_week * week + np.arange(rest)])
    return donors


def random_content(payload, splits, target_nonempty, has_content, week, rng):
    n = len(splits)
    source = np.arange(n)
    fallback = 0
    for split in (0, 1, 2):
        idx = np.flatnonzero(splits == split)
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            raise ValueError("split is not contiguous")
        chosen = idx[0] + week_donors(len(idx), week, rng)
        usable = idx[has_content[idx]]
        for pos, donor in zip(idx, chosen):
            if target_nonempty[pos] and not has_content[donor]:
                donor = usable[np.argmin(np.abs(usable - donor))]
                fallback += 1
            source[pos] = donor
    out = {}
    for key in CONTENT_KEYS:
        real = np.asarray(payload[key])
        values = real[source].copy()
        values[~target_nonempty] = 0
        out[key] = values
    return out, source, fallback


def noise_content(payload, splits, target_nonempty, bins_per_day, rng):
    n = len(splits)
    train = (splits == 0) & target_nonempty
    pairs = np.flatnonzero(train[:-bins_per_day] & train[bins_per_day:])
    out = {}
    for key in CONTENT_KEYS:
        real = np.asarray(payload[key]).astype(np.float64)
        reference = real[train]
        mean, std = reference.mean(0), reference.std(0)
        low, high = reference.min(0), reference.max(0)
        a, b = real[pairs] - mean, real[pairs + bins_per_day] - mean
        denominator = np.sqrt((a * a).sum(0) * (b * b).sum(0))
        rho = np.divide((a * b).sum(0), denominator, out=np.zeros_like(mean), where=denominator > 0)
        phi = np.clip(rho, 0.01, 0.999) ** (1.0 / bins_per_day)
        innovation = np.sqrt(1.0 - phi * phi)
        state = rng.standard_normal(mean.shape)
        series = np.empty(real.shape)
        for t in range(n):
            series[t] = state
            state = phi * state + innovation * rng.standard_normal(mean.shape)
        series = np.clip(mean + std * series, low, high)
        series[~target_nonempty] = 0
        out[key] = series.astype(payload[key].dtype)
    return out


def build(dataset: str, seed: int) -> None:
    source_name, prefix, time_col, dayfirst = DATASETS[dataset]
    with np.load(CACHE / source_name, allow_pickle=False) as payload:
        base = {key: np.asarray(payload[key]).copy() for key in payload.files}
        metadata = json.loads(str(payload["metadata_json"].item()))
    dates = pd.to_datetime(pd.Series(base["dates"]))
    steps = dates.diff().dropna().unique()
    if len(steps) != 1:
        raise ValueError(f"{dataset}: bank is not on a regular grid: {steps[:5]}")
    bins_per_day = int(round(pd.Timedelta(days=1) / pd.Timedelta(steps[0])))
    week = 7 * bins_per_day
    splits = split_ids(pd.DatetimeIndex(dates), prefix, time_col, dayfirst)
    has_content = np.zeros(len(dates), dtype=bool)
    for key in CONTENT_KEYS:
        has_content |= np.any(base[key] != 0, axis=1)
    target_nonempty = has_content | (base["relevance_mass"] != 0) | (base["in_force_doc_count"] != 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in ("random", "noise"):
        rng = np.random.default_rng([seed, 0 if variant == "random" else 1])
        extra = {}
        if variant == "random":
            content, donors, fallback = random_content(base, splits, target_nonempty, has_content, week, rng)
            extra = {"placebo_block": "1 week", "placebo_min_gap_weeks": MIN_WEEK_GAP, "placebo_fallback_bins": fallback}
            np.save(OUT_DIR / f"{dataset}_random_donors.npy", donors)
        else:
            content = noise_content(base, splits, target_nonempty, bins_per_day, rng)
        out = {**base, **content}
        meta = dict(metadata, ablation_mode=f"{variant}_placebo_matched_mass", ablation_source_bank=source_name, ablation_seed=seed, **extra)
        out["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=True))
        np.savez_compressed(OUT_DIR / f"{dataset}_{variant}_placebo.npz", **out)
        print(OUT_DIR / f"{dataset}_{variant}_placebo.npz", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    for dataset in DATASETS:
        build(dataset, args.seed)


if __name__ == "__main__":
    main()
