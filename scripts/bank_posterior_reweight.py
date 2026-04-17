#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from src.delta_v3.schema_refine_v2 import REGIME_KEYS, TOPIC_TAGS


def load_bank(path: str) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    dates = pd.to_datetime([str(x) for x in payload["dates"]]).normalize()

    if "regime_vec_short" in payload.files:
        short_mass = np.asarray(payload["relevance_mass_short"], dtype=np.float32).reshape(-1, 1)
        mid_mass = np.asarray(payload["relevance_mass_mid"], dtype=np.float32).reshape(-1, 1)
        long_mass = np.asarray(payload["relevance_mass_long"], dtype=np.float32).reshape(-1, 1)
        total_mass = (short_mass + mid_mass + long_mass).clip(min=1e-6)
        regime_vec = (
            np.asarray(payload["regime_vec_short"], dtype=np.float32) * short_mass
            + np.asarray(payload["regime_vec_mid"], dtype=np.float32) * mid_mass
            + np.asarray(payload["regime_vec_long"], dtype=np.float32) * long_mass
        ) / total_mass
        topic_tag_mass = (
            np.asarray(payload["topic_tag_mass_short"], dtype=np.float32) * short_mass
            + np.asarray(payload["topic_tag_mass_mid"], dtype=np.float32) * mid_mass
            + np.asarray(payload["topic_tag_mass_long"], dtype=np.float32) * long_mass
        ) / total_mass
        relevance_mass = np.asarray(payload["relevance_mass"], dtype=np.float32).reshape(-1, 1)
    else:
        regime_vec = np.asarray(payload["regime_vec"], dtype=np.float32)
        topic_tag_mass = np.asarray(payload["topic_tag_mass"], dtype=np.float32)
        relevance_mass = np.asarray(payload["relevance_mass"], dtype=np.float32).reshape(-1, 1)

    return {
        "dates": dates,
        "regime_vec": regime_vec,
        "topic_tag_mass": topic_tag_mass,
        "relevance_mass": relevance_mass,
    }


def load_daily_targets(csv_paths: list[str], time_col: str, value_col: str, day_first: bool) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        if not path:
            continue
        df = pd.read_csv(path)
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=day_first, errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[time_col, value_col])
        frames.append(df[[time_col, value_col]])
    if not frames:
        raise FileNotFoundError("No usable series files were provided.")

    raw = pd.concat(frames, axis=0, ignore_index=True).sort_values(time_col)
    raw["day"] = raw[time_col].dt.floor("D")
    agg = raw.groupby("day")[value_col].agg(
        mean="mean",
        max="max",
        p95=lambda s: float(np.quantile(s, 0.95)),
    )

    def _spikes(s):
        med = float(np.median(s))
        iqr = float(np.quantile(s, 0.75) - np.quantile(s, 0.25))
        thr = med + 3.0 * max(iqr, 1e-6)
        return int((s > thr).sum())

    agg["spike_count"] = raw.groupby("day")[value_col].apply(_spikes)
    agg = agg.reset_index().sort_values("day")
    agg["dow"] = agg["day"].dt.dayofweek
    agg["resid_mean"] = np.nan
    agg["resid_max"] = np.nan
    for dow in range(7):
        mask = agg["dow"] == dow
        sub = agg.loc[mask, ["mean", "max"]].rolling(window=3, min_periods=2).mean().shift(1)
        agg.loc[mask, "resid_mean"] = agg.loc[mask, "mean"] - sub["mean"]
        agg.loc[mask, "resid_max"] = agg.loc[mask, "max"] - sub["max"]
    return agg.set_index("day")


def score_dimensions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    if mask.sum() < 20:
        return np.zeros((X.shape[1],), dtype=np.float32)
    X_ = np.asarray(X[mask], dtype=np.float64)
    y_ = np.asarray(y[mask], dtype=np.float64)
    if np.allclose(np.std(y_), 0.0):
        return np.zeros((X.shape[1],), dtype=np.float32)
    scores = mutual_info_regression(X_, y_, n_neighbors=3, random_state=0)
    return np.asarray(scores, dtype=np.float32)


def align(bank: dict[str, np.ndarray], daily: pd.DataFrame, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_dates = bank["dates"] + pd.Timedelta(days=int(lag))
    keep = target_dates.isin(daily.index)
    if keep.sum() < 20:
        return np.zeros((0, bank["regime_vec"].shape[1]), dtype=np.float32), np.zeros((0, bank["topic_tag_mass"].shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    idx = [daily.index.get_loc(day) for day in target_dates[keep]]
    y = daily.iloc[idx]
    return (
        bank["regime_vec"][keep],
        bank["topic_tag_mass"][keep],
        y[["resid_mean", "resid_max", "spike_count", "mean", "max"]].to_numpy(dtype=np.float32),
    )


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size <= 0:
        return arr
    pivot = float(np.median(arr))
    arr = np.maximum(0.0, arr - pivot)
    peak = float(arr.max())
    if peak <= 1e-8:
        return np.ones_like(arr, dtype=np.float32)
    return arr / peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime_bank", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", default="")
    parser.add_argument("--test_file", default="")
    parser.add_argument("--time_col", default="date")
    parser.add_argument("--value_col", default="value")
    parser.add_argument("--day_first", action="store_true", default=False)
    parser.add_argument("--lag_max", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank = load_bank(args.regime_bank)
    daily = load_daily_targets(
        [args.train_file, args.val_file, args.test_file],
        time_col=args.time_col,
        value_col=args.value_col,
        day_first=bool(args.day_first),
    )

    regime_scores = np.zeros((len(REGIME_KEYS),), dtype=np.float32)
    topic_scores = np.zeros((len(TOPIC_TAGS),), dtype=np.float32)
    for lag in range(max(0, int(args.lag_max)) + 1):
        regime_x, topic_x, y_mat = align(bank, daily, lag)
        if regime_x.shape[0] <= 0:
            continue
        for col_idx in range(y_mat.shape[1]):
            regime_scores = np.maximum(regime_scores, score_dimensions(regime_x, y_mat[:, col_idx]))
            topic_scores = np.maximum(topic_scores, score_dimensions(topic_x, y_mat[:, col_idx]))

    regime_weights = normalize_scores(regime_scores)
    topic_weights = normalize_scores(topic_scores)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        regime_key_weights=regime_weights.astype(np.float32),
        topic_tag_weights=topic_weights.astype(np.float32),
        regime_keys=np.asarray(REGIME_KEYS, dtype="<U32"),
        topic_tags=np.asarray(TOPIC_TAGS, dtype="<U32"),
    )
    report = {
        "regime_bank": args.regime_bank,
        "output": str(out_path),
        "regime_key_weights": {key: float(val) for key, val in zip(REGIME_KEYS, regime_weights)},
        "topic_tag_weights": {key: float(val) for key, val in zip(TOPIC_TAGS, topic_weights)},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
