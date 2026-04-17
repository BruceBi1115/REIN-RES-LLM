#!/usr/bin/env python
"""Phase-I diagnostic: does news carry any predictive signal for NSW price?

Tests, for each regime_bank × news source:
  1. Mutual information MI(news_features[t], price_target[t+lag]) for lags in {0,1,2,3} days
  2. Permutation test: shuffle date→news mapping and re-compute MI null distribution
  3. Report per-feature-group MI, p-values, and an overall verdict

Targets are daily price statistics (mean/max/p95/spike_count) and residuals after
a seasonal-naive baseline (last-week-same-DOW mean). Residual MI is the relevant
signal for delta's job (predict what base cannot).

Usage:
  python scripts/diagnose_news_price_signal.py \
      --regime_banks _shared_refine_cache/v4/regime_bank_pv_magazine_australia_news.npz \
                     _shared_refine_cache/v4/regime_bank_watt_pv_mag.npz \
                     _shared_refine_cache/v4/regime_bank_reneweconomy_web_stories_2024.npz \
      --train_file dataset/2024NSWelecPRICE/2024NSWelecPRICE_trainset.csv \
      --val_file   dataset/2024NSWelecPRICE/2024NSWelecPRICE_valset.csv \
      --test_file  dataset/2024NSWelecPRICE/2024NSWelecPRICE_testset.csv \
      --time_col date --value_col RRP --day_first \
      --output results/diagnose_news_price_signal.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


FEATURE_GROUPS = {
    "regime_vec": "regime_vec",
    "topic_tag_mass": "topic_tag_mass",
    "relevance_mass": "relevance_mass",
    "text_emb_pca8": "text_emb",  # reduced via PCA to 8 dims
}

TARGETS = ["mean", "max", "p95", "spike_count", "resid_mean", "resid_max"]


def load_regime_bank(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    dates = pd.to_datetime([str(x) for x in d["dates"]]).normalize()
    return {
        "dates": dates,
        "regime_vec": d["regime_vec"],
        "topic_tag_mass": d["topic_tag_mass"],
        "text_emb": d["text_emb"],
        "relevance_mass": d["relevance_mass"].reshape(-1, 1),
    }


def pca_reduce(X: np.ndarray, k: int) -> np.ndarray:
    if X.shape[0] <= 1 or X.shape[1] <= k:
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :k] * S[:k])


def load_price_daily(csv_paths: List[str], time_col: str, value_col: str, day_first: bool) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=day_first, errors="coerce")
        df = df.dropna(subset=[time_col, value_col])
        frames.append(df[[time_col, value_col]])
    if not frames:
        raise FileNotFoundError(f"No price files loaded from {csv_paths}")
    raw = pd.concat(frames, ignore_index=True).sort_values(time_col)
    raw["day"] = raw[time_col].dt.floor("D")

    agg = raw.groupby("day")[value_col].agg(
        mean="mean",
        max="max",
        p95=lambda s: float(np.quantile(s, 0.95)),
        std="std",
    )
    # spike_count per day: points exceeding 3×IQR above median
    def _spikes(s):
        med = float(np.median(s))
        iqr = float(np.quantile(s, 0.75) - np.quantile(s, 0.25))
        thr = med + 3.0 * max(iqr, 1e-6)
        return int((s > thr).sum())

    agg["spike_count"] = raw.groupby("day")[value_col].apply(_spikes)

    # seasonal-naive residual: today - mean(last-week same DOW, previous 3 weeks)
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


def align_features(bank: Dict[str, np.ndarray], price_daily: pd.DataFrame, lag: int) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    bank_dates = bank["dates"]
    # target day = news_day + lag
    target_dates = bank_dates + pd.Timedelta(days=lag)
    idx = price_daily.index
    mask_keep = target_dates.isin(idx)
    if mask_keep.sum() < 20:
        return {}, pd.DataFrame()

    news_idx = np.where(mask_keep)[0]
    target_idx = [idx.get_loc(d) for d in target_dates[mask_keep]]
    feats = {}
    feats["regime_vec"] = bank["regime_vec"][news_idx]
    feats["topic_tag_mass"] = bank["topic_tag_mass"][news_idx]
    feats["relevance_mass"] = bank["relevance_mass"][news_idx]
    feats["text_emb"] = pca_reduce(bank["text_emb"][news_idx], k=8)

    y = price_daily.iloc[target_idx][TARGETS].reset_index(drop=True)
    return feats, y


def mi_for_group(X: np.ndarray, y: np.ndarray, seed: int = 0, n_neighbors: int = 3) -> float:
    """Aggregate MI: mean MI across dims of X against scalar y."""
    mask = ~np.isnan(y)
    if mask.sum() < 20:
        return float("nan")
    X_ = X[mask]
    y_ = y[mask].astype(np.float64)
    if np.allclose(y_.std(), 0):
        return 0.0
    mi = mutual_info_regression(X_, y_, n_neighbors=n_neighbors, random_state=seed)
    return float(np.mean(mi))


def permutation_pvalue(X: np.ndarray, y: np.ndarray, n_perms: int, seed: int) -> Tuple[float, float, float]:
    """Return (true_mi, null_mean, p_value) where p = P(null >= true)."""
    true_mi = mi_for_group(X, y, seed=seed)
    if not np.isfinite(true_mi):
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(y)
    X_ = X[mask]
    y_ = y[mask].astype(np.float64)
    null = np.zeros(n_perms, dtype=np.float64)
    for i in range(n_perms):
        perm = rng.permutation(len(y_))
        mi = mutual_info_regression(X_, y_[perm], n_neighbors=3, random_state=i)
        null[i] = float(np.mean(mi))
    null_mean = float(null.mean())
    p = float(((null >= true_mi).sum() + 1) / (n_perms + 1))
    return true_mi, null_mean, p


def run_for_bank(bank_path: str, price_daily: pd.DataFrame, lags: List[int], n_perms: int, seed: int) -> Dict:
    bank = load_regime_bank(bank_path)
    out = {"bank_path": bank_path, "dates_covered": len(bank["dates"]), "per_lag": {}}
    for lag in lags:
        feats, y_df = align_features(bank, price_daily, lag)
        if not feats:
            out["per_lag"][lag] = {"error": "not enough aligned days"}
            continue
        per_target = {}
        for target in TARGETS:
            y = y_df[target].values
            per_group = {}
            for group_name, bank_key in FEATURE_GROUPS.items():
                if group_name == "text_emb_pca8":
                    X = feats["text_emb"]
                else:
                    X = feats[bank_key]
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                true_mi, null_mean, p = permutation_pvalue(X, y, n_perms=n_perms, seed=seed)
                per_group[group_name] = {
                    "mi": true_mi,
                    "null_mean": null_mean,
                    "p_value": p,
                    "significant": bool(p < 0.05 and np.isfinite(true_mi) and true_mi > null_mean),
                }
            per_target[target] = per_group
        out["per_lag"][lag] = per_target
    return out


def format_report(results: List[Dict]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("Phase-I diagnostic: MI(news → price) with permutation test")
    lines.append("=" * 100)
    for r in results:
        name = Path(r["bank_path"]).stem
        lines.append(f"\n## {name}  (dates_covered={r['dates_covered']})")
        for lag, per_target in r["per_lag"].items():
            if isinstance(per_target, dict) and "error" in per_target:
                lines.append(f"  lag={lag}d: {per_target['error']}")
                continue
            lines.append(f"  lag={lag}d")
            lines.append(
                f"    {'target':<14} {'group':<16} {'MI':>8} {'null':>8} {'p':>7} {'sig':>5}"
            )
            for target, per_group in per_target.items():
                for group_name, stats in per_group.items():
                    mi = stats["mi"]
                    nm = stats["null_mean"]
                    p = stats["p_value"]
                    sig = "YES" if stats["significant"] else "no"
                    mi_s = f"{mi:.4f}" if np.isfinite(mi) else "nan"
                    nm_s = f"{nm:.4f}" if np.isfinite(nm) else "nan"
                    p_s = f"{p:.3f}" if np.isfinite(p) else "nan"
                    lines.append(
                        f"    {target:<14} {group_name:<16} {mi_s:>8} {nm_s:>8} {p_s:>7} {sig:>5}"
                    )
    # verdict summary
    lines.append("\n" + "=" * 100)
    lines.append("Verdict")
    lines.append("=" * 100)
    for r in results:
        name = Path(r["bank_path"]).stem
        sig_counts = 0
        total = 0
        best_mi = 0.0
        best_loc = ""
        for lag, per_target in r["per_lag"].items():
            if not isinstance(per_target, dict) or "error" in per_target:
                continue
            for target, per_group in per_target.items():
                for group_name, stats in per_group.items():
                    total += 1
                    if stats["significant"]:
                        sig_counts += 1
                        if stats["mi"] > best_mi:
                            best_mi = stats["mi"]
                            best_loc = f"lag={lag}d target={target} group={group_name}"
        if total == 0:
            lines.append(f"  {name}: no data")
            continue
        pct = 100 * sig_counts / total
        hint = "news has meaningful signal" if pct > 20 else (
            "marginal signal" if pct > 5 else "signal indistinguishable from noise"
        )
        lines.append(
            f"  {name}: {sig_counts}/{total} significant ({pct:.1f}%)  "
            f"best MI={best_mi:.4f} at [{best_loc}]  -->  {hint}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime_banks", nargs="+", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", default="")
    p.add_argument("--test_file", default="")
    p.add_argument("--time_col", default="date")
    p.add_argument("--value_col", default="RRP")
    p.add_argument("--day_first", action="store_true")
    p.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n_perms", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/diagnose_news_price_signal.json")
    args = p.parse_args()

    price_paths = [q for q in [args.train_file, args.val_file, args.test_file] if q]
    price_daily = load_price_daily(price_paths, args.time_col, args.value_col, args.day_first)
    print(f"[info] price_daily spans {price_daily.index.min().date()} to {price_daily.index.max().date()} ({len(price_daily)} days)", file=sys.stderr)

    results = []
    for bank_path in args.regime_banks:
        if not os.path.exists(bank_path):
            print(f"[warn] skipping missing bank: {bank_path}", file=sys.stderr)
            continue
        print(f"[info] running on {bank_path}", file=sys.stderr)
        results.append(run_for_bank(bank_path, price_daily, args.lags, args.n_perms, args.seed))

    report = format_report(results)
    print(report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2, default=str)
    print(f"\n[info] saved JSON to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
