#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BANK = "_shared_refine_cache/v4/regime_bank_gdelt_bitcoin_2024.npz"
DEFAULT_REFINED = "_shared_refine_cache/v4/refined_gdelt_bitcoin_2024.jsonl"
DEFAULT_OUTPUT = "results/regime_ema_demo_gdelt_bitcoin_2024.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw regime-bank relevance aggregation against the saved causal EMA series."
    )
    parser.add_argument("--bank-npz", default=DEFAULT_BANK)
    parser.add_argument("--refined-jsonl", default=DEFAULT_REFINED)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="", help="Optional plot start, e.g. 2024-03-01")
    parser.add_argument("--end", default="", help="Optional plot end, e.g. 2024-05-01")
    return parser.parse_args()


def to_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def parse_news_timestamp(value) -> pd.Timestamp:
    text = str(value or "").strip()
    if not text:
        return pd.NaT
    for fmt in ("%d-%m-%Y %I:%M:%S %p", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return to_naive_timestamp(datetime.strptime(text, fmt))
        except Exception:
            pass
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, dayfirst=False, errors="coerce")
    return to_naive_timestamp(parsed)


def timestamp_text_has_time(value) -> bool:
    text = str(value or "").strip()
    return ":" in text or "AM" in text.upper() or "PM" in text.upper()


def load_refined_rows(path: str, *, is_subdaily: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not bool(row.get("is_actionable", False)):
                continue

            published_at = parse_news_timestamp(row.get("published_at", ""))
            if pd.isna(published_at):
                continue
            published_at = published_at.floor("s")
            if is_subdaily and not timestamp_text_has_time(row.get("published_at", "")):
                published_at = published_at.normalize() + pd.Timedelta(days=1)
            elif not is_subdaily:
                published_at = published_at.normalize()

            horizon_days = int(max(0, min(14, int(row.get("horizon_days", 0) or 0))))
            confidence = float(max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))))
            if horizon_days <= 0 or confidence <= 0.0:
                continue
            rows.append(
                {
                    "published_at": published_at,
                    "horizon_days": horizon_days,
                    "confidence": confidence,
                }
            )
    return rows


def reconstruct_raw_relevance(timestamps: pd.DatetimeIndex, rows: list[dict[str, object]], tau_days: float) -> np.ndarray:
    raw = np.zeros((len(timestamps),), dtype=np.float32)
    tau_days = float(max(1e-3, tau_days))
    for time_idx, timestamp in enumerate(timestamps):
        mass = 0.0
        for row in rows:
            published_at = row["published_at"]
            horizon_end = published_at + pd.Timedelta(days=int(row["horizon_days"]))
            if published_at <= timestamp <= horizon_end:
                age_days = max(0.0, float((timestamp - published_at) / pd.Timedelta(days=1)))
                mass += float(row["confidence"]) * math.exp(-age_days / tau_days)
        raw[time_idx] = mass
    return raw


def main() -> None:
    args = parse_args()
    with np.load(args.bank_npz, allow_pickle=False) as bank:
        dates = pd.to_datetime([str(x) for x in bank["dates"]], errors="coerce")
        ema_relevance = np.asarray(bank["relevance_mass"], dtype=np.float32).reshape(-1)
        metadata = json.loads(str(bank["metadata_json"].item())) if "metadata_json" in bank.files else {}

    freq_min = int(metadata.get("freq_min", 0) or 0)
    is_subdaily = 0 < freq_min < 1440
    tau_days = float(metadata.get("tau_days", 5.0) or 5.0)
    ema_alpha = float(metadata.get("ema_alpha", 0.5) or 0.5)

    rows = load_refined_rows(args.refined_jsonl, is_subdaily=is_subdaily)
    raw_relevance = reconstruct_raw_relevance(dates, rows, tau_days=tau_days)

    plot_mask = np.ones((len(dates),), dtype=bool)
    if args.start:
        plot_mask &= dates >= pd.Timestamp(args.start)
    if args.end:
        plot_mask &= dates <= pd.Timestamp(args.end)

    plot_dates = dates[plot_mask]
    raw_plot = raw_relevance[plot_mask]
    ema_plot = ema_relevance[plot_mask]

    if len(plot_dates) <= 0:
        raise ValueError("No timestamps left after applying --start/--end.")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(plot_dates, raw_plot, color="#0b3d91", linewidth=2.0, alpha=0.9, label="Raw relevance mass")
    ax.plot(plot_dates, ema_plot, color="#d14f3f", linewidth=3.0, label=f"Causal EMA, alpha={ema_alpha:g}")

    ax.set_title("Regime Bank Aggregation -> EMA Smoothing")
    ax.set_ylabel("relevance_mass")
    ax.set_xlabel("time bin")
    ax.grid(True, axis="y", alpha=0.25)
    legend = ax.legend(loc="upper right", frameon=True)
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.95)

    subtitle = (
        f"bank={Path(args.bank_npz).name} | refined_docs={len(rows)} | "
        f"tau_days={tau_days:g} | freq_min={freq_min}"
    )
    fig.text(0.01, 0.01, subtitle, fontsize=9, color="#555555")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
