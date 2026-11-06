#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_HORIZON = 48
DEFAULT_TRAIN_FILE = "dataset/2024NSWelecLOAD/2024NSWelecLOAD_trainset.csv"
DEFAULT_VALUE_COL = "TOTALDEMAND"
DEFAULT_OUTPUT_DIR = "results/load_base_target_correction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot NSW load base-forecast vs target vs REIN-corrected forecast for one horizon. "
            "This is intended to clarify what the residual correction is correcting."
        )
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--debug-csv", default="", help="Path to test_delta_residual_debug_*.csv.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--value-col", default=DEFAULT_VALUE_COL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-idx", type=int, default=-1, help="Sample index to plot. Negative selects best improvement.")
    parser.add_argument("--num-panels", type=int, default=4, help="Number of small-multiple examples to plot.")
    return parser.parse_args()


def default_debug_csv(horizon: int) -> Path:
    task = (
        "delta_v3_nsw_load_dlinear_[load]freeNews_reneweconomy_pvmag_watt_"
        f"1_{int(horizon)}_1e-4_0.7_intfloat-e5-small-v2_0.001"
    )
    return Path("checkpoints") / task / f"test_delta_residual_debug_{task}.csv"


def parse_array(value: str) -> np.ndarray:
    parsed = ast.literal_eval(str(value))
    return np.asarray(parsed, dtype=np.float32)


def parse_times(value: str) -> pd.DatetimeIndex:
    parsed = ast.literal_eval(str(value))
    return pd.to_datetime(parsed, errors="coerce")


def train_zstats(train_file: str, value_col: str) -> tuple[float, float]:
    df = pd.read_csv(train_file)
    values = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if values.size <= 0:
        raise ValueError(f"No numeric values found in {train_file}:{value_col}")
    center = float(values.mean())
    scale = max(float(values.std()), 1e-6)
    return center, scale


def denorm(values_z: np.ndarray, center: float, scale: float) -> np.ndarray:
    return np.asarray(values_z, dtype=np.float32) * float(scale) + float(center)


def load_windows(debug_csv: Path, center: float, scale: float) -> list[dict[str, object]]:
    df = pd.read_csv(debug_csv)
    required = {"sample_idx", "target_times", "pred_z", "base_pred_z", "true_residual_z", "residual_hat_z"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"{debug_csv} is missing required columns: {missing}")

    windows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        times = parse_times(row["target_times"])
        pred_z = parse_array(row["pred_z"])
        base_z = parse_array(row["base_pred_z"])
        true_z = base_z + parse_array(row["true_residual_z"])
        correction_z = parse_array(row["residual_hat_z"])
        pred = denorm(pred_z, center, scale)
        base = denorm(base_z, center, scale)
        true = denorm(true_z, center, scale)
        correction = correction_z * float(scale)
        base_mae = float(np.mean(np.abs(base - true)))
        pred_mae = float(np.mean(np.abs(pred - true)))
        windows.append(
            {
                "sample_idx": int(row["sample_idx"]),
                "times": times,
                "base": base,
                "pred": pred,
                "true": true,
                "correction": correction,
                "base_mae": base_mae,
                "pred_mae": pred_mae,
                "mae_gain": base_mae - pred_mae,
                "relevance_mass": float(row.get("relevance_mass", 0.0)),
                "lambda_delta_final": float(row.get("lambda_delta_final", float("nan"))),
                "shape_gain": float(row.get("shape_gain", float("nan"))),
            }
        )
    return windows


def choose_window(windows: list[dict[str, object]], sample_idx: int) -> dict[str, object]:
    if sample_idx >= 0:
        for window in windows:
            if int(window["sample_idx"]) == int(sample_idx):
                return window
        raise ValueError(f"sample_idx={sample_idx} was not found in debug CSV.")
    return max(windows, key=lambda item: float(item["mae_gain"]))


def format_axes(ax) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))


def plot_single(window: dict[str, object], out_path: Path) -> None:
    times = window["times"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(times, window["true"], color="#111111", linewidth=2.4, label="Target")
    axes[0].plot(times, window["base"], color="#7f7f7f", linewidth=2.1, linestyle="--", label="Base forecast")
    axes[0].plot(times, window["pred"], color="#1f77b4", linewidth=2.2, label="REIN corrected")
    axes[0].set_ylabel("NSW load (MW)")
    axes[0].legend(loc="upper left", ncols=3)
    axes[0].set_title(
        "NSW Electricity Load: Base Forecast, Target, and REIN Correction "
        f"(H={len(times)}, sample={window['sample_idx']})"
    )
    format_axes(axes[0])

    axes[1].axhline(0.0, color="#333333", linewidth=0.8, alpha=0.65)
    axes[1].plot(times, window["correction"], color="#d62728", linewidth=2.0, label="REIN residual correction")
    axes[1].set_ylabel("Correction (MW)")
    axes[1].set_xlabel("Target time")
    axes[1].legend(loc="upper left")
    format_axes(axes[1])

    note = (
        f"Base MAE={float(window['base_mae']):.1f} MW | "
        f"REIN MAE={float(window['pred_mae']):.1f} MW | "
        f"MAE gain={float(window['mae_gain']):.1f} MW | "
        f"relevance={float(window['relevance_mass']):.3f}"
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_panels(windows: list[dict[str, object]], out_path: Path, num_panels: int) -> None:
    selected = sorted(windows, key=lambda item: float(item["mae_gain"]), reverse=True)[: max(1, int(num_panels))]
    cols = 2
    rows = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.8 * rows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")

    for ax, window in zip(axes.reshape(-1), selected):
        ax.axis("on")
        times = window["times"]
        ax.plot(times, window["true"], color="#111111", linewidth=2.0, label="Target")
        ax.plot(times, window["base"], color="#7f7f7f", linewidth=1.8, linestyle="--", label="Base")
        ax.plot(times, window["pred"], color="#1f77b4", linewidth=1.9, label="REIN")
        ax.set_title(
            f"sample={window['sample_idx']} | gain={float(window['mae_gain']):.1f} MW | "
            f"rel={float(window['relevance_mass']):.2f}",
            fontsize=10,
        )
        ax.set_ylabel("MW")
        format_axes(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3)
    fig.suptitle("NSW Load Examples Where REIN Corrects the Base Forecast", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(window: dict[str, object], debug_csv: Path, out_path: Path) -> None:
    payload = {
        "debug_csv": str(debug_csv),
        "selected_sample_idx": int(window["sample_idx"]),
        "horizon": int(len(window["times"])),
        "base_mae_mw": float(window["base_mae"]),
        "rein_mae_mw": float(window["pred_mae"]),
        "mae_gain_mw": float(window["mae_gain"]),
        "relevance_mass": float(window["relevance_mass"]),
        "lambda_delta_final": float(window["lambda_delta_final"]),
        "shape_gain": float(window["shape_gain"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def main() -> None:
    args = parse_args()
    debug_csv = Path(args.debug_csv) if args.debug_csv else default_debug_csv(int(args.horizon))
    if not debug_csv.exists():
        raise FileNotFoundError(
            f"Debug CSV not found: {debug_csv}\n"
            "Run the non-ablation NSW load experiment for one horizon first, e.g.\n"
            "  HORIZONS_CSV=48 bash scripts/run_nsw_load.sh\n"
            "or pass an existing file with --debug-csv."
        )

    center, scale = train_zstats(args.train_file, args.value_col)
    windows = load_windows(debug_csv, center, scale)
    if not windows:
        raise RuntimeError(f"No windows found in {debug_csv}")
    selected = choose_window(windows, int(args.sample_idx))

    out_dir = Path(args.output_dir)
    plot_single(selected, out_dir / f"nsw_load_h{int(args.horizon)}_base_target_correction.png")
    plot_panels(windows, out_dir / f"nsw_load_h{int(args.horizon)}_base_target_correction_examples.png", int(args.num_panels))
    write_summary(selected, debug_csv, out_dir / f"nsw_load_h{int(args.horizon)}_base_target_correction_summary.json")

    print(f"Wrote {out_dir / f'nsw_load_h{int(args.horizon)}_base_target_correction.png'}")
    print(f"Wrote {out_dir / f'nsw_load_h{int(args.horizon)}_base_target_correction_examples.png'}")
    print(f"Wrote {out_dir / f'nsw_load_h{int(args.horizon)}_base_target_correction_summary.json'}")


if __name__ == "__main__":
    main()
