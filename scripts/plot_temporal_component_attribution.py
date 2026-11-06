#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DEBUG_CSV = (
    "checkpoints/delta_v3_nsw_price_dlinear_[price]freeNews_reneweconomy_pvmag_watt_1_48_1e-4_0.7_intfloat-e5-small-v2_0.0001/"
    "test_delta_residual_debug_delta_v3_nsw_price_dlinear_[price]freeNews_reneweconomy_pvmag_watt_1_48_1e-4_0.7_intfloat-e5-small-v2_0.0001.csv"
)
DEFAULT_OUTPUT = "results/temporal_component_attribution_price_h48.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot REIN temporal component attribution from a residual debug CSV: "
            "slow/shape/spike correction over forecast horizon plus relevance/lambda controls."
        )
    )
    parser.add_argument("--debug-csv", default=DEFAULT_DEBUG_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-idx", type=int, default=None, help="Exact debug sample_idx to plot.")
    parser.add_argument(
        "--select",
        choices=("active_median_relevance", "active_top_relevance", "top_spike_component", "first_active", "first"),
        default="active_median_relevance",
        help="Sample selection strategy when --sample-idx is not provided.",
    )
    return parser.parse_args()


def parse_json_vector(value: str) -> np.ndarray:
    try:
        parsed = json.loads(str(value or "[]"))
    except Exception:
        parsed = []
    return np.asarray(parsed, dtype=np.float32).reshape(-1)


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def parse_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return int(default)


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_row(rows: list[dict[str, str]], sample_idx: int | None, select: str) -> dict[str, str]:
    if not rows:
        raise RuntimeError("Debug CSV has no rows.")
    if sample_idx is not None:
        for row in rows:
            if parse_int(row, "sample_idx", -1) == int(sample_idx):
                return row
        raise RuntimeError(f"No row with sample_idx={sample_idx}.")

    if select == "first":
        return rows[0]

    active_rows = [row for row in rows if parse_int(row, "regime_active", 0) > 0]
    if not active_rows:
        return rows[0]
    if select == "first_active":
        return active_rows[0]
    if select == "active_median_relevance":
        relevance = np.asarray([parse_float(row, "relevance_mass", 0.0) for row in active_rows], dtype=np.float32)
        median_relevance = float(np.median(relevance))
        return min(active_rows, key=lambda row: abs(parse_float(row, "relevance_mass", 0.0) - median_relevance))
    if select == "top_spike_component":
        def _spike_score(row: dict[str, str]) -> float:
            spike_component = parse_json_vector(row.get("spike_component_z", "[]"))
            if spike_component.size <= 0:
                spike_gate_mean = parse_float(row, "spike_gate_mean", 0.0)
                spike_component = spike_gate_mean * parse_json_vector(row.get("spike_ts", "[]"))
            return float(np.max(np.abs(spike_component))) if spike_component.size > 0 else 0.0

        return max(active_rows, key=_spike_score)
    return max(active_rows, key=lambda row: parse_float(row, "relevance_mass", 0.0))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.debug_csv)
    row = choose_row(rows, args.sample_idx, args.select)

    shape_ts = parse_json_vector(row.get("shape_ts", "[]"))
    shape_z = parse_json_vector(row.get("shape_z", "[]"))
    spike_ts = parse_json_vector(row.get("spike_ts", "[]"))
    spike_gate = parse_json_vector(row.get("spike_gate", "[]"))
    spike_component_z = parse_json_vector(row.get("spike_component_z", "[]"))
    residual_hat = parse_json_vector(row.get("residual_hat_z", "[]"))
    horizon = int(max(len(shape_ts), len(shape_z), len(spike_ts), len(spike_gate), len(spike_component_z), len(residual_hat)))
    if horizon <= 0:
        raise RuntimeError("Selected row has no horizon vectors to plot.")

    steps = np.arange(1, horizon + 1)
    slow_ts = parse_float(row, "slow_ts", 0.0)
    lambda_delta_final = parse_float(row, "lambda_delta_final", parse_float(row, "lambda_base", 1.0))
    shape_gain = parse_float(row, "shape_gain", 1.0)
    spike_gate_mean = parse_float(row, "spike_gate_mean", 0.0)
    relevance_mass = parse_float(row, "relevance_mass", 0.0)
    spike_bias = parse_float(row, "spike_bias", 0.0)
    hard_gate = parse_float(row, "hard_gate", 1.0)

    if len(shape_ts) != horizon:
        shape_ts = np.resize(shape_ts, horizon)
    has_exact_shape = len(shape_z) == horizon
    if not has_exact_shape:
        shape_z = shape_gain * shape_ts
    if len(spike_ts) != horizon:
        spike_ts = np.resize(spike_ts, horizon)
    has_exact_spike = len(spike_component_z) == horizon
    if not has_exact_spike:
        spike_component_z = spike_gate_mean * spike_ts
    if len(residual_hat) != horizon:
        residual_hat = np.resize(residual_hat, horizon)

    final_scale = hard_gate * lambda_delta_final
    slow_component = np.full((horizon,), final_scale * slow_ts, dtype=np.float32)
    shape_component = final_scale * shape_z
    spike_component = final_scale * spike_component_z
    spike_label = "spike component" if has_exact_spike else "spike component (mean-gate approx.)"

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.2),
        gridspec_kw={"height_ratios": [3.0, 1.15]},
        sharex=True,
    )

    ax_top.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.55)
    ax_top.plot(steps, slow_component, color="#3b6fb6", linewidth=2.2, linestyle="--", label="slow component")
    ax_top.plot(steps, shape_component, color="#2b9c73", linewidth=2.2, label="shape component")
    ax_top.plot(steps, spike_component, color="#c44e52", linewidth=2.0, linestyle=":", label=spike_label)
    ax_top.plot(steps, residual_hat, color="#111111", linewidth=2.8, alpha=0.95, label="model residual correction")
    ax_top.set_ylabel("residual correction magnitude")
    ax_top.set_title("REIN Text-Controlled Residual Correction by Temporal Component")
    ax_top.grid(True, axis="y", alpha=0.25)
    ax_top.legend(loc="upper right", frameon=True)

    ax_bottom.plot(steps, np.full((horizon,), relevance_mass), color="#0b3d91", linewidth=2.4, label="relevance_mass")
    ax_bottom.set_ylabel("relevance_mass")
    ax_bottom.grid(True, axis="y", alpha=0.25)
    ax_bottom.set_xlabel("forecast horizon steps")

    ax_lambda = ax_bottom.twinx()
    ax_lambda.plot(steps, np.full((horizon,), lambda_delta_final), color="#d14f3f", linewidth=2.4, label="lambda_delta_final")
    ax_lambda.set_ylabel("lambda_delta_final")
    rel_span = max(abs(relevance_mass) * 0.12, 1.0)
    lam_span = max(abs(lambda_delta_final) * 0.22, 0.05)
    ax_bottom.set_ylim(relevance_mass - 0.72 * rel_span, relevance_mass + 0.28 * rel_span)
    ax_lambda.set_ylim(lambda_delta_final - 0.32 * lam_span, lambda_delta_final + 0.68 * lam_span)

    bottom_lines, bottom_labels = ax_bottom.get_legend_handles_labels()
    lambda_lines, lambda_labels = ax_lambda.get_legend_handles_labels()
    ax_bottom.legend(bottom_lines + lambda_lines, bottom_labels + lambda_labels, loc="upper right", frameon=True)

    summary = (
        f"sample_idx={parse_int(row, 'sample_idx', -1)} | target_time={row.get('target_time', '')}\n"
        f"active={parse_int(row, 'regime_active', 0)} | docs={parse_int(row, 'regime_docs_used', 0)} | "
        f"relevance_mass={relevance_mass:.3f} | lambda_delta_final={lambda_delta_final:.3f}\n"
        f"shape_gain={shape_gain:.3f} | spike_bias={spike_bias:.3f} | "
        f"spike_gate_mean={spike_gate_mean:.3f} | hard_gate={hard_gate:.1f}"
    )
    ax_top.text(
        0.015,
        0.025,
        summary,
        transform=ax_top.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#444444", "alpha": 0.92},
    )

    source_note = f"debug_csv={Path(args.debug_csv).name}"
    fig.text(0.01, 0.01, source_note, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.025, 1, 1))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
