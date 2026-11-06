#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.delta_v3.targets import ResidualTargetDecomposer


OUTPUT_DIR_DEFAULT = "results/synthetic_component_validation"
FREQ_DEFAULT = "30min"
START_TIME_DEFAULT = "2024-01-01 00:00:00"

DATASET_PRESETS: dict[str, dict[str, str]] = {
    "price": {
        "dataset": "price",
        "dataset_key": "nsw_price",
        "freq": "30min",
        "start_time": "2024-01-01 00:00:00",
    },
    "load": {
        "dataset": "load",
        "dataset_key": "nsw_load",
        "freq": "30min",
        "start_time": "2024-01-01 00:00:00",
    },
    "gas_demand": {
        "dataset": "gas_demand",
        "dataset_key": "gas_demand",
        "freq": "60min",
        "start_time": "2025-01-01 00:00:00",
    },
    "traffic": {
        "dataset": "traffic",
        "dataset_key": "traffic_count",
        "freq": "60min",
        "start_time": "2024-01-01 00:00:00",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate REIN's slow/shape/spike residual target decomposition under "
            "controlled synthetic perturbations with known component-level ground truth."
        )
    )
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "price", "load", "gas_demand", "traffic", "all"],
        default="synthetic",
        help=(
            "Dataset preset for calendar frequency/output naming. Use 'all' to run "
            "price, load, gas_demand, and traffic."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["price", "load", "gas_demand", "traffic"],
        default=None,
        help="Explicit dataset preset list; overrides --dataset.",
    )
    parser.add_argument("--freq", default=FREQ_DEFAULT)
    parser.add_argument("--start-time", default=START_TIME_DEFAULT)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--level-amp", type=float, default=1.0)
    parser.add_argument("--shape-amp", type=float, default=1.0)
    parser.add_argument("--spike-amp", type=float, default=4.0)
    parser.add_argument("--num-spikes", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument(
        "--spike-threshold-abs",
        type=float,
        default=2.0,
        help="Absolute spike threshold used by ResidualTargetDecomposer.",
    )
    parser.add_argument("--spike-target-pct", type=float, default=0.10)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return list(dict.fromkeys(args.datasets))
    if args.dataset == "all":
        return ["price", "load", "gas_demand", "traffic"]
    return [str(args.dataset)]


def dataset_config(args: argparse.Namespace, dataset: str) -> dict[str, str]:
    if dataset == "synthetic":
        return {
            "dataset": "synthetic",
            "dataset_key": "synthetic",
            "freq": str(args.freq),
            "start_time": str(args.start_time),
        }
    cfg = dict(DATASET_PRESETS[dataset])
    if str(args.freq) != FREQ_DEFAULT:
        cfg["freq"] = str(args.freq)
    if str(args.start_time) != START_TIME_DEFAULT:
        cfg["start_time"] = str(args.start_time)
    return cfg


def make_target_times(num_samples: int, horizon: int, start_time: str, freq: str) -> list[list[str]]:
    base = pd.Timestamp(start_time)
    offset = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
    rows: list[list[str]] = []
    for sample_idx in range(num_samples):
        sample_start = base + sample_idx * horizon * offset
        times = pd.date_range(sample_start, periods=horizon, freq=freq)
        rows.append([ts.strftime("%Y-%m-%d %H:%M:%S") for ts in times])
    return rows


def centered_shape_pattern(horizon: int, phase: float, harmonic: int = 1) -> np.ndarray:
    t = np.arange(horizon, dtype=np.float32)
    pattern = np.sin(2.0 * np.pi * harmonic * t / float(horizon) + float(phase))
    pattern += 0.35 * np.cos(4.0 * np.pi * t / float(horizon) + 0.5 * float(phase))
    pattern = pattern - float(pattern.mean())
    norm = float(np.sqrt(np.mean(pattern**2)))
    return (pattern / max(norm, 1e-6)).astype(np.float32)


def make_components(args: argparse.Namespace, mode: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = int(args.num_samples)
    h = int(args.horizon)
    level = np.zeros((n, h), dtype=np.float32)
    shape = np.zeros((n, h), dtype=np.float32)
    spike = np.zeros((n, h), dtype=np.float32)
    spike_mask = np.zeros((n, h), dtype=bool)

    if mode in {"level", "mixed"}:
        signs = rng.choice([-1.0, 1.0], size=(n, 1)).astype(np.float32)
        amps = rng.uniform(0.75, 1.25, size=(n, 1)).astype(np.float32) * float(args.level_amp)
        level = np.repeat(signs * amps, h, axis=1).astype(np.float32)

    if mode in {"shape", "mixed"}:
        for i in range(n):
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            harmonic = int(rng.choice([1, 2]))
            sign = float(rng.choice([-1.0, 1.0]))
            amp = float(rng.uniform(0.75, 1.25)) * float(args.shape_amp)
            shape[i] = sign * amp * centered_shape_pattern(h, phase=phase, harmonic=harmonic)

    if mode in {"spike", "mixed"}:
        k = max(1, min(h, int(args.num_spikes)))
        for i in range(n):
            idx = rng.choice(h, size=k, replace=False)
            signs = rng.choice([-1.0, 1.0], size=k).astype(np.float32)
            amps = rng.uniform(0.85, 1.15, size=k).astype(np.float32) * float(args.spike_amp)
            spike[i, idx] = signs * amps
            spike_mask[i, idx] = True

    noise = np.zeros((n, h), dtype=np.float32)
    if float(args.noise_std) > 0:
        noise = rng.normal(0.0, float(args.noise_std), size=(n, h)).astype(np.float32)

    residual = level + shape + spike + noise
    return {
        "residual": residual.astype(np.float32),
        "level": level.astype(np.float32),
        "shape": shape.astype(np.float32),
        "spike": spike.astype(np.float32),
        "spike_mask": spike_mask,
    }


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if int(valid.sum()) < 2:
        return float("nan")
    aa = aa[valid]
    bb = bb[valid]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def energy(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(np.mean(arr * arr))


def spike_scores(pred_mask: np.ndarray, true_mask: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)
    tp = float(np.logical_and(pred, true).sum())
    fp = float(np.logical_and(pred, ~true).sum())
    fn = float(np.logical_and(~pred, true).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def summarize(
    dataset: str,
    dataset_key: str,
    mode: str,
    truth: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
) -> dict[str, float | str]:
    true_level_scalar = truth["level"].mean(axis=-1)
    pred_slow_scalar = pred["slow"]
    true_shape = truth["shape"]
    pred_shape = pred["shape"]
    true_spike = truth["spike"]
    pred_spike = pred["spike"]
    true_spike_mask = truth["spike_mask"]
    pred_spike_mask = pred["spike_mask"].astype(bool)

    intended_energy = {
        "level": energy(truth["level"]),
        "shape": energy(truth["shape"]),
        "spike": energy(truth["spike"]),
        "mixed": energy(truth["residual"]),
    }[mode]
    denom = max(intended_energy, 1e-12)
    precision, recall, f1 = spike_scores(pred_spike_mask, true_spike_mask)

    return {
        "dataset": dataset,
        "dataset_key": dataset_key,
        "mode": mode,
        "slow_mae": float(np.mean(np.abs(pred_slow_scalar - true_level_scalar))),
        "slow_corr": corr_flat(pred_slow_scalar, true_level_scalar),
        "shape_mae": float(np.mean(np.abs(pred_shape - true_shape))),
        "shape_corr": corr_flat(pred_shape, true_shape),
        "spike_mae": float(np.mean(np.abs(pred_spike - true_spike))),
        "spike_precision": precision,
        "spike_recall": recall,
        "spike_f1": f1,
        "pred_slow_energy_ratio": energy(pred_slow_scalar[:, None]) / denom,
        "pred_shape_energy_ratio": energy(pred_shape) / denom,
        "pred_spike_energy_ratio": energy(pred_spike) / denom,
    }


def plot_example(
    out_path: Path,
    mode: str,
    truth: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    sample_idx: int = 0,
) -> None:
    steps = np.arange(1, truth["residual"].shape[-1] + 1)
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.5), sharex=True)

    axes[0].plot(steps, truth["residual"][sample_idx], color="#111111", linewidth=2.4, label="synthetic residual")
    axes[0].set_title(f"Synthetic Component Validation: {mode}")
    axes[0].legend(loc="upper right")

    axes[1].plot(steps, truth["level"][sample_idx], color="#8b8b8b", linewidth=2.0, label="true level")
    axes[1].plot(
        steps,
        np.full_like(steps, pred["slow"][sample_idx], dtype=np.float32),
        color="#3b6fb6",
        linewidth=2.2,
        linestyle="--",
        label="decomposed slow",
    )
    axes[1].legend(loc="upper right")

    axes[2].plot(steps, truth["shape"][sample_idx], color="#8b8b8b", linewidth=2.0, label="true shape")
    axes[2].plot(steps, pred["shape"][sample_idx], color="#2b9c73", linewidth=2.2, label="decomposed shape")
    axes[2].legend(loc="upper right")

    axes[3].stem(
        steps,
        truth["spike"][sample_idx],
        linefmt="#8b8b8b",
        markerfmt="o",
        basefmt=" ",
        label="true spike",
    )
    axes[3].plot(steps, pred["spike"][sample_idx], color="#c44e52", linewidth=2.0, linestyle=":", label="decomposed spike")
    axes[3].scatter(
        steps[pred["spike_mask"][sample_idx].astype(bool)],
        pred["spike"][sample_idx][pred["spike_mask"][sample_idx].astype(bool)],
        color="#c44e52",
        s=28,
        zorder=4,
        label="predicted spike mask",
    )
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("forecast horizon step")

    for ax in axes:
        ax.axhline(0.0, color="#222222", linewidth=0.7, alpha=0.45)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("z residual")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def resolve_output_dir(args: argparse.Namespace, dataset: str, is_multi: bool) -> Path:
    base = Path(args.output_dir)
    if args.output_dir != OUTPUT_DIR_DEFAULT:
        return base / dataset if is_multi else base
    if dataset == "synthetic":
        return Path(f"{OUTPUT_DIR_DEFAULT}_h{int(args.horizon)}")
    return Path(f"{OUTPUT_DIR_DEFAULT}_{dataset}_h{int(args.horizon)}")


def run_one_dataset(args: argparse.Namespace, dataset: str, is_multi: bool) -> list[dict[str, float | str]]:
    cfg = dataset_config(args, dataset)
    rng = np.random.default_rng(int(args.seed))
    output_dir = resolve_output_dir(args, dataset, is_multi)
    output_dir.mkdir(parents=True, exist_ok=True)

    decomposer = ResidualTargetDecomposer(
        dow_hod_mean=np.zeros((7, 48), dtype=np.float32),
        spike_sigma=1.0,
        spike_threshold_abs=float(args.spike_threshold_abs),
        spike_k=3.0,
        spike_target_pct=float(args.spike_target_pct),
    )
    target_times = make_target_times(
        int(args.num_samples),
        int(args.horizon),
        cfg["start_time"],
        cfg["freq"],
    )

    rows: list[dict[str, float | str]] = []
    examples: dict[str, dict[str, str | list[float] | list[int]]] = {}
    for mode in ["level", "shape", "spike", "mixed"]:
        truth = make_components(args, mode, rng)
        residual = torch.tensor(truth["residual"], dtype=torch.float32)
        parts = decomposer.decompose(residual, target_times)
        pred = {
            "slow": parts["slow"].detach().cpu().numpy().astype(np.float32),
            "shape": parts["shape"].detach().cpu().numpy().astype(np.float32),
            "spike": parts["spike"].detach().cpu().numpy().astype(np.float32),
            "spike_mask": parts["spike_mask"].detach().cpu().numpy().astype(bool),
        }
        rows.append(summarize(cfg["dataset"], cfg["dataset_key"], mode, truth, pred))
        plot_example(output_dir / f"{mode}_example.png", mode, truth, pred, sample_idx=0)
        examples[mode] = {
            "dataset": cfg["dataset"],
            "dataset_key": cfg["dataset_key"],
            "freq": cfg["freq"],
            "start_time": cfg["start_time"],
            "synthetic_residual": truth["residual"][0].astype(float).tolist(),
            "true_level": truth["level"][0].astype(float).tolist(),
            "true_shape": truth["shape"][0].astype(float).tolist(),
            "true_spike": truth["spike"][0].astype(float).tolist(),
            "decomposed_slow": [float(pred["slow"][0])] * int(args.horizon),
            "decomposed_shape": pred["shape"][0].astype(float).tolist(),
            "decomposed_spike": pred["spike"][0].astype(float).tolist(),
            "decomposed_spike_mask": pred["spike_mask"][0].astype(int).tolist(),
        }

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "examples.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(examples, handle, ensure_ascii=True, indent=2)

    print(f"Wrote metrics: {csv_path}")
    print(f"Wrote examples: {json_path}")
    for row in rows:
        printable = {
            key: (round(value, 4) if isinstance(value, float) and math.isfinite(value) else value)
            for key, value in row.items()
        }
        print(json.dumps(printable, ensure_ascii=True))
    return rows


def write_corr_long_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    corr_fields = [key for key in rows[0].keys() if key.endswith("_corr")]
    long_rows: list[dict[str, float | str]] = []
    for row in rows:
        for field in corr_fields:
            long_rows.append(
                {
                    "dataset": row["dataset"],
                    "dataset_key": row["dataset_key"],
                    "mode": row["mode"],
                    "component": field[: -len("_corr")],
                    "corr": row[field],
                }
            )

    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "dataset_key", "mode", "component", "corr"])
        writer.writeheader()
        writer.writerows(long_rows)


def write_intended_summary_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    labels = {
        "level": "Level shift",
        "shape": "Shape distortion",
        "spike": "Localized spike",
        "mixed": "Mixed effects",
    }
    intended_metrics = {
        "level": [
            ("Slow channel", "Slow correlation", "slow_corr"),
            ("Slow channel", "Slow energy ratio", "pred_slow_energy_ratio"),
        ],
        "shape": [
            ("Shape channel", "Shape correlation", "shape_corr"),
            ("Shape channel", "Shape energy ratio", "pred_shape_energy_ratio"),
        ],
        "spike": [
            ("Spike channel", "Spike precision", "spike_precision"),
            ("Spike channel", "Spike recall", "spike_recall"),
            ("Spike channel", "Spike F1", "spike_f1"),
        ],
        "mixed": [
            ("Slow channel", "Slow correlation", "slow_corr"),
            ("Shape channel", "Shape correlation", "shape_corr"),
            ("Spike channel", "Spike F1", "spike_f1"),
        ],
    }

    summary_rows: list[dict[str, float | str]] = []
    for row in rows:
        mode = str(row["mode"])
        for intended_effect, metric_name, field in intended_metrics[mode]:
            summary_rows.append(
                {
                    "dataset": row["dataset"],
                    "dataset_key": row["dataset_key"],
                    "synthetic_setting": labels[mode],
                    "intended_effect": intended_effect,
                    "main_validation_metric": metric_name,
                    "result": row[field],
                }
            )

    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "dataset_key",
                "synthetic_setting",
                "intended_effect",
                "main_validation_metric",
                "result",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    datasets = selected_datasets(args)
    is_multi = len(datasets) > 1
    all_rows: list[dict[str, float | str]] = []
    for dataset in datasets:
        all_rows.extend(run_one_dataset(args, dataset, is_multi=is_multi))

    if is_multi:
        summary_dir = Path(args.output_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"metrics_all_h{int(args.horizon)}.csv"
        with open(summary_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Wrote aggregate metrics: {summary_path}")
        corr_long_path = summary_dir / f"metrics_corr_long_h{int(args.horizon)}.csv"
        write_corr_long_csv(all_rows, corr_long_path)
        print(f"Wrote aggregate correlations: {corr_long_path}")
        intended_path = summary_dir / f"component_validation_intended_h{int(args.horizon)}.csv"
        write_intended_summary_csv(all_rows, intended_path)
        print(f"Wrote intended-effect summary: {intended_path}")


if __name__ == "__main__":
    main()
