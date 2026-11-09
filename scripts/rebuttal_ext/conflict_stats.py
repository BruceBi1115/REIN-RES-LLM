#!/usr/bin/env python
"""Experiment #4 diagnostic: how often in-force articles disagree, and whether errors are larger there.

A time bin is a conflict bin if, on any signed regime axis (tightness, demand_outlook, renewable_surplus), both
the positive and the negative articles hold at least --min_share of the bin's total weight. Non-negative axes
cannot conflict in this sense. Weights use the submitted tau and confidence weighting.

For each submitted checkpoint, news-active test windows are split by the conflict flag of their bank bin, and
REIN and frozen-base MAE are reported on each group. Evaluation only: no training, and only --out_csv is written.

Run with cwd = a scratch directory that has a `dataset` symlink, as for the reviewer_q1 and reviewer_q2 scripts:
  python conflict_stats.py --ckpt_root DIR --bank_dir DIR --out_csv FILE --save_dir DIR [--filter gas_demand]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "reviewer_q1_semantic_cf"))
sys.dont_write_bytecode = True

from make_sensitivity_banks import in_force_weights, load_docs, load_source  # noqa: E402
from q1_semantic_counterfactuals import DATASET_OF, bank_index, load_run  # noqa: E402
from src.delta_v3 import regime_bank as rb  # noqa: E402
from src.delta_v3.trainer import (  # noqa: E402
    _base_forward, _compute_residual_history, _sample_history_times, _z_batch_tensors,
)

SIGNED_AXES = ("tightness", "demand_outlook", "renewable_surplus")
CONTENT_KEYS = ("regime_vec", "topic_tag_mass", "text_emb", "relevance_mass")


def conflict_flags(dataset: str, min_share: float) -> np.ndarray:
    _, meta, dates_ns = load_source(dataset)
    docs = load_docs(dataset, meta, with_text=False)
    axes = [list(rb.REGIME_KEYS).index(key) for key in SIGNED_AXES]
    flags = np.zeros(len(dates_ns), dtype=bool)
    tau = float(meta["tau_days"])
    for t, timestamp in enumerate(dates_ns):
        idx, weight = in_force_weights(docs, int(timestamp), tau, True)
        if idx.size < 2:
            continue
        total = float(weight.sum())
        for axis in axes:
            values = docs.regime[idx, axis]
            if min(float(weight[values > 0].sum()), float(weight[values < 0].sum())) >= min_share * total:
                flags[t] = True
                break
    return flags


def window_errors(run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    args, cfg, bundle, device = run["args"], run["cfg"], run["bundle"], run["device"]
    model, base, stats, bank = run["model"], run["base"], run["stats"], run["bank"]
    center, scale = float(stats["center_global"]), float(stats["scale_global"])
    n_bank = len(bank.dates)
    arrays = {key: np.concatenate([np.asarray(getattr(bank, key), np.float32).reshape(n_bank, -1),
                                   np.zeros((1, int(np.prod(np.asarray(getattr(bank, key)).shape[1:]) or 1)), np.float32)])
              for key in CONTENT_KEYS}
    counts = np.concatenate([np.asarray(bank.in_force_doc_count).reshape(-1), [0]])
    keep = {"idx": [], "rein": [], "base": []}
    with torch.no_grad():
        for batch in bundle["test_loader"]:
            history, _, _ = _z_batch_tensors(batch, args, global_zstats=stats)
            history = history.to(device)
            base_pred, base_hidden = _base_forward(base, history)
            idx = np.asarray([bank_index(bank, t) for t in batch["target_time"]])
            rows = np.where(idx >= 0, idx, n_bank)
            pack = {key: torch.tensor(arrays[key][rows], device=device) for key in CONTENT_KEYS}
            pack["in_force_doc_count"] = torch.tensor(counts[rows], dtype=torch.long, device=device)
            out = model(
                history_z=history,
                history_times=_sample_history_times(batch),
                base_pred_z=base_pred,
                base_hidden=base_hidden if model.cfg.use_base_hidden else None,
                residual_history=_compute_residual_history(history) if cfg.residual_history_channel else None,
                regime_pack=pack,
            )
            truth = batch["target_value"].numpy().astype(np.float32)
            rein = out["pred_z"].cpu().numpy() * scale + center
            plain = base_pred.cpu().numpy() * scale + center
            truth = truth.reshape(rein.shape)
            keep["idx"].append(idx)
            keep["rein"].append(np.abs(rein - truth).mean(-1))
            keep["base"].append(np.abs(plain - truth).mean(-1))
    return tuple(np.concatenate(keep[key]) for key in ("idx", "rein", "base"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", required=True)
    parser.add_argument("--bank_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--min_share", type=float, default=0.2)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    flags_of, rows = {}, []
    for name in sorted(os.listdir(args.ckpt_root)):
        if name.startswith("time_mmd") or args.filter not in name:
            continue
        run = load_run(os.path.join(args.ckpt_root, name), args.bank_dir, args.save_dir)
        cfg = run["cfg"]
        if cfg.dataset_key not in flags_of:
            flags_of[cfg.dataset_key] = conflict_flags(cfg.dataset_key, args.min_share)
        flags = flags_of[cfg.dataset_key]
        idx, rein, base = window_errors(run)
        # float32, as in the model: a mass equal to theta must not count as active.
        mass = np.asarray(run["bank"].relevance_mass, np.float32).reshape(-1)
        ok = idx >= 0
        active = np.zeros(len(idx), dtype=bool)
        active[ok] = mass[idx[ok]] > np.float32(cfg.active_mass_threshold)
        conflict = np.zeros(len(idx), dtype=bool)
        conflict[ok] = flags[idx[ok]]
        row = {"dataset": DATASET_OF.get(cfg.dataset_key, cfg.dataset_key), "horizon": int(cfg.horizon),
               "n_active": int(active.sum()), "conflict_share": float(conflict[active].mean()) if active.any() else float("nan")}
        for label, mask in (("conflict", active & conflict), ("no_conflict", active & ~conflict)):
            row[f"n_{label}"] = int(mask.sum())
            row[f"rein_mae_{label}"] = float(rein[mask].mean()) if mask.any() else float("nan")
            row[f"base_mae_{label}"] = float(base[mask].mean()) if mask.any() else float("nan")
        rows.append(row)
        print(row, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_csv}")
    share = pd.DataFrame(rows).groupby("dataset").conflict_share.mean().round(4).to_dict()
    print("mean conflict share of news-active windows:", share)


if __name__ == "__main__":
    main()
