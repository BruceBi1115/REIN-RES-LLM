"""Reviewer Q2: does REIN fall back to the numerical base when news is inactive?

For each of the paper's 20 checkpoints, on the news-INACTIVE test subset (true relevance mass <= theta)
and, for contrast, on the news-active subset, report:

  * MAE / MSE of Base, Active (REIN, true pack), Blank (zeroed pack) and Perm (paper protocol,
    whole pack date-permuted, seed 2024);
  * |y_REIN - y_base| in raw units and in sigma units (z-space): mean / p50 / p90 / p99 / max over
    all horizon steps, plus the mean over windows of the per-window max;
  * |y_REIN - y_Blank| (how different the true-pack and blank predictions are);
  * the realised trust coefficient lambda (mean / max) and the soft-gate value.

A run with no inactive windows gets n_windows = 0 and empty metric fields for that subset.

Usage: python q2_inactive_fallback.py --ckpt_root DIR --bank_dir DIR --out_csv FILE
       (run with cwd = a scratch work dir that contains a `dataset` symlink; see run_q2.sh)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reviewer_q1_semantic_cf"))
sys.dont_write_bytecode = True

from q1_semantic_counterfactuals import DATASET_OF, bank_index, load_run  # noqa: E402  (shared run loader)
from src.delta_v3.trainer import (  # noqa: E402
    _base_forward, _blank_regime_pack, _compute_residual_history, _sample_history_times, _z_batch_tensors,
)

CONTENT_KEYS = ("regime_vec", "topic_tag_mass", "text_emb")


def _stats(x, prefix):
    if x.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "p50", "p90", "p99", "max")}
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_p50": float(np.quantile(x, .5)),
            f"{prefix}_p90": float(np.quantile(x, .9)), f"{prefix}_p99": float(np.quantile(x, .99)),
            f"{prefix}_max": float(x.max())}


def evaluate(run):
    args, cfg, bundle, device = run["args"], run["cfg"], run["bundle"], run["device"]
    model, base, stats, bank = run["model"], run["base"], run["stats"], run["bank"]
    c, s = float(stats["center_global"]), float(stats["scale_global"])
    th = float(cfg.active_mass_threshold)
    n_bank = len(bank.dates)
    arr = {k: np.concatenate([np.asarray(getattr(bank, k), np.float32).reshape(n_bank, -1),
                              np.zeros((1, int(np.prod(np.asarray(getattr(bank, k)).shape[1:]) or 1)), np.float32)])
           for k in CONTENT_KEYS + ("relevance_mass",)}
    doc = np.concatenate([np.asarray(bank.in_force_doc_count).reshape(-1), [0]])
    perm = np.random.default_rng(int(cfg.eval_permutation_seed)).permutation(n_bank)  # == bank.shuffled(seed)

    def pack(idx, m):
        ok = idx >= 0
        i = np.where(ok, m[np.clip(idx, 0, n_bank - 1)], n_bank)
        p = {k: torch.tensor(arr[k][i], device=device) for k in CONTENT_KEYS + ("relevance_mass",)}
        p["in_force_doc_count"] = torch.tensor(doc[i], dtype=torch.long, device=device)
        return p

    ident = np.arange(n_bank)
    keep = {k: [] for k in ["true", "base", "active", "blank", "perm", "lam", "gate", "mass"]}
    with torch.no_grad():
        for batch in bundle["test_loader"]:
            hz, _, _ = _z_batch_tensors(batch, args, global_zstats=stats)
            hz = hz.to(device)
            ht = _sample_history_times(batch)
            bp, bh = _base_forward(base, hz)
            kw = dict(history_z=hz, history_times=ht, base_pred_z=bp,
                      base_hidden=bh if model.cfg.use_base_hidden else None,
                      residual_history=_compute_residual_history(hz) if cfg.residual_history_channel else None)
            idx = np.asarray([bank_index(bank, t) for t in batch["target_time"]])
            real = pack(idx, ident)
            out = model(regime_pack=real, **kw)
            keep["active"].append(out["pred_z"].cpu().numpy())
            keep["blank"].append(model(regime_pack=_blank_regime_pack(real), **kw)["pred_z"].cpu().numpy())
            keep["perm"].append(model(regime_pack=pack(idx, perm), **kw)["pred_z"].cpu().numpy())
            keep["base"].append(bp.cpu().numpy())
            keep["true"].append(batch["target_value"].numpy().astype(np.float32))
            keep["lam"].append(out["lambda_delta_final"].cpu().numpy())
            mass = real["relevance_mass"].view(-1)
            mh = model.modulation_heads
            keep["gate"].append(torch.sigmoid((mass - mh.active_mass_threshold) / mh.active_soft_tau).cpu().numpy())
            keep["mass"].append(mass.cpu().numpy())
    K = {k: np.concatenate(v) for k, v in keep.items()}
    raw = {k: K[k] * s + c for k in ("base", "active", "blank", "perm")}
    y = K["true"].reshape(raw["base"].shape)
    inactive = K["mass"] <= th
    # sanity check against the paper log (full-test REIN MAE and active-subset Blank/Perm MAE)
    lg = run["logged"]
    act = ~inactive
    repro = bool(act.any() and abs(np.abs(raw["active"][act] - y[act]).mean() - lg["active"]) / lg["active"] < 1e-4
                 and abs(np.abs(raw["perm"][act] - y[act]).mean() - lg["perm"]) / lg["perm"] < 1e-4)
    rows = []
    for subset, m in (("news_inactive", inactive), ("news_active", act)):
        r = {"dataset": DATASET_OF.get(cfg.dataset_key, cfg.dataset_key), "dataset_key": cfg.dataset_key,
             "horizon": int(cfg.horizon), "theta": th, "lambda_min": float(cfg.lambda_min),
             "lambda_max": float(cfg.lambda_max), "subset": subset, "n_windows": int(m.sum()),
             "n_test_windows": int(m.size), "frac_of_test": float(m.mean()),
             "reproduces_paper_log": repro}
        for k in ("base", "active", "blank", "perm"):
            r[f"{k}_mae"] = float(np.abs(raw[k][m] - y[m]).mean()) if m.any() else float("nan")
            r[f"{k}_mse"] = float(((raw[k][m] - y[m]) ** 2).mean()) if m.any() else float("nan")
        r["active_vs_base_mae_pct"] = (100 * (r["base_mae"] - r["active_mae"]) / r["base_mae"]) if m.any() else float("nan")
        dev = np.abs(raw["active"][m] - raw["base"][m])
        r.update(_stats(dev.reshape(-1), "abs_rein_minus_base_raw"))
        r.update(_stats((dev / s).reshape(-1), "abs_rein_minus_base_sigma"))
        r["abs_rein_minus_base_raw_window_max_mean"] = float(dev.max(-1).mean()) if m.any() else float("nan")
        r["rel_abs_rein_minus_base_vs_base_mae"] = (float(dev.mean()) / r["base_mae"]) if m.any() else float("nan")
        db = np.abs(raw["active"][m] - raw["blank"][m])
        r.update(_stats(db.reshape(-1), "abs_rein_minus_blank_raw"))
        r["lambda_mean"] = float(K["lam"][m].mean()) if m.any() else float("nan")
        r["lambda_max_observed"] = float(K["lam"][m].max()) if m.any() else float("nan")
        r["soft_gate_mean"] = float(K["gate"][m].mean()) if m.any() else float("nan")
        r["relevance_mass_mean"] = float(K["mass"][m].mean()) if m.any() else float("nan")
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--save_dir", default="./_run_logs")
    a = ap.parse_args()
    runs = sorted(d for d in os.listdir(a.ckpt_root)
                  if os.path.isdir(os.path.join(a.ckpt_root, d)) and not d.startswith("time_mmd"))
    print(f"[Q2] {len(runs)} runs under {a.ckpt_root}", flush=True)
    all_rows = []
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    for i, d in enumerate(runs, 1):
        run = load_run(os.path.join(a.ckpt_root, d), a.bank_dir, a.save_dir)
        rows = evaluate(run)
        all_rows += rows
        pd.DataFrame(all_rows).to_csv(a.out_csv, index=False)
        ia = rows[0]
        print(f"[Q2] ({i}/{len(runs)}) {run['cfg'].dataset_key} H={run['cfg'].horizon} repro={ia['reproduces_paper_log']} "
              f"inactive n={ia['n_windows']} ({100 * ia['frac_of_test']:.1f}%) base={ia['base_mae']:.4g} "
              f"rein={ia['active_mae']:.4g} |rein-base| mean={ia['abs_rein_minus_base_raw_mean']:.4g} "
              f"p99={ia['abs_rein_minus_base_raw_p99']:.4g} lambda={ia['lambda_mean']:.3f}", flush=True)
        del run
        torch.cuda.empty_cache()
    print(f"[Q2] wrote {a.out_csv}", flush=True)


if __name__ == "__main__":
    main()
