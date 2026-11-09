"""Reviewer Q1: semantic-only counterfactuals for the paper's 20 REIN checkpoints.

Every variant below keeps the timestamps, the relevance mass (hence the activity flag and the soft
gate) and the calendar position of each test window fixed, and perturbs only the semantic content
(regime_vec, topic_tag_mass, text_emb). The original full-pack date permutation is re-run with
several seeds for comparison.

  active                      true regime pack (reference)
  perm_full        (5 seeds)  paper protocol: whole pack incl. mass permuted over dates (seed 2024 = paper)
  content_global   (5 seeds)  content permuted over all dates, mass fixed
  content_month    (5 seeds)  content permuted only among bins in the same calendar month and same
                              time-of-day slot, mass fixed
  content_weekday  (5 seeds)  content permuted only among bins with the same weekday and same
                              time-of-day slot, mass fixed
  content_lag_kd              content taken from k days EARLIER (k = 1, 3, 7, 14), mass fixed
  content_lead_kd             content taken from k days LATER   (k = 1, 3, 7, 14), mass fixed

Metrics are computed on the news-active test subset (true mass > theta, the paper's subset) and on
the full test set. delta_* > 0 means the true pack beats the counterfactual. CIs: moving-block
bootstrap over test windows (block length = horizon), 1000 resamples.

Usage: python q1_semantic_counterfactuals.py --ckpt_root DIR --bank_dir DIR --out_csv FILE
       (run with cwd = a scratch work dir that contains a `dataset` symlink; see run_q1.sh)
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.dont_write_bytecode = True

from src.test_from_checkpoint import (  # noqa: E402
    _build_parser, _load_delta_cfg, _sync_args_from_base_meta, _sync_args_from_delta_cfg,
)
from src.base.common import _coerce_global_zstats  # noqa: E402
from src.base.stage import setup_env_and_data  # noqa: E402
from src.base_backbone import load_base_backbone_checkpoint  # noqa: E402
from src.delta_v3.model import build_delta_v3_model  # noqa: E402
from src.delta_v3.regime_bank import _format_axis_timestamp, _freq_alias, _to_naive_timestamp  # noqa: E402
from src.delta_v3.trainer import (  # noqa: E402
    _base_forward, _compute_residual_history, _prepare_regime_bank, _sample_history_times, _z_batch_tensors,
)

SEEDS = [2024, 1, 2, 3, 4]
SHIFT_DAYS = [1, 3, 7, 14]
CONTENT_KEYS = ("regime_vec", "topic_tag_mass", "text_emb")
DATASET_OF = {"nsw_load": "Electricity Demand", "nsw_price": "Electricity Price",
              "traffic_count": "Traffic", "gas_demand": "Gas Demand"}


# ------------------------------------------------------------------------------ run loading
def _parse_run_args(log_path):
    out, inside = {}, False
    for line in open(log_path, encoding="utf-8"):
        if "[RUN_ARGS] ----" in line:
            if inside:
                break
            inside = True
            continue
        m = re.match(r".*\|\s+- ([A-Za-z0-9_]+): (.*)$", line.rstrip("\n"))
        if inside and m:
            out[m.group(1)] = m.group(2)
    return out


def _cast_like(default, raw):
    if isinstance(default, bool):
        return raw.strip() in {"1", "True", "true"}
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


def _logged_counterfactual(log_path):
    line = [l for l in open(log_path, encoding="utf-8") if "[TEST][COUNTERFACTUAL]" in l][-1]
    get = lambda k: float(re.search(rf" {k}=([-\d.e+na]+)", line).group(1))
    return {"active": get("active_mae"), "blank": get("blank_active"), "perm": get("perm_active")}


def load_run(run_dir, bank_dir, save_dir):
    log = os.path.join(run_dir, [f for f in os.listdir(run_dir) if f.endswith(".log")][0])
    ra = _parse_run_args(log)
    args = _build_parser().parse_args(["--train_file", ra["train_file"], "--val_file", ra["val_file"],
                                       "--test_file", ra["test_file"]])
    for k, v in ra.items():
        if k != "news_api_key_path" and hasattr(args, k):
            setattr(args, k, _cast_like(getattr(args, k), v))
    args.save_dir, args.stage, args.news_api_enable, args.eval_progress_bar = save_dir, "all", 0, 0
    base_path = os.path.join(run_dir, [d for d in os.listdir(run_dir) if d.startswith("best_base")][0])
    delta_path = os.path.join(run_dir, [d for d in os.listdir(run_dir) if d.startswith("best_delta")][0])
    _sync_args_from_base_meta(args, base_path)
    cfg = _load_delta_cfg(delta_path, args)
    _sync_args_from_delta_cfg(args, cfg)
    cfg.cache_dir = cfg.legacy_cache_dir = bank_dir
    cfg.regime_bank_path = os.path.join(bank_dir, os.path.basename(cfg.regime_bank_path))
    cfg.regime_bank_legacy_path = os.path.join(bank_dir, "__none__.npz")
    if not os.path.exists(cfg.regime_bank_path):
        raise FileNotFoundError(cfg.regime_bank_path)
    bundle = setup_env_and_data(args)
    device = bundle["device"]
    base, bmeta = load_base_backbone_checkpoint(base_path, device=device, is_trainable=False)
    stats = _coerce_global_zstats(bmeta, args, required=False) or _coerce_global_zstats(
        bundle.get("global_zstats"), args, required=True)
    bank, _, used = _prepare_regime_bank(args, bundle, cfg)
    if os.path.abspath(used) != os.path.abspath(cfg.regime_bank_path):
        raise RuntimeError(f"unexpected bank {used}")
    model = build_delta_v3_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(delta_path, "model.pt"), map_location="cpu")["state_dict"])
    model.eval(); base.eval()
    return dict(args=args, cfg=cfg, bundle=bundle, device=device, base=base, stats=stats, bank=bank,
                model=model, logged=_logged_counterfactual(log))


def bank_index(bank, target_time):
    """Same resolution logic as RegimeBank.lookup, returning the row index (-1 if none)."""
    ts = _to_naive_timestamp(target_time)
    if pd.isna(ts):
        return -1
    idx = bank._lookup.get(_format_axis_timestamp(ts, bank.freq_min))
    if idx is None and bank._is_subdaily:
        idx = bank._lookup.get(_format_axis_timestamp(ts.floor(_freq_alias(bank.freq_min)), bank.freq_min))
    if idx is None and bank._is_subdaily and bank._timestamp_ns.size > 0:
        valid = bank._timestamp_ns[bank._timestamp_ns != np.iinfo(np.int64).min]
        pos = int(np.searchsorted(valid, int(ts.value), side="right") - 1)
        if pos >= 0:
            idx = pos
    return -1 if idx is None else int(idx)


# ------------------------------------------------------------------------------ variants
def group_permutation(groups, rng):
    """Permutation of 0..N-1 that only moves rows within the same group id."""
    perm = np.arange(len(groups))
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    bounds = np.flatnonzero(np.diff(g_sorted)) + 1
    for block in np.split(order, bounds):
        perm[block] = block[rng.permutation(len(block))]
    return perm


def build_variants(bank):
    n = len(bank.dates)
    t = pd.to_datetime(pd.Series(bank.dates))
    tod = (t.dt.hour * 60 + t.dt.minute).to_numpy()
    month_key = ((t.dt.year * 12 + t.dt.month) * 1440 + tod).to_numpy()
    wday_key = (t.dt.dayofweek * 1440 + tod).to_numpy()
    step_min = max(1, int(round(pd.Series(t).diff().median().total_seconds() / 60)))
    bins_per_day = int(round(1440 / step_min))
    ident = np.arange(n)
    v = []  # (family, label, content_map, mass_map)
    for s in SEEDS:
        full = np.random.default_rng(int(s)).permutation(n)  # identical to RegimeBank.shuffled(s)
        v.append(("perm_full", f"seed{s}", full, full))
        v.append(("content_global", f"seed{s}", np.random.default_rng([s, 1]).permutation(n), ident))
        v.append(("content_month", f"seed{s}", group_permutation(month_key, np.random.default_rng([s, 2])), ident))
        v.append(("content_weekday", f"seed{s}", group_permutation(wday_key, np.random.default_rng([s, 3])), ident))
    for k in SHIFT_DAYS:
        off = k * bins_per_day
        v.append(("content_lag", f"{k}d", np.clip(ident - off, 0, n - 1), ident))
        v.append(("content_lead", f"{k}d", np.clip(ident + off, 0, n - 1), ident))
    return v


def block_bootstrap_ci(d, block, n_boot=1000, seed=0):
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    block = int(max(1, min(block, n)))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        means.append(d[idx].mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ------------------------------------------------------------------------------ evaluation
def evaluate(run):
    args, cfg, bundle, device = run["args"], run["cfg"], run["bundle"], run["device"]
    model, base, stats, bank = run["model"], run["base"], run["stats"], run["bank"]
    c, s = float(stats["center_global"]), float(stats["scale_global"])
    th = float(cfg.active_mass_threshold)
    arr = {k: np.concatenate([np.asarray(getattr(bank, k), np.float32).reshape(len(bank.dates), -1),
                              np.zeros((1, int(np.prod(np.asarray(getattr(bank, k)).shape[1:]) or 1)), np.float32)])
           for k in CONTENT_KEYS + ("relevance_mass",)}
    doc = np.concatenate([np.asarray(bank.in_force_doc_count).reshape(-1), [0]])
    variants = build_variants(bank)
    n_bank = len(bank.dates)

    def pack_from(idx, cmap, mmap):
        ok = idx >= 0
        ci = np.where(ok, cmap[np.clip(idx, 0, n_bank - 1)], n_bank)
        mi = np.where(ok, mmap[np.clip(idx, 0, n_bank - 1)], n_bank)
        p = {k: torch.tensor(arr[k][ci], device=device) for k in CONTENT_KEYS}
        p["relevance_mass"] = torch.tensor(arr["relevance_mass"][mi], device=device)
        p["in_force_doc_count"] = torch.tensor(doc[mi], dtype=torch.long, device=device)
        return p

    ident = np.arange(n_bank)
    names = ["active"] + [f"{f}:{l}" for f, l, _, _ in variants]
    ae = {n: [] for n in names}
    se = {n: [] for n in names}
    act = []
    with torch.no_grad():
        for batch in bundle["test_loader"]:
            hz, tz, _ = _z_batch_tensors(batch, args, global_zstats=stats)
            hz = hz.to(device)
            true = batch["target_value"].to(device, dtype=torch.float32)
            ht = _sample_history_times(batch)
            bp, bh = _base_forward(base, hz)
            kw = dict(history_z=hz, history_times=ht, base_pred_z=bp,
                      base_hidden=bh if model.cfg.use_base_hidden else None,
                      residual_history=_compute_residual_history(hz) if cfg.residual_history_channel else None)
            idx = np.asarray([bank_index(bank, t) for t in batch["target_time"]])
            real = pack_from(idx, ident, ident)
            act.append((real["relevance_mass"].view(-1) > th).cpu().numpy())
            for name, (cmap, mmap) in [("active", (ident, ident))] + [
                    (f"{f}:{l}", (cm, mm)) for f, l, cm, mm in variants]:
                pr = model(regime_pack=pack_from(idx, cmap, mmap), **kw)["pred_z"] * s + c
                ae[name].append((pr - true).abs().mean(-1).cpu().numpy())
                se[name].append(((pr - true) ** 2).mean(-1).cpu().numpy())
    A = np.concatenate(act)
    AE = {k: np.concatenate(v) for k, v in ae.items()}
    SE = {k: np.concatenate(v) for k, v in se.items()}
    # reproduce the paper's logged numbers as a sanity check (seed 2024 full permutation)
    lg = run["logged"]
    repro_ok = (abs(AE["active"][A].mean() - lg["active"]) / lg["active"] < 1e-4 and
                abs(AE["perm_full:seed2024"][A].mean() - lg["perm"]) / lg["perm"] < 1e-4)
    rows = []
    H = int(cfg.horizon)
    for subset, m in (("news_active", A), ("full_test", np.ones_like(A))):
        if not m.any():
            continue
        a_mae, a_mse = AE["active"][m].mean(), SE["active"][m].mean()
        for name in names:
            fam, _, lab = name.partition(":")
            mae, mse = AE[name][m].mean(), SE[name][m].mean()
            lo, hi = block_bootstrap_ci(AE[name][m] - AE["active"][m], H) if name != "active" else (0.0, 0.0)
            rows.append({
                "dataset": DATASET_OF.get(cfg.dataset_key, cfg.dataset_key), "dataset_key": cfg.dataset_key,
                "horizon": H, "theta": th, "subset": subset, "n_windows": int(m.sum()),
                "variant_family": fam, "variant": lab or "true_pack",
                "mae": float(mae), "mse": float(mse),
                "delta_mae_pct": float(100 * (mae - a_mae) / a_mae),
                "delta_mse_pct": float(100 * (mse - a_mse) / a_mse),
                "delta_mae_abs": float(mae - a_mae), "delta_mae_ci95_low": lo, "delta_mae_ci95_high": hi,
                "frac_windows_true_better": float((AE[name][m] > AE["active"][m]).mean()) if name != "active" else float("nan"),
                "reproduces_paper_log": bool(repro_ok),
            })
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    df = df[df["variant_family"] != "active"].copy()
    df["ci_excludes_zero"] = ((df["delta_mae_ci95_low"] > 0) | (df["delta_mae_ci95_high"] < 0)).astype(float)
    g = df.groupby(["dataset", "dataset_key", "horizon", "subset", "variant_family"], sort=False)
    return g.agg(n_variants=("variant", "count"), n_windows=("n_windows", "first"),
                 delta_mae_pct_mean=("delta_mae_pct", "mean"), delta_mae_pct_std=("delta_mae_pct", "std"),
                 delta_mae_pct_min=("delta_mae_pct", "min"), delta_mae_pct_max=("delta_mae_pct", "max"),
                 delta_mse_pct_mean=("delta_mse_pct", "mean"),
                 frac_variants_ci_excludes_zero=("ci_excludes_zero", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--save_dir", default="./_run_logs")
    a = ap.parse_args()
    runs = sorted(d for d in os.listdir(a.ckpt_root)
                  if os.path.isdir(os.path.join(a.ckpt_root, d)) and not d.startswith("time_mmd"))
    print(f"[Q1] {len(runs)} runs under {a.ckpt_root}", flush=True)
    all_rows = []
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    for i, d in enumerate(runs, 1):
        run = load_run(os.path.join(a.ckpt_root, d), a.bank_dir, a.save_dir)
        rows = evaluate(run)
        all_rows += rows
        pd.DataFrame(all_rows).to_csv(a.out_csv, index=False)  # rewrite after each run (partial results survive)
        r0 = [r for r in rows if r["subset"] == "news_active"] or rows
        msg = " ".join(f"{fam}={np.mean([r['delta_mae_pct'] for r in r0 if r['variant_family'] == fam]):+.2f}%"
                       for fam in ["perm_full", "content_global", "content_month", "content_weekday",
                                   "content_lag", "content_lead"])
        print(f"[Q1] ({i}/{len(runs)}) {run['cfg'].dataset_key} H={run['cfg'].horizon} "
              f"repro={rows[0]['reproduces_paper_log']} {msg}", flush=True)
        del run
        torch.cuda.empty_cache()
    summ = summarize(all_rows)
    summ_path = a.out_csv.replace(".csv", "_summary.csv")
    summ.to_csv(summ_path, index=False)
    print(f"[Q1] wrote {a.out_csv} and {summ_path}", flush=True)


if __name__ == "__main__":
    main()
