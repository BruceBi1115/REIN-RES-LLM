#!/usr/bin/env python
"""Build the isolated regime banks for the rebuttal experiments #2, #3 and #4.

Reads the refined corpus and the shipped banks, and writes only to rebuttal_ext_runs/banks/.
No LLM calls are made: every variant reuses the cached refined corpus.

  tau1, tau2, tau5, tau10  exponential decay constant in days (each dataset skips its submitted value)
  eventtau                 event-specific decay tau_i = h_i / 3, so an article reaches about 5% of its weight at the
                           end of its LLM-estimated duration (as tau = 5 does at the 14-day maximum)
  noconf                   confidence weighting removed (c_i = 1)
  latest                   content taken from the most recent in-force article; relevance mass unchanged
  real, blank              copies of the submitted and no-news banks

Before writing anything, the submitted settings are rebuilt and compared with the shipped bank for each
dataset, and the script aborts on any mismatch. Every output carries `ablation_mode`, so the trainer loads
it as is and never rebuilds a cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.delta_v3 import regime_bank as rb  # noqa: E402

CACHE = REPO_ROOT / "_shared_refine_cache" / "v4"
OUT_DIR = REPO_ROOT / "rebuttal_ext_runs" / "banks"
STEMS = {
    "gas_demand": "gdelt_gas_nl_2025",
    "nsw_load": "load_freeNews_reneweconomy_pvmag_watt",
    "nsw_price": "price_freeNews_reneweconomy_pvmag_watt",
    "traffic_count": "gdelt_2024",
}
TAU_GRID = (1.0, 2.0, 5.0, 10.0)
NS_PER_DAY = 86_400_000_000_000
TOLERANCE = 1e-4


@dataclass
class Docs:
    published_ns: np.ndarray
    end_ns: np.ndarray
    horizon_days: np.ndarray
    confidence: np.ndarray
    regime: np.ndarray
    topic: np.ndarray
    text: np.ndarray | None


def load_source(dataset: str) -> tuple[dict, dict, np.ndarray]:
    with np.load(CACHE / f"regime_bank_{STEMS[dataset]}.npz", allow_pickle=False) as payload:
        base = {key: np.asarray(payload[key]).copy() for key in payload.files}
    meta = json.loads(str(base["metadata_json"].item()))
    dates_ns = pd.to_datetime(pd.Series(base["dates"])).to_numpy("datetime64[ns]").astype("int64")
    return base, meta, dates_ns


def load_docs(dataset: str, meta: dict, with_text: bool = True) -> Docs:
    """Same article filter as rb.build_regime_bank."""
    freq = rb._effective_freq_minutes(meta["freq_min"])
    subdaily = 0 < int(freq) < 1440
    variant = meta["schema_variant"]
    published, horizon, confidence, regime, topic, summaries = [], [], [], [], [], []
    for row in rb._load_jsonl(str(CACHE / f"refined_{STEMS[dataset]}.jsonl")):
        if not bool(row.get("is_actionable", False)):
            continue
        when, has_time = rb._parse_news_timestamp_with_precision(row.get("published_at", ""))
        if pd.isna(when):
            continue
        when = when.floor("s")
        if subdaily and not has_time:
            when = when.normalize() + pd.Timedelta(days=1)
        elif not subdaily:
            when = when.normalize()
        days = int(max(0, min(rb.refinement_horizon_days(variant), int(row.get("horizon_days", 0) or 0))))
        conf = float(max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))))
        if days <= 0 or conf <= 0.0:
            continue
        payload = row.get("regime_vec", {}) if isinstance(row.get("regime_vec"), dict) else {}
        published.append(when.value)
        horizon.append(days)
        confidence.append(conf)
        regime.append([float(payload.get(key, 0.0) or 0.0) for key in rb.REGIME_KEYS])
        topic.append(rb._topic_mass(list(row.get("topic_tags", []) or [])))
        summaries.append(str(row.get("summary", "") or "").strip())
    text = None
    if with_text:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = OUT_DIR / f"_emb_{dataset}.npy"
        if cache_path.exists():
            text = np.load(cache_path)
        else:
            text = rb._encode_texts(summaries, meta["encoder_model_id"], max_length=int(meta["max_length"]))
            np.save(cache_path, text)
        if len(text) != len(summaries):
            raise ValueError(f"{dataset}: cached embeddings do not match the corpus ({len(text)} vs {len(summaries)})")
    published_ns = np.asarray(published, dtype=np.int64)
    return Docs(
        published_ns=published_ns,
        end_ns=published_ns + np.asarray(horizon, dtype=np.int64) * NS_PER_DAY,
        horizon_days=np.asarray(horizon, dtype=np.float64),
        confidence=np.asarray(confidence, dtype=np.float64),
        regime=np.asarray(regime, dtype=np.float64),
        topic=np.asarray(topic, dtype=np.float64),
        text=None if text is None else np.asarray(text, dtype=np.float64),
    )


def in_force_weights(docs: Docs, timestamp_ns: int, tau: float | None, confidence_weighting: bool):
    """Weights of the articles in force at a time bin. tau=None uses the event-specific decay h_i / 3."""
    idx = np.flatnonzero((docs.published_ns <= timestamp_ns) & (timestamp_ns <= docs.end_ns))
    if idx.size == 0:
        return idx, np.zeros(0)
    age_days = (timestamp_ns - docs.published_ns[idx]) / NS_PER_DAY
    factor = docs.confidence[idx] if confidence_weighting else 1.0
    decay = docs.horizon_days[idx] / 3.0 if tau is None else float(tau)
    return idx, factor * np.exp(-age_days / decay)


def aggregate(docs: Docs, dates_ns: np.ndarray, *, tau: float | None, confidence_weighting: bool, mode: str, alpha: float) -> dict:
    n = len(dates_ns)
    regime = np.zeros((n, docs.regime.shape[1]))
    topic = np.zeros((n, docs.topic.shape[1]))
    text = np.zeros((n, docs.text.shape[1]))
    mass = np.zeros(n)
    count = np.zeros(n, dtype=np.int64)
    for t, timestamp in enumerate(dates_ns):
        idx, weight = in_force_weights(docs, int(timestamp), tau, confidence_weighting)
        if idx.size == 0:
            continue
        raw = float(weight.sum())
        mass[t], count[t] = raw, idx.size
        if mode == "mean":
            share = weight / max(1e-6, raw)
            regime[t], topic[t], text[t] = share @ docs.regime[idx], share @ docs.topic[idx], share @ docs.text[idx]
        elif mode == "latest":
            newest = idx[np.lexsort((docs.confidence[idx], docs.published_ns[idx]))[-1]]
            regime[t], topic[t], text[t] = docs.regime[newest], docs.topic[newest], docs.text[newest]
        else:
            raise ValueError(mode)
    return {
        "regime_vec": rb._ema_smooth(regime.astype(np.float32), alpha),
        "topic_tag_mass": rb._ema_smooth(topic.astype(np.float32), alpha),
        "text_emb": rb._ema_smooth(text.astype(np.float32), alpha),
        "relevance_mass": rb._ema_smooth(mass.astype(np.float32)[:, None], alpha).reshape(-1),
        "in_force_doc_count": count,
    }


def check_reproduces(dataset: str, base: dict, built: dict) -> None:
    for key in ("regime_vec", "topic_tag_mass", "text_emb", "relevance_mass"):
        gap = float(np.abs(base[key].astype(np.float64) - built[key].astype(np.float64)).max())
        print(f"  check {dataset}.{key}: max_abs_diff={gap:.2e}", flush=True)
        if gap > TOLERANCE:
            raise SystemExit(f"ABORT: {dataset}.{key} does not reproduce the shipped bank (gap {gap:.3e})")
    if not np.array_equal(base["in_force_doc_count"].astype(np.int64), built["in_force_doc_count"]):
        raise SystemExit(f"ABORT: {dataset}.in_force_doc_count does not reproduce the shipped bank")


def write_bank(path: Path, base: dict, meta: dict, arrays: dict, **extra) -> None:
    out = {**base, **arrays}
    out["metadata_json"] = np.asarray(json.dumps({**meta, **extra}, ensure_ascii=True))
    np.savez_compressed(path, **out)
    print(f"  wrote {path.name}", flush=True)


def write_copy(dataset: str, name: str, source: Path, mode: str) -> None:
    with np.load(source, allow_pickle=False) as payload:
        data = {key: np.asarray(payload[key]).copy() for key in payload.files}
    meta = json.loads(str(data["metadata_json"].item()))
    meta.setdefault("ablation_mode", mode)
    data["metadata_json"] = np.asarray(json.dumps(meta, ensure_ascii=True))
    np.savez_compressed(OUT_DIR / f"{dataset}_{name}.npz", **data)
    print(f"  wrote {dataset}_{name}.npz from {source.name}", flush=True)


def build(dataset: str, check_only: bool) -> None:
    print(f"== {dataset}", flush=True)
    base, meta, dates_ns = load_source(dataset)
    docs = load_docs(dataset, meta)
    tau0, alpha = float(meta["tau_days"]), float(meta["ema_alpha"])
    check_reproduces(dataset, base, aggregate(docs, dates_ns, tau=tau0, confidence_weighting=True, mode="mean", alpha=alpha))
    if check_only:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tau in TAU_GRID:
        if tau == tau0:
            continue
        name = f"tau{tau:g}"
        arrays = aggregate(docs, dates_ns, tau=tau, confidence_weighting=True, mode="mean", alpha=alpha)
        write_bank(OUT_DIR / f"{dataset}_{name}.npz", base, meta, arrays, ablation_mode=f"sens_{name}", tau_days=tau)
    arrays = aggregate(docs, dates_ns, tau=None, confidence_weighting=True, mode="mean", alpha=alpha)
    write_bank(OUT_DIR / f"{dataset}_eventtau.npz", base, meta, arrays, ablation_mode="sens_eventtau", decay="event_specific_h_over_3")
    arrays = aggregate(docs, dates_ns, tau=tau0, confidence_weighting=False, mode="mean", alpha=alpha)
    write_bank(OUT_DIR / f"{dataset}_noconf.npz", base, meta, arrays, ablation_mode="sens_noconf", confidence_weighting=False)
    arrays = aggregate(docs, dates_ns, tau=tau0, confidence_weighting=True, mode="latest", alpha=alpha)
    write_bank(OUT_DIR / f"{dataset}_latest.npz", base, meta, arrays, ablation_mode="sens_latest", content_aggregation="latest")
    write_copy(dataset, "real", CACHE / f"regime_bank_{STEMS[dataset]}.npz", "direct_real")
    write_copy(dataset, "blank", CACHE / "ablations" / "true_nonews" / f"{dataset}_blank_news_features.npz", "blank_news_features")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only verify that the submitted banks are reproduced")
    args = parser.parse_args()
    for dataset in STEMS:
        build(dataset, args.check)


if __name__ == "__main__":
    main()
