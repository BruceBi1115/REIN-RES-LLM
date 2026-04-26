from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from .schema_refine_v2 import REGIME_KEYS, TOPIC_TAGS, compute_news_corpus_signature, schema_version_for_variant


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _encode_texts(texts: list[str], model_id: str, max_length: int, batch_size: int = 32) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start : start + max(1, batch_size)]
            prefixed = [f"passage: {text}" for text in chunk]
            enc = tokenizer(
                prefixed,
                padding=True,
                truncation=True,
                max_length=max(1, int(max_length)),
                return_tensors="pt",
            )
            out = model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            outputs.append(pooled.cpu().numpy().astype(np.float32))
    if not outputs:
        return np.zeros((0, 384), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _parse_news_timestamp(value) -> pd.Timestamp:
    text = str(value or "").strip()
    if not text:
        return pd.NaT
    for fmt in ("%d-%m-%Y %I:%M:%S %p", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return _to_naive_timestamp(datetime.strptime(text, fmt))
        except Exception:
            continue
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, dayfirst=False, errors="coerce")
    return _to_naive_timestamp(parsed)


def _to_naive_timestamp(value) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _timestamp_text_has_time(value) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(re.search(r"\d{1,2}:\d{2}", text) or re.search(r"\b(?:AM|PM)\b", text, flags=re.IGNORECASE))


def _parse_news_timestamp_with_precision(value) -> tuple[pd.Timestamp, bool]:
    ts = _parse_news_timestamp(value)
    return ts, _timestamp_text_has_time(value)


def _effective_freq_minutes(freq_min) -> int:
    try:
        value = int(float(str(freq_min or "").strip()))
    except Exception:
        return 0
    return value if value > 0 else 0


def _freq_alias(freq_minutes: int) -> str:
    return f"{int(freq_minutes)}min" if int(freq_minutes) > 0 else "D"


def _align_bank_time_bounds(date_start, date_end, freq_minutes: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = _to_naive_timestamp(date_start)
    end_ts = _to_naive_timestamp(date_end)
    if pd.isna(start_ts) or pd.isna(end_ts):
        return pd.NaT, pd.NaT
    if int(freq_minutes) > 0 and int(freq_minutes) < 1440:
        freq = _freq_alias(freq_minutes)
        return start_ts.floor(freq), end_ts.ceil(freq)
    return start_ts.normalize(), end_ts.normalize()


def _format_axis_timestamp(value, freq_minutes: int) -> str:
    ts = _to_naive_timestamp(value)
    if pd.isna(ts):
        return ""
    if int(freq_minutes) > 0 and int(freq_minutes) < 1440:
        return ts.floor("s").strftime("%Y-%m-%d %H:%M:%S")
    return str(ts.normalize().date())


def _ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size <= 0:
        return values
    out = np.zeros_like(values, dtype=np.float32)
    out[0] = values[0]
    a = float(max(0.0, min(1.0, alpha)))
    for idx in range(1, len(values)):
        out[idx] = a * values[idx] + (1.0 - a) * out[idx - 1]
    return out


def _topic_mass(tags: list[str]) -> np.ndarray:
    out = np.zeros((len(TOPIC_TAGS),), dtype=np.float32)
    for tag in tags:
        if tag in TOPIC_TAGS:
            out[TOPIC_TAGS.index(tag)] = 1.0
    return out


def build_regime_bank(
    refined_jsonl: str,
    out_path: str,
    *,
    encoder_model_id: str,
    max_length: int,
    date_start,
    date_end,
    source_news_path: str = "",
    schema_variant: str = "",
    dataset_key: str = "",
    tau_days: float = 5.0,
    ema_alpha: float = 0.5,
    ema_window: int = 5,
    freq_min: int | str | None = None,
    batch_size: int = 32,
) -> None:
    _ = ema_window
    freq_minutes = _effective_freq_minutes(freq_min)
    is_subdaily = 0 < int(freq_minutes) < 1440
    rows = _load_jsonl(refined_jsonl)
    actionable_rows: list[dict] = []
    summaries: list[str] = []
    for row in rows:
        if not bool(row.get("is_actionable", False)):
            continue
        published_at, has_published_time = _parse_news_timestamp_with_precision(row.get("published_at", ""))
        if pd.isna(published_at):
            continue
        published_at = published_at.floor("s")
        if is_subdaily and not has_published_time:
            # A date-only article can have been published at any time during that day.
            # For sub-daily forecasts, make it available only after that day has ended.
            published_at = published_at.normalize() + pd.Timedelta(days=1)
        elif not is_subdaily:
            published_at = published_at.normalize()
        horizon_days = int(max(0, min(14, int(row.get("horizon_days", 0) or 0))))
        confidence = float(max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))))
        if horizon_days <= 0 or confidence <= 0.0:
            continue
        regime_payload = row.get("regime_vec", {}) if isinstance(row.get("regime_vec"), dict) else {}
        actionable_rows.append(
            {
                "published_at": published_at,
                "horizon_days": horizon_days,
                "confidence": confidence,
                "regime_vec": np.asarray(
                    [float(regime_payload.get(key, 0.0) or 0.0) for key in REGIME_KEYS],
                    dtype=np.float32,
                ),
                "topic_mass": _topic_mass(list(row.get("topic_tags", []) or [])),
                "summary": str(row.get("summary", "") or "").strip(),
            }
        )
        summaries.append(str(row.get("summary", "") or "").strip())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    start_ts, end_ts = _align_bank_time_bounds(date_start, date_end, freq_minutes)
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts < start_ts:
        raise ValueError(f"Invalid regime bank date range: start={date_start} end={date_end}")

    timestamps = pd.date_range(start_ts, end_ts, freq=_freq_alias(freq_minutes))
    if len(timestamps) <= 0:
        raise ValueError("Regime bank time range is empty.")

    if actionable_rows:
        text_embeddings = _encode_texts(summaries, encoder_model_id, max_length=max_length, batch_size=batch_size)
        for idx, row in enumerate(actionable_rows):
            row["text_emb"] = text_embeddings[idx]
    else:
        text_embeddings = np.zeros((0, 384), dtype=np.float32)

    regime_vec = np.zeros((len(timestamps), len(REGIME_KEYS)), dtype=np.float32)
    topic_tag_mass = np.zeros((len(timestamps), len(TOPIC_TAGS)), dtype=np.float32)
    text_emb = np.zeros((len(timestamps), int(text_embeddings.shape[-1]) if text_embeddings.size > 0 else 384), dtype=np.float32)
    relevance_mass = np.zeros((len(timestamps),), dtype=np.float32)
    in_force_doc_count = np.zeros((len(timestamps),), dtype=np.int64)

    tau_days = float(max(1e-3, tau_days))
    for time_idx, timestamp in enumerate(timestamps):
        in_force: list[tuple[dict, float]] = []
        for row in actionable_rows:
            published_at = row["published_at"]
            horizon_end = published_at + pd.Timedelta(days=int(row["horizon_days"]))
            if published_at <= timestamp <= horizon_end:
                age_days = max(0.0, float((timestamp - published_at) / pd.Timedelta(days=1)))
                weight = float(row["confidence"]) * math.exp(-float(age_days) / tau_days)
                if weight > 0:
                    in_force.append((row, weight))

        if not in_force:
            continue

        raw_mass = float(sum(weight for _, weight in in_force))
        norm = max(1e-6, raw_mass)
        relevance_mass[time_idx] = raw_mass
        in_force_doc_count[time_idx] = int(len(in_force))
        for row, weight in in_force:
            w = float(weight) / norm
            regime_vec[time_idx] += w * row["regime_vec"]
            topic_tag_mass[time_idx] += w * row["topic_mass"]
            text_emb[time_idx] += w * row["text_emb"]

    regime_vec = _ema_smooth(regime_vec, ema_alpha)
    topic_tag_mass = _ema_smooth(topic_tag_mass, ema_alpha)
    text_emb = _ema_smooth(text_emb, ema_alpha)
    relevance_mass = _ema_smooth(relevance_mass[:, None], ema_alpha).reshape(-1)
    metadata = {
        "source_news_file": os.path.basename(str(source_news_path or "").strip()),
        "source_news_stem": os.path.splitext(os.path.basename(str(source_news_path or "").strip()))[0],
        "schema_variant": str(schema_variant or "").strip(),
        "dataset_key": str(dataset_key or "").strip(),
        "encoder_model_id": str(encoder_model_id or "").strip(),
        "max_length": int(max_length),
        "date_start": _format_axis_timestamp(start_ts, freq_minutes),
        "date_end": _format_axis_timestamp(end_ts, freq_minutes),
        "tau_days": float(tau_days),
        "ema_alpha": float(ema_alpha),
        "ema_window": int(ema_window),
        "freq_min": int(freq_minutes),
        "time_granularity": "subdaily" if is_subdaily else "daily",
        "causal_cutoff": "published_at_lte_target_time",
        "date_only_news_policy": "available_next_day_for_subdaily",
    }
    if str(schema_variant or "").strip() == "bitcoin":
        metadata["schema_version"] = schema_version_for_variant(schema_variant)
    if str(source_news_path or "").strip():
        try:
            news_signature = compute_news_corpus_signature(source_news_path)
            metadata["source_news_doc_count"] = int(news_signature.get("doc_count", 0) or 0)
            metadata["source_news_digest"] = str(news_signature.get("digest", "") or "").strip()
        except Exception:
            pass

    np.savez_compressed(
        out_path,
        dates=np.asarray([_format_axis_timestamp(ts, freq_minutes) for ts in timestamps], dtype="<U19"),
        regime_vec=regime_vec.astype(np.float32),
        topic_tag_mass=topic_tag_mass.astype(np.float32),
        text_emb=text_emb.astype(np.float32),
        relevance_mass=relevance_mass.astype(np.float32),
        in_force_doc_count=in_force_doc_count.astype(np.int64),
        topic_tags=np.asarray(TOPIC_TAGS, dtype="<U32"),
        regime_keys=np.asarray(REGIME_KEYS, dtype="<U32"),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=True)),
    )


@dataclass
class RegimeBank:
    dates: np.ndarray
    regime_vec: np.ndarray
    topic_tag_mass: np.ndarray
    text_emb: np.ndarray
    relevance_mass: np.ndarray
    in_force_doc_count: np.ndarray
    metadata: dict[str, object] | None = None

    def __post_init__(self):
        self.metadata = dict(self.metadata or {})
        self.freq_min = _effective_freq_minutes(self.metadata.get("freq_min", 0))
        self._is_subdaily = 0 < int(self.freq_min) < 1440
        parsed = pd.to_datetime(self.dates.tolist(), errors="coerce")
        lookup: dict[str, int] = {}
        timestamp_ns: list[int] = []
        for idx, ts in enumerate(parsed):
            if pd.isna(ts):
                timestamp_ns.append(np.iinfo(np.int64).min)
                continue
            clean_ts = _to_naive_timestamp(ts)
            lookup[_format_axis_timestamp(clean_ts, self.freq_min)] = idx
            timestamp_ns.append(int(clean_ts.value))
        self._lookup = lookup
        self._timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64)

    @property
    def text_dim(self) -> int:
        return int(self.text_emb.shape[-1]) if self.text_emb.ndim == 2 else 0

    @property
    def topic_dim(self) -> int:
        return int(self.topic_tag_mass.shape[-1]) if self.topic_tag_mass.ndim == 2 else 0

    @property
    def regime_dim(self) -> int:
        return int(self.regime_vec.shape[-1]) if self.regime_vec.ndim == 2 else 0

    def lookup(self, target_date) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        regime_vec = np.zeros((self.regime_dim,), dtype=np.float32)
        topic_tag_mass = np.zeros((self.topic_dim,), dtype=np.float32)
        text_emb = np.zeros((self.text_dim,), dtype=np.float32)
        relevance_mass = np.zeros((1,), dtype=np.float32)
        in_force_doc_count = np.zeros((1,), dtype=np.int64)

        target_ts = _to_naive_timestamp(target_date)
        if pd.isna(target_ts):
            return regime_vec, topic_tag_mass, text_emb, relevance_mass, in_force_doc_count
        key = _format_axis_timestamp(target_ts, self.freq_min)
        idx = self._lookup.get(key)
        if idx is None and self._is_subdaily:
            aligned_ts = target_ts.floor(_freq_alias(self.freq_min))
            idx = self._lookup.get(_format_axis_timestamp(aligned_ts, self.freq_min))
        if idx is None and self._is_subdaily and self._timestamp_ns.size > 0:
            target_ns = int(target_ts.value)
            valid_ns = self._timestamp_ns[self._timestamp_ns != np.iinfo(np.int64).min]
            pos = int(np.searchsorted(valid_ns, target_ns, side="right") - 1)
            if pos >= 0:
                idx = pos
        if idx is None:
            return regime_vec, topic_tag_mass, text_emb, relevance_mass, in_force_doc_count
        regime_vec = self.regime_vec[idx].astype(np.float32, copy=True)
        topic_tag_mass = self.topic_tag_mass[idx].astype(np.float32, copy=True)
        text_emb = self.text_emb[idx].astype(np.float32, copy=True)
        relevance_mass = np.asarray([self.relevance_mass[idx]], dtype=np.float32)
        in_force_doc_count = np.asarray([self.in_force_doc_count[idx]], dtype=np.int64)
        return regime_vec, topic_tag_mass, text_emb, relevance_mass, in_force_doc_count

    def shuffled(self, seed: int) -> "RegimeBank":
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(len(self.dates))
        return RegimeBank(
            dates=self.dates.copy(),
            regime_vec=self.regime_vec[perm].copy(),
            topic_tag_mass=self.topic_tag_mass[perm].copy(),
            text_emb=self.text_emb[perm].copy(),
            relevance_mass=self.relevance_mass[perm].copy(),
            in_force_doc_count=self.in_force_doc_count[perm].copy(),
            metadata=dict(self.metadata or {}),
        )


def _read_metadata_from_payload(payload) -> dict[str, str | float | int]:
    if "metadata_json" not in payload.files:
        return {}
    raw = payload["metadata_json"]
    if getattr(raw, "shape", ()) == ():
        text = str(raw.item())
    else:
        text = str(np.asarray(raw).reshape(-1)[0])
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_regime_bank(path: str) -> RegimeBank:
    with np.load(path, allow_pickle=False) as payload:
        required = {"dates", "regime_vec", "topic_tag_mass", "text_emb", "relevance_mass", "in_force_doc_count"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(
                f"Regime bank at {path} is missing fields {sorted(missing)}. "
                "This usually means it was built by the removed daily-news scheme; rebuild with delta_v3_refined_bank_build=1."
            )
        return RegimeBank(
            dates=np.asarray(payload["dates"]).copy(),
            regime_vec=np.asarray(payload["regime_vec"], dtype=np.float32).copy(),
            topic_tag_mass=np.asarray(payload["topic_tag_mass"], dtype=np.float32).copy(),
            text_emb=np.asarray(payload["text_emb"], dtype=np.float32).copy(),
            relevance_mass=np.asarray(payload["relevance_mass"], dtype=np.float32).reshape(-1).copy(),
            in_force_doc_count=np.asarray(payload["in_force_doc_count"], dtype=np.int64).reshape(-1).copy(),
            metadata=_read_metadata_from_payload(payload),
        )


def read_regime_bank_metadata(path: str) -> dict[str, str | float | int]:
    with np.load(path, allow_pickle=False) as payload:
        return _read_metadata_from_payload(payload)
