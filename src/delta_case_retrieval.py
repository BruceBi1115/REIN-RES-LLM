from __future__ import annotations

import json
import math
import os
import hashlib
import zlib
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

from .delta_news_hooks import extract_structured_events, refine_news_text
from .news_rules import get_candidates, select_news


EVENT_TYPE_BUCKETS = {
    "outage": {"outage", "trip", "plant_outage", "generator_outage"},
    "weather": {"weather", "storm", "temperature", "rain", "heat", "cold"},
    "policy": {"policy", "regulation", "government", "rule"},
    "transmission": {"transmission", "interconnector", "network", "line"},
    "fuel": {"fuel", "gas", "coal", "oil"},
    "demand": {"demand", "load", "consumption"},
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _safe_array(x: Any, default_len: int = 0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0 and default_len > 0:
        return np.zeros((default_len,), dtype=np.float32)
    return arr


def _build_text_vector(text: Any, dim: int = 128) -> np.ndarray:
    s = str(text or "").strip().lower()
    if not s:
        return np.zeros((0,), dtype=np.float32)
    toks = re.findall(r"[a-z0-9]+", s)
    if len(toks) == 0:
        return np.zeros((0,), dtype=np.float32)
    d = int(max(16, dim))
    vec = np.zeros((d,), dtype=np.float32)
    for tok in toks:
        h = hashlib.sha1(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little", signed=False) % d
        sign = 1.0 if (int(h[4]) & 1) == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    if n <= 1e-8:
        return np.zeros((0,), dtype=np.float32)
    vec = vec / n
    return vec.astype(np.float32, copy=False)


def _case_text_vector(case_like: dict, dim: int = 128) -> np.ndarray:
    if not isinstance(case_like, dict):
        return np.zeros((0,), dtype=np.float32)
    tv = _safe_array(case_like.get("text_vector", []))
    if tv.size > 0:
        return tv
    return _build_text_vector(case_like.get("refined_news", ""), dim=dim)


def _text_similarity_from_query_vec(query_vec: np.ndarray, case_like: dict, dim: int = 128) -> float:
    q = _safe_array(query_vec)
    c = _case_text_vector(case_like, dim=dim)
    if q.size == 0 or c.size == 0:
        return 0.0
    d = min(int(q.size), int(c.size))
    if d <= 0:
        return 0.0
    qv = q[:d].astype(np.float32, copy=False)
    cv = c[:d].astype(np.float32, copy=False)
    qn = float(np.linalg.norm(qv))
    cn = float(np.linalg.norm(cv))
    if qn <= 1e-8 or cn <= 1e-8:
        return 0.0
    cos = float(np.dot(qv, cv) / (qn * cn))
    return float(max(0.0, min(1.0, 0.5 * (cos + 1.0))))


def _refine_cache_key_for_case(
    *,
    raw_news_texts: list[str],
    context: dict,
    mode: str,
    model: str,
    max_tokens: int,
) -> str:
    clean = [str(x).strip() for x in raw_news_texts if str(x).strip()]
    payload = {
        "mode": str(mode or "").lower().strip(),
        "model": str(model or "").strip(),
        "max_tokens": int(max(1, max_tokens)),
        "target_time": str(context.get("target_time", "")).strip(),
        "region": str(context.get("region", "")).strip(),
        "description": str(context.get("description", "")).strip(),
        "news": clean,
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _refine_doc_cache_key_for_case(
    *,
    raw_news_text: str,
    context: dict,
    mode: str,
    model: str,
    max_tokens: int,
) -> str:
    txt = str(raw_news_text or "").strip()
    payload = {
        "kind": "doc",
        "mode": str(mode or "").lower().strip(),
        "model": str(model or "").strip(),
        "max_tokens": int(max(1, max_tokens)),
        "region": str(context.get("region", "")).strip(),
        "description": str(context.get("description", "")).strip(),
        "news": txt,
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "doc::" + hashlib.sha1(s.encode("utf-8")).hexdigest()


def _truncate_with_tokenizer_case(text: str, tokenizer, max_tokens: int) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    n = int(max(1, max_tokens))
    if tokenizer is None:
        return s[: n * 4]
    try:
        enc = tokenizer(
            s,
            add_special_tokens=False,
            truncation=True,
            max_length=n,
            return_attention_mask=False,
        )
        ids = enc.get("input_ids", []) if isinstance(enc, dict) else []
        if len(ids) == 0:
            return s[: n * 4]
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except Exception:
        return s[: n * 4]


def _refine_one_news_doc_for_case(
    *,
    raw_news_text: str,
    tokenizer,
    max_tokens: int,
    args,
    api_adapter=None,
):
    clean = str(raw_news_text or "").strip()
    if not clean:
        return ""

    mode = str(getattr(args, "news_refine_mode", "local") or "local").lower().strip()
    model = str(getattr(api_adapter, "model", getattr(args, "news_api_model", "")))
    context = {
        "target_time": "",
        "region": str(getattr(args, "region", "")),
        "description": str(getattr(args, "description", "")),
    }

    cache_enabled = bool(getattr(args, "_refine_cache_enabled", False))
    cache_store = getattr(args, "_refine_cache_store", None)
    cache_key = ""
    if cache_enabled and isinstance(cache_store, dict):
        cache_key = _refine_doc_cache_key_for_case(
            raw_news_text=clean,
            context=context,
            mode=mode,
            model=model,
            max_tokens=max_tokens,
        )
        cached = cache_store.get(cache_key, "")
        if isinstance(cached, str) and cached.strip():
            setattr(
                args,
                "_refine_cache_hits",
                int(getattr(args, "_refine_cache_hits", 0)) + 1,
            )
            return cached.strip()

    use_api = mode == "api" and api_adapter is not None and hasattr(api_adapter, "refine_news")
    if use_api:
        refined = refine_news_text(
            raw_news_texts=[clean],
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            mode=mode,
            api_adapter=api_adapter,
            context=context,
        )
    else:
        refined = _truncate_with_tokenizer_case(clean, tokenizer, max_tokens=max_tokens)
    refined = str(refined or "").strip()

    if cache_enabled:
        setattr(
            args,
            "_refine_cache_misses",
            int(getattr(args, "_refine_cache_misses", 0)) + 1,
        )
        if isinstance(cache_store, dict) and cache_key and refined:
            cache_store[cache_key] = refined
            setattr(args, "_refine_cache_store", cache_store)
            setattr(args, "_refine_cache_dirty", True)
    return refined


def _refine_news_from_doc_cache_for_case(
    *,
    raw_news_texts: list[str],
    tokenizer,
    max_tokens: int,
    args,
    api_adapter=None,
) -> str:
    items = [str(x).strip() for x in raw_news_texts if str(x).strip()]
    if len(items) == 0:
        return ""

    snippets = []
    seen = set()
    for item in items:
        refined_item = _refine_one_news_doc_for_case(
            raw_news_text=item,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            args=args,
            api_adapter=api_adapter,
        )
        if not refined_item:
            continue
        if refined_item in seen:
            continue
        seen.add(refined_item)
        snippets.append(refined_item)
    if len(snippets) == 0:
        return ""
    merged = "\n".join([f"- {s}" for s in snippets])
    return _truncate_with_tokenizer_case(merged, tokenizer, max_tokens=max_tokens)


def _to_utc_time(ts_like: Any) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(ts_like, errors="coerce", utc=True)
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        return None
    return ts


def _recency_similarity(query_time: Any, case_time: Any, tau_hours: float = 168.0) -> float:
    qt = _to_utc_time(query_time)
    ct = _to_utc_time(case_time)
    if qt is None or ct is None:
        return 0.0
    tau = max(1e-3, float(tau_hours))
    diff_h = abs(float((qt - ct).total_seconds()) / 3600.0)
    return float(math.exp(-diff_h / tau))


def _regime_similarity(query_regime: Any, case_regime: Any) -> float:
    q = str(query_regime or "").strip().lower()
    c = str(case_regime or "").strip().lower()
    if not q or not c:
        return 0.0
    return 1.0 if q == c else 0.0


def _parse_time_features(target_time: str) -> tuple[float, float, float, float]:
    try:
        dt = pd.to_datetime(str(target_time), errors="coerce")
    except Exception:
        dt = pd.NaT
    if pd.isna(dt):
        return 0.0, 0.0, 0.0, 0.0
    hour = float(dt.hour)
    dow = float(dt.dayofweek)
    hour_rad = 2.0 * math.pi * hour / 24.0
    dow_rad = 2.0 * math.pi * dow / 7.0
    return (
        float(math.sin(hour_rad)),
        float(math.cos(hour_rad)),
        float(math.sin(dow_rad)),
        float(math.cos(dow_rad)),
    )


def _trend_strength(x: np.ndarray) -> float:
    x = _safe_array(x)
    if x.size < 2:
        return 0.0
    idx = np.arange(x.size, dtype=np.float32)
    idx = idx - idx.mean()
    denom = float((idx * idx).sum())
    if denom <= 1e-8:
        return 0.0
    slope = float((idx * (x - x.mean())).sum() / denom)
    return float(slope)


def _event_type_bucket(event_type: Any) -> str:
    t = str(event_type or "").strip().lower()
    if not t:
        return "other"
    for bucket, terms in EVENT_TYPE_BUCKETS.items():
        if t == bucket or t in terms:
            return bucket
        for term in terms:
            if term in t:
                return bucket
    if t in {"mixed", "general"}:
        return "mixed"
    return "other"


def _bucket_level(v: Any, low: float, high: float, names: tuple[str, str, str]) -> str:
    x = _safe_float(v, default=0.0)
    if x < low:
        return names[0]
    if x < high:
        return names[1]
    return names[2]


def normalize_structured_events(events: dict | None) -> dict:
    if not isinstance(events, dict) or len(events) == 0:
        return {}
    direction_raw = events.get("direction", 0)
    try:
        direction_f = float(direction_raw)
    except Exception:
        direction_f = 0.0
    if direction_f > 0.15:
        direction = "up"
    elif direction_f < -0.15:
        direction = "down"
    else:
        direction = "uncertain"

    strength_f = _safe_float(events.get("strength", 0.0), default=0.0)
    persistence_f = _safe_float(events.get("persistence", 0.0), default=0.0)
    confidence_f = _safe_float(events.get("confidence", 0.0), default=0.0)

    return {
        "event_type": _event_type_bucket(events.get("event_type", "other")),
        "direction": direction,
        "strength": _bucket_level(strength_f, 0.33, 0.66, ("weak", "medium", "strong")),
        "persistence": _bucket_level(persistence_f, 0.33, 0.66, ("short", "medium", "long")),
        "confidence": _bucket_level(confidence_f, 0.33, 0.66, ("low", "medium", "high")),
        "strength_value": float(max(0.0, min(1.0, strength_f))),
        "persistence_value": float(max(0.0, min(1.0, persistence_f))),
        "confidence_value": float(max(0.0, min(1.0, confidence_f))),
    }


def _regime_from_vol(volatility: float) -> str:
    if volatility < 0.8:
        return "low"
    if volatility < 1.5:
        return "mid"
    return "high"


def _state_vector(history_z: np.ndarray, base_pred_z: np.ndarray, target_time: str) -> tuple[np.ndarray, dict]:
    h = _safe_array(history_z)
    b = _safe_array(base_pred_z)

    h_mean = float(h.mean()) if h.size else 0.0
    h_std = float(h.std()) if h.size else 0.0
    h_last = float(h[-1]) if h.size else 0.0
    h_slope = _trend_strength(h)

    b_mean = float(b.mean()) if b.size else 0.0
    b_std = float(b.std()) if b.size else 0.0
    b_last = float(b[-1]) if b.size else 0.0
    b_slope = _trend_strength(b)

    volatility = h_std
    trend = h_slope
    sin_hour, cos_hour, sin_dow, cos_dow = _parse_time_features(target_time)

    state = np.asarray(
        [
            h_mean,
            h_std,
            h_last,
            h_slope,
            b_mean,
            b_std,
            b_last,
            b_slope,
            volatility,
            trend,
            sin_hour,
            cos_hour,
            sin_dow,
            cos_dow,
        ],
        dtype=np.float32,
    )
    state_features = {
        "volatility": float(volatility),
        "trend": float(trend),
        "regime": _regime_from_vol(float(volatility)),
        "sin_hour": float(sin_hour),
        "cos_hour": float(cos_hour),
        "sin_dow": float(sin_dow),
        "cos_dow": float(cos_dow),
    }
    return state, state_features


def build_case_record(
    *,
    sample_id: str,
    split: str,
    target_time: str,
    history_z: list[float] | np.ndarray,
    base_pred_z: list[float] | np.ndarray,
    target_z: list[float] | np.ndarray | None = None,
    refined_news: str | None = None,
    structured_events: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    history_arr = _safe_array(history_z)
    base_arr = _safe_array(base_pred_z)
    target_arr = _safe_array(target_z, default_len=base_arr.size if target_z is not None else 0)
    if target_z is None:
        residual_arr = np.zeros_like(base_arr, dtype=np.float32)
    else:
        if target_arr.size != base_arr.size:
            n = min(target_arr.size, base_arr.size)
            if n == 0:
                residual_arr = np.zeros_like(base_arr, dtype=np.float32)
            else:
                residual_arr = target_arr[:n] - base_arr[:n]
                if n < base_arr.size:
                    pad = np.zeros((base_arr.size - n,), dtype=np.float32)
                    residual_arr = np.concatenate([residual_arr, pad], axis=0)
        else:
            residual_arr = target_arr - base_arr

    state_vec, state_features = _state_vector(history_arr, base_arr, target_time=str(target_time))
    norm_events = normalize_structured_events(structured_events)
    text_vec = _build_text_vector(refined_news, dim=128)

    return {
        "sample_id": str(sample_id),
        "split": str(split),
        "target_time": str(target_time),
        "history_z": history_arr.tolist(),
        "base_pred_z": base_arr.tolist(),
        "target_z": target_arr.tolist() if target_z is not None else [],
        "residual_z": residual_arr.tolist(),
        "state_vector": state_vec.tolist(),
        "state_features": state_features,
        "refined_news": str(refined_news or ""),
        "text_vector": text_vec.tolist(),
        "structured_events": norm_events,
        "metadata": dict(metadata or {}),
    }


def _cosine_similarity_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if matrix.shape[1] != query.shape[0]:
        d = min(matrix.shape[1], query.shape[0])
        if d <= 0:
            return np.zeros((matrix.shape[0],), dtype=np.float32)
        matrix = matrix[:, :d]
        query = query[:d]

    m_norm = np.linalg.norm(matrix, axis=1) + 1e-8
    q_norm = float(np.linalg.norm(query) + 1e-8)
    sims = (matrix @ query) / (m_norm * q_norm)
    return sims.astype(np.float32)


def _event_similarity(query_event: dict, case_event: dict) -> float:
    if not query_event or not case_event:
        return 0.0
    score = 0.0
    weight = 0.0

    wt = 0.35
    score += wt * (1.0 if query_event.get("event_type") == case_event.get("event_type") else 0.0)
    weight += wt

    wd = 0.25
    qd = str(query_event.get("direction", "uncertain"))
    cd = str(case_event.get("direction", "uncertain"))
    if qd == "uncertain" or cd == "uncertain":
        score += wd * 0.5
    elif qd == cd:
        score += wd * 1.0
    else:
        score += wd * 0.0
    weight += wd

    ws = 0.2
    score += ws * (1.0 - abs(_safe_float(query_event.get("strength_value", 0.0)) - _safe_float(case_event.get("strength_value", 0.0))))
    weight += ws

    wp = 0.1
    score += wp * (1.0 - abs(_safe_float(query_event.get("persistence_value", 0.0)) - _safe_float(case_event.get("persistence_value", 0.0))))
    weight += wp

    wc = 0.1
    score += wc * (1.0 - abs(_safe_float(query_event.get("confidence_value", 0.0)) - _safe_float(case_event.get("confidence_value", 0.0))))
    weight += wc

    if weight <= 1e-8:
        return 0.0
    return float(max(0.0, min(1.0, score / weight)))


def rerank_cases_by_event(
    query_case: dict,
    candidates: list[dict],
    *,
    mode: str = "price_event",
    alpha_price: float = 0.85,
    alpha_event: float = 0.15,
    alpha_text: float = 0.20,
    alpha_recency: float = 0.10,
    alpha_regime: float = 0.05,
    recency_tau_hours: float = 168.0,
) -> list[dict]:
    if not candidates:
        return []
    use_mode = str(mode or "price_event").lower().strip()
    q_event = query_case.get("structured_events", {})
    q_time = query_case.get("target_time", "")
    q_regime = query_case.get("state_features", {}).get("regime", "")
    q_text_vec = _case_text_vector(query_case, dim=128)
    use_event = (
        use_mode == "price_event"
        and isinstance(q_event, dict)
        and len(q_event) > 0
        and float(alpha_event) > 0.0
    )
    use_text = q_text_vec.size > 0 and float(alpha_text) > 0.0

    w_price = float(max(0.0, alpha_price))
    w_event = float(max(0.0, alpha_event if use_event else 0.0))
    w_text = float(max(0.0, alpha_text if use_text else 0.0))
    w_recency = float(max(0.0, alpha_recency))
    w_regime = float(max(0.0, alpha_regime))
    denom = w_price + w_event + w_text + w_recency + w_regime
    if denom <= 1e-8:
        w_price, w_event, w_text, w_recency, w_regime, denom = 1.0, 0.0, 0.0, 0.0, 0.0, 1.0

    ranked = []
    for c in candidates:
        price_score = _safe_float(c.get("price_score", 0.0), default=0.0)
        event_score = 0.0
        if use_event:
            event_score = _event_similarity(q_event, c.get("structured_events", {}))
        text_score = 0.0
        if use_text:
            text_score = _text_similarity_from_query_vec(q_text_vec, c, dim=128)
        recency_score = _recency_similarity(
            q_time,
            c.get("target_time", ""),
            tau_hours=float(recency_tau_hours),
        )
        regime_score = _regime_similarity(
            q_regime,
            c.get("state_features", {}).get("regime", ""),
        )
        final_score = (
            w_price * price_score
            + w_event * event_score
            + w_text * text_score
            + w_recency * recency_score
            + w_regime * regime_score
        ) / denom
        cc = dict(c)
        cc["event_score"] = float(event_score)
        cc["text_score"] = float(text_score)
        cc["recency_score"] = float(recency_score)
        cc["regime_score"] = float(regime_score)
        cc["final_score"] = float(final_score)
        ranked.append(cc)
    ranked.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    return ranked


def reject_low_confidence_retrieval(
    query_case: dict,
    ranked_cases: list[dict],
    *,
    min_top_score: float = 0.12,
    min_candidates: int = 2,
    min_direction_agreement: float = 0.45,
    max_event_mismatch: float = 0.8,
) -> dict:
    if not ranked_cases:
        return {
            "retrieval_valid": False,
            "retrieval_confidence": 0.0,
            "reject_reason": "empty",
            "direction_agreement": 0.0,
            "event_mismatch_rate": 1.0,
        }

    top = ranked_cases[0]
    top_score = _safe_float(top.get("final_score", top.get("price_score", 0.0)))
    if top_score < float(min_top_score):
        return {
            "retrieval_valid": False,
            "retrieval_confidence": max(0.0, min(1.0, top_score)),
            "reject_reason": "low_top_score",
            "direction_agreement": 0.0,
            "event_mismatch_rate": 1.0,
        }

    if len(ranked_cases) < int(max(1, min_candidates)):
        return {
            "retrieval_valid": False,
            "retrieval_confidence": max(0.0, min(1.0, top_score)),
            "reject_reason": "too_few_candidates",
            "direction_agreement": 0.0,
            "event_mismatch_rate": 1.0,
        }

    signs = []
    mismatch = []
    q_event_type = str(query_case.get("structured_events", {}).get("event_type", "other"))
    for c in ranked_cases:
        residual = _safe_array(c.get("residual_z", []))
        r_mean = float(residual.mean()) if residual.size else 0.0
        signs.append(1 if r_mean > 0 else (-1 if r_mean < 0 else 0))
        c_type = str(c.get("structured_events", {}).get("event_type", "other"))
        mismatch.append(0.0 if q_event_type in {"", "other"} or c_type == q_event_type else 1.0)

    signs = np.asarray(signs, dtype=np.int32)
    nonzero = signs[np.abs(signs) > 0]
    if nonzero.size == 0:
        direction_agreement = 0.0
    else:
        pos_frac = float((nonzero > 0).mean())
        neg_frac = float((nonzero < 0).mean())
        direction_agreement = max(pos_frac, neg_frac)

    event_mismatch_rate = float(np.mean(mismatch)) if mismatch else 0.0
    if direction_agreement < float(min_direction_agreement):
        return {
            "retrieval_valid": False,
            "retrieval_confidence": max(0.0, min(1.0, 0.5 * (top_score + direction_agreement))),
            "reject_reason": "direction_disagree",
            "direction_agreement": float(direction_agreement),
            "event_mismatch_rate": float(event_mismatch_rate),
        }

    if event_mismatch_rate > float(max_event_mismatch):
        return {
            "retrieval_valid": False,
            "retrieval_confidence": max(0.0, min(1.0, 0.5 * (top_score + (1.0 - event_mismatch_rate)))),
            "reject_reason": "event_mismatch",
            "direction_agreement": float(direction_agreement),
            "event_mismatch_rate": float(event_mismatch_rate),
        }

    conf = 0.5 * top_score + 0.3 * direction_agreement + 0.2 * (1.0 - event_mismatch_rate)
    conf = float(max(0.0, min(1.0, conf)))
    return {
        "retrieval_valid": True,
        "retrieval_confidence": conf,
        "reject_reason": "",
        "direction_agreement": float(direction_agreement),
        "event_mismatch_rate": float(event_mismatch_rate),
    }


def _compute_retrieval_soft_weight(reject_meta: dict, n_candidates: int) -> float:
    if int(max(0, n_candidates)) <= 0:
        return 0.0
    conf = float(max(0.0, min(1.0, _safe_float(reject_meta.get("retrieval_confidence", 0.0)))))
    valid = bool(reject_meta.get("retrieval_valid", False))
    reason = str(reject_meta.get("reject_reason", "")).strip().lower()
    direction_agree = float(max(0.0, min(1.0, _safe_float(reject_meta.get("direction_agreement", 0.0)))))
    mismatch = float(max(0.0, min(1.0, _safe_float(reject_meta.get("event_mismatch_rate", 1.0)))))

    if valid:
        base = 0.5 + 0.5 * conf
    elif reason == "direction_disagree":
        base = 0.35 * conf
    elif reason == "event_mismatch":
        base = 0.30 * conf
    elif reason == "low_top_score":
        base = 0.25 * conf
    elif reason == "too_few_candidates":
        base = 0.20 * conf
    else:
        base = 0.0

    base *= (0.6 + 0.4 * direction_agree)
    base *= (0.7 + 0.3 * (1.0 - mismatch))
    return float(max(0.0, min(1.0, base)))


def _compute_knn_delta_prior(
    ranked_cases: list[dict],
    *,
    horizon: int = 0,
    temperature: float = 0.20,
) -> np.ndarray:
    if not ranked_cases:
        return np.zeros((max(0, int(horizon))), dtype=np.float32)

    horizon_int = int(max(0, horizon))
    if horizon_int <= 0:
        horizon_int = max(int(_safe_array(c.get("residual_z", [])).size) for c in ranked_cases)
    if horizon_int <= 0:
        return np.zeros((0,), dtype=np.float32)

    res_rows = []
    score_rows = []
    for c in ranked_cases:
        r = _safe_array(c.get("residual_z", []), default_len=horizon_int)
        if r.size > horizon_int:
            r = r[:horizon_int]
        elif r.size < horizon_int:
            pad = np.zeros((horizon_int - r.size,), dtype=np.float32)
            r = np.concatenate([r, pad], axis=0)
        res_rows.append(r.astype(np.float32, copy=False))
        score_rows.append(_safe_float(c.get("final_score", c.get("price_score", 0.0)), default=0.0))

    res_m = np.asarray(res_rows, dtype=np.float32)
    scores = np.asarray(score_rows, dtype=np.float32).reshape(-1)
    if res_m.ndim != 2 or res_m.shape[0] == 0:
        return np.zeros((horizon_int,), dtype=np.float32)

    if res_m.shape[0] == 1:
        weights = np.ones((1,), dtype=np.float32)
    else:
        t = max(1e-4, float(temperature))
        logits = (scores - float(scores.max(initial=0.0))) / t
        w = np.exp(logits)
        z = float(w.sum())
        if not math.isfinite(z) or z <= 1e-12:
            weights = np.full((res_m.shape[0],), 1.0 / float(res_m.shape[0]), dtype=np.float32)
        else:
            weights = (w / z).astype(np.float32)

    prior = (weights.reshape(-1, 1) * res_m).sum(axis=0)
    return prior.astype(np.float32, copy=False)


def retrieve_similar_cases(
    *,
    query_case: dict,
    case_bank: dict | None,
    top_n: int = 5,
    mode: str = "price_event",
    alpha_price: float = 0.85,
    alpha_event: float = 0.15,
    alpha_text: float = 0.20,
    alpha_recency: float = 0.10,
    alpha_regime: float = 0.05,
    recency_tau_hours: float = 168.0,
    min_top_score: float = 0.12,
    min_candidates: int = 2,
    min_direction_agreement: float = 0.45,
    max_event_mismatch: float = 0.8,
    knn_temperature: float = 0.20,
    knn_horizon: int = 0,
) -> dict:
    top_n = int(max(1, top_n))
    if not isinstance(case_bank, dict) or len(case_bank.get("cases", [])) == 0:
        return {
            "candidates": [],
            "retrieval_valid": False,
            "retrieval_confidence": 0.0,
            "reject_reason": "empty_bank",
            "direction_agreement": 0.0,
            "event_mismatch_rate": 1.0,
            "mode": str(mode),
            "retrieval_soft_weight": 0.0,
            "knn_delta_prior_z": [],
        }

    cases = case_bank.get("cases", [])
    state_matrix = np.asarray(case_bank.get("state_matrix", []), dtype=np.float32)
    if state_matrix.ndim != 2 or state_matrix.shape[0] != len(cases):
        state_matrix = np.asarray([_safe_array(c.get("state_vector", [])) for c in cases], dtype=np.float32)

    query_vec = _safe_array(query_case.get("state_vector", []), default_len=state_matrix.shape[1] if state_matrix.ndim == 2 else 0)
    sims = _cosine_similarity_matrix(state_matrix, query_vec)
    if sims.size == 0:
        return {
            "candidates": [],
            "retrieval_valid": False,
            "retrieval_confidence": 0.0,
            "reject_reason": "empty_similarity",
            "direction_agreement": 0.0,
            "event_mismatch_rate": 1.0,
            "mode": str(mode),
            "retrieval_soft_weight": 0.0,
            "knn_delta_prior_z": [],
        }

    qid = str(query_case.get("sample_id", ""))
    candidates = []
    for idx, score in enumerate(sims.tolist()):
        c = cases[idx]
        if qid and str(c.get("sample_id", "")) == qid:
            continue
        cc = dict(c)
        cc["price_score"] = float(score)
        candidates.append(cc)

    if not candidates:
        return {
            "candidates": [],
            "retrieval_valid": False,
            "retrieval_confidence": 0.0,
            "reject_reason": "all_filtered_self",
            "direction_agreement": 0.0,
            "event_mismatch_rate": 1.0,
            "mode": str(mode),
            "retrieval_soft_weight": 0.0,
            "knn_delta_prior_z": [],
        }

    candidates.sort(key=lambda x: float(x.get("price_score", 0.0)), reverse=True)
    pre_top = candidates[: max(top_n * 4, top_n)]

    use_mode = str(mode or "price_event").lower().strip()
    if use_mode == "random":
        seed = zlib.adler32(str(qid or "random").encode("utf-8"))
        rng = np.random.default_rng(seed)
        pick_n = min(top_n, len(candidates))
        if pick_n <= 0:
            ranked = []
        else:
            picked = rng.choice(len(candidates), size=pick_n, replace=False).tolist()
            ranked = [dict(candidates[int(i)]) for i in picked]
            q_event = query_case.get("structured_events", {})
            q_time = query_case.get("target_time", "")
            q_regime = query_case.get("state_features", {}).get("regime", "")
            for cc in ranked:
                cc["event_score"] = _event_similarity(q_event, cc.get("structured_events", {}))
                cc["recency_score"] = _recency_similarity(
                    q_time,
                    cc.get("target_time", ""),
                    tau_hours=float(recency_tau_hours),
                )
                cc["regime_score"] = _regime_similarity(
                    q_regime,
                    cc.get("state_features", {}).get("regime", ""),
                )
                cc["final_score"] = float(cc.get("price_score", 0.0))
    elif use_mode == "price":
        ranked = rerank_cases_by_event(
            query_case=query_case,
            candidates=pre_top,
            mode="price",
            alpha_price=1.0,
            alpha_event=0.0,
            alpha_text=float(alpha_text),
            alpha_recency=float(alpha_recency),
            alpha_regime=float(alpha_regime),
            recency_tau_hours=float(recency_tau_hours),
        )
    else:
        ranked = rerank_cases_by_event(
            query_case=query_case,
            candidates=pre_top,
            mode="price_event",
            alpha_price=float(alpha_price),
            alpha_event=float(alpha_event),
            alpha_text=float(alpha_text),
            alpha_recency=float(alpha_recency),
            alpha_regime=float(alpha_regime),
            recency_tau_hours=float(recency_tau_hours),
        )

    ranked = ranked[:top_n]
    reject = reject_low_confidence_retrieval(
        query_case=query_case,
        ranked_cases=ranked,
        min_top_score=float(min_top_score),
        min_candidates=int(min_candidates),
        min_direction_agreement=float(min_direction_agreement),
        max_event_mismatch=float(max_event_mismatch),
    )
    soft_w = _compute_retrieval_soft_weight(reject, n_candidates=len(ranked))
    knn_prior = _compute_knn_delta_prior(
        ranked,
        horizon=int(max(0, knn_horizon)),
        temperature=float(knn_temperature),
    )
    out = {
        "candidates": ranked,
        "mode": use_mode,
        "retrieval_soft_weight": float(soft_w),
        "knn_delta_prior_z": knn_prior.tolist(),
    }
    out.update(reject)
    return out


def build_retrieval_features(
    *,
    query_case: dict,
    retrieval_output: dict,
    feature_dim: int = 12,
) -> tuple[np.ndarray, dict]:
    feature_dim = int(max(8, feature_dim))
    candidates = list(retrieval_output.get("candidates", []))
    valid = float(bool(retrieval_output.get("retrieval_valid", False)))
    confidence = float(_safe_float(retrieval_output.get("retrieval_confidence", 0.0)))
    soft_weight = float(_safe_float(retrieval_output.get("retrieval_soft_weight", valid * confidence)))
    soft_weight = float(max(0.0, min(1.0, soft_weight)))
    knn_delta_prior = _safe_array(retrieval_output.get("knn_delta_prior_z", []))

    if not candidates:
        feat = np.zeros((feature_dim,), dtype=np.float32)
        feat[0] = soft_weight
        feat[1] = confidence
        feat[2] = valid
        meta = {
            "retrieval_valid": bool(valid > 0.5),
            "retrieval_confidence": confidence,
            "retrieval_soft_weight": soft_weight,
            "retrieval_mode": str(retrieval_output.get("mode", "off")),
            "reject_reason": str(retrieval_output.get("reject_reason", "empty")),
            "coverage": float(valid),
            "knn_delta_prior_z": knn_delta_prior.tolist(),
        }
        return feat, meta

    price_scores = np.asarray([_safe_float(c.get("price_score", 0.0)) for c in candidates], dtype=np.float32)
    event_scores = np.asarray([_safe_float(c.get("event_score", 0.0)) for c in candidates], dtype=np.float32)
    recency_scores = np.asarray([_safe_float(c.get("recency_score", 0.0)) for c in candidates], dtype=np.float32)
    regime_scores = np.asarray([_safe_float(c.get("regime_score", 0.0)) for c in candidates], dtype=np.float32)
    final_scores = np.asarray([_safe_float(c.get("final_score", 0.0)) for c in candidates], dtype=np.float32)

    residual_means = []
    residual_abs_means = []
    signs = []
    q_type = str(query_case.get("structured_events", {}).get("event_type", "other"))
    type_matches = []
    for c in candidates:
        r = _safe_array(c.get("residual_z", []))
        r_mean = float(r.mean()) if r.size else 0.0
        residual_means.append(r_mean)
        residual_abs_means.append(float(np.abs(r).mean()) if r.size else 0.0)
        signs.append(1 if r_mean > 0 else (-1 if r_mean < 0 else 0))
        c_type = str(c.get("structured_events", {}).get("event_type", "other"))
        type_matches.append(1.0 if q_type in {"", "other"} or q_type == c_type else 0.0)

    signs_arr = np.asarray(signs, dtype=np.int32)
    nonzero = signs_arr[np.abs(signs_arr) > 0]
    if nonzero.size == 0:
        direction_agree = 0.0
        pos_frac = 0.5
    else:
        pos_frac = float((nonzero > 0).mean())
        neg_frac = float((nonzero < 0).mean())
        direction_agree = max(pos_frac, neg_frac)

    feat_vals = [
        soft_weight,
        confidence,
        valid,
        float(price_scores.max(initial=0.0)),
        float(price_scores.mean() if price_scores.size else 0.0),
        float(price_scores.std() if price_scores.size else 0.0),
        float(event_scores.max(initial=0.0)),
        float(event_scores.mean() if event_scores.size else 0.0),
        float(recency_scores.mean() if recency_scores.size else 0.0),
        float(regime_scores.mean() if regime_scores.size else 0.0),
        float(np.mean(residual_abs_means) if residual_abs_means else 0.0),
        float(np.std(residual_abs_means) if residual_abs_means else 0.0),
        float(pos_frac),
        float(direction_agree),
        float(np.mean(type_matches) if type_matches else 0.0),
        float(final_scores.max(initial=0.0)),
        float(final_scores.mean() if final_scores.size else 0.0),
        float(np.mean(residual_means) if residual_means else 0.0),
    ]

    feat = np.zeros((feature_dim,), dtype=np.float32)
    n = min(feature_dim, len(feat_vals))
    feat[:n] = np.asarray(feat_vals[:n], dtype=np.float32)

    meta = {
        "retrieval_valid": bool(valid > 0.5),
        "retrieval_confidence": confidence,
        "retrieval_soft_weight": soft_weight,
        "retrieval_mode": str(retrieval_output.get("mode", "off")),
        "reject_reason": str(retrieval_output.get("reject_reason", "")),
        "direction_agreement": float(_safe_float(retrieval_output.get("direction_agreement", direction_agree))),
        "event_mismatch_rate": float(
            _safe_float(
                retrieval_output.get(
                    "event_mismatch_rate",
                    1.0 - np.mean(type_matches) if type_matches else 1.0,
                )
            )
        ),
        "topk": int(len(candidates)),
        "knn_delta_prior_z": knn_delta_prior.tolist(),
    }
    return feat, meta


def _build_news_context_for_case(
    *,
    target_time: str,
    news_df: pd.DataFrame | None,
    args,
    tokenizer,
    policy_name: str,
    policy_kw: list[str] | None,
    api_adapter: Any = None,
) -> tuple[str, dict]:
    if news_df is None or len(news_df) == 0:
        return "", {}
    text_col = str(getattr(args, "news_text_col", "content"))
    target = pd.to_datetime(str(target_time), errors="coerce")
    if pd.isna(target):
        return "", {}
    cand = get_candidates(
        news_df,
        str(getattr(args, "news_time_col", "date")),
        target,
        int(getattr(args, "news_window_days", 1)),
        int(getattr(args, "news_topM", 20)),
    )
    if cand is None or len(cand) == 0:
        return "", {}
    selected, _ = select_news(
        cand,
        str(policy_name or "smart"),
        text_col,
        policy_kw or [],
        int(getattr(args, "news_topK", 5)),
        args=args,
    )
    if selected is None or len(selected) == 0:
        return "", {}
    raw_news_texts = selected[text_col].fillna("").astype(str).tolist()
    max_tokens = int(getattr(args, "token_budget", 700) * float(getattr(args, "token_budget_news_frac", 0.9)))
    context = {
        "target_time": str(target_time),
        "region": str(getattr(args, "region", "")),
        "description": str(getattr(args, "description", "")),
    }
    refined_news = _refine_news_from_doc_cache_for_case(
        raw_news_texts=raw_news_texts,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        args=args,
        api_adapter=api_adapter,
    )

    if not refined_news:
        refine_mode = str(getattr(args, "news_refine_mode", "local"))
        refined_news = refine_news_text(
            raw_news_texts=raw_news_texts,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            mode=refine_mode,
            api_adapter=api_adapter,
            context=context,
        )

    if not refined_news:
        return "", {}
    events = extract_structured_events(
        raw_or_refined_news=refined_news,
        mode=str(getattr(args, "news_structured_mode", "off")),
        api_adapter=api_adapter,
        context=context,
    )
    return refined_news, events


@torch.no_grad()
def build_case_bank(
    *,
    train_loader,
    base_model,
    args,
    global_zstats: dict,
    news_df: pd.DataFrame | None = None,
    tokenizer=None,
    policy_name: str = "smart",
    policy_kw: list[str] | None = None,
    live_logger=None,
    api_adapter: Any = None,
) -> dict:
    if base_model is None:
        return {"cases": [], "state_matrix": np.zeros((0, 0), dtype=np.float32)}

    mu = _safe_float(global_zstats.get("mu_global", global_zstats.get("mu", 0.0)), default=0.0)
    sigma = _safe_float(global_zstats.get("sigma_global", global_zstats.get("sigma", 1.0)), default=1.0)
    sigma = max(sigma, 1e-6)

    device = next(base_model.parameters()).device
    base_model.eval()
    cases: list[dict] = []
    bank_limit = int(max(0, getattr(args, "case_retrieval_bank_max", 0)))
    use_news_context = int(getattr(args, "case_retrieval_enable", 0)) == 1
    for batch in train_loader:
        history_raw = batch["history_value"].to(torch.float32)
        target_raw = batch["target_value"].to(torch.float32)
        history_z = (history_raw - mu) / sigma
        target_z = (target_raw - mu) / sigma

        pred_z = base_model(history_z.to(device)).to(torch.float32).cpu()
        B = pred_z.size(0)
        for i in range(B):
            target_time = str(batch["target_time"][i])
            series_id = str(batch.get("series_id", ["Not Specified"] * B)[i])
            sample_id = f"{series_id}::{target_time}"

            refined_news = ""
            structured_events = {}
            if use_news_context and news_df is not None and len(news_df) > 0:
                refined_news, structured_events = _build_news_context_for_case(
                    target_time=target_time,
                    news_df=news_df,
                    args=args,
                    tokenizer=tokenizer,
                    policy_name=policy_name,
                    policy_kw=policy_kw,
                    api_adapter=api_adapter,
                )

            case = build_case_record(
                sample_id=sample_id,
                split="train",
                target_time=target_time,
                history_z=history_z[i].cpu().numpy(),
                base_pred_z=pred_z[i].cpu().numpy(),
                target_z=target_z[i].cpu().numpy(),
                refined_news=refined_news,
                structured_events=structured_events,
                metadata={
                    "series_id": series_id,
                },
            )
            cases.append(case)
            # with open("case711.json", "a", encoding="utf-8") as _f:
            #     _f.write(json.dumps(case, ensure_ascii=False) + "\n")
            if bank_limit > 0 and len(cases) >= bank_limit:
                break
        if bank_limit > 0 and len(cases) >= bank_limit:
            break

    state_matrix = np.asarray([_safe_array(c.get("state_vector", [])) for c in cases], dtype=np.float32)
    bank = {
        "cases": cases,
        "state_matrix": state_matrix,
        "created_at": datetime.utcnow().isoformat(),
        "feature_dim": int(state_matrix.shape[1]) if state_matrix.ndim == 2 and state_matrix.size else 0,
    }
    if live_logger is not None:
        live_logger.info(
            f"[CASE_BANK] built train-only bank: n_cases={len(cases)} "
            f"state_dim={bank['feature_dim']} policy={policy_name}"
        )
    return bank


def save_case_bank(path: str, case_bank: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(case_bank or {})
    state_matrix = payload.get("state_matrix", None)
    if isinstance(state_matrix, np.ndarray):
        payload["state_matrix"] = state_matrix.tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_case_bank(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sm = payload.get("state_matrix", [])
    payload["state_matrix"] = np.asarray(sm, dtype=np.float32)
    return payload
