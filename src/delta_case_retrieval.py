from __future__ import annotations

import json
import math
import os
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
    alpha_price: float = 0.85,
    alpha_event: float = 0.15,
) -> list[dict]:
    if not candidates:
        return []
    q_event = query_case.get("structured_events", {})
    use_event = isinstance(q_event, dict) and len(q_event) > 0 and alpha_event > 0.0
    ranked = []
    for c in candidates:
        price_score = _safe_float(c.get("price_score", 0.0), default=0.0)
        event_score = 0.0
        if use_event:
            event_score = _event_similarity(q_event, c.get("structured_events", {}))
        final_score = alpha_price * price_score + alpha_event * event_score
        cc = dict(c)
        cc["event_score"] = float(event_score)
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


def retrieve_similar_cases(
    *,
    query_case: dict,
    case_bank: dict | None,
    top_n: int = 5,
    mode: str = "price_event",
    alpha_price: float = 0.85,
    alpha_event: float = 0.15,
    min_top_score: float = 0.12,
    min_candidates: int = 2,
    min_direction_agreement: float = 0.45,
    max_event_mismatch: float = 0.8,
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
        }

    candidates.sort(key=lambda x: float(x.get("price_score", 0.0)), reverse=True)
    pre_top = candidates[: max(top_n * 4, top_n)]

    use_mode = str(mode or "price_event").lower().strip()
    if use_mode == "price":
        ranked = []
        for c in pre_top:
            cc = dict(c)
            cc["event_score"] = 0.0
            cc["final_score"] = float(cc.get("price_score", 0.0))
            ranked.append(cc)
    else:
        ap = float(max(0.5, min(1.0, alpha_price)))
        ae = float(max(0.0, min(0.5, alpha_event)))
        if ap <= ae:
            ap = 0.7
            ae = 0.3
        ranked = rerank_cases_by_event(
            query_case=query_case,
            candidates=pre_top,
            alpha_price=ap,
            alpha_event=ae,
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
    out = {
        "candidates": ranked,
        "mode": use_mode,
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

    if not candidates:
        feat = np.zeros((feature_dim,), dtype=np.float32)
        feat[0] = valid
        feat[1] = confidence
        meta = {
            "retrieval_valid": bool(valid > 0.5),
            "retrieval_confidence": confidence,
            "retrieval_mode": str(retrieval_output.get("mode", "off")),
            "reject_reason": str(retrieval_output.get("reject_reason", "empty")),
            "coverage": float(valid),
        }
        return feat, meta

    price_scores = np.asarray([_safe_float(c.get("price_score", 0.0)) for c in candidates], dtype=np.float32)
    event_scores = np.asarray([_safe_float(c.get("event_score", 0.0)) for c in candidates], dtype=np.float32)
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
        valid,
        confidence,
        float(price_scores.max(initial=0.0)),
        float(price_scores.mean() if price_scores.size else 0.0),
        float(price_scores.std() if price_scores.size else 0.0),
        float(event_scores.max(initial=0.0)),
        float(event_scores.mean() if event_scores.size else 0.0),
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
    refined_news = refine_news_text(
        raw_news_texts=raw_news_texts,
        tokenizer=tokenizer,
        max_tokens=int(getattr(args, "token_budget", 700) * float(getattr(args, "token_budget_news_frac", 0.9))),
        mode=str(getattr(args, "news_refine_mode", "local")),
        api_adapter=api_adapter,
        context={"target_time": str(target_time), "region": str(getattr(args, "region", ""))},
    )
    if not refined_news:
        return "", {}
    events = extract_structured_events(
        raw_or_refined_news=refined_news,
        mode=str(getattr(args, "news_structured_mode", "off")),
        api_adapter=api_adapter,
        context={"target_time": str(target_time), "region": str(getattr(args, "region", ""))},
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
    use_events = str(getattr(args, "news_structured_mode", "off")).lower().strip() in {"heuristic", "api"}
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
            if use_events and news_df is not None and len(news_df) > 0:
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
            with open("case711.json", "a", encoding="utf-8") as _f:
                _f.write(json.dumps(case, ensure_ascii=False) + "\n")
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
