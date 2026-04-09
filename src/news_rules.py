import json
import os
import re
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

from .news_datetime import normalize_news_datetime, parse_news_datetime


def _parse_news_datetime(raw):
    dt = parse_news_datetime(raw, dayfirst=True)
    if dt is not None:
        return pd.Timestamp(dt)
    return pd.to_datetime(raw, dayfirst=True, errors="coerce")


def _normalize_news_identity_title_local(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def _normalize_news_identity_time_local(value) -> str:
    norm = normalize_news_datetime(value, dayfirst=True, floor="s")
    if norm:
        return norm
    ts = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return ""
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.floor("s").isoformat()


def _normalize_news_identity_url_local(url: str) -> str:
    return re.sub(r"\s+", "", str(url or "").strip())


def _deduplicate_news_identity_rows(df: pd.DataFrame, time_col: str, *, source_path: str = "") -> pd.DataFrame:
    if df is None or len(df) == 0 or "title" not in df.columns or time_col not in df.columns:
        return df

    work = df.copy()
    work["_identity_title"] = work["title"].apply(_normalize_news_identity_title_local)
    work["_identity_time"] = work[time_col].apply(_normalize_news_identity_time_local)
    work["_identity_url"] = (
        work["url"].apply(_normalize_news_identity_url_local) if "url" in work.columns else ""
    )
    valid_identity = work["_identity_title"].ne("") & work["_identity_time"].ne("")
    if not valid_identity.any():
        return df

    dup_mask = valid_identity & work.duplicated(
        subset=["_identity_title", "_identity_time", "_identity_url"],
        keep=False,
    )
    if not dup_mask.any():
        return df

    content_col = "content" if "content" in work.columns else ""
    url_col = "url" if "url" in work.columns else ""
    work["_content_len"] = (
        work[content_col].fillna("").astype(str).str.len() if content_col else 0
    )
    work["_canonical_url_score"] = (
        work[url_col]
        .fillna("")
        .astype(str)
        .apply(lambda s: 0 if re.search(r"(?:-|%3A)0/?$", s.strip()) else 1)
        if url_col
        else 0
    )
    work["_original_order"] = np.arange(len(work), dtype=np.int64)

    deduped = (
        work.sort_values(
            by=[
                "_identity_title",
                "_identity_time",
                "_identity_url",
                "_content_len",
                "_canonical_url_score",
                "_original_order",
            ],
            ascending=[True, True, True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["_identity_title", "_identity_time", "_identity_url"], keep="first")
        .sort_values(by=["_original_order"], kind="mergesort")
        .drop(
            columns=[
                "_identity_title",
                "_identity_time",
                "_identity_url",
                "_content_len",
                "_canonical_url_score",
                "_original_order",
            ]
        )
        .reset_index(drop=True)
    )

    removed = int(len(df) - len(deduped))
    dup_groups = int(
        work.loc[dup_mask, ["_identity_title", "_identity_time", "_identity_url"]].drop_duplicates().shape[0]
    )
    warnings.warn(
        f"[NEWS] Deduplicated {removed} rows across {dup_groups} duplicate title+date+url identities from {source_path or 'news source'}. "
        "The framework keeps the most informative row per identity to preserve strict refined-cache matching.",
        RuntimeWarning,
        stacklevel=2,
    )
    return deduped


def load_news(path: str, time_col: str, tz: str) -> pd.DataFrame:
    """
    Load news ONLY from a JSON file that contains an array of objects.
    - path: path to a .json file
    - time_col: name of the datetime column in the JSON objects
    - tz: target timezone string (e.g., "Australia/Sydney")
    """
    if not path.endswith(".json"):
        raise ValueError(f"Only .json files are supported, got: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON array of news objects, got: {type(payload).__name__}")
    df = pd.DataFrame(payload)

    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in JSON file.")

    # Parse using the input data's own time basis; only localize/convert if the caller
    # explicitly provides a timezone.
    df[time_col] = df[time_col].apply(_parse_news_datetime)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    if tz:
        if getattr(df[time_col].dt, "tz", None) is None:
            df[time_col] = df[time_col].dt.tz_localize(tz)
        else:
            df[time_col] = df[time_col].dt.tz_convert(tz)

    df = _deduplicate_news_identity_rows(df, time_col, source_path=path)

    # print(df.sort_values(time_col).reset_index(drop=True))

    return df.sort_values(time_col).reset_index(drop=True)


def _align_ts_to_series_tz(ts, ref_series: pd.Series) -> pd.Timestamp:
    """将 ts 解析为 pandas.Timestamp，并对齐到 ref_series 的时区（或去掉时区）。"""
    
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return ts
    # 取新闻列的时区（可能为 None）
    tz = getattr(ref_series.dt, "tz", None)
    # print("ref_series tz:", tz)
    # print("ts.tzinfo = ",ts.tzinfo)
    if tz is not None:
        # ref 是 tz-aware
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
    else:
        # ref 是 tz-naive
        if ts.tzinfo:
            ts = ts.tz_localize(None)  # 保留输入数据的本地时间，不强制转 UTC
    return ts

def get_candidates(news_df, time_col, target_time, window_days, topM):
    # ✅ 先把传入的 target_time 对齐到新闻列的时区
    # print("original target_time:", target_time)
    target_time = _align_ts_to_series_tz(target_time, news_df[time_col])

    # print("Target time aligned to news timezone:", target_time)
    if pd.isna(target_time):
        return news_df.iloc[0:0]  # 空 DataFrame：没有候选

    start = target_time - pd.Timedelta(days=window_days)
    # print("Start time for candidates:", start)
    
    cand = news_df[(news_df[time_col] >= start) & (news_df[time_col] < target_time)]
    # print(f"Found {len(cand)} candidates in the window from {start} to {target_time}")
    # 取最近 topM 条
    return cand.sort_values(time_col, ascending=False).head(topM)




def policy_by_keywords(cand: pd.DataFrame, text_col: str, keywords: List[str], K: int) -> pd.DataFrame:
    if not keywords:
        return cand.head(K)
    m = cand[text_col].fillna('').astype(str).str.lower()
    mask = m.apply(lambda x: any(kw.strip() in x for kw in keywords))
    filtered = cand[mask]
    if len(filtered) == 0:
        return cand.head(K)
    return filtered.head(K)


def _arg_or_default(args, key: str, default):
    if args is None:
        return default
    v = getattr(args, key, default)
    return default if v is None else v


def _normalize01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _keyword_ratio(text_series: pd.Series, keywords: List[str]) -> np.ndarray:
    if not keywords:
        return np.zeros((len(text_series),), dtype=np.float32)
    kws = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
    if not kws:
        return np.zeros((len(text_series),), dtype=np.float32)

    lowered = text_series.fillna("").astype(str).str.lower()
    vals = []
    for s in lowered.tolist():
        hits = sum(1 for kw in kws if kw in s)
        vals.append(float(hits) / float(len(kws)))
    return np.asarray(vals, dtype=np.float32)


def _smart_select_news(
    cand: pd.DataFrame,
    text_col: str,
    policy_kw: List[str],
    K: int,
    args=None,
) -> pd.DataFrame:
    """
    Hybrid retrieval:
      1) relevance (TF-IDF query similarity)
      2) time decay (recent first)
      3) optional rating prior
      4) MMR diversification + near-duplicate suppression
    """
    if len(cand) == 0 or K <= 0:
        return cand.head(0)

    texts = cand[text_col].fillna("").astype(str)
    n = len(texts)
    if n == 1:
        return cand.head(1)

    try:
        vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
        mat = vec.fit_transform(texts.tolist())  # (N,V)
    except Exception:
        # Fallback to recency when vectorization fails on degenerate text.
        return cand.head(K)

    # relevance from keyword query
    kw_query = " ".join([kw.strip() for kw in (policy_kw or []) if kw and kw.strip()])
    if kw_query.strip():
        qv = vec.transform([kw_query])
        rel = cosine_similarity(mat, qv).reshape(-1).astype(np.float32)
    else:
        rel = np.zeros((n,), dtype=np.float32)

    kw_ratio = _keyword_ratio(texts, policy_kw)

    # rating prior (if available)
    if "rate" in cand.columns:
        rate = cand["rate"].fillna(0.0).astype(float).to_numpy()
        rate = _normalize01(rate)
    else:
        rate = np.zeros((n,), dtype=np.float32)

    # recency prior: cand is already sorted by time descending in get_candidates
    rec_tau = float(_arg_or_default(args, "smart_recency_tau", 8.0))
    rec_tau = max(1e-6, rec_tau)
    rank = np.arange(n, dtype=np.float32)
    recency = np.exp(-rank / rec_tau).astype(np.float32)
    recency = _normalize01(recency)

    rel_w = float(_arg_or_default(args, "smart_rel_weight", 0.55))
    kw_w = float(_arg_or_default(args, "smart_kw_weight", 0.15))
    rate_w = float(_arg_or_default(args, "smart_rate_weight", 0.15))
    rec_w = float(_arg_or_default(args, "smart_recency_weight", 0.15))
    w_sum = rel_w + kw_w + rate_w + rec_w
    if w_sum <= 1e-12:
        rel_w, kw_w, rate_w, rec_w = 1.0, 0.0, 0.0, 0.0
        w_sum = 1.0
    rel_w, kw_w, rate_w, rec_w = rel_w / w_sum, kw_w / w_sum, rate_w / w_sum, rec_w / w_sum

    base_score = (
        rel_w * _normalize01(rel)
        + kw_w * _normalize01(kw_ratio)
        + rate_w * rate
        + rec_w * recency
    ).astype(np.float32)

    sim = cosine_similarity(mat).astype(np.float32)
    mmr_lambda = float(_arg_or_default(args, "smart_mmr_lambda", 0.75))
    mmr_lambda = max(0.0, min(1.0, mmr_lambda))
    dedup_th = float(_arg_or_default(args, "smart_dedup_threshold", 0.92))
    dedup_th = max(0.0, min(1.0, dedup_th))

    selected = []
    pool = list(range(n))
    while pool and len(selected) < K:
        best_i, best_val = None, -1e18
        for i in pool:
            if selected:
                redundancy = float(np.max(sim[i, selected]))
            else:
                redundancy = 0.0
            mmr = mmr_lambda * float(base_score[i]) - (1.0 - mmr_lambda) * redundancy
            if mmr > best_val:
                best_val = mmr
                best_i = i

        if best_i is None:
            break
        pool.remove(best_i)

        # hard de-dup for near-identical summaries
        if selected:
            max_dup = float(np.max(sim[best_i, selected]))
            if max_dup >= dedup_th:
                continue
        selected.append(best_i)

    # fill if overly strict de-dup removed too many items
    if len(selected) < K:
        remain = [i for i in np.argsort(-base_score).tolist() if i not in selected]
        selected.extend(remain[: max(0, K - len(selected))])

    return cand.iloc[selected[:K]]


def rerank_selected_news_by_utility(
    selected: pd.DataFrame,
    target_time,
    time_col: str,
    text_col: str,
    policy_kw: List[str],
    args=None,
) -> pd.DataFrame:
    """
    Utility rerank for already-selected news items.
    Produces a `utility_score` column and sorts rows descending by utility.
    """
    if selected is None or len(selected) == 0:
        return selected

    out = selected.copy()
    texts = out[text_col].fillna("").astype(str)
    n = len(out)

    kw = _normalize01(_keyword_ratio(texts, policy_kw))

    # recency utility: closer to target_time => higher score
    recency = np.ones((n,), dtype=np.float32)
    if time_col in out.columns and target_time is not None:
        ts = pd.to_datetime(out[time_col], errors="coerce")
        tt = _align_ts_to_series_tz(target_time, ts)
        if not pd.isna(tt):
            age_h = (tt - ts).dt.total_seconds() / 3600.0
            age_h = age_h.fillna(1e9).clip(lower=0.0)
            tau_h = float(_arg_or_default(args, "utility_recency_tau_hours", 24.0))
            tau_h = max(1e-6, tau_h)
            recency = np.exp(-(age_h.to_numpy(dtype=np.float32) / tau_h)).astype(np.float32)
            recency = _normalize01(recency)

    # external rating prior
    if "rate" in out.columns:
        rate = _normalize01(out["rate"].fillna(0.0).astype(float).to_numpy(dtype=np.float32))
    else:
        rate = np.zeros((n,), dtype=np.float32)

    # sentiment magnitude as a weak "event strength" signal
    sent_w_raw = float(_arg_or_default(args, "utility_sentiment_weight", 0.0))
    if sent_w_raw > 0.0:
        sent = texts.apply(lambda x: abs(float(TextBlob(x).sentiment.polarity))).to_numpy(dtype=np.float32)
        sent = _normalize01(sent)
    else:
        sent = np.zeros((n,), dtype=np.float32)

    kw_w = float(_arg_or_default(args, "utility_keyword_weight", 0.35))
    rec_w = float(_arg_or_default(args, "utility_recency_weight", 0.25))
    rate_w = float(_arg_or_default(args, "utility_rate_weight", 0.35))
    sent_w = max(0.0, sent_w_raw)
    w_sum = kw_w + rec_w + rate_w + sent_w
    if w_sum <= 1e-12:
        kw_w, rec_w, rate_w, sent_w, w_sum = 1.0, 0.0, 0.0, 0.0, 1.0
    kw_w, rec_w, rate_w, sent_w = kw_w / w_sum, rec_w / w_sum, rate_w / w_sum, sent_w / w_sum

    base_score = (kw_w * kw + rec_w * recency + rate_w * rate + sent_w * sent).astype(np.float32)

    # Optional MMR-style diversification at rerank stage.
    use_mmr = int(_arg_or_default(args, "utility_mmr_enable", 1)) == 1 and n > 1
    if use_mmr:
        try:
            vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)
            mat = vec.fit_transform(texts.tolist())
            sim = cosine_similarity(mat).astype(np.float32)
            lam = float(_arg_or_default(args, "utility_mmr_lambda", 0.8))
            lam = max(0.0, min(1.0, lam))
            dedup_th = float(_arg_or_default(args, "utility_dedup_threshold", 0.95))
            dedup_th = max(0.0, min(1.0, dedup_th))

            order = []
            pool = list(range(n))
            while pool:
                best_i, best_v = None, -1e18
                for i in pool:
                    redundancy = float(np.max(sim[i, order])) if order else 0.0
                    mmr = lam * float(base_score[i]) - (1.0 - lam) * redundancy
                    if mmr > best_v:
                        best_i, best_v = i, mmr
                if best_i is None:
                    break
                pool.remove(best_i)
                if order and float(np.max(sim[best_i, order])) >= dedup_th:
                    continue
                order.append(best_i)

            if len(order) < n:
                remaining = [i for i in np.argsort(-base_score).tolist() if i not in order]
                order.extend(remaining)
        except Exception:
            order = np.argsort(-base_score).tolist()
    else:
        order = np.argsort(-base_score).tolist()

    out["utility_score"] = base_score
    out = out.iloc[order].copy()

    keep_k = int(_arg_or_default(args, "utility_keep_topk", -1))
    if keep_k > 0:
        out = out.head(keep_k)

    min_s = float(_arg_or_default(args, "utility_min_score", -1.0))
    if min_s > -0.999999:
        out = out[out["utility_score"] >= min_s]
        if len(out) == 0:
            # Never return empty when caller already had selected news.
            out = selected.head(1).copy()
            out["utility_score"] = float(base_score[np.argsort(-base_score)[0]])

    return out.reset_index(drop=True)

def select_by_sentiment_polarity_high(cand: pd.DataFrame, text_col: str, K: int) -> pd.DataFrame:
    if len(cand) == 0:
        return cand
    sentiments = cand[text_col].fillna('').astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    order = np.argsort(-sentiments.values)
    return cand.iloc[order].head(K)

def select_by_sentiment_polarity_low(cand: pd.DataFrame, text_col: str, K: int) -> pd.DataFrame:
    if len(cand) == 0:
        return cand
    sentiments = cand[text_col].fillna('').astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    order = np.argsort(sentiments.values)
    return cand.iloc[order].head(K)

def keyword_polarity_high_hybrid(cand: pd.DataFrame, text_col: str, keywords, K: int) -> pd.DataFrame:
    if len(cand) == 0 or K <= 0:
        return cand.head(0)

    s = cand[text_col].fillna('').astype(str)
    sentiments = s.apply(lambda x: TextBlob(x).sentiment.polarity)

    # 关键词命中
    if keywords:
        kws = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
        mask = s.str.lower().apply(lambda x: any(kw in x for kw in kws))
    else:
        mask = pd.Series(False, index=cand.index)

    # 先在命中集内按情感排序
    idx_pref = sentiments[mask].sort_values(ascending=False).index.tolist()
    chosen = idx_pref[:K]

    # 不足则从非命中集里按情感排序补齐
    if len(chosen) < K:
        need = K - len(chosen)
        idx_rest = sentiments[~mask].sort_values(ascending=False).index.tolist()
        chosen += idx_rest[:need]

    return cand.loc[chosen]

def keyword_polarity_low_hybrid(cand: pd.DataFrame, text_col: str, keywords, K: int) -> pd.DataFrame:
    if len(cand) == 0 or K <= 0:
        return cand.head(0)

    s = cand[text_col].fillna('').astype(str)
    sentiments = s.apply(lambda x: TextBlob(x).sentiment.polarity)

    # 关键词命中
    if keywords:
        kws = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
        mask = s.str.lower().apply(lambda x: any(kw in x for kw in kws))
    else:
        mask = pd.Series(False, index=cand.index)

    # 先在命中集内按情感排序
    idx_pref = sentiments[mask].sort_values(ascending=True).index.tolist()
    chosen = idx_pref[:K]

    # 不足则从非命中集里按情感排序补齐
    if len(chosen) < K:
        need = K - len(chosen)
        idx_rest = sentiments[~mask].sort_values(ascending=True).index.tolist()
        chosen += idx_rest[:need]

    return cand.loc[chosen]

def select_news(
    cand: pd.DataFrame,
    policy: str,
    text_col: str,
    policy_kw: List[str],
    K: int,
    args=None,
) -> tuple[pd.DataFrame, float]:
    # print(cand)
    selected = cand
    if policy == 'keywords':
        selected = policy_by_keywords(cand, text_col, policy_kw, K) #这里把命中关键词的都选上
    elif policy == 'polarity_high':
        selected = select_by_sentiment_polarity_high(cand, text_col, K)
    elif policy == 'polarity_low':
        selected = select_by_sentiment_polarity_low(cand, text_col, K)
    elif policy == 'keyword_polarity_high_hybrid':
        selected = keyword_polarity_high_hybrid(cand, text_col, policy_kw, K)#先按keyword筛选，然后选个
    elif policy == 'keyword_polarity_low_hybrid':
        selected = keyword_polarity_low_hybrid(cand, text_col, policy_kw, 1)#先按keyword筛选，然后选个
    elif policy == "all":
        selected = cand.head(K)
    elif policy in {"smart", "auto"}:
        selected = _smart_select_news(cand, text_col, policy_kw, K, args=args)
    else:
        raise ValueError("Unknown news select policy")

    return selected

def lead3(text: str, max_sentences: int = 3) -> str:
    seps = ['。', '.', '；', ';', '！', '!', '？', '?', '\n']
    tmp = text
    for s in seps:
        tmp = tmp.replace(s, '.')
    parts = [p.strip() for p in tmp.split('.') if p.strip()]
    return ' '.join(parts[:max_sentences])
