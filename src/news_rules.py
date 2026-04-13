from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
import pandas as pd

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
    work["_identity_url"] = work["url"].apply(_normalize_news_identity_url_local) if "url" in work.columns else ""
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
    work["_content_len"] = work[content_col].fillna("").astype(str).str.len() if content_col else 0
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
        "The framework keeps the most informative row per identity.",
        RuntimeWarning,
        stacklevel=2,
    )
    return deduped


def load_news(path: str, time_col: str, tz: str) -> pd.DataFrame:
    if not path.endswith(".json"):
        raise ValueError(f"Only .json files are supported, got: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON array of news objects, got: {type(payload).__name__}")

    df = pd.DataFrame(payload)
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in JSON file.")

    df[time_col] = df[time_col].apply(_parse_news_datetime)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    if tz:
        if getattr(df[time_col].dt, "tz", None) is None:
            df[time_col] = df[time_col].dt.tz_localize(tz)
        else:
            df[time_col] = df[time_col].dt.tz_convert(tz)

    df = _deduplicate_news_identity_rows(df, time_col, source_path=path)
    return df.sort_values(time_col).reset_index(drop=True)
