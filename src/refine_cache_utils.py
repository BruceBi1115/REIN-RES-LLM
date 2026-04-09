from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any


def _abs_path(path: Any) -> str:
    s = str(path or "").strip()
    if not s:
        return ""
    try:
        return os.path.abspath(s)
    except Exception:
        return s


def _normalize_cache_name_part(text: Any, fallback: str = "na") -> str:
    s = str(text or "").strip()
    if not s:
        return fallback
    s = os.path.basename(s)
    s = os.path.splitext(s)[0]
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()
    return s or fallback


def _dataset_source(args) -> str:
    for attr in ("train_file", "val_file", "test_file"):
        raw = _abs_path(getattr(args, attr, ""))
        if not raw:
            continue
        parent = os.path.basename(os.path.dirname(raw))
        if parent:
            return parent
        stem = os.path.splitext(os.path.basename(raw))[0]
        if stem:
            return stem
    return (
        str(getattr(args, "taskName", "") or "").strip()
        or str(getattr(args, "description", "") or "").strip()
        or "dataset"
    )


def _news_source(args) -> str:
    raw = _abs_path(getattr(args, "news_path", ""))
    if raw:
        stem = os.path.splitext(os.path.basename(raw))[0]
        if stem:
            return stem
    return _dataset_source(args)


def resolve_refine_description(args) -> str:
    desc = str(getattr(args, "description", "") or "").strip()
    if desc:
        return desc
    
    
    # if args.description isss not provided
    value_col = str(getattr(args, "value_col", "") or "").strip() or "target series"
    dataset_source = _dataset_source(args)
    dataset_name = _normalize_cache_name_part(dataset_source, fallback="dataset").replace("_", " ")
    return f"This dataset forecasts {value_col} for {dataset_name}."


def build_refine_context(args, target_time: Any = "") -> dict:
    return {
        "target_time": str(target_time or "").strip(),
        "region": str(getattr(args, "region", "") or "").strip(),
        "description": resolve_refine_description(args),
    }


def make_refine_news_cache_key(
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


def make_refine_doc_cache_key(
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


def make_structured_doc_cache_key(
    *,
    news_text: str,
    context: dict,
    mode: str,
    model: str,
    source_kind: str = "refined",
) -> str:
    txt = str(news_text or "").strip()
    payload = {
        "kind": "structured_doc",
        "mode": str(mode or "").lower().strip(),
        "model": str(model or "").strip(),
        "source_kind": str(source_kind or "refined").lower().strip(),
        "region": str(context.get("region", "")).strip(),
        "description": str(context.get("description", "")).strip(),
        "news_text": txt,
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "structured::" + hashlib.sha1(s.encode("utf-8")).hexdigest()


def _dataset_refine_cache_slug(args) -> str:
    news_part = _normalize_cache_name_part(_news_source(args), fallback="news")
    return f"{news_part}"


def _stable_refine_cache_tag(args) -> str:
    payload = {
        "news_path": _abs_path(getattr(args, "news_path", "")),
        "train_file": _abs_path(getattr(args, "train_file", "")),
        "val_file": _abs_path(getattr(args, "val_file", "")),
        "test_file": _abs_path(getattr(args, "test_file", "")),
        "news_text_col": str(getattr(args, "news_text_col", "content") or "content"),
        "refine_mode": str(getattr(args, "news_refine_mode", "local") or "local").lower().strip(),
        "api_model": str(getattr(args, "news_api_model", "") or "").strip(),
        "time_col": str(getattr(args, "time_col", "") or "").strip(),
        "value_col": str(getattr(args, "value_col", "") or "").strip(),
        "region": str(getattr(args, "region", "") or "").strip(),
        "description": resolve_refine_description(args),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def resolve_refine_cache_path(args) -> str:
    return resolve_news_doc_cache_path(args)


def resolve_news_doc_cache_path(args) -> str:
    p = str(getattr(args, "news_doc_cache_path", "") or "").strip()
    if p:
        return p
    p = str(getattr(args, "news_refine_cache_path", "") or "").strip()
    if p:
        return p
    p = str(getattr(args, "news_structured_cache_path", "") or "").strip()
    if p:
        return p
    dataset_slug = _dataset_refine_cache_slug(args)
    return os.path.join("./checkpoints", "_shared_refine_cache", f"news_doc_cache_{dataset_slug}.json")


def resolve_structured_cache_path(args) -> str:
    return resolve_news_doc_cache_path(args)
