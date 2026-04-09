from __future__ import annotations

import json
import math
import os
import re
import hashlib
import sys
import textwrap

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..delta_news_hooks import discover_news_api_key, extract_structured_events, merge_structured_events, refine_news_text
from ..news_datetime import normalize_news_datetime
from ..refine_cache_utils import (
    build_refine_context,
    make_refine_doc_cache_key,
    make_structured_doc_cache_key,
    resolve_refine_cache_path,
    resolve_structured_cache_path,
)

def _refine_doc_cache_key(
    *,
    raw_news_text: str,
    context: dict,
    mode: str,
    model: str,
    max_tokens: int,
) -> str:
    return make_refine_doc_cache_key(
        raw_news_text=raw_news_text,
        context=context,
        mode=mode,
        model=model,
        max_tokens=max_tokens,
    )

def _truncate_with_tokenizer(text: str, tokenizer, max_tokens: int) -> str:
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
        ids = enc.get("input_ids", []) if hasattr(enc, "get") else []
        if len(ids) == 0:
            return s[: n * 4]
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except Exception:
        return s[: n * 4]

def _refine_cache_path(args) -> str:
    return resolve_refine_cache_path(args)

def _parse_refine_cache_read_paths(raw_value) -> list[str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return []
    out = []
    seen = set()
    for part in raw.replace("\n", ",").split(","):
        p = str(part or "").strip()
        if not p:
            continue
        norm = os.path.abspath(p)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(p)
    return out

def _load_refine_cache_file(path: str) -> dict[str, str]:
    store = {}
    if not path or (not os.path.exists(path)):
        return store
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            if isinstance(obj.get("cache_key"), str) and isinstance(obj.get("refined_news"), str):
                k = str(obj.get("cache_key", "")).strip()
                v = str(obj.get("refined_news", ""))
                store = {k: v} if k else {}
            else:
                store = {str(k): str(v) for k, v in obj.items() if isinstance(v, str)}
        elif isinstance(obj, list):
            parsed = {}
            for rec in obj:
                if not isinstance(rec, dict):
                    continue
                key = rec.get("cache_key", rec.get("key", rec.get("id", "")))
                value = rec.get("refined_news", rec.get("value", rec.get("text", "")))
                if isinstance(key, str) and isinstance(value, str):
                    k = key.strip()
                    if k:
                        parsed[k] = value
            store = parsed
    except Exception:
        parsed = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = str(line or "").strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    key = rec.get("cache_key", rec.get("key", rec.get("id", "")))
                    value = rec.get("refined_news", rec.get("value", rec.get("text", "")))
                    if isinstance(key, str) and isinstance(value, str):
                        k = key.strip()
                        if k:
                            parsed[k] = value
        except Exception:
            parsed = {}
        store = parsed
    return store

def _load_structured_cache_file(path: str) -> dict[str, dict]:
    store = {}
    if not path or (not os.path.exists(path)):
        return store
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            if isinstance(obj.get("structured_cache_key"), str) and isinstance(obj.get("structured_events"), dict):
                kk = str(obj.get("structured_cache_key", "")).strip()
                if kk:
                    store[kk] = dict(obj.get("structured_events", {}))
            else:
                for k, v in obj.items():
                    if isinstance(v, dict):
                        kk = str(k).strip()
                        if kk:
                            store[kk] = dict(v)
        elif isinstance(obj, list):
            parsed = {}
            for rec in obj:
                if not isinstance(rec, dict):
                    continue
                key = rec.get("structured_cache_key", rec.get("key", rec.get("id", "")))
                value = rec.get("structured_events", rec.get("value", {}))
                if isinstance(key, str) and isinstance(value, dict):
                    k = key.strip()
                    if k:
                        parsed[k] = dict(value)
            store = parsed
    except Exception:
        parsed = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = str(line or "").strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    key = rec.get("structured_cache_key", rec.get("key", rec.get("id", "")))
                    value = rec.get("structured_events", rec.get("value", {}))
                    if isinstance(key, str) and isinstance(value, dict):
                        k = key.strip()
                        if k:
                            parsed[k] = dict(value)
        except Exception:
            parsed = {}
        store = parsed
    return store

def _load_news_doc_cache_file(path: str, *, args=None) -> dict[str, dict]:
    store = {}
    if not path or (not os.path.exists(path)):
        return store

    def _ingest_record(rec: dict):
        if not isinstance(rec, dict):
            return
        norm = _normalize_news_doc_cache_record(rec)
        primary = _news_doc_primary_key_from_meta(
            norm,
            dayfirst=bool(getattr(args, "dayFirst", False)) if args is not None else False,
            origin=f"cache_file_record[path={path}]",
        )
        store[primary] = _merge_news_doc_cache_records(store.get(primary, {}), norm)

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        parsed_records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = str(line or "").strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                parsed_records.append(rec)
        for rec in parsed_records:
            _ingest_record(rec)
        return store

    if isinstance(obj, dict):
        _ingest_record(obj)
    elif isinstance(obj, list):
        for rec in obj:
            _ingest_record(rec)
    return store

def _log_news_doc_cache_detection(live_logger, *, tag: str, path: str, object_count: int):
    if live_logger is None:
        return
    clean_path = str(path or "").strip()
    detected = bool(clean_path) and os.path.exists(clean_path) and int(object_count) > 0
    if clean_path:
        live_logger.info(
            f"[{tag}] cache_detected={int(detected)} file={clean_path} objects={int(object_count)}"
        )
    else:
        live_logger.info(
            f"[{tag}] cache_detected=0 file=<EMPTY> objects={int(object_count)}"
        )

def _normalize_news_identity_title(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def _normalize_news_identity_time(value, *, dayfirst: bool) -> str:
    norm = normalize_news_datetime(value, dayfirst=dayfirst, floor="s")
    if norm:
        return norm
    ts = pd.to_datetime(value, errors="coerce", dayfirst=bool(dayfirst))
    if pd.isna(ts):
        return ""
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        ts = ts.floor("s")
        return ts.isoformat()
    return str(ts)

def _normalize_news_identity_url(url: str) -> str:
    return re.sub(r"\s+", "", str(url or "").strip())


def _news_identity_key(title: str, published_at, url: str, *, dayfirst: bool) -> tuple[str, str, str]:
    return (
        _normalize_news_identity_title(title),
        _normalize_news_identity_time(published_at, dayfirst=dayfirst),
        _normalize_news_identity_url(url),
    )


def _news_doc_primary_key_from_identity_keys(title_key: str, time_key: str, url_key: str) -> str:
    payload = "\x1f".join([str(title_key), str(time_key), str(url_key)])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"identity::{digest}"


def _news_doc_primary_key_from_meta(meta: dict | None, *, dayfirst: bool, origin: str) -> str:
    meta = dict(meta or {})
    title = str(meta.get("title", "") or "").strip()
    published_at = meta.get("date", "")
    url = str(meta.get("url", "") or "").strip()
    title_key, time_key, url_key = _news_identity_key(
        title,
        published_at,
        url,
        dayfirst=bool(dayfirst),
    )
    if not title_key:
        raise ValueError(
            f"[NEWS_DOC_CACHE] missing title for {origin}. "
            f"title={title!r} date={str(published_at or '').strip()!r} url={url!r}"
        )
    if not time_key:
        raise ValueError(
            f"[NEWS_DOC_CACHE] invalid publish time for {origin}. "
            f"title={title!r} date={str(published_at or '').strip()!r} url={url!r}"
        )
    return _news_doc_primary_key_from_identity_keys(title_key, time_key, url_key)


def _require_news_identity_key(meta: dict, args, *, origin: str) -> tuple[str, str, str]:
    meta = dict(meta or {})
    title = str(meta.get("title", "") or "").strip()
    published_at = meta.get("date", "")
    url = str(meta.get("url", "") or "").strip()
    title_key, time_key, url_key = _news_identity_key(
        title,
        published_at,
        url,
        dayfirst=bool(getattr(args, "dayFirst", False)),
    )
    cache_path = str(getattr(args, "_news_doc_cache_path", "") or "").strip()
    if not title_key:
        raise ValueError(
            f"[NEWS_DOC_CACHE] missing title for {origin}. "
            f"title={title!r} date={str(published_at or '').strip()!r} url={url!r} cache_path={cache_path or '<EMPTY>'}"
        )
    if not time_key:
        raise ValueError(
            f"[NEWS_DOC_CACHE] invalid publish time for {origin}. "
            f"title={title!r} date={str(published_at or '').strip()!r} url={url!r} cache_path={cache_path or '<EMPTY>'}"
        )
    return title_key, time_key, url_key


def _warn_news_doc_cache_skip(args, *, news_meta: dict | None, reason: str):
    warned = getattr(args, "_news_doc_cache_warned_missing", None)
    if not isinstance(warned, set):
        warned = set()
    meta = dict(news_meta or {})
    key = (
        str(reason or "").strip(),
        str(meta.get("title", "") or "").strip(),
        str(meta.get("date", "") or "").strip(),
        str(meta.get("url", "") or "").strip(),
    )
    if key in warned:
        return
    warned.add(key)
    setattr(args, "_news_doc_cache_warned_missing", warned)

    count = int(getattr(args, "_news_doc_cache_warned_missing_count", 0))
    max_warn = int(max(1, getattr(args, "news_doc_cache_missing_warn_limit", 10) or 10))
    if count < max_warn:
        print(
            "[NEWS_DOC_CACHE][WARN] skipping missing cache-backed news. "
            f"reason={str(reason or '').strip() or 'missing'} "
            f"title={str(meta.get('title', '') or '').strip()!r} "
            f"date={str(meta.get('date', '') or '').strip()!r} "
            f"url={str(meta.get('url', '') or '').strip()!r}",
            file=sys.stderr,
        )
    elif count == max_warn:
        print(
            "[NEWS_DOC_CACHE][WARN] additional missing cache-backed news warnings suppressed.",
            file=sys.stderr,
        )
    setattr(args, "_news_doc_cache_warned_missing_count", count + 1)


def _news_cache_mode(args) -> str:
    return str(getattr(args, "_news_doc_cache_mode", "") or "").strip()


def _news_cache_is_read_only(args) -> bool:
    return _news_cache_mode(args) == "read_only"


def _news_cache_is_build_mode(args) -> bool:
    return _news_cache_mode(args) == "build_mode"


def _resolve_news_doc_cache_mode(args) -> dict:
    explicit_requested = int(getattr(args, "news_doc_cache_explicit", 0) or 0) == 1
    news_path = str(getattr(args, "news_path", "") or "").strip()
    cache_path = str(resolve_refine_cache_path(args) or "").strip()
    file_exists = bool(cache_path) and os.path.isfile(cache_path)

    if not news_path:
        mode = "disabled"
        source = "no_news_path"
    elif explicit_requested:
        if not file_exists:
            raise FileNotFoundError(
                f"Explicit refined news cache path was provided but does not exist: {cache_path}"
            )
        mode = "read_only"
        source = "explicit"
    elif file_exists:
        mode = "read_only"
        source = "auto_discovered"
    else:
        mode = "build_mode"
        source = "build_missing"

    state = {
        "mode": mode,
        "source": source,
        "path": cache_path,
        "exists": int(file_exists),
        "explicit": int(explicit_requested),
    }
    setattr(args, "_news_doc_cache_mode", mode)
    setattr(args, "_news_doc_cache_source", source)
    setattr(args, "_news_doc_cache_path", cache_path)
    setattr(args, "_news_doc_cache_exists", bool(file_exists))
    setattr(args, "_news_doc_cache_explicit", bool(explicit_requested))
    return state


def _should_require_news_api_adapter(args) -> bool:
    if not _news_cache_is_build_mode(args):
        return False
    refine_mode = str(getattr(args, "news_refine_mode", "local") or "local").lower().strip()
    structured_mode = str(getattr(args, "news_structured_mode", "off") or "off").lower().strip()
    return (refine_mode == "api") or (structured_mode == "api")


def _prime_news_api_key_state(args) -> dict:
    key, key_source = discover_news_api_key(args)
    state = {
        "detected": int(bool(key)),
        "source": key_source or "",
    }
    setattr(args, "_news_api_key_detected", bool(key))
    setattr(args, "_news_api_key_source", key_source or "")
    setattr(args, "_require_news_api_adapter", _should_require_news_api_adapter(args))
    return state


def _ascii_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_col_widths: list[int] | tuple[int, ...] | None = None,
) -> str:
    hdr = [str(x) for x in headers]
    ncols = len(hdr)
    body = []
    for row in rows:
        norm = [str(x) for x in row[:ncols]]
        if len(norm) < ncols:
            norm.extend([""] * (ncols - len(norm)))
        body.append(norm)

    width_limits = [0] * ncols
    if max_col_widths is not None:
        for idx in range(min(ncols, len(max_col_widths))):
            width_limits[idx] = int(max(1, int(max_col_widths[idx])))

    def _cell_lines(text: str, idx: int) -> list[str]:
        clean = str(text or "")
        raw_lines = clean.splitlines() or [""]
        limit = width_limits[idx]
        out = []
        for line in raw_lines:
            if limit > 0 and len(line) > limit:
                wrapped = textwrap.wrap(
                    line,
                    width=limit,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                out.extend(wrapped or [""])
            else:
                out.append(line)
        return out or [""]

    wrapped_hdr = [_cell_lines(cell, idx) for idx, cell in enumerate(hdr)]
    wrapped_rows = [[_cell_lines(cell, idx) for idx, cell in enumerate(row)] for row in body]

    widths = [0] * ncols
    for idx, lines in enumerate(wrapped_hdr):
        widths[idx] = max(widths[idx], max(len(line) for line in lines))
    for row in wrapped_rows:
        for idx, lines in enumerate(row):
            widths[idx] = max(widths[idx], max(len(line) for line in lines))

    def _rule(sep: str = "+", fill: str = "-") -> str:
        return sep + sep.join(fill * (w + 2) for w in widths) + sep

    def _fmt_multiline(row_lines: list[list[str]]) -> list[str]:
        height = max(len(lines) for lines in row_lines) if row_lines else 1
        out = []
        for line_idx in range(height):
            rendered = []
            for col_idx, lines in enumerate(row_lines):
                cell = lines[line_idx] if line_idx < len(lines) else ""
                rendered.append(cell.ljust(widths[col_idx]))
            out.append("| " + " | ".join(rendered) + " |")
        return out

    lines = [_rule()]
    lines.extend(_fmt_multiline(wrapped_hdr))
    lines.append(_rule())
    for row in wrapped_rows:
        lines.extend(_fmt_multiline(row))
    lines.append(_rule())
    return "\n".join(lines)


def _cache_decision_rows(args) -> list[list[str]]:
    return [
        ["news_path", str(getattr(args, "news_path", "") or "").strip() or "<EMPTY>"],
        ["cache_path", str(getattr(args, "_news_doc_cache_path", "") or "").strip() or "<EMPTY>"],
        ["cache_mode", _news_cache_mode(args) or "<EMPTY>"],
        ["cache_source", str(getattr(args, "_news_doc_cache_source", "") or "").strip() or "<EMPTY>"],
        ["explicit_cache", str(int(bool(getattr(args, "_news_doc_cache_explicit", False))))],
        ["cache_exists", str(int(bool(getattr(args, "_news_doc_cache_exists", False))))],
        ["refine_mode", str(getattr(args, "news_refine_mode", "") or "").strip() or "<EMPTY>"],
        ["structured_mode", str(getattr(args, "news_structured_mode", "") or "").strip() or "<EMPTY>"],
        ["api_key_detected", str(int(bool(getattr(args, "_news_api_key_detected", False))))],
        ["api_key_source", str(getattr(args, "_news_api_key_source", "") or "").strip() or "<NONE>"],
    ]


def _selected_news_meta_records(selected: pd.DataFrame, args, *, text_col: str, time_col: str) -> list[dict]:
    if len(selected) == 0:
        return []
    if "title" not in selected.columns:
        raise KeyError("Selected news rows must contain a 'title' column for refined cache identity matching.")

    out = []
    for row_idx, (_, row) in enumerate(selected.iterrows()):
        ts_value = row.get(time_col, "")
        if hasattr(ts_value, "isoformat"):
            ts_value = ts_value.isoformat()
        meta = {
            "raw_news_text": str(row.get(text_col, "") or "").strip(),
            "title": str(row.get("title", "") or "").strip(),
            "date": str(ts_value or "").strip(),
            "url": str(row.get("url", "") or "").strip() if "url" in selected.columns else "",
            "row_index": int(row_idx),
        }
        _require_news_identity_key(meta, args, origin=f"selected_news[row_index={row_idx}]")
        out.append(meta)
    return out


def _validate_news_identity_source(news_df: pd.DataFrame, args, *, time_col: str, text_col: str):
    if news_df is None or len(news_df) == 0:
        return
    if "title" not in news_df.columns:
        raise KeyError("News source file must contain a 'title' column for refined cache identity matching.")

    seen = {}
    for row_idx, (_, row) in enumerate(news_df.iterrows()):
        ts_value = row.get(time_col, "")
        if hasattr(ts_value, "isoformat"):
            ts_value = ts_value.isoformat()
        meta = {
            "raw_news_text": str(row.get(text_col, "") or "").strip(),
            "title": str(row.get("title", "") or "").strip(),
            "date": str(ts_value or "").strip(),
            "url": str(row.get("url", "") or "").strip() if "url" in news_df.columns else "",
        }
        key = _require_news_identity_key(meta, args, origin=f"news_source[row_index={row_idx}]")
        if key in seen:
            prev_idx = seen[key]
            raise ValueError(
                "[NEWS_DOC_CACHE] duplicate news source identity detected. "
                f"title={meta['title']!r} date={meta['date']!r} url={meta.get('url', '')!r} first_row={prev_idx} duplicate_row={row_idx}"
            )
        seen[key] = row_idx


def _rebuild_news_doc_cache_indexes(args):
    store = getattr(args, "_news_doc_cache_store", None)
    if not isinstance(store, dict):
        setattr(args, "_news_doc_cache_structured_index", {})
        setattr(args, "_news_doc_cache_identity_index", {})
        setattr(args, "_news_doc_cache_identity_duplicate_count", 0)
        return

    structured_index = {}
    identity_index = {}
    duplicate_count = 0
    for primary_key, rec in store.items():
        if not isinstance(rec, dict):
            continue
        norm = _normalize_news_doc_cache_record(rec)
        s_key = str(norm.get("structured_cache_key", "") or "").strip()
        if s_key:
            structured_index[s_key] = str(primary_key)

        has_payload = bool(
            str(norm.get("refined_news", "") or "").strip()
            or bool(norm.get("structured_events", {}))
            or str(norm.get("raw_news_text", "") or "").strip()
        )
        if not has_payload:
            continue
        ident = _require_news_identity_key(
            norm,
            args,
            origin=f"cache_record[primary_key={str(primary_key)}]",
        )
        existing = identity_index.get(ident, [])
        if isinstance(existing, str):
            existing = [existing]
        if existing:
            duplicate_count += 1
        existing.append(str(primary_key))
        identity_index[ident] = existing

    setattr(args, "_news_doc_cache_structured_index", structured_index)
    setattr(args, "_news_doc_cache_identity_index", identity_index)
    setattr(args, "_news_doc_cache_identity_duplicate_count", int(duplicate_count))


def _news_doc_cache_record_choice_key(primary_key: str, rec: dict) -> tuple:
    norm = _normalize_news_doc_cache_record(rec)
    has_refined = int(bool(str(norm.get("refined_news", "") or "").strip()))
    has_structured = int(bool(dict(norm.get("structured_events", {}) or {})))
    refined_len = len(str(norm.get("refined_news", "") or "").strip())
    raw_len = len(str(norm.get("raw_news_text", "") or "").strip())
    struct_size = len(dict(norm.get("structured_events", {}) or {}))
    return (
        -has_refined,
        -has_structured,
        -refined_len,
        -struct_size,
        -raw_len,
        str(primary_key),
    )


def _lookup_news_doc_cache_record_by_meta(
    args,
    *,
    news_meta: dict | None,
    require_refined: bool = False,
    require_structured: bool = False,
) -> dict:
    store = getattr(args, "_news_doc_cache_store", None)
    identity_index = getattr(args, "_news_doc_cache_identity_index", None)
    if not isinstance(store, dict) or not isinstance(identity_index, dict):
        raise RuntimeError("[NEWS_DOC_CACHE] cache store is not initialized.")

    meta = dict(news_meta or {})
    ident = _require_news_identity_key(
        meta,
        args,
        origin=f"cache_lookup[title={str(meta.get('title', '') or '').strip()!r}]",
    )
    primary_keys = identity_index.get(ident, [])
    if isinstance(primary_keys, str):
        primary_keys = [primary_keys]
    primary_keys = [str(k) for k in primary_keys if str(k).strip()]
    if not primary_keys:
        raise KeyError(
            "[NEWS_DOC_CACHE] missing cache entry for selected news. "
            f"title={str(meta.get('title', '') or '').strip()!r} "
            f"date={str(meta.get('date', '') or '').strip()!r} "
            f"url={str(meta.get('url', '') or '').strip()!r} "
            f"cache_path={str(getattr(args, '_news_doc_cache_path', '') or '').strip() or '<EMPTY>'}"
        )

    candidates = []
    for primary_key in primary_keys:
        rec = _normalize_news_doc_cache_record(store.get(primary_key, {}))
        has_refined = bool(str(rec.get("refined_news", "") or "").strip())
        has_structured = bool(dict(rec.get("structured_events", {}) or {}))
        if require_refined and not has_refined:
            continue
        if require_structured and not has_structured:
            continue
        candidates.append((primary_key, rec))

    if not candidates:
        primary_key = primary_keys[0]
        rec = _normalize_news_doc_cache_record(store.get(primary_key, {}))
    else:
        primary_key, rec = sorted(
            candidates,
            key=lambda item: _news_doc_cache_record_choice_key(item[0], item[1]),
        )[0]

    if require_refined and not str(rec.get("refined_news", "") or "").strip():
        raise KeyError(
            "[NEWS_DOC_CACHE] selected news is missing refined_news in cache. "
            f"title={str(meta.get('title', '') or '').strip()!r} "
            f"date={str(meta.get('date', '') or '').strip()!r} "
            f"url={str(meta.get('url', '') or '').strip()!r} key={primary_key}"
        )
    if require_structured and not dict(rec.get("structured_events", {}) or {}):
        raise KeyError(
            "[NEWS_DOC_CACHE] selected news is missing structured_events in cache. "
            f"title={str(meta.get('title', '') or '').strip()!r} "
            f"date={str(meta.get('date', '') or '').strip()!r} "
            f"url={str(meta.get('url', '') or '').strip()!r} key={primary_key}"
        )
    return rec


def _news_doc_cache_sort_key(args, rec: dict) -> tuple[str, str, str]:
    norm = _normalize_news_doc_cache_record(rec)
    date_key = _normalize_news_identity_time(
        norm.get("date", ""),
        dayfirst=bool(getattr(args, "dayFirst", False)),
    )
    title_key = _normalize_news_identity_title(norm.get("title", ""))
    url_key = _normalize_news_identity_url(norm.get("url", ""))
    return date_key, title_key, url_key


def _build_news_doc_meta_index(news_df: pd.DataFrame, *, text_col: str, time_col: str) -> dict[str, dict]:
    idx = {}
    if text_col not in news_df.columns:
        return idx
    title_col = "title" if "title" in news_df.columns else ""
    url_col = "url" if "url" in news_df.columns else ""
    for _, row in news_df.iterrows():
        raw_text = str(row.get(text_col, "") or "").strip()
        if not raw_text:
            continue
        ts_value = row.get(time_col, "")
        if hasattr(ts_value, "isoformat"):
            ts_value = ts_value.isoformat()
        meta = {
            "title": str(row.get(title_col, "") or "").strip() if title_col else "",
            "date": str(ts_value or "").strip(),
            "url": str(row.get(url_col, "") or "").strip() if url_col else "",
        }
        if raw_text in idx and idx[raw_text] != meta:
            idx[raw_text] = {}
            continue
        idx[raw_text] = meta
    return idx

def _lookup_news_doc_meta(args, raw_news_text: str) -> dict:
    clean = str(raw_news_text or "").strip()
    if not clean:
        return {}
    meta_index = getattr(args, "_news_doc_meta_by_text", None)
    if not isinstance(meta_index, dict):
        return {}
    meta = meta_index.get(clean, {})
    return dict(meta) if isinstance(meta, dict) else {}

def _normalize_news_doc_cache_record(rec: dict | None) -> dict:
    rec = dict(rec or {})
    out = {
        "structured_cache_key": str(rec.get("structured_cache_key", "") or "").strip(),
        "raw_news_text": str(rec.get("raw_news_text", "") or "").strip(),
        "title": str(rec.get("title", "") or "").strip(),
        "date": str(rec.get("date", "") or "").strip(),
        "url": str(rec.get("url", "") or "").strip(),
        "refined_news": str(rec.get("refined_news", "") or "").strip(),
        "structured_events": dict(rec.get("structured_events", {})) if isinstance(rec.get("structured_events", {}), dict) else {},
        "structured_source_kind": str(rec.get("structured_source_kind", "") or "").strip(),
        "region": str(rec.get("region", "") or "").strip(),
        "description": str(rec.get("description", "") or "").strip(),
        "news_path": str(rec.get("news_path", "") or "").strip(),
        "api_model": str(rec.get("api_model", "") or "").strip(),
        "refine_mode": str(rec.get("refine_mode", "") or "").strip(),
        "structured_mode": str(rec.get("structured_mode", "") or "").strip(),
        "refine_max_tokens": int(rec.get("refine_max_tokens", 0) or 0),
    }
    return out


def _merge_news_doc_cache_records(existing: dict | None, incoming: dict | None) -> dict:
    cur = _normalize_news_doc_cache_record(existing)
    new = _normalize_news_doc_cache_record(incoming)
    out = dict(cur)

    def _prefer_longer(left: str, right: str) -> str:
        left_clean = str(left or "").strip()
        right_clean = str(right or "").strip()
        if len(right_clean) > len(left_clean):
            return right_clean
        return left_clean

    for field in [
        "structured_cache_key",
        "title",
        "date",
        "url",
        "structured_source_kind",
        "region",
        "description",
        "news_path",
        "api_model",
        "refine_mode",
        "structured_mode",
    ]:
        if not out.get(field) and new.get(field):
            out[field] = new[field]

    out["raw_news_text"] = _prefer_longer(cur.get("raw_news_text", ""), new.get("raw_news_text", ""))
    out["refined_news"] = _prefer_longer(cur.get("refined_news", ""), new.get("refined_news", ""))

    cur_events = dict(cur.get("structured_events", {}) or {})
    new_events = dict(new.get("structured_events", {}) or {})
    if len(new_events) > len(cur_events):
        out["structured_events"] = new_events
    else:
        out["structured_events"] = cur_events

    out["refine_max_tokens"] = int(max(int(cur.get("refine_max_tokens", 0) or 0), int(new.get("refine_max_tokens", 0) or 0)))
    return _normalize_news_doc_cache_record(out)

def _upsert_news_doc_cache_record(
    args,
    *,
    structured_cache_key: str = "",
    raw_news_text: str = "",
    news_meta: dict | None = None,
    refined_news: str = "",
    structured_events: dict | None = None,
    source_kind: str = "",
    context: dict | None = None,
    model: str = "",
    max_tokens: int | None = None,
):
    if _news_cache_is_read_only(args):
        raise RuntimeError("[NEWS_DOC_CACHE] read_only cache mode forbids cache mutation.")
    store = getattr(args, "_news_doc_cache_store", None)
    pending = getattr(args, "_news_doc_cache_pending", None)
    structured_index = getattr(args, "_news_doc_cache_structured_index", None)
    if not isinstance(store, dict) or not isinstance(pending, dict) or not isinstance(structured_index, dict):
        return

    meta = dict(news_meta or {})
    if not meta:
        meta = _lookup_news_doc_meta(args, raw_news_text)
    primary_key = _news_doc_primary_key_from_meta(
        meta,
        dayfirst=bool(getattr(args, "dayFirst", False)),
        origin="cache_upsert",
    )

    record = _normalize_news_doc_cache_record(store.get(primary_key, {}))
    if structured_cache_key:
        record["structured_cache_key"] = str(structured_cache_key)
        structured_index[str(structured_cache_key)] = primary_key
    if raw_news_text:
        record["raw_news_text"] = str(raw_news_text)
    if refined_news:
        record["refined_news"] = str(refined_news)
    if isinstance(structured_events, dict) and len(structured_events) > 0:
        record["structured_events"] = dict(structured_events)
    if source_kind:
        record["structured_source_kind"] = str(source_kind)
    if meta.get("title"):
        record["title"] = str(meta.get("title", ""))
    if meta.get("date"):
        record["date"] = str(meta.get("date", ""))
    if meta.get("url"):
        record["url"] = str(meta.get("url", ""))
    ctx = dict(context or {})
    if ctx.get("region"):
        record["region"] = str(ctx.get("region", ""))
    if ctx.get("description"):
        record["description"] = str(ctx.get("description", ""))
    if model:
        record["api_model"] = str(model)
    if max_tokens is not None:
        record["refine_max_tokens"] = int(max(1, max_tokens))
    news_path = str(getattr(args, "news_path", "") or "").strip()
    if news_path:
        record["news_path"] = news_path

    record = _normalize_news_doc_cache_record(record)
    store[primary_key] = record
    pending[primary_key] = dict(record)
    setattr(args, "_news_doc_cache_store", store)
    setattr(args, "_news_doc_cache_pending", pending)
    setattr(args, "_news_doc_cache_structured_index", structured_index)
    _rebuild_news_doc_cache_indexes(args)

def _save_news_doc_cache(args, live_logger=None, force: bool = False):
    if _news_cache_is_read_only(args):
        return
    path = str(getattr(args, "_news_doc_cache_path", "")).strip()
    store = getattr(args, "_news_doc_cache_store", None)
    pending = getattr(args, "_news_doc_cache_pending", None)
    dirty = bool(getattr(args, "_news_doc_cache_dirty", False))
    if not path or not isinstance(store, dict) or not isinstance(pending, dict):
        return
    if (not force) and (not dirty):
        return

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    payload = [
        _normalize_news_doc_cache_record(v)
        for _, v in sorted(store.items(), key=lambda x: _news_doc_cache_sort_key(args, x[1]))
        if isinstance(v, dict)
    ]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

    appended_count = len([1 for _, v in pending.items() if isinstance(v, dict)])
    setattr(args, "_news_doc_cache_pending", {})
    setattr(args, "_news_doc_cache_dirty", False)
    if live_logger is not None:
        live_logger.info(
            f"[NEWS_DOC_CACHE] saved entries={len(store)} updated={appended_count} format=json_array"
        )

def _resolve_refine_model_name(args, api_adapter) -> str:
    return str(getattr(api_adapter, "model", getattr(args, "news_api_model", "")))

def _doc_refine_context(args) -> dict:
    return build_refine_context(args, target_time="")

def _refine_one_news_doc(
    *,
    raw_news_text: str,
    news_meta: dict | None = None,
    tokenizer,
    max_tokens: int,
    args,
    api_adapter=None,
):
    clean = str(raw_news_text or "").strip()
    if not clean:
        return ""

    mode = str(getattr(args, "news_refine_mode", "local") or "local").lower().strip()
    model = _resolve_refine_model_name(args, api_adapter)
    context = _doc_refine_context(args)

    cache_store = getattr(args, "_refine_cache_store", None)
    cache_enabled = bool(getattr(args, "_refine_cache_enabled", False))
    cache_key = ""
    if _news_cache_is_read_only(args):
        try:
            rec = _lookup_news_doc_cache_record_by_meta(
                args,
                news_meta=news_meta,
                require_refined=True,
            )
        except KeyError as exc:
            _warn_news_doc_cache_skip(
                args,
                news_meta=news_meta,
                reason=str(exc),
            )
            return ""
        setattr(
            args,
            "_refine_cache_hits",
            int(getattr(args, "_refine_cache_hits", 0)) + 1,
        )
        return str(rec.get("refined_news", "") or "").strip()
    if cache_enabled and isinstance(cache_store, dict):
        cache_key = _refine_doc_cache_key(
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
            refined_cached = cached.strip()
            _upsert_news_doc_cache_record(
                args,
                raw_news_text=clean,
                news_meta=news_meta,
                refined_news=refined_cached,
                context=context,
                model=model,
                max_tokens=max_tokens,
            )
            return refined_cached

    use_api = mode == "api" and api_adapter is not None and hasattr(api_adapter, "refine_news")
    if mode == "api" and not use_api:
        raise RuntimeError(
            "API refine mode requires an initialized adapter while building refined news cache."
        )
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
        refined = _truncate_with_tokenizer(clean, tokenizer, max_tokens=max_tokens)
    refined = str(refined or "").strip()

    if cache_enabled:
        setattr(
            args,
            "_refine_cache_misses",
            int(getattr(args, "_refine_cache_misses", 0)) + 1,
        )
        if isinstance(cache_store, dict) and cache_key and refined:
            cache_store[cache_key] = refined
            pending_store = getattr(args, "_refine_cache_pending", None)
            if isinstance(pending_store, dict):
                pending_store[cache_key] = refined
                setattr(args, "_refine_cache_pending", pending_store)
            setattr(args, "_refine_cache_store", cache_store)
            setattr(args, "_refine_cache_dirty", True)
            setattr(args, "_news_doc_cache_dirty", True)
            _upsert_news_doc_cache_record(
                args,
                raw_news_text=clean,
                news_meta=news_meta,
                refined_news=refined,
                context=context,
                model=model,
                max_tokens=max_tokens,
            )
    return refined

def _merge_refined_news_docs(
    snippets: list[str],
    *,
    tokenizer,
    max_tokens: int,
) -> str:
    if len(snippets) == 0:
        return ""
    merged = "\n".join([f"- {s}" for s in snippets])
    return _truncate_with_tokenizer(merged, tokenizer, max_tokens=max_tokens)

def _refine_news_docs_from_doc_cache(
    *,
    raw_news_texts: list[str],
    news_metas: list[dict] | None = None,
    tokenizer,
    max_tokens: int,
    args,
    api_adapter=None,
) -> list[str]:
    aligned = _refine_news_docs_aligned_from_doc_cache(
        raw_news_texts=raw_news_texts,
        news_metas=news_metas,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        args=args,
        api_adapter=api_adapter,
    )
    snippets = []
    seen = set()
    for refined_item in aligned:
        clean = str(refined_item or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        snippets.append(clean)
    return snippets

def _refine_news_docs_aligned_from_doc_cache(
    *,
    raw_news_texts: list[str],
    news_metas: list[dict] | None = None,
    tokenizer,
    max_tokens: int,
    args,
    api_adapter=None,
) -> list[str]:
    items = [str(x or "").strip() for x in raw_news_texts]
    if len(items) == 0:
        return []

    aligned = []
    local_cache = {}
    metas = list(news_metas or [])
    for idx, item in enumerate(items):
        meta = metas[idx] if idx < len(metas) else None
        if not item:
            aligned.append("")
            continue
        local_key = (
            str(meta.get("title", "") or "").strip(),
            str(meta.get("date", "") or "").strip(),
            str(meta.get("url", "") or "").strip(),
            item,
        ) if isinstance(meta, dict) else item
        if local_key not in local_cache:
            local_cache[local_key] = _refine_one_news_doc(
                raw_news_text=item,
                news_meta=meta,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                args=args,
                api_adapter=api_adapter,
            )
        aligned.append(str(local_cache[local_key] or "").strip())
    return aligned

def _align_refined_docs_to_history_steps(
    *,
    history_times: list[str],
    news_times,
    refined_news_docs: list[str],
    tokenizer,
    max_tokens: int,
) -> list[str]:
    L = len(history_times)
    if L <= 0:
        return [""]

    hist_series = pd.Series(pd.to_datetime(history_times, errors="coerce"))
    step_docs = [[] for _ in range(L)]
    last_idx = L - 1

    for raw_ts, doc_txt in zip(list(news_times), refined_news_docs):
        txt = str(doc_txt or "").strip()
        if not txt:
            continue
        ts = _align_ts_to_ref_tz(raw_ts, hist_series)
        idx = last_idx
        if not pd.isna(ts):
            idx = 0
            for step_idx, step_ts in enumerate(hist_series.tolist()):
                if pd.isna(step_ts):
                    continue
                if ts < step_ts:
                    break
                idx = step_idx
        step_docs[int(max(0, min(last_idx, idx)))].append(txt)

    return [
        _merge_refined_news_docs(snippets, tokenizer=tokenizer, max_tokens=max_tokens) if len(snippets) > 0 else ""
        for snippets in step_docs
    ]

def _init_refine_cache(args, live_logger=None):
    enabled = int(getattr(args, "news_refine_cache_enable", 1)) == 1
    setattr(args, "_refine_cache_enabled", bool(enabled))
    setattr(args, "_refine_cache_store", {})
    setattr(args, "_refine_cache_pending", {})
    setattr(args, "_news_doc_cache_store", {})
    setattr(args, "_news_doc_cache_pending", {})
    setattr(args, "_news_doc_cache_structured_index", {})
    setattr(args, "_news_doc_cache_identity_index", {})
    setattr(args, "_news_doc_cache_dirty", False)
    setattr(args, "_news_doc_cache_warned_missing", set())
    setattr(args, "_news_doc_cache_warned_missing_count", 0)
    setattr(args, "_refine_cache_dirty", False)
    setattr(args, "_refine_cache_hits", 0)
    setattr(args, "_refine_cache_misses", 0)
    if not enabled:
        if live_logger is not None:
            live_logger.info("[NEWS_REFINE_CACHE] disabled.")
        return

    path = str(getattr(args, "_news_doc_cache_path", "") or _refine_cache_path(args)).strip()
    setattr(args, "_refine_cache_path", path)
    setattr(args, "_news_doc_cache_path", path)
    read_paths = _parse_refine_cache_read_paths(getattr(args, "news_refine_cache_read_path", ""))
    merged_read_paths = list(read_paths)
    path_abs = os.path.abspath(path)
    if path_abs not in {os.path.abspath(p) for p in merged_read_paths}:
        merged_read_paths.append(path)
    setattr(args, "_refine_cache_read_paths", merged_read_paths)
    dirpath = os.path.dirname(path)
    if dirpath and (not _news_cache_is_read_only(args)):
        os.makedirs(dirpath, exist_ok=True)
    store = {}
    doc_store = _load_news_doc_cache_file(path, args=args)
    _log_news_doc_cache_detection(
        live_logger,
        tag="NEWS_REFINE_CACHE",
        path=path,
        object_count=len(doc_store),
    )
    structured_index = {}
    for primary_key, rec in doc_store.items():
        if not isinstance(rec, dict):
            continue
        s_key = str(rec.get("structured_cache_key", "") or "").strip()
        if s_key:
            structured_index[s_key] = str(primary_key)
    source_summaries = []
    for src in merged_read_paths:
        loaded = _load_refine_cache_file(src)
        if loaded:
            store.update(loaded)
        source_summaries.append(f"{src}:{len(loaded)}")
    setattr(args, "_refine_cache_store", store)
    setattr(args, "_news_doc_cache_store", doc_store)
    _rebuild_news_doc_cache_indexes(args)
    if live_logger is not None:
        if read_paths:
            live_logger.info(
                f"[NEWS_REFINE_CACHE] enabled mode={_news_cache_mode(args)} write_path={path} entries={len(store)} "
                f"preload_sources={source_summaries}"
            )
        else:
            live_logger.info(
                f"[NEWS_REFINE_CACHE] enabled mode={_news_cache_mode(args)} path={path} entries={len(store)}"
            )

def _save_refine_cache(args, live_logger=None, force: bool = False):
    if not bool(getattr(args, "_refine_cache_enabled", False)):
        return
    if _news_cache_is_read_only(args):
        return
    dirty = bool(getattr(args, "_refine_cache_dirty", False))
    if (not force) and (not dirty):
        return
    path = str(getattr(args, "_refine_cache_path", "")).strip()
    store = getattr(args, "_refine_cache_store", None)
    pending_store = getattr(args, "_refine_cache_pending", None)
    if not path or not isinstance(store, dict):
        return
    if not isinstance(pending_store, dict):
        pending_store = {}
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    setattr(args, "_refine_cache_pending", {})
    setattr(args, "_refine_cache_dirty", False)
    setattr(args, "_news_doc_cache_dirty", True)
    _save_news_doc_cache(args, live_logger=live_logger, force=force)
    if live_logger is not None:
        hits = int(getattr(args, "_refine_cache_hits", 0))
        misses = int(getattr(args, "_refine_cache_misses", 0))
        write_items = [
            (str(k), str(v))
            for k, v in sorted(pending_store.items(), key=lambda x: str(x[0]))
            if isinstance(v, str)
        ]
        live_logger.info(
            f"[NEWS_REFINE_CACHE] saved entries={len(store)} appended={len(write_items)} "
            f"path={path} hits={hits} misses={misses}"
        )

def _structured_cache_path(args) -> str:
    return resolve_structured_cache_path(args)

def _structured_cache_enabled(args) -> bool:
    if str(getattr(args, "news_structured_mode", "off")).lower().strip() != "api":
        return False
    return int(getattr(args, "news_structured_cache_enable", 1)) == 1

def _init_structured_cache(args, live_logger=None):
    enabled = _structured_cache_enabled(args)
    setattr(args, "_structured_cache_enabled", bool(enabled))
    setattr(args, "_structured_cache_store", {})
    setattr(args, "_structured_cache_dirty", False)
    setattr(args, "_structured_cache_hits", 0)
    setattr(args, "_structured_cache_misses", 0)
    if not enabled:
        if live_logger is not None:
            live_logger.info("[NEWS_STRUCTURED_CACHE] disabled.")
        return

    path = str(getattr(args, "_news_doc_cache_path", "") or _structured_cache_path(args)).strip()
    setattr(args, "_structured_cache_path", path)
    setattr(args, "_news_doc_cache_path", path)
    existing_doc_store = getattr(args, "_news_doc_cache_store", None)
    loaded_doc_store = _load_news_doc_cache_file(path, args=args)
    _log_news_doc_cache_detection(
        live_logger,
        tag="NEWS_STRUCTURED_CACHE",
        path=path,
        object_count=len(loaded_doc_store),
    )
    if loaded_doc_store or (not isinstance(existing_doc_store, dict)):
        setattr(args, "_news_doc_cache_store", loaded_doc_store)
    if not isinstance(getattr(args, "_news_doc_cache_pending", None), dict):
        setattr(args, "_news_doc_cache_pending", {})
    if not isinstance(getattr(args, "_news_doc_cache_structured_index", None), dict):
        setattr(args, "_news_doc_cache_structured_index", {})
    if not hasattr(args, "_news_doc_cache_dirty"):
        setattr(args, "_news_doc_cache_dirty", False)
    read_paths = _parse_refine_cache_read_paths(getattr(args, "news_structured_cache_read_path", ""))
    merged_read_paths = list(read_paths)
    path_abs = os.path.abspath(path)
    if path_abs not in {os.path.abspath(p) for p in merged_read_paths}:
        merged_read_paths.append(path)
    setattr(args, "_structured_cache_read_paths", merged_read_paths)
    dirpath = os.path.dirname(path)
    if dirpath and (not _news_cache_is_read_only(args)):
        os.makedirs(dirpath, exist_ok=True)
    store = {}
    source_summaries = []
    for src in merged_read_paths:
        loaded = _load_structured_cache_file(src)
        if loaded:
            store.update(loaded)
        source_summaries.append(f"{src}:{len(loaded)}")
    setattr(args, "_structured_cache_store", store)
    _rebuild_news_doc_cache_indexes(args)
    if live_logger is not None:
        if read_paths:
            live_logger.info(
                f"[NEWS_STRUCTURED_CACHE] enabled mode={_news_cache_mode(args)} write_path={path} entries={len(store)} "
                f"preload_sources={source_summaries}"
            )
        else:
            live_logger.info(
                f"[NEWS_STRUCTURED_CACHE] enabled mode={_news_cache_mode(args)} path={path} entries={len(store)}"
            )
        if len(store) == 0:
            live_logger.info(
                "[NEWS_STRUCTURED_CACHE] cache is currently empty; "
                "this is expected before structured prewarm or first structured extraction."
            )

def _save_structured_cache(args, live_logger=None, force: bool = False):
    if not bool(getattr(args, "_structured_cache_enabled", False)):
        return
    if _news_cache_is_read_only(args):
        return
    dirty = bool(getattr(args, "_structured_cache_dirty", False))
    if (not force) and (not dirty):
        return
    path = str(getattr(args, "_structured_cache_path", "")).strip()
    store = getattr(args, "_structured_cache_store", None)
    if not path or not isinstance(store, dict):
        return
    setattr(args, "_structured_cache_dirty", False)
    setattr(args, "_news_doc_cache_dirty", True)
    _save_news_doc_cache(args, live_logger=live_logger, force=force)
    if live_logger is not None:
        hits = int(getattr(args, "_structured_cache_hits", 0))
        misses = int(getattr(args, "_structured_cache_misses", 0))
        live_logger.info(
            f"[NEWS_STRUCTURED_CACHE] saved entries={len(store)} path={path} hits={hits} misses={misses}"
        )

def _structured_doc_source_kind(args) -> str:
    mode = str(getattr(args, "news_structured_mode", "off") or "off").lower().strip()
    return "raw" if mode == "api" else "refined"

def _extract_structured_event_one_doc(
    *,
    news_text: str,
    news_meta: dict | None = None,
    args,
    api_adapter=None,
    context: dict | None = None,
    source_kind: str = "refined",
    raw_news_text: str | None = None,
    refine_max_tokens: int | None = None,
) -> dict:
    clean = str(news_text or "").strip()
    if not clean:
        return {}

    mode = str(getattr(args, "news_structured_mode", "off") or "off").lower().strip()
    use_api = mode == "api" and api_adapter is not None and hasattr(api_adapter, "extract_events")
    model = _resolve_refine_model_name(args, api_adapter)
    ctx = dict(context or _doc_refine_context(args))

    cache_store = getattr(args, "_structured_cache_store", None)
    cache_enabled = bool(getattr(args, "_structured_cache_enabled", False)) and use_api
    cache_key = ""
    if _news_cache_is_read_only(args):
        try:
            rec = _lookup_news_doc_cache_record_by_meta(
                args,
                news_meta=news_meta,
                require_refined=False,
                require_structured=True,
            )
        except KeyError as exc:
            _warn_news_doc_cache_skip(
                args,
                news_meta=news_meta,
                reason=str(exc),
            )
            return {}
        setattr(
            args,
            "_structured_cache_hits",
            int(getattr(args, "_structured_cache_hits", 0)) + 1,
        )
        return dict(rec.get("structured_events", {}) or {})
    if cache_enabled and isinstance(cache_store, dict):
        cache_key = make_structured_doc_cache_key(
            news_text=clean,
            context=ctx,
            mode=mode,
            model=model,
            source_kind=source_kind,
        )
        cached = cache_store.get(cache_key, None)
        if isinstance(cached, dict) and len(cached) > 0:
            setattr(
                args,
                "_structured_cache_hits",
                int(getattr(args, "_structured_cache_hits", 0)) + 1,
            )
            cached_events = dict(cached)
            _upsert_news_doc_cache_record(
                args,
                structured_cache_key=cache_key,
                raw_news_text=str(raw_news_text or clean).strip(),
                news_meta=news_meta,
                structured_events=cached_events,
                source_kind=source_kind,
                context=ctx,
                model=model,
                max_tokens=refine_max_tokens,
            )
            return cached_events

    if mode == "api" and not use_api:
        raise RuntimeError(
            "API structured mode requires an initialized adapter while building refined news cache."
        )
    events = extract_structured_events(
        raw_or_refined_news=clean,
        mode=mode,
        api_adapter=api_adapter,
        context=ctx,
    )
    events = dict(events) if isinstance(events, dict) else {}

    if cache_enabled:
        setattr(
            args,
            "_structured_cache_misses",
            int(getattr(args, "_structured_cache_misses", 0)) + 1,
        )
        if isinstance(cache_store, dict) and cache_key and events:
            cache_store[cache_key] = dict(events)
            setattr(args, "_structured_cache_store", cache_store)
            setattr(args, "_structured_cache_dirty", True)
            setattr(args, "_news_doc_cache_dirty", True)
            raw_clean = str(raw_news_text or clean).strip()
            _upsert_news_doc_cache_record(
                args,
                structured_cache_key=cache_key,
                raw_news_text=raw_clean,
                news_meta=news_meta,
                structured_events=events,
                source_kind=source_kind,
                context=ctx,
                model=model,
                max_tokens=refine_max_tokens,
            )
    return events

def _extract_structured_events_from_refined_docs_detailed(
    *,
    raw_news_texts: list[str] | None = None,
    refined_news_texts: list[str],
    news_metas: list[dict] | None = None,
    args,
    api_adapter=None,
    context: dict | None = None,
) -> tuple[dict, list[dict]]:
    source_kind = _structured_doc_source_kind(args)
    raw_items_all = [str(x).strip() for x in (raw_news_texts or []) if str(x).strip()]
    meta_items = list(news_metas or [])
    refine_max_tokens = int(args.token_budget * args.token_budget_news_frac)
    if source_kind == "raw":
        source_items = list(raw_items_all)
        fallback_items = [str(x).strip() for x in refined_news_texts if str(x).strip()]
    else:
        source_items = [str(x).strip() for x in refined_news_texts if str(x).strip()]
        fallback_items = []

    items = source_items
    if len(items) == 0:
        items = fallback_items
        source_kind = "refined"
    if len(items) == 0:
        return {}, []

    seen = set()
    merged_event_items = []
    doc_records = []
    local_cache = {}
    for idx, item in enumerate(items):
        meta_item = meta_items[idx] if idx < len(meta_items) else None
        local_key = (
            str(meta_item.get("title", "") or "").strip(),
            str(meta_item.get("date", "") or "").strip(),
            str(meta_item.get("url", "") or "").strip(),
            item,
        ) if isinstance(meta_item, dict) else item
        dedup_reused = local_key in local_cache
        if dedup_reused:
            ev = dict(local_cache[local_key])
        else:
            raw_item = item if source_kind == "raw" else (raw_items_all[idx] if idx < len(raw_items_all) else item)
            ev = _extract_structured_event_one_doc(
                news_text=item,
                news_meta=meta_item,
                args=args,
                api_adapter=api_adapter,
                context=context,
                source_kind=source_kind,
                raw_news_text=raw_item,
                refine_max_tokens=refine_max_tokens,
            )
            ev = dict(ev) if isinstance(ev, dict) else {}
            local_cache[local_key] = dict(ev)
        if item not in seen:
            seen.add(item)
            if len(ev) > 0:
                merged_event_items.append(dict(ev))
        doc_records.append(
            {
                "doc_index": int(idx),
                "source_kind": str(source_kind),
                "news_text": str(item),
                "events": dict(ev),
                "has_events": bool(len(ev) > 0),
                "dedup_reused": bool(dedup_reused),
            }
        )
    return merge_structured_events(merged_event_items), doc_records

def _extract_structured_events_from_refined_docs(
    *,
    raw_news_texts: list[str] | None = None,
    refined_news_texts: list[str],
    news_metas: list[dict] | None = None,
    args,
    api_adapter=None,
    context: dict | None = None,
) -> dict:
    merged, _doc_records = _extract_structured_events_from_refined_docs_detailed(
        raw_news_texts=raw_news_texts,
        refined_news_texts=refined_news_texts,
        news_metas=news_metas,
        args=args,
        api_adapter=api_adapter,
        context=context,
    )
    return merged

STRUCTURED_EVENT_TYPE_ORDER = [
    "outage",
    "weather",
    "policy",
    "transmission",
    "fuel",
    "demand",
    "other",
]

STRUCTURED_EVENT_TYPE_BUCKETS = {
    "outage": {"outage", "trip", "plant_outage", "generator_outage"},
    "weather": {"weather", "storm", "temperature", "rain", "heat", "cold"},
    "policy": {"policy", "regulation", "government", "rule"},
    "transmission": {"transmission", "interconnector", "network", "line"},
    "fuel": {"fuel", "gas", "coal", "oil"},
    "demand": {"demand", "load", "consumption"},
}

def _safe_structured_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)

def _structured_event_type_bucket(event_type) -> str:
    t = str(event_type or "").strip().lower()
    if not t:
        return "other"
    for bucket, terms in STRUCTURED_EVENT_TYPE_BUCKETS.items():
        if t == bucket or t in terms:
            return bucket
        for term in terms:
            if term in t:
                return bucket
    if t in {"mixed", "general"}:
        return "mixed"
    return "other"

def _structured_bucket_level(v, low: float, high: float, names: tuple[str, str, str]) -> str:
    x = _safe_structured_float(v, default=0.0)
    if x < low:
        return names[0]
    if x < high:
        return names[1]
    return names[2]

def _normalize_structured_events(events: dict | None) -> dict:
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

    strength_f = _safe_structured_float(events.get("strength", 0.0), default=0.0)
    persistence_f = _safe_structured_float(events.get("persistence", 0.0), default=0.0)
    confidence_f = _safe_structured_float(events.get("confidence", 0.0), default=0.0)

    return {
        "event_type": _structured_event_type_bucket(events.get("event_type", "other")),
        "direction": direction,
        "strength": _structured_bucket_level(strength_f, 0.33, 0.66, ("weak", "medium", "strong")),
        "persistence": _structured_bucket_level(persistence_f, 0.33, 0.66, ("short", "medium", "long")),
        "confidence": _structured_bucket_level(confidence_f, 0.33, 0.66, ("low", "medium", "high")),
        "strength_value": float(max(0.0, min(1.0, strength_f))),
        "persistence_value": float(max(0.0, min(1.0, persistence_f))),
        "confidence_value": float(max(0.0, min(1.0, confidence_f))),
    }

def _structured_events_to_feature_vec(events: dict | None, dim: int = 12) -> np.ndarray:
    d = int(max(1, dim))
    vec = np.zeros((d,), dtype=np.float32)
    if not isinstance(events, dict) or len(events) == 0:
        return vec

    norm = _normalize_structured_events(events)
    if not norm:
        return vec

    relevance = float(max(0.0, min(1.0, float(events.get("relevance", 0.0) or 0.0))))
    direction_label = str(norm.get("direction", "uncertain"))
    if direction_label == "up":
        direction_value = 1.0
    elif direction_label == "down":
        direction_value = -1.0
    else:
        direction_value = 0.0
    strength = float(max(0.0, min(1.0, float(norm.get("strength_value", 0.0) or 0.0))))
    persistence = float(max(0.0, min(1.0, float(norm.get("persistence_value", 0.0) or 0.0))))
    confidence = float(max(0.0, min(1.0, float(norm.get("confidence_value", 0.0) or 0.0))))

    base_feats = [relevance, direction_value, strength, persistence, confidence]
    for idx, v in enumerate(base_feats[:d]):
        vec[idx] = float(v)

    event_type = str(norm.get("event_type", "other") or "other").strip().lower()
    type_offset = len(base_feats)
    if type_offset < d:
        try:
            type_idx = STRUCTURED_EVENT_TYPE_ORDER.index(event_type)
        except ValueError:
            type_idx = len(STRUCTURED_EVENT_TYPE_ORDER) - 1
        one_hot_width = min(len(STRUCTURED_EVENT_TYPE_ORDER), d - type_offset)
        if type_idx < one_hot_width:
            vec[type_offset + type_idx] = 1.0
    return vec

def _align_ts_to_ref_tz(ts, ref_series: pd.Series):
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return ts
    tz = getattr(ref_series.dt, "tz", None)
    if tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
    elif ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts

def _prewarm_refine_cache(
    *,
    args,
    news_df,
    train_df,
    val_df,
    test_df,
    tokenizer,
    live_logger,
    api_adapter=None,
):
    if _news_cache_is_read_only(args):
        if live_logger is not None:
            live_logger.info("[NEWS_PREPROCESS_CACHE] prewarm skipped: read_only cache mode.")
        return
    refine_cache_enabled = bool(getattr(args, "_refine_cache_enabled", False))
    refine_prewarm_enabled = int(getattr(args, "news_refine_prewarm", 1)) == 1
    structured_mode = str(getattr(args, "news_structured_mode", "off")).lower().strip()
    structured_cache_enabled = bool(getattr(args, "_structured_cache_enabled", False))
    structured_prewarm_enabled = int(getattr(args, "news_structured_prewarm", 1)) == 1
    run_refine = bool(refine_cache_enabled and refine_prewarm_enabled)
    run_structured = bool(structured_mode != "off" and structured_cache_enabled and structured_prewarm_enabled)
    if (not run_refine) and (not run_structured):
        if live_logger is not None:
            live_logger.info(
                "[NEWS_PREPROCESS_CACHE] prewarm skipped: "
                f"refine_enabled={int(run_refine)} structured_enabled={int(run_structured)}"
            )
        return
    if news_df is None or len(news_df) == 0:
        return

    text_col = str(getattr(args, "news_text_col", "content"))
    if text_col not in news_df.columns:
        if live_logger is not None:
            live_logger.info(f"[NEWS_PREPROCESS_CACHE] prewarm skipped: text_col not found: {text_col}")
        return

    max_items = int(getattr(args, "news_refine_prewarm_max_batches", -1))
    before_n = len(getattr(args, "_refine_cache_store", {}))
    hit0 = int(getattr(args, "_refine_cache_hits", 0))
    miss0 = int(getattr(args, "_refine_cache_misses", 0))
    structured_before_n = len(getattr(args, "_structured_cache_store", {}))
    structured_hit0 = int(getattr(args, "_structured_cache_hits", 0))
    structured_miss0 = int(getattr(args, "_structured_cache_misses", 0))

    time_col = str(getattr(args, "news_time_col", "date"))
    in_scope = news_df
    if time_col in news_df.columns:
        news_ts = pd.to_datetime(
            news_df[time_col], errors="coerce", dayfirst=args.dayFirst
        )
        news_tz = getattr(news_ts.dt, "tz", None)

        def _align_series_to_news_tz(series_like):
            ts = pd.to_datetime(series_like, errors="coerce", dayfirst=args.dayFirst)
            cur_tz = getattr(ts.dt, "tz", None)
            if news_tz is not None:
                if cur_tz is None:
                    ts = ts.dt.tz_localize(news_tz)
                else:
                    ts = ts.dt.tz_convert(news_tz)
            elif cur_tz is not None:
                ts = ts.dt.tz_localize(None)
            return ts

        lo = None
        hi = None
        for df in [train_df, val_df, test_df]:
            if isinstance(df, pd.DataFrame) and (args.time_col in df.columns) and len(df) > 0:
                ts = _align_series_to_news_tz(df[args.time_col])
                ts = ts.dropna()
                if len(ts) == 0:
                    continue
                cur_lo = ts.min()
                cur_hi = ts.max()
                lo = cur_lo if lo is None else min(lo, cur_lo)
                hi = cur_hi if hi is None else max(hi, cur_hi)
        if lo is not None and hi is not None:
            pad_days = float(max(0.0, float(getattr(args, "news_window_days", 1) or 1)))
            lo = lo - pd.Timedelta(days=pad_days + 1.0)
            nts = news_ts
            mask = nts.ge(lo) & nts.le(hi)
            in_scope = news_df.loc[mask.fillna(False)].copy()

    meta_records = _selected_news_meta_records(
        in_scope,
        args,
        text_col=text_col,
        time_col=time_col,
    )
    if max_items > 0:
        meta_records = meta_records[:max_items]

    show_progress = (
        int(getattr(args, "news_refine_show_progress", 1)) == 1
        or int(getattr(args, "news_structured_show_progress", 1)) == 1
    )
    if live_logger is not None:
        doc_cache_path = str(getattr(args, "_news_doc_cache_path", "") or "").strip()
        doc_cache_objects = len(getattr(args, "_news_doc_cache_store", {}) or {})
        live_logger.info(
            "[NEWS_PREPROCESS_CACHE] start: "
            f"news_file={str(getattr(args, 'news_path', '') or '').strip() or '<NONE>'}, "
            f"news_rows={len(news_df)}, in_scope_rows={len(in_scope)}, unique_docs={len(meta_records)}, "
            f"refine_enabled={int(run_refine)} structured_enabled={int(run_structured)} "
            f"refine_write_path={str(getattr(args, '_refine_cache_path', '')).strip() or '<DISABLED>'} "
            f"structured_write_path={str(getattr(args, '_structured_cache_path', '')).strip() or '<DISABLED>'} "
            f"doc_cache_detected={int(bool(doc_cache_path) and os.path.exists(doc_cache_path) and doc_cache_objects > 0)} "
            f"doc_cache_file={doc_cache_path or '<DISABLED>'} "
            f"doc_cache_objects={doc_cache_objects}"
        )
        live_logger.info(
            "[NEWS_PREPROCESS_CACHE] sources: "
            f"refine_reads={list(getattr(args, '_refine_cache_read_paths', []))} "
            f"structured_reads={list(getattr(args, '_structured_cache_read_paths', []))}"
        )

    pbar = tqdm(
        meta_records,
        desc="[DELTA][NEWS_API_PREPROCESS]",
        leave=show_progress,
        dynamic_ncols=True,
        mininterval=0.3,
        disable=(not show_progress),
    )
    news_budget = int(args.token_budget * args.token_budget_news_frac)
    total_docs = len(meta_records)
    structured_context = _doc_refine_context(args)
    structured_source_kind = _structured_doc_source_kind(args)
    for idx, news_meta in enumerate(pbar, start=1):
        raw_text = str(news_meta.get("raw_news_text", "") or "").strip()
        if not raw_text:
            continue
        refined_doc = ""
        if run_refine:
            refined_doc = _refine_one_news_doc(
                raw_news_text=raw_text,
                news_meta=news_meta,
                tokenizer=tokenizer,
                max_tokens=news_budget,
                args=args,
                api_adapter=api_adapter,
            )
        if run_structured:
            structured_input = raw_text if structured_source_kind == "raw" else refined_doc
            if (not structured_input) and refined_doc:
                structured_input = refined_doc
            if structured_input:
                _ = _extract_structured_event_one_doc(
                    news_text=structured_input,
                    news_meta=news_meta,
                    args=args,
                    api_adapter=api_adapter,
                    context=structured_context,
                    source_kind=("raw" if structured_input == raw_text else "refined"),
                    raw_news_text=raw_text,
                    refine_max_tokens=news_budget,
                )
        if show_progress and (idx == 1 or idx % 10 == 0 or idx == total_docs):
            hit_now = int(getattr(args, "_refine_cache_hits", 0))
            miss_now = int(getattr(args, "_refine_cache_misses", 0))
            structured_hit_now = int(getattr(args, "_structured_cache_hits", 0))
            structured_miss_now = int(getattr(args, "_structured_cache_misses", 0))
            pbar.set_postfix(
                {
                    "ref_hit": max(0, hit_now - hit0),
                    "ref_new": max(0, miss_now - miss0),
                    "str_hit": max(0, structured_hit_now - structured_hit0),
                    "str_new": max(0, structured_miss_now - structured_miss0),
                },
                refresh=False,
            )
    _save_refine_cache(args, live_logger=live_logger, force=True)
    after_n = len(getattr(args, "_refine_cache_store", {}))
    hit1 = int(getattr(args, "_refine_cache_hits", 0))
    miss1 = int(getattr(args, "_refine_cache_misses", 0))
    _save_structured_cache(args, live_logger=live_logger, force=True)
    structured_after_n = len(getattr(args, "_structured_cache_store", {}))
    structured_hit1 = int(getattr(args, "_structured_cache_hits", 0))
    structured_miss1 = int(getattr(args, "_structured_cache_misses", 0))
    if live_logger is not None:
        live_logger.info(
            "[NEWS_PREPROCESS_CACHE] done: "
            f"unique_docs={len(meta_records)} "
            f"refine_entries_before={before_n} refine_entries_after={after_n} "
            f"refine_hits_delta={hit1 - hit0} refine_misses_delta={miss1 - miss0} "
            f"structured_entries_before={structured_before_n} structured_entries_after={structured_after_n} "
            f"structured_hits_delta={structured_hit1 - structured_hit0} "
            f"structured_misses_delta={structured_miss1 - structured_miss0}"
        )
