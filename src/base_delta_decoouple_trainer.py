# trainer.py (REGRESSION version - full file)
# [RESIDUAL BASE+DELTA, NO TRUE-vs-SHUFFLED]
# Refactor: split BASE and DELTA into separate runnable stages via args.stage in {"all","base","delta"}.
# - stage=base : train/save best_base only
# - stage=delta: load existing base checkpoint and train/save best_delta (+test)
# - stage=all  : base -> delta (original behavior)
#
# Notes:
# - This file assumes run.py (argparse) provides optional args:
#   --stage, --base_ckpt, --base_epochs, --delta_epochs
#   If not provided, getattr defaults will be used.

from __future__ import annotations

import csv
import gc
import os
import json
import math
import hashlib
import re
from collections import deque
from contextlib import nullcontext
import shutil

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

from src.data_construction.DataStatistic import DataStatistic
from .data_construction.data import make_loader
from .news_rules import (
    load_news,
    get_candidates,
    select_news,
    rerank_selected_news_by_utility,
)
from .data_construction.prompt import load_templates, build_prompt
from .utils.logger import setup_live_logger

from .model2 import build_delta_model, load_checkpoint, save_checkpoint
from .base_backbone import (
    build_base_backbone,
    save_base_backbone_checkpoint,
    load_base_backbone_checkpoint,
)
from .delta_news_hooks import (
    refine_news_text,
    extract_structured_events,
    merge_structured_events,
    format_structured_events_for_prompt,
    reflect_hard_samples,
    build_news_api_adapter,
)
from .refine_cache_utils import (
    build_refine_context,
    make_refine_doc_cache_key,
    make_refine_news_cache_key,
    make_structured_doc_cache_key,
    resolve_refine_cache_path,
    resolve_structured_cache_path,
)
from .utils.residual_utils import split_two_stage_epochs

from .utils.utils import (
    set_seed,
    device_from_id,
    compute_volatility_bin,
    draw_pred_true,
    print_prompt_stats,
    record_test_results_csv,
)

dataStatistic = DataStatistic()


# ----------------------------
# helpers
# ----------------------------
def _adapter_off(peft_model):
    # peft >= 0.8 common: disable_adapter()
    if hasattr(peft_model, "disable_adapter"):
        return peft_model.disable_adapter()
    if hasattr(peft_model, "disable_adapters"):
        return peft_model.disable_adapters()
    return nullcontext()


def _single_device_map(args):
    """
    Ensure the whole model loads onto ONE GPU, matching device_from_id(args.gpu).
    Important to avoid cuda:0/cuda:1 mismatch.
    """
    if torch.cuda.is_available():
        return {"": int(args.gpu)}
    return None


def _zstats(x, eps: float = 1e-6):
    x = np.asarray(x, dtype=np.float32)
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma < eps:
        sigma = eps
    return mu, sigma


def _zscore(x, mu, sigma):
    x = np.asarray(x, dtype=np.float32)
    return ((x - mu) / sigma).tolist()


def _inv_zscore(z, mu, sigma):
    z = np.asarray(z, dtype=np.float32)
    return (z * sigma + mu).tolist()


def _json_csv_cell(value):
    return json.dumps(value, ensure_ascii=False)


def _batch_time_seq_for_sample(batch_field, sample_idx: int) -> list[str]:
    out = []
    if not isinstance(batch_field, (list, tuple)):
        return out
    for step in batch_field:
        if isinstance(step, (list, tuple)):
            if sample_idx < len(step):
                out.append(str(step[sample_idx]))
        else:
            out.append(str(step))
    return out


def _sign_match_pct(true_vals, pred_vals, eps: float = 1e-8) -> float | None:
    true_arr = np.asarray(true_vals, dtype=np.float32).reshape(-1)
    pred_arr = np.asarray(pred_vals, dtype=np.float32).reshape(-1)
    if true_arr.size == 0 or pred_arr.size == 0 or true_arr.size != pred_arr.size:
        return None

    true_sign = np.where(true_arr > eps, 1, np.where(true_arr < -eps, -1, 0))
    pred_sign = np.where(pred_arr > eps, 1, np.where(pred_arr < -eps, -1, 0))
    return 100.0 * float((true_sign == pred_sign).mean())


def _open_residual_debug_csv(path: str | None):
    if not path:
        return None, None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "split",
        "sample_idx",
        "series_id",
        "target_time",
        "history_start",
        "history_end",
        "target_start",
        "target_end",
        "history_times",
        "target_times",
        "z_input",
        "target_z",
        "base_pred_z",
        "true_residual_z",
        "delta_branch_output",
        "pred_residual_z",
        "pred_residual_sign_match_pct_additive",
        "final_pred_z",
        "gate",
        "gate_logits",
        "sign_logits",
        "sign_soft",
        "magnitude",
        "magnitude_raw",
        "gate_mean",
        "news_count",
        "policy",
        "template_id",
    ]
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


def _eval_iter(data_loader, args, desc: str):
    use_pbar = int(getattr(args, "eval_progress_bar", 1)) == 1
    leave = int(getattr(args, "eval_progress_leave", 0)) == 1
    if use_pbar:
        return tqdm(data_loader, desc=desc, dynamic_ncols=True, leave=leave), True
    return data_loader, False


def _normalize_delta_val_mode(raw_mode) -> str:
    mode = str(raw_mode or "each_epoch").lower().strip()
    alias = {
        "epoch": "each_epoch",
        "per_epoch": "each_epoch",
        "end": "end_only",
        "final": "end_only",
        "last": "end_only",
        "off": "none",
        "disable": "none",
        "disabled": "none",
        "no": "none",
    }
    mode = alias.get(mode, mode)
    if mode not in {"each_epoch", "end_only", "none"}:
        mode = "each_epoch"
    return mode


def _coerce_global_zstats(global_zstats, args, required: bool = True):
    if isinstance(global_zstats, dict):
        mu = global_zstats.get("mu_global", global_zstats.get("mu", None))
        sigma = global_zstats.get("sigma_global", global_zstats.get("sigma", None))
        if mu is not None and sigma is not None:
            eps = float(getattr(args, "zscore_eps", 1e-6))
            sigma = max(float(sigma), eps)
            return {"mu_global": float(mu), "sigma_global": float(sigma)}
    if required:
        raise ValueError("global_zstats must contain mu_global and sigma_global.")
    return None


def _compute_global_zstats_from_train_df(train_df: pd.DataFrame, args):
    if train_df is None or len(train_df) == 0:
        raise ValueError("Cannot compute global z-score stats from empty train_df.")
    if args.value_col not in train_df.columns:
        raise KeyError(f"value_col not found in train_df: {args.value_col}")
    vals = pd.to_numeric(train_df[args.value_col], errors="coerce").to_numpy(dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Train data has no finite values for global z-score statistics.")
    mu, sigma = _zstats(vals, eps=float(getattr(args, "zscore_eps", 1e-6)))
    return {"mu_global": float(mu), "sigma_global": float(sigma)}


def _maybe_news_dropout(news_str: str, args) -> str:
    p = float(getattr(args, "news_dropout", 0.0) or 0.0)
    if p <= 0:
        return news_str
    if np.random.rand() < p:
        return ""
    return news_str


def _refine_cache_key(
    *,
    raw_news_texts: list[str],
    context: dict,
    mode: str,
    model: str,
    max_tokens: int,
) -> str:
    return make_refine_news_cache_key(
        raw_news_texts=raw_news_texts,
        context=context,
        mode=mode,
        model=model,
        max_tokens=max_tokens,
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
        ids = enc.get("input_ids", []) if isinstance(enc, dict) else []
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
                key = rec.get("structured_cache_key", rec.get("cache_key", rec.get("key", rec.get("id", ""))))
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
                    key = rec.get("structured_cache_key", rec.get("cache_key", rec.get("key", rec.get("id", ""))))
                    value = rec.get("structured_events", rec.get("value", {}))
                    if isinstance(key, str) and isinstance(value, dict):
                        k = key.strip()
                        if k:
                            parsed[k] = dict(value)
        except Exception:
            parsed = {}
        store = parsed
    return store


def _load_news_doc_cache_file(path: str) -> dict[str, dict]:
    store = {}
    if not path or (not os.path.exists(path)):
        return store

    def _ingest_record(rec: dict):
        if not isinstance(rec, dict):
            return
        norm = _normalize_news_doc_cache_record(rec)
        primary = str(
            norm.get(
                "cache_key",
                rec.get("doc_cache_key", norm.get("structured_cache_key", rec.get("key", rec.get("id", "")))),
            )
        ).strip()
        if not primary:
            return
        store[primary] = norm

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            _ingest_record(obj)
        elif isinstance(obj, list):
            for rec in obj:
                _ingest_record(rec)
    except Exception:
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
                    _ingest_record(rec)
        except Exception:
            store = {}
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


def _build_news_doc_meta_index(news_df: pd.DataFrame, *, text_col: str, time_col: str) -> dict[str, dict]:
    idx = {}
    if text_col not in news_df.columns:
        return idx
    title_col = "title" if "title" in news_df.columns else ""
    url_col = "url" if "url" in news_df.columns else ""
    for _, row in news_df.iterrows():
        raw_text = str(row.get(text_col, "") or "").strip()
        if not raw_text or raw_text in idx:
            continue
        ts_value = row.get(time_col, "")
        if hasattr(ts_value, "isoformat"):
            ts_value = ts_value.isoformat()
        meta = {
            "title": str(row.get(title_col, "") or "").strip() if title_col else "",
            "date": str(ts_value or "").strip(),
            "url": str(row.get(url_col, "") or "").strip() if url_col else "",
        }
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
        "cache_key": str(rec.get("cache_key", "") or "").strip(),
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


def _upsert_news_doc_cache_record(
    args,
    *,
    refine_cache_key: str = "",
    structured_cache_key: str = "",
    raw_news_text: str = "",
    refined_news: str = "",
    structured_events: dict | None = None,
    source_kind: str = "",
    context: dict | None = None,
    model: str = "",
    max_tokens: int | None = None,
):
    store = getattr(args, "_news_doc_cache_store", None)
    pending = getattr(args, "_news_doc_cache_pending", None)
    structured_index = getattr(args, "_news_doc_cache_structured_index", None)
    if not isinstance(store, dict) or not isinstance(pending, dict) or not isinstance(structured_index, dict):
        return

    primary_key = ""
    if refine_cache_key and refine_cache_key in store:
        primary_key = refine_cache_key
    elif structured_cache_key and structured_cache_key in structured_index:
        primary_key = structured_index.get(structured_cache_key, "")
    elif refine_cache_key:
        primary_key = refine_cache_key
    elif structured_cache_key:
        primary_key = structured_cache_key
    else:
        return

    record = _normalize_news_doc_cache_record(store.get(primary_key, {}))
    meta = _lookup_news_doc_meta(args, raw_news_text)
    if refine_cache_key:
        record["cache_key"] = str(refine_cache_key)
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


def _save_news_doc_cache(args, live_logger=None, force: bool = False):
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
        for _, v in sorted(store.items(), key=lambda x: str(x[0]))
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
                refine_cache_key=cache_key,
                raw_news_text=clean,
                refined_news=refined,
                context=context,
                model=model,
                max_tokens=max_tokens,
            )
    return refined


def _refine_news_from_doc_cache(
    *,
    raw_news_texts: list[str],
    tokenizer,
    max_tokens: int,
    args,
    api_adapter=None,
) -> str:
    snippets = _refine_news_docs_from_doc_cache(
        raw_news_texts=raw_news_texts,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        args=args,
        api_adapter=api_adapter,
    )
    return _merge_refined_news_docs(snippets, tokenizer=tokenizer, max_tokens=max_tokens)


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
    tokenizer,
    max_tokens: int,
    args,
    api_adapter=None,
) -> list[str]:
    items = [str(x).strip() for x in raw_news_texts if str(x).strip()]
    if len(items) == 0:
        return []

    snippets = []
    seen = set()
    for item in items:
        refined_item = _refine_one_news_doc(
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
    return snippets


def _init_refine_cache(args, live_logger=None):
    enabled = int(getattr(args, "news_refine_cache_enable", 1)) == 1
    setattr(args, "_refine_cache_enabled", bool(enabled))
    setattr(args, "_refine_cache_store", {})
    setattr(args, "_refine_cache_pending", {})
    setattr(args, "_news_doc_cache_store", {})
    setattr(args, "_news_doc_cache_pending", {})
    setattr(args, "_news_doc_cache_structured_index", {})
    setattr(args, "_news_doc_cache_dirty", False)
    setattr(args, "_refine_cache_dirty", False)
    setattr(args, "_refine_cache_hits", 0)
    setattr(args, "_refine_cache_misses", 0)
    if not enabled:
        if live_logger is not None:
            live_logger.info("[NEWS_REFINE_CACHE] disabled.")
        return

    path = _refine_cache_path(args)
    setattr(args, "_refine_cache_path", path)
    setattr(args, "_news_doc_cache_path", path)
    read_paths = _parse_refine_cache_read_paths(getattr(args, "news_refine_cache_read_path", ""))
    merged_read_paths = list(read_paths)
    path_abs = os.path.abspath(path)
    if path_abs not in {os.path.abspath(p) for p in merged_read_paths}:
        merged_read_paths.append(path)
    setattr(args, "_refine_cache_read_paths", merged_read_paths)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    store = {}
    doc_store = _load_news_doc_cache_file(path)
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
    setattr(args, "_news_doc_cache_structured_index", structured_index)
    if live_logger is not None:
        if read_paths:
            live_logger.info(
                f"[NEWS_REFINE_CACHE] enabled write_path={path} entries={len(store)} "
                f"preload_sources={source_summaries}"
            )
        else:
            live_logger.info(
                f"[NEWS_REFINE_CACHE] enabled path={path} entries={len(store)}"
            )


def _save_refine_cache(args, live_logger=None, force: bool = False):
    if not bool(getattr(args, "_refine_cache_enabled", False)):
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

    path = _structured_cache_path(args)
    setattr(args, "_structured_cache_path", path)
    setattr(args, "_news_doc_cache_path", path)
    existing_doc_store = getattr(args, "_news_doc_cache_store", None)
    loaded_doc_store = _load_news_doc_cache_file(path)
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
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    store = {}
    source_summaries = []
    for src in merged_read_paths:
        loaded = _load_structured_cache_file(src)
        if loaded:
            store.update(loaded)
        source_summaries.append(f"{src}:{len(loaded)}")
    setattr(args, "_structured_cache_store", store)
    structured_index = getattr(args, "_news_doc_cache_structured_index", None)
    if isinstance(structured_index, dict):
        structured_index.clear()
        for primary_key, rec in getattr(args, "_news_doc_cache_store", {}).items():
            if not isinstance(rec, dict):
                continue
            s_key = str(rec.get("structured_cache_key", "") or "").strip()
            if s_key:
                structured_index[s_key] = str(primary_key)
        setattr(args, "_news_doc_cache_structured_index", structured_index)
    if live_logger is not None:
        if read_paths:
            live_logger.info(
                f"[NEWS_STRUCTURED_CACHE] enabled write_path={path} entries={len(store)} "
                f"preload_sources={source_summaries}"
            )
        else:
            live_logger.info(
                f"[NEWS_STRUCTURED_CACHE] enabled path={path} entries={len(store)}"
            )
        if len(store) == 0:
            live_logger.info(
                "[NEWS_STRUCTURED_CACHE] cache is currently empty; "
                "this is expected before structured prewarm or first structured extraction."
            )


def _save_structured_cache(args, live_logger=None, force: bool = False):
    if not bool(getattr(args, "_structured_cache_enabled", False)):
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
            return dict(cached)

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
            refine_cache_key = ""
            if raw_clean and refine_max_tokens is not None:
                refine_cache_key = make_refine_doc_cache_key(
                    raw_news_text=raw_clean,
                    context=ctx,
                    mode=str(getattr(args, "news_refine_mode", "local") or "local").lower().strip(),
                    model=_resolve_refine_model_name(args, api_adapter),
                    max_tokens=int(max(1, refine_max_tokens)),
                )
            _upsert_news_doc_cache_record(
                args,
                refine_cache_key=refine_cache_key,
                structured_cache_key=cache_key,
                raw_news_text=raw_clean,
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
    args,
    api_adapter=None,
    context: dict | None = None,
) -> tuple[dict, list[dict]]:
    source_kind = _structured_doc_source_kind(args)
    raw_items_all = [str(x).strip() for x in (raw_news_texts or []) if str(x).strip()]
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
        dedup_reused = item in local_cache
        if dedup_reused:
            ev = dict(local_cache[item])
        else:
            raw_item = item if source_kind == "raw" else (raw_items_all[idx] if idx < len(raw_items_all) else item)
            ev = _extract_structured_event_one_doc(
                news_text=item,
                args=args,
                api_adapter=api_adapter,
                context=context,
                source_kind=source_kind,
                raw_news_text=raw_item,
                refine_max_tokens=refine_max_tokens,
            )
            ev = dict(ev) if isinstance(ev, dict) else {}
            local_cache[item] = dict(ev)
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
    args,
    api_adapter=None,
    context: dict | None = None,
) -> dict:
    merged, _doc_records = _extract_structured_events_from_refined_docs_detailed(
        raw_news_texts=raw_news_texts,
        refined_news_texts=refined_news_texts,
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


def _bounded_sigmoid_gate(logits: torch.Tensor, args) -> torch.Tensor:
    """
    Sample-wise gate in [floor, 1], controlled by temperature.
    """
    temperature = float(getattr(args, "news_gate_temperature", 1.0) or 1.0)
    floor = float(getattr(args, "news_gate_floor", 0.0) or 0.0)
    temperature = max(1e-6, temperature)
    floor = max(0.0, min(1.0, floor))
    gate = torch.sigmoid(logits / temperature)
    return floor + (1.0 - floor) * gate


def _match_gate_shape(gate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    g = gate.to(torch.float32)
    if reference.ndim != 2:
        return g
    horizon = int(reference.size(1))
    if g.ndim == 1:
        return g.unsqueeze(1).expand(-1, horizon)
    if g.ndim == 2:
        if g.size(1) == horizon:
            return g
        if g.size(1) == 1:
            return g.expand(-1, horizon)
        if g.size(1) > horizon:
            return g[:, :horizon]
        pad = g[:, -1:].expand(-1, horizon - g.size(1))
        return torch.cat([g, pad], dim=1)
    return g.reshape(g.size(0), -1)


def _build_horizon_gate_targets(
    *,
    pred_real_z: torch.Tensor,
    pred_null_z: torch.Tensor,
    targets_z: torch.Tensor,
    args,
) -> torch.Tensor:
    err_real_h = torch.abs(pred_real_z.to(torch.float32) - targets_z.to(torch.float32))
    err_null_h = torch.abs(pred_null_z.to(torch.float32) - targets_z.to(torch.float32))
    margin = float(getattr(args, "cf_pseudo_margin", 0.01) or 0.0)
    temp = max(1e-6, float(getattr(args, "cf_pseudo_temp", 0.2) or 0.2))
    gain = err_null_h - err_real_h - margin
    if int(getattr(args, "cf_pseudo_hard", 0)) == 1:
        return (gain > 0).to(dtype=torch.float32)
    return torch.sigmoid(gain / temp).to(dtype=torch.float32)


def _weighted_sample_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    v = values.to(torch.float32)
    w = weights.to(torch.float32)
    if v.ndim > 1:
        v = v.reshape(v.size(0), -1).mean(dim=1)
    else:
        v = v.reshape(-1)
    if w.ndim > 1:
        w = w.reshape(w.size(0), -1).mean(dim=1)
    else:
        w = w.reshape(-1)
    denom = w.sum().clamp_min(1e-6)
    return (v * w).sum() / denom


def _build_news_usefulness_weights(
    *,
    has_news: torch.Tensor,
    news_counts: torch.Tensor | None,
    structured_feats: torch.Tensor | None,
    enabled: bool,
) -> torch.Tensor:
    base = torch.ones_like(has_news, dtype=torch.float32)
    if not enabled:
        return base

    w = base.clone()
    has_news_f = has_news.to(torch.float32)
    w = w - 0.25 * (1.0 - has_news_f)

    if news_counts is not None:
        counts = news_counts.to(torch.float32).clamp_min(0.0)
        w = w + 0.10 * torch.clamp(counts, 0.0, 4.0) / 4.0

    if structured_feats is not None and structured_feats.ndim == 2 and structured_feats.size(1) >= 5:
        sf = structured_feats.to(torch.float32)
        relevance = sf[:, 0].clamp(0.0, 1.0)
        confidence = sf[:, 4].clamp(0.0, 1.0)
        w = w + 0.25 * relevance * confidence

    return w.clamp(0.5, 1.5)


def _structured_consistency_losses(
    *,
    out_delta: dict,
    structured_feats: torch.Tensor,
    sample_weight: torch.Tensor,
    gate_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    device = structured_feats.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if structured_feats.ndim != 2 or structured_feats.size(1) < 5:
        return zero, {
            "sign": zero,
            "scale": zero,
            "decay": zero,
            "mask": zero,
        }

    sign_logit = out_delta.get("struct_sign_logit", None)
    struct_scale = out_delta.get("struct_scale", None)
    struct_decay = out_delta.get("struct_decay", None)
    struct_mask_logits = out_delta.get("struct_mask_logits", None)
    if sign_logit is None and struct_scale is None and struct_decay is None and struct_mask_logits is None:
        return zero, {
            "sign": zero,
            "scale": zero,
            "decay": zero,
            "mask": zero,
        }

    sf = structured_feats.to(torch.float32)
    relevance = sf[:, 0].clamp(0.0, 1.0)
    direction = sf[:, 1].clamp(-1.0, 1.0)
    strength = sf[:, 2].clamp(0.0, 1.0)
    persistence = sf[:, 3].clamp(0.0, 1.0)
    active_mask = (relevance > 0.0).to(torch.float32)
    sample_w = sample_weight.to(torch.float32) * active_mask

    loss_sign = zero
    if sign_logit is not None:
        pred_sign = torch.tanh(sign_logit.to(torch.float32))
        pred_sign = pred_sign.reshape(pred_sign.size(0), -1).mean(dim=1)
        sign_mask = (active_mask > 0.0) * (direction.abs() > 0.0).to(torch.float32)
        if float(sign_mask.sum().detach().cpu()) > 0.0:
            loss_sign = _weighted_sample_mean(
                torch.abs(pred_sign - direction),
                sample_weight.to(torch.float32) * sign_mask,
            )

    loss_scale = zero
    if struct_scale is not None:
        pred_scale = 1.0 - torch.exp(-struct_scale.to(torch.float32).reshape(struct_scale.size(0), -1).mean(dim=1).clamp_min(0.0))
        if float(active_mask.sum().detach().cpu()) > 0.0:
            loss_scale = _weighted_sample_mean(torch.abs(pred_scale - strength), sample_w)

    loss_decay = zero
    if struct_decay is not None:
        pred_persistence = torch.exp(-struct_decay.to(torch.float32).reshape(struct_decay.size(0), -1).mean(dim=1).clamp_min(0.0))
        if float(active_mask.sum().detach().cpu()) > 0.0:
            loss_decay = _weighted_sample_mean(torch.abs(pred_persistence - persistence), sample_w)

    loss_mask = zero
    if struct_mask_logits is not None:
        logits = struct_mask_logits.to(torch.float32)
        if gate_targets is not None:
            target = gate_targets.to(torch.float32)
            if target.shape != logits.shape:
                target = _match_gate_shape(target, logits)
        else:
            target = relevance.unsqueeze(1).expand_as(logits)
        mask_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weight_h = sample_weight.to(torch.float32).unsqueeze(1) * active_mask.unsqueeze(1).expand_as(mask_loss)
        denom = weight_h.sum().clamp_min(1.0)
        loss_mask = (mask_loss * weight_h).sum() / denom

    total = loss_sign + loss_scale + loss_decay + loss_mask
    return total, {
        "sign": loss_sign,
        "scale": loss_scale,
        "decay": loss_decay,
        "mask": loss_mask,
    }


def _all_gates_disabled(args) -> bool:
    return int(getattr(args, "disable_all_gates", 0)) == 1


def _final_gate_enabled(args) -> bool:
    return (not _all_gates_disabled(args)) and int(getattr(args, "news_gate_enable", 1)) == 1


def _epoch_ramp_scale(epoch_idx: int, warmup_epochs: int, curriculum_epochs: int) -> float:
    """
    Piecewise ramp:
      - 0 before warmup end
      - linearly to 1 during curriculum window
      - 1 afterwards
    """
    e = int(max(0, epoch_idx))
    w = int(max(0, warmup_epochs))
    c = int(max(1, curriculum_epochs))
    if e < w:
        return 0.0
    progress = float(e - w + 1) / float(c)
    return float(max(0.0, min(1.0, progress)))


def _step_ramp_scale(step_idx: int, warmup_steps: int, ramp_steps: int) -> float:
    """
    Step-wise ramp used for null regularization:
      - 0 before warmup steps
      - linearly to 1 during ramp window
      - 1 afterwards
    """
    s = int(max(0, step_idx))
    w = int(max(0, warmup_steps))
    r = int(max(0, ramp_steps))
    if s < w:
        return 0.0
    if r <= 0:
        return 1.0
    progress = float(s - w + 1) / float(r)
    return float(max(0.0, min(1.0, progress)))


def _delta_gate_is_modeled_internally(args) -> bool:
    return int(getattr(args, "delta_internal_gate_in_model", getattr(args, "delta_internal_gate", 1))) == 1


def _resolve_delta_residual_mode(args) -> str:
    if _delta_gate_is_modeled_internally(args):
        return "additive"
    mode = str(getattr(args, "delta_residual_mode", "additive")).lower().strip()
    if mode not in {"additive", "relative"}:
        mode = "additive"
    return mode


def _resolve_delta_sign_mode(args) -> str:
    mode = str(getattr(args, "delta_sign_mode", "signnet_binary") or "signnet_binary").lower().strip()
    if mode != "signnet_binary":
        raise ValueError(
            f"Unsupported delta_sign_mode={mode!r}. This framework now supports only 'signnet_binary'."
        )
    return mode


def _use_external_signnet(args) -> bool:
    return _resolve_delta_sign_mode(args) == "signnet_binary"


def _z_to_raw_tensor(x_z: torch.Tensor, mu_global: float, sigma_global: float) -> torch.Tensor:
    return x_z.to(torch.float32) * float(sigma_global) + float(mu_global)


def _raw_to_z_tensor(x_raw: torch.Tensor, mu_global: float, sigma_global: float) -> torch.Tensor:
    sigma = max(float(sigma_global), 1e-6)
    return (x_raw.to(torch.float32) - float(mu_global)) / sigma


def _safe_signed_denom_tensor(base_raw: torch.Tensor, args) -> torch.Tensor:
    floor = float(getattr(args, "delta_relative_denom_floor", 1.0) or 1.0)
    floor = max(1e-6, floor)
    sign = torch.where(base_raw >= 0, torch.ones_like(base_raw), -torch.ones_like(base_raw))
    return torch.where(base_raw.abs() >= floor, base_raw, sign * floor)


def _convert_knn_prior_z_to_relative_np(
    prior_z: np.ndarray,
    base_pred_z: np.ndarray,
    args,
    mu_global: float,
    sigma_global: float,
) -> np.ndarray:
    prior = np.asarray(prior_z, dtype=np.float32).reshape(-1)
    base = np.asarray(base_pred_z, dtype=np.float32).reshape(-1)
    if prior.size == 0:
        return prior
    if base.size < prior.size:
        pad = np.zeros((prior.size - base.size,), dtype=np.float32)
        base = np.concatenate([base, pad], axis=0)
    elif base.size > prior.size:
        base = base[: prior.size]

    prior_raw = prior * float(sigma_global)
    base_raw = base * float(sigma_global) + float(mu_global)

    floor = float(getattr(args, "delta_relative_denom_floor", 1.0) or 1.0)
    floor = max(1e-6, floor)
    sign = np.where(base_raw >= 0.0, 1.0, -1.0).astype(np.float32)
    denom = np.where(np.abs(base_raw) >= floor, base_raw, sign * floor).astype(np.float32)
    ratio = prior_raw / denom

    ratio_clip = float(getattr(args, "delta_relative_ratio_clip", 0.0) or 0.0)
    if ratio_clip > 0.0:
        ratio = np.clip(ratio, -ratio_clip, ratio_clip)
    return ratio.astype(np.float32, copy=False)


def _fuse_base_and_delta(
    *,
    base_pred_z: torch.Tensor,
    delta_pred: torch.Tensor,
    gate_h: torch.Tensor,
    args,
    mu_global: float,
    sigma_global: float,
) -> torch.Tensor:
    mode = _resolve_delta_residual_mode(args)
    base_z = base_pred_z.to(torch.float32)
    delta = delta_pred.to(torch.float32)
    gate = gate_h.to(torch.float32)
    if mode == "additive":
        if _delta_gate_is_modeled_internally(args):
            return base_z + delta
        return base_z + gate * delta

    base_raw = _z_to_raw_tensor(base_z, mu_global=mu_global, sigma_global=sigma_global)
    ratio = gate * delta
    ratio_clip = float(getattr(args, "delta_relative_ratio_clip", 0.0) or 0.0)
    if ratio_clip > 0.0:
        ratio = ratio.clamp(min=-ratio_clip, max=ratio_clip)
    pred_raw = base_raw * (1.0 + ratio)
    return _raw_to_z_tensor(pred_raw, mu_global=mu_global, sigma_global=sigma_global)


def _build_delta_targets(
    targets_z: torch.Tensor,
    base_pred: torch.Tensor,
    mu_global: float,
    sigma_global: float,
    args,
) -> torch.Tensor:
    if _delta_gate_is_modeled_internally(args):
        delta_target = (targets_z.to(torch.float32) - base_pred.to(torch.float32)).detach()
        target_clip = float(getattr(args, "delta_target_clip", 0.0) or 0.0)
        if target_clip > 0.0:
            delta_target = delta_target.clamp(min=-target_clip, max=target_clip)
        return delta_target

    mode = _resolve_delta_residual_mode(args)
    if mode == "relative":
        target_raw = _z_to_raw_tensor(targets_z.to(torch.float32), mu_global=mu_global, sigma_global=sigma_global)
        base_raw = _z_to_raw_tensor(base_pred.to(torch.float32), mu_global=mu_global, sigma_global=sigma_global)
        denom = _safe_signed_denom_tensor(base_raw, args)
        delta_target = ((target_raw - base_raw) / denom).detach()
    else:
        delta_target = (targets_z.to(torch.float32) - base_pred.to(torch.float32)).detach()
    target_clip = float(getattr(args, "delta_target_clip", 0.0) or 0.0)
    if target_clip > 0.0:
        delta_target = delta_target.clamp(min=-target_clip, max=target_clip)
    return delta_target


def _resolve_delta_mag_target(args) -> str:
    mode = str(getattr(args, "delta_mag_target", "log1p") or "log1p").lower().strip()
    return mode if mode in {"raw", "log1p"} else "log1p"


def _transform_delta_magnitude_target(x: torch.Tensor, args) -> torch.Tensor:
    x_pos = x.to(torch.float32).clamp_min(0.0)
    if _resolve_delta_mag_target(args) == "log1p":
        return torch.log1p(x_pos)
    return x_pos


def _build_delta_residual_position_weights(abs_target: torch.Tensor, args) -> torch.Tensor:
    scale = float(getattr(args, "delta_residual_weight_scale", 1.0) or 0.0)
    if scale <= 0.0:
        return torch.ones_like(abs_target, dtype=torch.float32)
    return 1.0 + scale * abs_target.to(torch.float32).clamp(0.0, 1.0)


def _masked_weighted_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    v = values.to(torch.float32)
    w = weights.to(torch.float32)
    if mask is not None:
        w = w * mask.to(torch.float32)
    denom = w.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return torch.zeros((), device=v.device, dtype=torch.float32)
    return (v * w).sum() / denom.clamp_min(1.0)


def _masked_binary_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = (logits.to(torch.float32) > 0).to(torch.float32)
    tgt = targets.to(torch.float32)
    correct = (pred == tgt).to(torch.float32)
    if mask is None:
        return correct.mean()
    m = mask.to(torch.float32)
    denom = m.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return torch.zeros((), device=correct.device, dtype=torch.float32)
    return (correct * m).sum() / denom.clamp_min(1.0)


def _masked_binary_balanced_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = logits.to(torch.float32) > 0.0
    tgt = targets.to(torch.float32) > 0.5
    if mask is None:
        m = torch.ones_like(targets, dtype=torch.float32)
    else:
        m = mask.to(torch.float32)
    valid = m > 0.5
    if int(valid.sum().detach().cpu().item()) <= 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    tp = ((pred & tgt) & valid).to(torch.float32).sum()
    fn = (((~pred) & tgt) & valid).to(torch.float32).sum()
    tn = (((~pred) & (~tgt)) & valid).to(torch.float32).sum()
    fp = ((pred & (~tgt)) & valid).to(torch.float32).sum()
    tpr = tp / (tp + fn).clamp_min(1.0)
    tnr = tn / (tn + fp).clamp_min(1.0)
    return 0.5 * (tpr + tnr)


def _masked_binary_pos_weight(
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight_floor: float = 0.5,
    weight_clip: float = 3.0,
) -> torch.Tensor:
    t = targets.to(torch.float32)
    m = mask.to(torch.float32)
    pos = (t * m).sum()
    neg = ((1.0 - t) * m).sum()
    if float(pos.detach().cpu()) <= 0.0 or float(neg.detach().cpu()) <= 0.0:
        return torch.ones((), device=t.device, dtype=torch.float32)
    w = neg / pos.clamp_min(1.0)
    lo = float(max(0.0, weight_floor))
    hi = float(max(lo + 1e-6, weight_clip))
    return w.clamp(min=lo, max=hi).to(torch.float32)


def _normalize_signnet_select_metric(raw_metric) -> str:
    metric = str(raw_metric or "acc").lower().strip()
    if metric in {"balanced_acc", "balanced", "balanced_accuracy", "bacc"}:
        return "balanced_acc"
    if metric == "loss":
        return "loss"
    return "acc"


class ResidualSignNet(nn.Module):
    """
    Independent sign classifier trained before DELTA.
    Inputs:
      - raw history values
      - base prediction in z-space
      - structured news feature vector
      - news count scalar
    Output:
      - horizon-wise binary sign logits (positive vs negative residual)
    """

    def __init__(
        self,
        history_len: int,
        horizon: int,
        structured_dim: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_len = int(max(1, history_len))
        self.horizon = int(max(1, horizon))
        self.structured_dim = int(max(0, structured_dim))
        hidden = int(max(32, hidden_size))
        drop = float(max(0.0, dropout))

        self.hist_norm = nn.LayerNorm(self.history_len)
        self.hist_proj = nn.Sequential(
            nn.Linear(self.history_len, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(hidden),
        )
        self.base_proj = nn.Sequential(
            nn.Linear(self.horizon, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(hidden),
        )
        if self.structured_dim > 0:
            self.structured_proj = nn.Sequential(
                nn.Linear(self.structured_dim, hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.LayerNorm(hidden),
            )
        else:
            self.structured_proj = None
        self.news_count_proj = nn.Sequential(
            nn.Linear(1, hidden // 2),
            nn.GELU(),
            nn.Dropout(drop),
        )

        fuse_in = hidden * 2 + (hidden if self.structured_proj is not None else 0) + hidden // 2
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(hidden),
        )
        self.out_head = nn.Linear(hidden, self.horizon)
        self.register_buffer("decision_bias", torch.zeros((), dtype=torch.float32))

    def forward(
        self,
        history_raw: torch.Tensor,
        base_pred_z: torch.Tensor,
        structured_feats: torch.Tensor | None,
        news_counts: torch.Tensor | None,
    ) -> torch.Tensor:
        h_raw = history_raw.to(torch.float32)
        if h_raw.ndim != 2:
            h_raw = h_raw.reshape(h_raw.size(0), -1)
        if h_raw.size(1) < self.history_len:
            pad = h_raw.new_zeros(h_raw.size(0), self.history_len - h_raw.size(1))
            h_raw = torch.cat([pad, h_raw], dim=1)
        elif h_raw.size(1) > self.history_len:
            h_raw = h_raw[:, -self.history_len :]

        base = base_pred_z.to(torch.float32)
        if base.ndim != 2:
            base = base.reshape(base.size(0), -1)
        if base.size(1) < self.horizon:
            pad = base.new_zeros(base.size(0), self.horizon - base.size(1))
            base = torch.cat([base, pad], dim=1)
        elif base.size(1) > self.horizon:
            base = base[:, : self.horizon]

        parts = [
            self.hist_proj(self.hist_norm(h_raw)),
            self.base_proj(base),
        ]
        if self.structured_proj is not None:
            sf = structured_feats
            if sf is None:
                sf = h_raw.new_zeros(h_raw.size(0), self.structured_dim)
            sf = sf.to(torch.float32)
            if sf.ndim != 2:
                sf = sf.reshape(sf.size(0), -1)
            if sf.size(1) < self.structured_dim:
                pad = sf.new_zeros(sf.size(0), self.structured_dim - sf.size(1))
                sf = torch.cat([sf, pad], dim=1)
            elif sf.size(1) > self.structured_dim:
                sf = sf[:, : self.structured_dim]
            parts.append(self.structured_proj(sf))

        nc = news_counts
        if nc is None:
            nc = h_raw.new_zeros(h_raw.size(0))
        nc = nc.to(torch.float32).reshape(-1, 1)
        parts.append(self.news_count_proj(nc))

        fused = self.fuse(torch.cat(parts, dim=-1))
        return self.out_head(fused)


def _history_raw_tensor_from_batch(batch, args, device) -> torch.Tensor:
    L = int(max(1, getattr(args, "history_len", 1)))
    B = len(batch["history_value"])
    arr = np.zeros((B, L), dtype=np.float32)
    for i in range(B):
        h_i = batch["history_value"][i]
        if torch.is_tensor(h_i):
            v = h_i.detach().cpu().numpy()
        else:
            v = np.asarray(h_i, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size <= 0:
            continue
        if v.size >= L:
            arr[i] = v[-L:]
        else:
            arr[i, -v.size :] = v
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _run_external_signnet(
    signnet_model: ResidualSignNet,
    history_raw: torch.Tensor,
    base_pred_z: torch.Tensor,
    structured_feats: torch.Tensor,
    news_counts: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = signnet_model(
        history_raw=history_raw,
        base_pred_z=base_pred_z.to(torch.float32),
        structured_feats=structured_feats.to(torch.float32),
        news_counts=news_counts.to(torch.float32),
    ).to(torch.float32)
    bias = getattr(signnet_model, "decision_bias", None)
    if torch.is_tensor(bias):
        logits = logits + bias.to(device=logits.device, dtype=logits.dtype)
    elif bias is not None:
        logits = logits + float(bias)
    t = max(1e-6, float(tau))
    sign_soft = torch.tanh(logits / t)
    return logits, sign_soft


def _compose_delta_with_external_sign(
    *,
    gate: torch.Tensor,
    magnitude: torch.Tensor,
    sign_soft: torch.Tensor,
    delta_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mag = magnitude.to(torch.float32)
    gate_h = _match_gate_shape(gate.to(torch.float32), mag)
    sign_h = _match_gate_shape(sign_soft.to(torch.float32), mag)
    delta = gate_h * sign_h * mag
    clip_v = float(delta_clip)
    if clip_v > 0.0:
        c = torch.tensor(clip_v, device=delta.device, dtype=delta.dtype)
        delta = c * torch.tanh(delta / c)
    return delta, gate_h


@torch.no_grad()
def _evaluate_external_signnet(
    *,
    signnet_model: ResidualSignNet,
    base_backbone,
    tokenizer,
    data_loader,
    templates,
    tpl_id: int,
    args,
    global_zstats,
    news_df,
    policy_name: str,
    policy_kw,
    device,
    volatility_bin,
    api_adapter=None,
    eval_desc: str = "[SIGNNET][VAL]",
    return_tensors: bool = False,
):
    if data_loader is None:
        if return_tensors:
            empty = torch.zeros(0, dtype=torch.float32)
            return 0.0, 0.0, 0.0, 0.0, {"logits": empty, "targets": empty, "mask": empty}
        return 0.0, 0.0, 0.0, 0.0
    signnet_model.eval()
    base_backbone.eval()
    use_news_weighting = int(getattr(args, "delta_sign_external_use_news_weighting", 0)) == 1
    use_residual_weighting = int(getattr(args, "delta_sign_external_use_residual_weighting", 0)) == 1
    use_pos_weight = int(getattr(args, "delta_sign_external_use_pos_weight", 1)) == 1
    pos_weight_floor = float(max(0.0, getattr(args, "delta_sign_external_pos_weight_floor", 0.5)))
    pos_weight_clip = float(max(pos_weight_floor + 1e-6, getattr(args, "delta_sign_external_pos_weight_clip", 3.0)))

    loss_sum = 0.0
    acc_sum = 0.0
    bacc_sum = 0.0
    n_samples = 0
    valid_sum = 0.0
    total_positions = 0.0
    all_logits = []
    all_targets = []
    all_masks = []
    loader, use_pbar = _eval_iter(data_loader, args, desc=eval_desc)

    for _, batch in enumerate(loader):
        history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
        history_z = history_z.to(device)
        base_pred = base_backbone(history_z).to(torch.float32)

        delta_inputs = build_delta_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            testing=False,
            force_no_news=False,
            news_dropout=False,
            api_adapter=api_adapter,
        )
        targets_z = delta_inputs["targets_z"].to(device=device, dtype=torch.float32)
        structured_feats = delta_inputs["structured_feats"].to(device=device, dtype=torch.float32)
        news_counts = delta_inputs["news_counts"].to(device=device, dtype=torch.float32)
        history_raw = _history_raw_tensor_from_batch(batch, args, device=device)
        sign_logits, _ = _run_external_signnet(
            signnet_model=signnet_model,
            history_raw=history_raw,
            base_pred_z=base_pred,
            structured_feats=structured_feats,
            news_counts=news_counts,
            tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
        )

        true_residual_z = targets_z - base_pred
        abs_residual = true_residual_z.abs()
        sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
        valid_mask = (abs_residual > sign_eps).to(torch.float32)
        sign_target_bin = (true_residual_z > 0).to(torch.float32)

        has_news = (news_counts > 0).to(torch.float32)
        if use_news_weighting:
            sample_weight = _build_news_usefulness_weights(
                has_news=has_news,
                news_counts=news_counts,
                structured_feats=structured_feats,
                enabled=True,
            )
        else:
            sample_weight = torch.ones_like(news_counts, dtype=torch.float32)
        if use_residual_weighting:
            position_weight = _build_delta_residual_position_weights(abs_residual, args)
        else:
            position_weight = torch.ones_like(abs_residual, dtype=torch.float32)
        sample_pos_weight = sample_weight.unsqueeze(1) * position_weight

        bce_kwargs = {"reduction": "none"}
        if use_pos_weight:
            bce_kwargs["pos_weight"] = _masked_binary_pos_weight(
                sign_target_bin,
                valid_mask,
                weight_floor=pos_weight_floor,
                weight_clip=pos_weight_clip,
            )
        sign_bce = F.binary_cross_entropy_with_logits(
            sign_logits,
            sign_target_bin,
            **bce_kwargs,
        )
        loss = _masked_weighted_mean(sign_bce, sample_pos_weight, mask=valid_mask)
        acc = _masked_binary_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)
        bacc = _masked_binary_balanced_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)

        bs = int(targets_z.size(0))
        loss_sum += float(loss.detach().cpu()) * bs
        acc_sum += float(acc.detach().cpu()) * bs
        bacc_sum += float(bacc.detach().cpu()) * bs
        n_samples += bs
        valid_sum += float(valid_mask.sum().detach().cpu())
        total_positions += float(valid_mask.numel())
        if return_tensors:
            all_logits.append(sign_logits.detach().cpu())
            all_targets.append(sign_target_bin.detach().cpu())
            all_masks.append(valid_mask.detach().cpu())
        if use_pbar:
            loader.set_postfix(loss=f"{loss_sum / max(1, n_samples):.6f}")

    eval_tuple = (
        loss_sum / max(1, n_samples),
        acc_sum / max(1, n_samples),
        valid_sum / max(1.0, total_positions),
        bacc_sum / max(1, n_samples),
    )
    if not return_tensors:
        return eval_tuple
    if all_logits:
        pack = {
            "logits": torch.cat(all_logits, dim=0).to(torch.float32),
            "targets": torch.cat(all_targets, dim=0).to(torch.float32),
            "mask": torch.cat(all_masks, dim=0).to(torch.float32),
        }
    else:
        empty = torch.zeros(0, dtype=torch.float32)
        pack = {"logits": empty, "targets": empty, "mask": empty}
    return eval_tuple[0], eval_tuple[1], eval_tuple[2], eval_tuple[3], pack


def _calibrate_external_signnet_bias(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    objective: str = "acc",
    max_abs_bias: float = 2.0,
) -> tuple[float, float, int]:
    if logits.numel() <= 0 or targets.numel() <= 0 or mask.numel() <= 0:
        return 0.0, 0.0, 0
    x = logits.reshape(-1).to(torch.float32)
    y = targets.reshape(-1).to(torch.float32)
    m = mask.reshape(-1).to(torch.float32) > 0.5
    if int(m.sum().item()) <= 0:
        return 0.0, 0.0, 0
    x = x[m]
    y = (y[m] > 0.5).to(torch.bool)
    if x.numel() <= 0:
        return 0.0, 0.0, 0

    q = torch.linspace(0.02, 0.98, 97, dtype=torch.float32, device=x.device)
    cand = torch.quantile(x, q)
    zero = torch.zeros(1, dtype=torch.float32, device=x.device)
    cand = torch.unique(torch.cat([cand, zero], dim=0))
    pred_mat = x.unsqueeze(0) > cand.unsqueeze(1)
    obj = _normalize_signnet_select_metric(objective)
    use_bacc = obj == "balanced_acc"
    if use_bacc:
        n_pos = int(y.to(torch.float32).sum().item())
        n_neg = int((~y).to(torch.float32).sum().item())
        if n_pos <= 0 or n_neg <= 0:
            use_bacc = False
    if use_bacc:
        y_row = y.unsqueeze(0)
        tp = (pred_mat & y_row).to(torch.float32).sum(dim=1)
        fn = ((~pred_mat) & y_row).to(torch.float32).sum(dim=1)
        tn = ((~pred_mat) & (~y_row)).to(torch.float32).sum(dim=1)
        fp = (pred_mat & (~y_row)).to(torch.float32).sum(dim=1)
        score_vec = 0.5 * (tp / (tp + fn).clamp_min(1.0) + tn / (tn + fp).clamp_min(1.0))
    else:
        score_vec = (pred_mat == y.unsqueeze(0)).to(torch.float32).mean(dim=1)

    best_idx = int(torch.argmax(score_vec).item())
    best_thr = float(cand[best_idx].item())
    bias = float(-best_thr)
    clip_v = float(max(0.0, max_abs_bias))
    if clip_v > 0.0:
        bias = float(max(-clip_v, min(clip_v, bias)))
    pred_final = (x + bias) > 0
    if use_bacc:
        tp = ((pred_final & y)).to(torch.float32).sum()
        fn = (((~pred_final) & y)).to(torch.float32).sum()
        tn = (((~pred_final) & (~y))).to(torch.float32).sum()
        fp = ((pred_final & (~y))).to(torch.float32).sum()
        final_score = float((0.5 * (tp / (tp + fn).clamp_min(1.0) + tn / (tn + fp).clamp_min(1.0))).item())
    else:
        final_score = float((pred_final == y).to(torch.float32).mean().item())
    return bias, final_score, int(x.numel())


def _train_external_signnet(
    *,
    args,
    base_backbone,
    tokenizer,
    templates,
    tpl_id: int,
    policy_name: str,
    policy_kw,
    train_loader,
    val_loader,
    test_loader,
    news_df,
    volatility_bin,
    volatility_bin_val,
    volatility_bin_test,
    global_zstats,
    device,
    live_logger,
    api_adapter=None,
) -> ResidualSignNet | None:
    if not _use_external_signnet(args):
        return None
    if val_loader is None:
        raise ValueError("delta_sign_mode=signnet_binary requires a non-empty val_loader for signnet selection.")

    epochs = int(max(0, getattr(args, "delta_sign_external_epochs", 0)))
    if epochs <= 0:
        raise ValueError("delta_sign_mode=signnet_binary requires delta_sign_external_epochs > 0.")

    structured_dim = int(getattr(args, "delta_structured_feature_dim", 12))
    signnet = ResidualSignNet(
        history_len=int(max(1, getattr(args, "history_len", 1))),
        horizon=int(max(1, getattr(args, "horizon", 1))),
        structured_dim=max(0, structured_dim),
        hidden_size=int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
        dropout=float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
    ).to(device)

    lr = float(getattr(args, "delta_sign_external_lr", args.lr))
    wd = float(getattr(args, "delta_sign_external_weight_decay", args.weight_decay))
    grad_clip = float(getattr(args, "delta_sign_external_grad_clip", 1.0))
    patience = int(max(0, getattr(args, "delta_sign_external_patience", 0)))
    min_delta = float(max(0.0, getattr(args, "delta_sign_external_min_delta", 1e-4)))
    select_metric = _normalize_signnet_select_metric(getattr(args, "delta_sign_external_select_metric", "acc"))
    lr_factor = float(getattr(args, "delta_sign_external_lr_factor", 0.5))
    lr_patience = int(max(0, getattr(args, "delta_sign_external_lr_patience", 1)))
    min_lr = float(max(0.0, getattr(args, "delta_sign_external_min_lr", 1e-5)))
    calibrate_bias = int(getattr(args, "delta_sign_external_calibrate_bias", 1)) == 1
    bias_clip = float(max(0.0, getattr(args, "delta_sign_external_bias_clip", 2.0)))
    signnet_news_dropout = int(getattr(args, "delta_sign_external_news_dropout", 0)) == 1
    use_news_weighting = int(getattr(args, "delta_sign_external_use_news_weighting", 0)) == 1
    use_residual_weighting = int(getattr(args, "delta_sign_external_use_residual_weighting", 0)) == 1
    use_pos_weight = int(getattr(args, "delta_sign_external_use_pos_weight", 1)) == 1
    pos_weight_floor = float(max(0.0, getattr(args, "delta_sign_external_pos_weight_floor", 0.5)))
    pos_weight_clip = float(max(pos_weight_floor + 1e-6, getattr(args, "delta_sign_external_pos_weight_clip", 3.0)))
    if hasattr(signnet, "decision_bias"):
        signnet.decision_bias.data.zero_()
    opt = AdamW(signnet.parameters(), lr=lr, weight_decay=wd)
    scheduler = None
    if 0.0 < lr_factor < 1.0:
        scheduler_mode = "min" if select_metric == "loss" else "max"
        scheduler = ReduceLROnPlateau(
            opt,
            mode=scheduler_mode,
            factor=lr_factor,
            patience=max(1, lr_patience),
            min_lr=min_lr,
        )

    best_val = float("inf")
    best_acc = float("-inf")
    best_bacc = float("-inf")
    best_state = None
    stale = 0
    if live_logger is not None:
        live_logger.info(
            "[SIGNNET] pretrain start: "
            f"epochs={epochs} lr={lr:.3e} wd={wd:.3e} hidden={int(max(32, getattr(args, 'delta_sign_external_hidden', 256)))} "
            f"dropout={float(max(0.0, getattr(args, 'delta_sign_external_dropout', 0.1))):.3f} "
            f"patience={patience} select_metric={select_metric} min_delta={min_delta:.1e} "
            f"lr_sched_factor={lr_factor:.3f} lr_sched_patience={max(1, lr_patience)} min_lr={min_lr:.3e} "
            f"news_dropout={int(signnet_news_dropout)} calibrate_bias={int(calibrate_bias)} bias_clip={bias_clip:.3f} "
            f"use_news_weighting={int(use_news_weighting)} use_residual_weighting={int(use_residual_weighting)} "
            f"use_pos_weight={int(use_pos_weight)} pos_weight_floor={pos_weight_floor:.3f} pos_weight_clip={pos_weight_clip:.3f}"
        )

    for epoch in range(epochs):
        signnet.train()
        base_backbone.eval()
        pbar = tqdm(train_loader, desc=f"[SIGNNET] Epoch {epoch+1}/{epochs}")
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_bacc_sum = 0.0
        n_samples = 0

        for _, batch in enumerate(pbar):
            with torch.no_grad():
                history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
                history_z = history_z.to(device)
                base_pred = base_backbone(history_z).to(torch.float32)

            delta_inputs = build_delta_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                testing=False,
                force_no_news=False,
                news_dropout=signnet_news_dropout,
                api_adapter=api_adapter,
            )
            targets_z = delta_inputs["targets_z"].to(device=device, dtype=torch.float32)
            structured_feats = delta_inputs["structured_feats"].to(device=device, dtype=torch.float32)
            news_counts = delta_inputs["news_counts"].to(device=device, dtype=torch.float32)
            history_raw = _history_raw_tensor_from_batch(batch, args, device=device)
            sign_logits, _ = _run_external_signnet(
                signnet_model=signnet,
                history_raw=history_raw,
                base_pred_z=base_pred,
                structured_feats=structured_feats,
                news_counts=news_counts,
                tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
            )

            true_residual_z = targets_z - base_pred
            abs_residual = true_residual_z.abs()
            sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
            valid_mask = (abs_residual > sign_eps).to(torch.float32)
            sign_target_bin = (true_residual_z > 0).to(torch.float32)

            has_news = (news_counts > 0).to(torch.float32)
            if use_news_weighting:
                sample_weight = _build_news_usefulness_weights(
                    has_news=has_news,
                    news_counts=news_counts,
                    structured_feats=structured_feats,
                    enabled=True,
                )
            else:
                sample_weight = torch.ones_like(news_counts, dtype=torch.float32)
            if use_residual_weighting:
                position_weight = _build_delta_residual_position_weights(abs_residual, args)
            else:
                position_weight = torch.ones_like(abs_residual, dtype=torch.float32)
            sample_pos_weight = sample_weight.unsqueeze(1) * position_weight

            bce_kwargs = {"reduction": "none"}
            if use_pos_weight:
                bce_kwargs["pos_weight"] = _masked_binary_pos_weight(
                    sign_target_bin,
                    valid_mask,
                    weight_floor=pos_weight_floor,
                    weight_clip=pos_weight_clip,
                )
            sign_bce = F.binary_cross_entropy_with_logits(
                sign_logits,
                sign_target_bin,
                **bce_kwargs,
            )
            loss = _masked_weighted_mean(sign_bce, sample_pos_weight, mask=valid_mask)
            acc = _masked_binary_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)
            bacc = _masked_binary_balanced_accuracy_from_logits(sign_logits, sign_target_bin, valid_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                clip_grad_norm_(signnet.parameters(), grad_clip)
            opt.step()

            bs = int(targets_z.size(0))
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_acc_sum += float(acc.detach().cpu()) * bs
            train_bacc_sum += float(bacc.detach().cpu()) * bs
            n_samples += bs
            pbar.set_postfix(
                loss=f"{train_loss_sum / max(1, n_samples):.6f}",
                acc=f"{train_acc_sum / max(1, n_samples):.4f}",
                bacc=f"{train_bacc_sum / max(1, n_samples):.4f}",
            )

        val_loss, val_acc, val_valid, val_bacc = _evaluate_external_signnet(
            signnet_model=signnet,
            base_backbone=base_backbone,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            api_adapter=api_adapter,
            eval_desc=f"[SIGNNET][VAL] Epoch {epoch+1}/{epochs}",
        )
        if live_logger is not None:
            curr_lr = float(opt.param_groups[0]["lr"])
            live_logger.info(
                f"[SIGNNET][EVAL] epoch={epoch+1} "
                f"train_loss={train_loss_sum / max(1, n_samples):.6f} "
                f"train_acc={train_acc_sum / max(1, n_samples):.4f} "
                f"train_bacc={train_bacc_sum / max(1, n_samples):.4f} "
                f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} val_bacc={val_bacc:.4f} val_valid={val_valid:.4f} lr={curr_lr:.3e}"
            )
        if scheduler is not None:
            if select_metric == "loss":
                scheduler.step(float(val_loss))
            elif select_metric == "balanced_acc":
                scheduler.step(float(val_bacc))
            else:
                scheduler.step(float(val_acc))

        improved = False
        if select_metric == "acc":
            if val_acc > best_acc + min_delta:
                improved = True
            elif abs(val_acc - best_acc) <= min_delta and val_loss < best_val - 1e-6:
                improved = True
        elif select_metric == "balanced_acc":
            if val_bacc > best_bacc + min_delta:
                improved = True
            elif abs(val_bacc - best_bacc) <= min_delta and val_loss < best_val - 1e-6:
                improved = True
        else:
            if val_loss < best_val - min_delta:
                improved = True

        if improved:
            best_val = float(val_loss)
            best_acc = float(val_acc)
            best_bacc = float(val_bacc)
            best_state = {k: v.detach().cpu().clone() for k, v in signnet.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                if live_logger is not None:
                    live_logger.info(f"[SIGNNET] early stop at epoch={epoch+1}")
                break

    if best_state is not None:
        signnet.load_state_dict(best_state, strict=False)
    signnet.eval()
    for p in signnet.parameters():
        p.requires_grad = False

    if calibrate_bias:
        _, _, _, _, val_pack = _evaluate_external_signnet(
            signnet_model=signnet,
            base_backbone=base_backbone,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            api_adapter=api_adapter,
            eval_desc="[SIGNNET][VAL][CALIBRATE]",
            return_tensors=True,
        )
        calibrate_objective = "balanced_acc" if select_metric == "balanced_acc" else "acc"
        bias, cal_score, n_valid = _calibrate_external_signnet_bias(
            logits=val_pack["logits"],
            targets=val_pack["targets"],
            mask=val_pack["mask"],
            objective=calibrate_objective,
            max_abs_bias=bias_clip,
        )
        if hasattr(signnet, "decision_bias"):
            signnet.decision_bias.data.fill_(float(bias))
        if live_logger is not None:
            live_logger.info(
                f"[SIGNNET][CALIBRATE] decision_bias={bias:.6f} objective={calibrate_objective} "
                f"val_score_cal={cal_score:.4f} n_valid={n_valid}"
            )

    if test_loader is not None:
        test_loss, test_acc, test_valid, test_bacc = _evaluate_external_signnet(
            signnet_model=signnet,
            base_backbone=base_backbone,
            tokenizer=tokenizer,
            data_loader=test_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_test,
            api_adapter=api_adapter,
            eval_desc="[SIGNNET][TEST]",
        )
    else:
        test_loss, test_acc, test_valid, test_bacc = float("nan"), float("nan"), float("nan"), float("nan")
    signnet_ckpt_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_signnet_{args.taskName}.pt")
    os.makedirs(os.path.dirname(signnet_ckpt_path), exist_ok=True)
    torch.save(
        {
            "state_dict": signnet.state_dict(),
            "history_len": int(max(1, getattr(args, "history_len", 1))),
            "horizon": int(max(1, getattr(args, "horizon", 1))),
            "structured_dim": int(max(0, structured_dim)),
            "hidden_size": int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
            "dropout": float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
            "best_val_loss": float(best_val),
            "best_val_acc": float(best_acc),
            "best_val_bacc": float(best_bacc),
            "decision_bias": float(getattr(signnet, "decision_bias", torch.zeros((), dtype=torch.float32)).detach().cpu().item())
            if torch.is_tensor(getattr(signnet, "decision_bias", None))
            else float(getattr(signnet, "decision_bias", 0.0)),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_bacc": float(test_bacc),
            "test_valid": float(test_valid),
        },
        signnet_ckpt_path,
    )
    if live_logger is not None:
        bias_now = 0.0
        if hasattr(signnet, "decision_bias"):
            try:
                bias_now = float(signnet.decision_bias.detach().cpu().item())
            except Exception:
                bias_now = 0.0
        live_logger.info(
            f"[SIGNNET][TEST] loss={test_loss:.6f} acc={test_acc:.4f} bacc={test_bacc:.4f} valid={test_valid:.4f} "
            f"bias={bias_now:.6f} "
            f"ckpt={signnet_ckpt_path}"
        )
    return signnet


def _select_metric(loss_v: float, mse_v: float, mae_v: float, select_metric: str) -> float:
    rm = str(select_metric).lower()
    if rm == "loss":
        return float(loss_v)
    if rm == "mse":
        return float(mse_v)
    return float(mae_v)


def _log_prompt_stats_if_available(live_logger, data_statistic, title: str, skip_msg: str) -> None:
    if live_logger is None:
        return
    if int(max(0, getattr(data_statistic, "prompt_num", 0))) <= 0:
        live_logger.info(skip_msg)
        return
    live_logger.info(title)
    print_prompt_stats(live_logger, data_statistic)
    live_logger.info("-----------------------------------------------------")


def _mask_sensitive_arg(key: str, value):
    k = str(key).lower()
    is_secret = any(tok in k for tok in ["api_key", "token", "secret", "password"])
    if not is_secret:
        return value
    s = str(value or "")
    if not s:
        return ""
    if len(s) <= 8:
        return "***"
    return f"{s[:4]}***{s[-4:]}"


def _log_run_args(args, live_logger):
    """
    Keep startup logging concise; detailed args are intentionally suppressed.
    """
    if live_logger is None:
        return
    live_logger.info("-----------------------------------------------------")
    live_logger.info("[CONFIG] Full argument dump suppressed; logging mechanism status only.")
    live_logger.info("-----------------------------------------------------")


def _log_enabled_mechanisms(args, live_logger, stage: str):
    if live_logger is None:
        return
    stage_norm = str(stage or "").lower().strip()
    delta_active = stage_norm in {"delta", "all"}
    news_api_enabled = int(getattr(args, "news_api_enable", 0)) == 1
    structured_enabled = int(getattr(args, "delta_structured_enable", 0)) == 1
    text_direct_enabled = int(getattr(args, "delta_text_direct_enable", 0)) == 1
    text_direct_effective = text_direct_enabled and abs(float(getattr(args, "delta_text_fuse_lambda", 0.0))) > 0.0
    doc_direct_enabled = int(getattr(args, "delta_doc_direct_enable", 0)) == 1
    doc_direct_effective = doc_direct_enabled and abs(float(getattr(args, "delta_doc_fuse_lambda", 0.0))) > 0.0
    non_degrade_enabled = float(getattr(args, "delta_non_degrade_lambda", 0.0)) > 0.0
    sign_enabled = float(getattr(args, "delta_sign_lambda", 0.0)) > 0.0

    live_logger.info(
        f"[MECH][NEWS_API] {'enabled' if news_api_enabled else 'disabled'}: "
        f"refine_mode={getattr(args, 'news_refine_mode', 'na')} "
        f"structured_mode={getattr(args, 'news_structured_mode', 'na')}"
    )
    live_logger.info(
        f"[MECH][STRUCTURED] {'enabled' if structured_enabled else 'disabled'}: "
        f"delta_structured_feature_dim={int(getattr(args, 'delta_structured_feature_dim', 0) or 0)}"
    )
    if _all_gates_disabled(args):
        live_logger.info(
            "[MECH][GATE] disabled: internal delta gate, final fusion gate, and text gate are all bypassed."
        )
    elif int(getattr(args, "news_gate_enable", 1)) != 1 and (not _delta_gate_is_modeled_internally(args)):
        live_logger.info(
            "[MECH][FINAL_GATE] disabled: final fusion gate is bypassed; internal delta gate/text gate remain active."
        )
    elif delta_active:
        live_logger.info(
            "[MECH][FINAL_GATE] enabled in-model: DELTA predicts gate * sign * magnitude directly, "
            "and trainer fusion is additive base + delta."
        )

    if delta_active:
        live_logger.info(
            "[MECH][DELTA_PROMPT] skipped in DELTA model path: template prompt tokens are not "
            "consumed by tiny_news_ts; news enters through structured and direct text/doc branches."
        )
    if text_direct_effective:
        if delta_active:
            live_logger.info(
                "[MECH][TEXT_DIRECT] enabled: refined-news text contributes to the fused news context "
                "when its branch weight is nonzero."
            )
        else:
            live_logger.info(
                "[MECH][TEXT_DIRECT] enabled in args, but current stage is BASE-only; "
                "it will not be executed in this run."
            )
    elif text_direct_enabled:
        live_logger.info(
            "[MECH][TEXT_DIRECT] disabled in practice: branch is enabled in args but fuse_lambda=0, "
            "so it does not contribute to the current run."
        )
    else:
        live_logger.info("[MECH][TEXT_DIRECT] disabled.")
    if doc_direct_effective:
        if delta_active:
            live_logger.info(
                "[MECH][DOC_DIRECT] enabled: article-level refined-news documents are encoded individually "
                "to produce document attention and document impact in the DELTA deformation path."
            )
        else:
            live_logger.info(
                "[MECH][DOC_DIRECT] enabled in args, but current stage is BASE-only; "
                "it will not be executed in this run."
            )
    elif doc_direct_enabled:
        live_logger.info(
            "[MECH][DOC_DIRECT] disabled in practice: branch is enabled in args but fuse_lambda=0, "
            "so it does not contribute to the current run."
        )
    else:
        live_logger.info("[MECH][DOC_DIRECT] disabled.")
    live_logger.info(f"[MECH][NON_DEGRADE] {'enabled' if non_degrade_enabled else 'disabled'}.")
    sign_mode = _resolve_delta_sign_mode(args)
    if sign_mode == "signnet_binary" and delta_active:
        live_logger.info(
            "[MECH][DELTA_SIGN] external signnet enabled: "
            "an independent sign classifier is pretrained first and then replaces DELTA's internal sign path."
        )
    else:
        live_logger.info(
            f"[MECH][DELTA_SIGN] {'enabled' if sign_enabled else 'disabled'} "
            f"(mode={sign_mode})."
        )


def _format_ts_range(ts_min, ts_max) -> str:
    if ts_min is None or ts_max is None or pd.isna(ts_min) or pd.isna(ts_max):
        return "N/A"
    return f"{pd.Timestamp(ts_min)} -> {pd.Timestamp(ts_max)}"


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


def _loader_series_time_range(loader, time_col: str):
    ds = getattr(loader, "dataset", None)
    if ds is None or not hasattr(ds, "groups"):
        return None, None
    vals = []
    for _, g in getattr(ds, "groups", []):
        if isinstance(g, pd.DataFrame) and time_col in g.columns and len(g) > 0:
            vals.append(pd.to_datetime(g[time_col], errors="coerce"))
    if not vals:
        return None, None
    s = pd.concat(vals, ignore_index=True)
    s = s.dropna()
    if len(s) == 0:
        return None, None
    return s.min(), s.max()


def _loader_target_time_range(loader, time_col: str):
    ds = getattr(loader, "dataset", None)
    if ds is None or not hasattr(ds, "index") or not hasattr(ds, "groups"):
        return None, None
    vals = []
    for gi, sidx in getattr(ds, "index", []):
        _, g = ds.groups[gi]
        target_idx = int(sidx) + int(getattr(ds, "L", 0))
        if target_idx >= len(g) or time_col not in g.columns:
            continue
        vals.append(pd.to_datetime(g.iloc[target_idx][time_col], errors="coerce"))
    if not vals:
        return None, None
    s = pd.Series(vals).dropna()
    if len(s) == 0:
        return None, None
    return s.min(), s.max()


def _matched_news_time_range(loader, news_df: pd.DataFrame, *, time_col: str, news_time_col: str, window_days: float):
    if news_df is None or len(news_df) == 0 or news_time_col not in news_df.columns:
        return None, None, 0
    target_min, target_max = _loader_target_time_range(loader, time_col)
    if target_min is None or target_max is None:
        return None, None, 0
    target_min = _align_ts_to_ref_tz(target_min, news_df[news_time_col])
    target_max = _align_ts_to_ref_tz(target_max, news_df[news_time_col])
    if pd.isna(target_min) or pd.isna(target_max):
        return None, None, 0
    start = target_min - pd.Timedelta(days=float(window_days))
    matched = news_df[(news_df[news_time_col] >= start) & (news_df[news_time_col] < target_max)]
    if len(matched) == 0:
        return None, None, 0
    return matched[news_time_col].min(), matched[news_time_col].max(), int(len(matched))


def _prepend_split_history(
    prev_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    *,
    time_col: str,
    history_len: int,
    id_col: str = "",
):
    cur = cur_df.copy()
    if len(cur) == 0 or time_col not in cur.columns:
        return cur, None

    cur = cur.sort_values(time_col).reset_index(drop=True)
    id_key = str(id_col or "").strip()
    history_len = int(max(0, int(history_len)))

    if history_len <= 0 or prev_df is None or len(prev_df) == 0 or time_col not in prev_df.columns:
        if id_key and id_key in cur.columns:
            target_start_by_id = (
                cur.groupby(id_key, sort=False)[time_col]
                .min()
                .dropna()
                .to_dict()
            )
            return cur, target_start_by_id
        return cur, cur[time_col].dropna().min()

    prev = prev_df.copy().sort_values(time_col).reset_index(drop=True)

    if id_key and id_key in cur.columns and id_key in prev.columns:
        combined_parts = []
        target_start_by_id = {}
        for gid, cur_g in cur.groupby(id_key, sort=False):
            cur_g = cur_g.sort_values(time_col).reset_index(drop=True)
            cur_start = cur_g[time_col].dropna().min()
            if pd.isna(cur_start):
                combined_parts.append(cur_g)
                continue
            target_start_by_id[gid] = cur_start
            prev_g = prev.loc[prev[id_key] == gid].sort_values(time_col)
            prev_g = prev_g.loc[prev_g[time_col] < cur_start]
            prefix = prev_g.tail(history_len).copy()
            merged = pd.concat([prefix, cur_g], ignore_index=True)
            merged = merged.drop_duplicates(subset=[id_key, time_col], keep="last")
            merged = merged.sort_values(time_col).reset_index(drop=True)
            combined_parts.append(merged)

        if not combined_parts:
            return cur, target_start_by_id

        out = pd.concat(combined_parts, ignore_index=True)
        out = out.sort_values([id_key, time_col]).reset_index(drop=True)
        return out, target_start_by_id

    cur_start = cur[time_col].dropna().min()
    if pd.isna(cur_start):
        return cur, None
    prefix = prev.loc[prev[time_col] < cur_start].tail(history_len).copy()
    out = pd.concat([prefix, cur], ignore_index=True)
    out = out.drop_duplicates(subset=[time_col], keep="last")
    out = out.sort_values(time_col).reset_index(drop=True)
    return out, cur_start

def _build_delta_optimizer(delta_model, args):
    base_lr = float(args.lr)
    wd = float(args.weight_decay)

    head_scale = float(getattr(args, "delta_head_lr_scale", 1.0))
    other_scale = float(getattr(args, "delta_other_lr_scale", 1.0))

    head_params, other_params = [], []
    for name, p in delta_model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if (
            lname.startswith("delta_head")
            or lname.startswith("delta_gate")
            or lname.startswith("delta_fuse")
            or lname.startswith("delta_mag_head")
            or lname.startswith("delta_text_ln")
            or lname.startswith("delta_log_scale")
            or lname.startswith("delta_rel_head")
            or lname.startswith("text_")
            or lname.startswith("doc_")
            or lname.startswith("rel_head")
        ):
            head_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr * head_scale, "weight_decay": wd})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr * other_scale, "weight_decay": wd})

    if not param_groups:
        raise ValueError("No trainable parameters found for DELTA optimizer.")

    optimizer = AdamW(param_groups)
    lr_info = {
        "base_lr": base_lr,
        "head_lr": base_lr * head_scale if head_params else 0.0,
        "other_lr": base_lr * other_scale if other_params else 0.0,
        "n_head": len(head_params),
        "n_other": len(other_params),
    }
    return optimizer, lr_info


def _z_batch_tensors(batch, args, global_zstats):
    """
    Build global z-scored tensors for pure TS backbone.
    Returns:
      history_z: (B, L)
      targets_z: (B, H)
      metas: list[{"mu_global": float, "sigma_global": float}]
    """
    L = int(args.history_len)
    H = int(args.horizon)
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    mu_global = float(stats["mu_global"])
    sigma_global = float(stats["sigma_global"])

    history_z_list = []
    targets_z_list = []
    metas = []

    B = len(batch["history_value"])
    for i in range(B):
        history = batch["history_value"][i].tolist()
        target = batch["target_value"][i].tolist()

        history_z = np.asarray(_zscore(history, mu_global, sigma_global), dtype=np.float32)
        target_z = np.asarray(_zscore(target, mu_global, sigma_global), dtype=np.float32)

        if history_z.shape[0] > L:
            history_z = history_z[-L:]
        elif history_z.shape[0] < L:
            pad = np.zeros((L,), dtype=np.float32)
            pad[-history_z.shape[0] :] = history_z
            history_z = pad

        if target_z.shape[0] > H:
            target_z = target_z[:H]
        elif target_z.shape[0] < H:
            pad = np.zeros((H,), dtype=np.float32)
            pad[: target_z.shape[0]] = target_z
            target_z = pad

        history_z_list.append(history_z)
        targets_z_list.append(target_z)
        metas.append(
            {
                "mu_global": float(mu_global),
                "sigma_global": float(sigma_global),
                # keep legacy keys for downstream compatibility
                "mu": float(mu_global),
                "sigma": float(sigma_global),
            }
        )

    history_z_t = torch.tensor(np.stack(history_z_list, axis=0), dtype=torch.float32)
    targets_z_t = torch.tensor(np.stack(targets_z_list, axis=0), dtype=torch.float32)
    return history_z_t, targets_z_t, metas


def _point_loss(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    m = str(mode).lower()
    if m == "mse":
        return F.mse_loss(pred, target, reduction="mean")
    if m == "mae":
        return F.l1_loss(pred, target, reduction="mean")
    return F.smooth_l1_loss(pred, target, reduction="mean")


def _make_patches(seq: list[float], patch_len: int, stride: int):
    """
    seq: length L list
    returns: patches (P, patch_len), mask (P,)
    """
    x = np.asarray(seq, dtype=np.float32)
    L = int(x.shape[0])
    patch_len = int(patch_len)
    stride = int(stride)

    if patch_len <= 0:
        raise ValueError("patch_len must be > 0")
    if stride <= 0:
        raise ValueError("patch_stride must be > 0")

    if L < patch_len:
        p = np.zeros((1, patch_len), dtype=np.float32)
        p[0, :L] = x
        m = np.ones((1,), dtype=np.int64)
        return p, m

    idxs = list(range(0, L - patch_len + 1, stride))
    patches = np.stack([x[i : i + patch_len] for i in idxs], axis=0).astype(np.float32)  # (P, patch_len)
    mask = np.ones((patches.shape[0],), dtype=np.int64)
    return patches, mask


def history_text(history_z: list[float], mu_global: float, sigma_global: float) -> str:
    hz = np.asarray(history_z, dtype=np.float32)
    last = hz.tolist() if len(hz) >= 8 else hz.tolist()
    slope = float(hz[-1] - hz[-9]) if len(hz) >= 9 else float(hz[-1] - hz[0]) if len(hz) >= 2 else 0.0
    return (
        f"History z-scored (global mu={mu_global:.4f}, std={sigma_global:.4f}). "
        f"Last {len(last)} z: {', '.join([f'{v:.3f}' for v in last])}. "
        f"Recent slope: {slope:.3f}."
    )



def _pad_2d_int(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(seqs)
    max_t = max(len(s) for s in seqs) if B > 0 else 1
    input_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_t), dtype=torch.long)
    for i, s in enumerate(seqs):
        t = len(s)
        if t == 0:
            continue
        input_ids[i, :t] = torch.tensor(s, dtype=torch.long)
        attn[i, :t] = 1
    return input_ids, attn


def _pad_3d_int(seqs_3d: list[list[list[int]]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(seqs_3d)
    max_docs = max((len(docs) for docs in seqs_3d), default=0)
    max_len = 0
    for docs in seqs_3d:
        for seq in docs:
            max_len = max(max_len, len(seq))
    max_docs = max(1, max_docs)
    max_len = max(1, max_len)
    input_ids = torch.full((B, max_docs, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_docs, max_len), dtype=torch.long)
    doc_mask = torch.zeros((B, max_docs), dtype=torch.long)
    for i, docs in enumerate(seqs_3d):
        for j, seq in enumerate(docs[:max_docs]):
            t = len(seq)
            if t == 0:
                continue
            input_ids[i, j, :t] = torch.tensor(seq, dtype=torch.long)
            attn[i, j, :t] = 1
            doc_mask[i, j] = 1
    return input_ids, attn, doc_mask


def _pad_patches(
    patches_list: list[np.ndarray], mask_list: list[np.ndarray], patch_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(patches_list)
    max_p = max(p.shape[0] for p in patches_list) if B > 0 else 1
    ts_patches = torch.zeros((B, max_p, patch_len), dtype=torch.float32)
    ts_patch_mask = torch.zeros((B, max_p), dtype=torch.long)
    for i, (p, pm) in enumerate(zip(patches_list, mask_list)):
        P_i = p.shape[0]
        ts_patches[i, :P_i, :] = torch.tensor(p, dtype=torch.float32)
        ts_patch_mask[i, :P_i] = torch.tensor(pm, dtype=torch.long)
    return ts_patches, ts_patch_mask



# ----------------------------
# batch build
# ----------------------------
def build_batch_inputs(
    batch,
    tokenizer,
    templates,
    tpl_id,
    args,
    global_zstats,
    news_df,
    policy_name,
    policy_kw,
    volatility_bin,
    epoch: int = -1,
    record_train_prompt: bool = False,
    testing: bool = False,
    force_no_news: bool = False,
    news_dropout: bool = False,
    prompt_path: str = None,
    api_adapter=None,
    build_prompt_inputs: bool = True,
):
    """
    Returns:
      input_ids, attn,
      ts_patches, ts_patch_mask,
      targets_z, metas,
      prompt_texts
    """
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    mu_global = float(stats["mu_global"])
    sigma_global = float(stats["sigma_global"])

    L, H = int(args.history_len), int(args.horizon)
    news_budget = int(args.token_budget * args.token_budget_news_frac)

    patch_len = int(getattr(args, "patch_len", 4))
    patch_stride = int(getattr(args, "patch_stride", patch_len))
    need_prompt_context = bool(build_prompt_inputs or record_train_prompt)

    tpl_text = templates[tpl_id]["text"]
    B = len(batch["history_value"])

    targets_z_list = []
    patches_list = []
    patchmask_list = []
    metas = []

    hist_strs = []
    news_str_list = []
    refined_news_list = []
    refined_news_docs_list = []
    structured_events_list = []
    structured_doc_events_list = []
    structured_feature_list = []
    rel_labels_list = []

    start_dates = []
    end_dates = []
    pred_starts = []
    pred_ends = []

    len_selected_news = []

    for i in range(B):
        history = batch["history_value"][i].tolist()
        target = batch["target_value"][i].tolist()
        t_target = batch["target_time"][i]

        history_z = _zscore(history, mu_global, sigma_global)
        target_z = _zscore(target, mu_global, sigma_global)

        p, pm = _make_patches(history_z, patch_len=patch_len, stride=patch_stride)
        news_text_col = args.news_text_col
        if policy_name == "no_sum":
            news_text_col = "no_sum"
        elif policy_name == "sum_v0":
            news_text_col = "sum_v0"
        # news
        avg_rate = 0.0
        if force_no_news or (news_df is None) or (len(news_df) == 0):
            selected = pd.DataFrame(columns=[args.news_time_col, news_text_col])
        else:
            cand = get_candidates(news_df, args.news_time_col, t_target, args.news_window_days, args.news_topM)
            policy_k = int(args.news_topK)

            selected, avg_rate = select_news(
                cand, policy_name, news_text_col, policy_kw, policy_k, args=args
            )

            if len(selected) > policy_k:
                selected = selected.head(policy_k)

            if int(getattr(args, "utility_rerank_enable", 1)) == 1 and len(selected) > 0:
                selected = rerank_selected_news_by_utility(
                    selected=selected,
                    target_time=t_target,
                    time_col=args.news_time_col,
                    text_col=news_text_col,
                    policy_kw=policy_kw,
                    args=args,
                )

            if (not selected.empty) and ("rate" in selected.columns):
                avg_rate = float(selected["rate"].mean())

        len_selected_news.append(len(selected))

        news_str = ""
        refined_news = ""
        refined_news_docs = []
        structured_events = {}
        structured_doc_events = []
        if (not force_no_news) and len(selected) > 0:
            raw_news_texts = selected[news_text_col].fillna("").astype(str).tolist()
            refine_context = build_refine_context(args, target_time=t_target)
            refined_news_docs = _refine_news_docs_from_doc_cache(
                raw_news_texts=raw_news_texts,
                tokenizer=tokenizer,
                max_tokens=news_budget,
                args=args,
                api_adapter=api_adapter,
            )
            refined_news = _merge_refined_news_docs(
                refined_news_docs,
                tokenizer=tokenizer,
                max_tokens=news_budget,
            )

            if not refined_news:
                # Last fallback keeps legacy behavior if doc-level refine yields empty.
                refine_mode = str(getattr(args, "news_refine_mode", "local"))
                refined_news = refine_news_text(
                    raw_news_texts=raw_news_texts,
                    tokenizer=tokenizer,
                    max_tokens=news_budget,
                    mode=refine_mode,
                    api_adapter=api_adapter,
                    context=refine_context,
                )
            refined_news_docs = [str(x or "").strip() for x in refined_news_docs if str(x or "").strip()]
            doc_cap = int(max(1, getattr(args, "delta_doc_max_docs", 4)))
            if doc_cap > 0:
                refined_news_docs = refined_news_docs[:doc_cap]

            pieces = [refined_news] if need_prompt_context and refined_news else []

            need_structured_for_prompt = need_prompt_context and int(getattr(args, "delta_include_structured_news", 0)) == 1
            need_structured_for_delta = int(getattr(args, "delta_structured_enable", 0)) == 1
            if need_structured_for_prompt or need_structured_for_delta:
                structured_events, structured_doc_events = _extract_structured_events_from_refined_docs_detailed(
                    raw_news_texts=raw_news_texts,
                    refined_news_texts=refined_news_docs,
                    args=args,
                    api_adapter=api_adapter,
                    context=refine_context,
                )
                if (not structured_events) and refined_news:
                    structured_events = extract_structured_events(
                        raw_or_refined_news=refined_news,
                        mode=str(getattr(args, "news_structured_mode", "off")),
                        api_adapter=api_adapter,
                        context=refine_context,
                    )
                    if len(structured_events) > 0:
                        structured_doc_events = [
                            {
                                "doc_index": 0,
                                "source_kind": "merged_fallback",
                                "news_text": str(refined_news),
                                "events": dict(structured_events),
                                "has_events": True,
                                "dedup_reused": False,
                            }
                        ]
            if need_structured_for_prompt:
                structured_text = format_structured_events_for_prompt(structured_events)
                if structured_text:
                    pieces.append(structured_text)

            if need_prompt_context:
                news_str = "\n\n".join([p for p in pieces if str(p).strip()])
                if news_dropout:
                    news_str = _maybe_news_dropout(news_str, args)

        structured_feature_list.append(
            _structured_events_to_feature_vec(
                structured_events,
                dim=int(max(1, getattr(args, "delta_structured_feature_dim", 12))),
            )
        )


        rel = float(avg_rate) if np.isfinite(avg_rate) else 0.0
        rel = max(0.0, min(1.0, rel))
        rel_labels_list.append(rel)

        # if len(selected) > 5:
        #     print(news_str)
        # time meta for prompt
        start_date = batch["history_times"][0][i]
        end_date = batch["history_times"][-1][i]
        prediction_start = batch["target_times"][0][i]
        prediction_end = batch["target_times"][-1][i]

        targets_z_list.append(np.asarray(target_z, dtype=np.float32))
        patches_list.append(p)
        patchmask_list.append(pm)
        metas.append(
            {
                "mu_global": float(mu_global),
                "sigma_global": float(sigma_global),
                # keep legacy keys for downstream compatibility
                "mu": float(mu_global),
                "sigma": float(sigma_global),
            }
        )

        if need_prompt_context:
            hist_strs.append(history_text(history_z, mu_global, sigma_global))
            news_str_list.append(news_str)
        refined_news_list.append(str(refined_news or ""))
        refined_news_docs_list.append(list(refined_news_docs))
        structured_events_list.append(dict(structured_events) if isinstance(structured_events, dict) else {})
        structured_doc_events_list.append(list(structured_doc_events))

        if need_prompt_context:
            start_dates.append(start_date)
            end_dates.append(end_date)
            pred_starts.append(prediction_start)
            pred_ends.append(prediction_end)

    # tokenize prompts
    ids_list = [[] for _ in range(B)]
    prompt_texts = []

    if need_prompt_context:
        for i in range(B):
            prompt = build_prompt(
                tpl_text,
                L,
                H,
                args.unit,
                args.description,
                hist_strs[i],
                news_str_list[i],
                start_date=start_dates[i],
                end_date=end_dates[i],
                freq=args.freq_min,
                value_col=args.value_col,
                pred_end=pred_ends[i],
                pred_start=pred_starts[i],
                region=args.region,
            )
            prompt = prompt + "\n\n[Output]\n" + f"Predict the next {H} steps (internally as z-values).\n"

            dataStatistic.news_num_stats_update(len_selected_news[i], prompt=prompt)

            if build_prompt_inputs:
                enc = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=int(args.max_seq_len),
                    return_attention_mask=False,
                )
                ids_list[i] = enc["input_ids"]
            prompt_texts.append(prompt)

            if record_train_prompt and epoch == 0:
                ckpt_dir = os.path.join("./checkpoints", args.taskName)
                os.makedirs(ckpt_dir, exist_ok=True)

                rec = {
                    "batch_idx": i,
                    "epoch_num": epoch + 1,
                    "template_id": int(tpl_id),
                    "policy": str(policy_name),
                    "force_no_news": bool(force_no_news),
                    "prompt": prompt,
                    "mu_global": float(metas[i]["mu_global"]),
                    "sigma_global": float(metas[i]["sigma_global"]),
                    "mu": float(metas[i]["mu"]),
                    "sigma": float(metas[i]["sigma"]),
                }
                with open(prompt_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    input_ids, attn = _pad_2d_int(ids_list, pad_id=tokenizer.pad_token_id)
    refined_ids_list = []
    refined_doc_ids_list = []
    refined_max_len = int(max(1, getattr(args, "delta_text_max_len", 160)))
    refined_doc_max_len = int(max(1, getattr(args, "delta_doc_max_len", 96)))
    refined_doc_max_docs = int(max(1, getattr(args, "delta_doc_max_docs", 4)))
    for refined_txt in refined_news_list:
        txt = str(refined_txt or "").strip()
        if not txt:
            refined_ids_list.append([])
            continue
        enc = tokenizer(
            txt,
            add_special_tokens=False,
            truncation=True,
            max_length=refined_max_len,
            return_attention_mask=False,
        )
        refined_ids_list.append(enc["input_ids"])
    for refined_docs in refined_news_docs_list:
        doc_ids = []
        docs_trim = list(refined_docs[:refined_doc_max_docs])
        for doc_txt in docs_trim:
            txt = str(doc_txt or "").strip()
            if not txt:
                continue
            enc = tokenizer(
                txt,
                add_special_tokens=False,
                truncation=True,
                max_length=refined_doc_max_len,
                return_attention_mask=False,
            )
            doc_ids.append(enc["input_ids"])
        refined_doc_ids_list.append(doc_ids)
    refined_news_ids, refined_news_attn = _pad_2d_int(refined_ids_list, pad_id=tokenizer.pad_token_id)
    refined_news_doc_ids, refined_news_doc_attn, refined_news_doc_mask = _pad_3d_int(
        refined_doc_ids_list,
        pad_id=tokenizer.pad_token_id,
    )
    ts_patches, ts_patch_mask = _pad_patches(patches_list, patchmask_list, patch_len=patch_len)
    targets_z = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets_z_list], dim=0)
    # print("max = ", len_selected_news)
    rel_labels = torch.tensor(rel_labels_list, dtype=torch.float32)
    news_counts = torch.tensor(len_selected_news, dtype=torch.float32)
    structured_feats = torch.tensor(np.stack(structured_feature_list, axis=0), dtype=torch.float32)
    return (
        input_ids,
        attn,
        ts_patches,
        ts_patch_mask,
        targets_z,
        metas,
        prompt_texts,
        rel_labels,
        news_counts,
        structured_events_list,
        structured_doc_events_list,
        structured_feats,
        refined_news_ids,
        refined_news_attn,
        refined_news_doc_ids,
        refined_news_doc_attn,
        refined_news_doc_mask,
    )


def build_delta_batch_inputs(
    *,
    batch,
    tokenizer,
    templates,
    tpl_id,
    args,
    global_zstats,
    news_df,
    policy_name,
    policy_kw,
    volatility_bin,
    testing: bool = False,
    force_no_news: bool = False,
    news_dropout: bool = False,
    api_adapter=None,
):
    (
        _input_ids,
        _attn,
        ts_patches,
        ts_patch_mask,
        targets_z,
        _metas,
        prompt_texts,
        rel_labels,
        news_counts,
        structured_events_list,
        structured_doc_events_list,
        structured_feats,
        refined_news_ids,
        refined_news_attn,
        refined_news_doc_ids,
        refined_news_doc_attn,
        refined_news_doc_mask,
    ) = build_batch_inputs(
        batch=batch,
        tokenizer=tokenizer,
        templates=templates,
        tpl_id=tpl_id,
        args=args,
        global_zstats=global_zstats,
        news_df=news_df,
        policy_name=policy_name,
        policy_kw=policy_kw,
        volatility_bin=volatility_bin,
        epoch=-1,
        record_train_prompt=False,
        testing=testing,
        force_no_news=force_no_news,
        news_dropout=news_dropout,
        api_adapter=api_adapter,
        build_prompt_inputs=False,
    )
    return {
        "ts_patches": ts_patches,
        "ts_patch_mask": ts_patch_mask,
        "targets_z": targets_z,
        "prompt_texts": prompt_texts,
        "rel_labels": rel_labels,
        "news_counts": news_counts,
        "structured_events": structured_events_list,
        "structured_doc_events": structured_doc_events_list,
        "structured_feats": structured_feats,
        "refined_news_ids": refined_news_ids,
        "refined_news_attn": refined_news_attn,
        "refined_news_doc_ids": refined_news_doc_ids,
        "refined_news_doc_attn": refined_news_doc_attn,
        "refined_news_doc_mask": refined_news_doc_mask,
    }

# ----------------------------
# eval
# ----------------------------
@torch.no_grad()
def evaluate_metrics_single(
    model,
    tokenizer,
    data_loader,
    templates,
    tpl_id,
    args,
    global_zstats,
    news_df,
    policy_name,
    policy_kw,
    device,
    volatility_bin,
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    news_dropout: bool = False,
    force_no_news: bool = False,
    filename: str = None,
    api_adapter=None,
):
    """
    Single-model evaluation: used for BASE stage.
    Returns:
      - loss_avg: z-space MSE
      - mse_avg: raw-scale MSE
      - mae_avg: raw-scale MAE
    """
    model.eval()
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    mu_global = float(stats["mu_global"])
    sigma_global = float(stats["sigma_global"])

    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")

    eval_desc = "[EVAL][SINGLE][TEST]" if testing else "[EVAL][SINGLE][VAL]"
    eval_loader, use_pbar = _eval_iter(data_loader, args, desc=eval_desc)
    for _, batch in enumerate(eval_loader):
        (
            input_ids,
            attn,
            ts_patches,
            ts_patch_mask,
            targets_z,
            metas,
            prompt_texts,
            rel_labels,
            n_selected,
            _structured_events,
            _structured_doc_events,
            _structured_feats,
            _refined_news_ids,
            _refined_news_attn,
            _refined_news_doc_ids,
            _refined_news_doc_attn,
            _refined_news_doc_mask,
        ) = build_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=stats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            epoch=-1,
            record_train_prompt=False,
            testing=testing,
            force_no_news=force_no_news,
            news_dropout=news_dropout,
            api_adapter=api_adapter,
        )

        input_ids = input_ids.to(device)
        attn = attn.to(device)
        ts_patches = ts_patches.to(device)
        ts_patch_mask = ts_patch_mask.to(device)
        targets_z = targets_z.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            ts_patches=ts_patches,
            ts_patch_mask=ts_patch_mask,
            targets=targets_z,
            rel_targets=rel_labels,
            rel_lambda=args.rel_lambda,
        )
        loss = out["loss_fore"]
        pred_z = out["pred"]

        bs = input_ids.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_samples += bs
        if use_pbar:
            eval_loader.set_postfix(zLoss=f"{loss_sum / max(1, n_samples):.6f}")

        pred_z_cpu = pred_z.detach().to(torch.float32).cpu().numpy()
        targets_cpu = batch["target_value"].detach().cpu().numpy()  # raw

        for i in range(bs):
            pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu_global, sigma_global)
            true_vals = targets_cpu[i].reshape(-1).tolist()
            true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

            pred = np.asarray(pred_denorm, dtype=np.float32)
            true = np.asarray(true_vals, dtype=np.float32)

            se_sum += float(((pred - true) ** 2).sum())
            ae_sum += float(np.abs(pred - true).sum())
            n_elems += int(args.horizon)

            if true_pred_csv_path is not None:
                with open(true_pred_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_denorm, true_vals))

            if testing:
                record = {
                    "test_prompt": prompt_texts[i],
                    "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                    "pred": [float(x) for x in pred_denorm],
                    "true": [float(x) for x in true_vals],
                    "mu_global": mu_global,
                    "sigma_global": sigma_global,
                    "mu": mu_global,
                    "sigma": sigma_global,
                }
                with open(ans_json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    return loss_avg, mse_avg, mae_avg


@torch.no_grad()
def evaluate_metrics_backbone(
    base_backbone,
    data_loader,
    args,
    global_zstats,
    device,
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    filename: str | None = None,
):
    """
    Pure TS backbone evaluation in z-space.
    """
    base_backbone.eval()
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    mu_global = float(stats["mu_global"])
    sigma_global = float(stats["sigma_global"])

    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0

    ans_json_path = None
    if testing and filename is not None:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")

    eval_desc = "[EVAL][BACKBONE][TEST]" if testing else "[EVAL][BACKBONE][VAL]"
    eval_loader, use_pbar = _eval_iter(data_loader, args, desc=eval_desc)
    for _, batch in enumerate(eval_loader):
        history_z, targets_z, metas = _z_batch_tensors(batch, args, global_zstats=stats)
        history_z = history_z.to(device)
        targets_z = targets_z.to(device)

        pred_z = base_backbone(history_z).to(torch.float32)
        loss = _point_loss(pred_z, targets_z.to(torch.float32), mode=getattr(args, "base_loss", "smooth_l1"))

        bs = pred_z.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_samples += bs
        if use_pbar:
            eval_loader.set_postfix(zLoss=f"{loss_sum / max(1, n_samples):.6f}")

        pred_z_cpu = pred_z.detach().cpu().numpy()
        targets_cpu = batch["target_value"].detach().cpu().numpy()

        for i in range(bs):
            pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu_global, sigma_global)
            true_vals = targets_cpu[i].reshape(-1).tolist()
            true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

            pred = np.asarray(pred_denorm, dtype=np.float32)
            true = np.asarray(true_vals, dtype=np.float32)
            se_sum += float(((pred - true) ** 2).sum())
            ae_sum += float(np.abs(pred - true).sum())
            n_elems += int(args.horizon)

            if true_pred_csv_path is not None:
                with open(true_pred_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_denorm, true_vals))

            if ans_json_path is not None:
                rec = {
                    "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                    "pred": [float(x) for x in pred_denorm],
                    "true": [float(x) for x in true_vals],
                    "mu_global": mu_global,
                    "sigma_global": sigma_global,
                    "mu": mu_global,
                    "sigma": sigma_global,
                }
                with open(ans_json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    return loss_avg, mse_avg, mae_avg


@torch.no_grad()
def evaluate_metrics_residual(
    base_model,
    delta_model,
    external_signnet: ResidualSignNet | None,
    tokenizer,
    data_loader,
    templates,
    tpl_id,
    args,
    global_zstats,
    news_df,
    policy_name,
    policy_kw,
    device,
    volatility_bin,
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    news_dropout: bool = False,
    filename: str = None,
    api_adapter=None,
    residual_debug_csv_path: str | None = None,
    residual_debug_split: str | None = None,
):
    """
    Residual evaluation:
      additive mode: final_pred = base_pred + delta_pred
      relative mode: legacy fallback only
    Returns:
      final_loss, final_mse, final_mae, base_loss, base_mse, base_mae
    """
    if base_model is None:
        raise ValueError("evaluate_metrics_residual requires a trained base backbone model.")
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    mu_global = float(stats["mu_global"])
    sigma_global = float(stats["sigma_global"])

    base_model.eval()
    delta_model.eval()
    if _use_external_signnet(args) and external_signnet is None:
        raise ValueError("delta_sign_mode=signnet_binary requires an external_signnet model during evaluation.")
    use_external_sign = _use_external_signnet(args) and (external_signnet is not None)
    if use_external_sign:
        external_signnet.eval()

    loss_sum, n_samples = 0.0, 0
    base_loss_sum = 0.0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0
    base_se_sum, base_ae_sum = 0.0, 0.0
    gate_sum = 0.0
    gate_active_sum = 0.0
    gate_count = 0
    sign_correct_sum = 0.0
    sign_valid_sum = 0.0
    mag_pred_sum = 0.0
    mag_true_sum = 0.0
    delta_abs_sum = 0.0

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")
    eval_desc = "[EVAL][RESIDUAL][TEST]" if testing else "[EVAL][RESIDUAL][VAL]"
    eval_loader, use_pbar = _eval_iter(data_loader, args, desc=eval_desc)
    debug_fh, debug_writer = _open_residual_debug_csv(residual_debug_csv_path)
    debug_split = str(
        residual_debug_split or ("test" if testing else "val")
    ).strip() or ("test" if testing else "val")
    sample_idx = 0
    try:
        for _, batch in enumerate(eval_loader):
            history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=stats)
            history_z = history_z.to(device)
            base_pred = base_model(history_z).to(torch.float32)  # (B,H)
            base_pred_cpu = base_pred.detach().cpu()

            # build delta (with news)
            delta_inputs = build_delta_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=stats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                testing=testing,
                force_no_news=False,
                news_dropout=news_dropout,
                api_adapter=api_adapter,
            )
            ts_p = delta_inputs["ts_patches"]
            ts_pm = delta_inputs["ts_patch_mask"]
            targets_z = delta_inputs["targets_z"]
            rel_labels_d = delta_inputs["rel_labels"]
            news_counts_d = delta_inputs["news_counts"]
            structured_events_d = delta_inputs["structured_events"]
            structured_doc_events_d = delta_inputs["structured_doc_events"]
            structured_feats_d = delta_inputs["structured_feats"]
            refined_news_ids_d = delta_inputs["refined_news_ids"]
            refined_news_attn_d = delta_inputs["refined_news_attn"]
            refined_news_doc_ids_d = delta_inputs["refined_news_doc_ids"]
            refined_news_doc_attn_d = delta_inputs["refined_news_doc_attn"]
            refined_news_doc_mask_d = delta_inputs["refined_news_doc_mask"]

            ts_p = ts_p.to(device)
            ts_pm = ts_pm.to(device)
            targets_z = targets_z.to(device)
            structured_feats_d = structured_feats_d.to(device=device, dtype=torch.float32)
            refined_news_ids_d = refined_news_ids_d.to(device)
            refined_news_attn_d = refined_news_attn_d.to(device)
            refined_news_doc_ids_d = refined_news_doc_ids_d.to(device)
            refined_news_doc_attn_d = refined_news_doc_attn_d.to(device)
            refined_news_doc_mask_d = refined_news_doc_mask_d.to(device)
            history_raw = (
                _history_raw_tensor_from_batch(batch, args, device=device)
                if use_external_sign
                else None
            )

            # delta pred: adapter on + with news
            out_delta = delta_model(
                ts_patches=ts_p,
                ts_patch_mask=ts_pm,
                refined_news_input_ids=refined_news_ids_d,
                refined_news_attention_mask=refined_news_attn_d,
                refined_news_doc_input_ids=refined_news_doc_ids_d,
                refined_news_doc_attention_mask=refined_news_doc_attn_d,
                refined_news_doc_mask=refined_news_doc_mask_d,
                targets=None,
                head_mode="delta",
                rel_targets=None,
                rel_lambda=0.0,
                structured_feats=structured_feats_d,
            )
            delta_corr_model = out_delta["pred"].to(torch.float32)
            gate_logits = out_delta.get("gate_logits", out_delta["rel_logits"]).to(torch.float32)
            gate = out_delta.get("gate", torch.sigmoid(gate_logits)).to(torch.float32)
            magnitude = out_delta.get("magnitude", torch.abs(delta_corr_model)).to(torch.float32)
            magnitude_raw = out_delta.get("magnitude_raw", magnitude).to(torch.float32)
            if use_external_sign:
                sign_logits, sign_soft = _run_external_signnet(
                    signnet_model=external_signnet,
                    history_raw=history_raw,
                    base_pred_z=base_pred,
                    structured_feats=structured_feats_d,
                    news_counts=news_counts_d.to(device=device, dtype=torch.float32),
                    tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
                )
                delta_corr, gate = _compose_delta_with_external_sign(
                    gate=gate,
                    magnitude=magnitude,
                    sign_soft=sign_soft,
                    delta_clip=float(getattr(delta_model, "delta_clip", getattr(args, "delta_clip", 0.0))),
                )
            else:
                delta_corr = delta_corr_model
                sign_logits = out_delta.get("sign_logits", torch.zeros_like(delta_corr)).to(torch.float32)
                sign_soft = out_delta.get("sign_soft", torch.zeros_like(delta_corr)).to(torch.float32)

            targets_cpu = batch["target_value"].detach().cpu().numpy()  # raw

            bs = ts_p.size(0)
            history_z_cpu = history_z.detach().cpu().numpy()
            targets_z_f = targets_z.to(torch.float32)
            targets_z_cpu = targets_z_f.detach().cpu().numpy()
            base_pred_f = base_pred.to(torch.float32)
            delta_corr_f = delta_corr.to(torch.float32)
            gate_f = _match_gate_shape(gate.to(torch.float32), delta_corr_f)
            gate_logits_f = _match_gate_shape(gate_logits.to(torch.float32), delta_corr_f)
            sign_logits_f = _match_gate_shape(sign_logits.to(torch.float32), delta_corr_f)
            sign_soft_f = _match_gate_shape(sign_soft.to(torch.float32), delta_corr_f)
            magnitude_f = _match_gate_shape(magnitude.to(torch.float32), delta_corr_f)
            magnitude_raw_f = _match_gate_shape(magnitude_raw.to(torch.float32), delta_corr_f)
            pred_z = _fuse_base_and_delta(
                base_pred_z=base_pred_f,
                delta_pred=delta_corr_f,
                gate_h=gate_f,
                args=args,
                mu_global=mu_global,
                sigma_global=sigma_global,
            )
            loss = F.l1_loss(pred_z, targets_z_f, reduction="mean")
            base_loss = F.l1_loss(base_pred_f, targets_z_f, reduction="mean")
            true_residual_batch = targets_z_f - base_pred_f
            abs_residual_batch = true_residual_batch.abs()
            sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
            valid_sign_mask = (abs_residual_batch > sign_eps).to(torch.float32)
            sign_target_bin = (true_residual_batch > 0).to(torch.float32)
            pred_z_cpu = pred_z.detach().cpu().numpy()
            base_pred_cpu = base_pred_f.detach().cpu().numpy()
            delta_corr_cpu = delta_corr_f.detach().cpu().numpy()
            gate_cpu = gate_f.detach().cpu().numpy()
            gate_logits_cpu = gate_logits_f.detach().cpu().numpy()
            sign_logits_cpu = sign_logits_f.detach().cpu().numpy()
            sign_soft_cpu = sign_soft_f.detach().cpu().numpy()
            magnitude_cpu = magnitude_f.detach().cpu().numpy()
            magnitude_raw_cpu = magnitude_raw_f.detach().cpu().numpy()
            loss_sum += float(loss.detach().cpu()) * bs
            base_loss_sum += float(base_loss.detach().cpu()) * bs
            n_samples += bs
            gate_sum += float(gate_f.sum().detach().cpu())
            gate_active_sum += float((gate_f > 0.5).to(torch.float32).sum().detach().cpu())
            gate_count += int(gate_f.numel())
            sign_correct_sum += float(
                (((sign_logits_f > 0).to(torch.float32) == sign_target_bin).to(torch.float32) * valid_sign_mask)
                .sum()
                .detach()
                .cpu()
            )
            sign_valid_sum += float(valid_sign_mask.sum().detach().cpu())
            mag_pred_sum += float(magnitude_f.sum().detach().cpu())
            mag_true_sum += float(abs_residual_batch.sum().detach().cpu())
            delta_abs_sum += float(delta_corr_f.abs().sum().detach().cpu())
            if use_pbar:
                eval_loader.set_postfix(zMAE=f"{loss_sum / max(1, n_samples):.6f}")

            for i in range(bs):
                pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu_global, sigma_global)
                base_denorm = _inv_zscore(base_pred_cpu[i].tolist(), mu_global, sigma_global)
                true_vals = targets_cpu[i].reshape(-1).tolist()
                true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

                pred = np.asarray(pred_denorm, dtype=np.float32)
                base_only = np.asarray(base_denorm, dtype=np.float32)
                true = np.asarray(true_vals, dtype=np.float32)

                se_sum += float(((pred - true) ** 2).sum())
                ae_sum += float(np.abs(pred - true).sum())
                base_se_sum += float(((base_only - true) ** 2).sum())
                base_ae_sum += float(np.abs(base_only - true).sum())
                n_elems += int(args.horizon)

                if true_pred_csv_path is not None:
                    with open(true_pred_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(zip(pred_denorm, true_vals))

                if debug_writer is not None:
                    history_times_i = _batch_time_seq_for_sample(batch.get("history_times"), i)
                    target_times_i = _batch_time_seq_for_sample(batch.get("target_times"), i)
                    true_residual_z = (
                        np.asarray(targets_z_cpu[i], dtype=np.float32)
                        - np.asarray(base_pred_cpu[i], dtype=np.float32)
                    )
                    pred_residual_z = (
                        np.asarray(pred_z_cpu[i], dtype=np.float32)
                        - np.asarray(base_pred_cpu[i], dtype=np.float32)
                    )
                    residual_mode = _resolve_delta_residual_mode(args)
                    sign_match_pct_additive = ""
                    if residual_mode == "additive":
                        sign_match_pct = _sign_match_pct(true_residual_z, pred_residual_z)
                        sign_match_pct_additive = "" if sign_match_pct is None else float(sign_match_pct)
                    series_id = ""
                    if "series_id" in batch and i < len(batch["series_id"]):
                        series_id = str(batch["series_id"][i])
                    target_time = ""
                    if "target_time" in batch and i < len(batch["target_time"]):
                        target_time = str(batch["target_time"][i])
                    debug_writer.writerow(
                        {
                            "split": debug_split,
                            "sample_idx": sample_idx,
                            "series_id": series_id,
                            "target_time": target_time,
                            "history_start": history_times_i[0] if history_times_i else "",
                            "history_end": history_times_i[-1] if history_times_i else "",
                            "target_start": target_times_i[0] if target_times_i else "",
                            "target_end": target_times_i[-1] if target_times_i else "",
                            "history_times": _json_csv_cell(history_times_i),
                            "target_times": _json_csv_cell(target_times_i),
                            "z_input": _json_csv_cell([float(x) for x in history_z_cpu[i].tolist()]),
                            "target_z": _json_csv_cell([float(x) for x in targets_z_cpu[i].tolist()]),
                            "base_pred_z": _json_csv_cell([float(x) for x in base_pred_cpu[i].tolist()]),
                            "true_residual_z": _json_csv_cell([float(x) for x in true_residual_z.tolist()]),
                            "delta_branch_output": _json_csv_cell([float(x) for x in delta_corr_cpu[i].tolist()]),
                            "pred_residual_z": _json_csv_cell([float(x) for x in pred_residual_z.tolist()]),
                            "pred_residual_sign_match_pct_additive": sign_match_pct_additive,
                            "final_pred_z": _json_csv_cell([float(x) for x in pred_z_cpu[i].tolist()]),
                            "gate": _json_csv_cell([float(x) for x in gate_cpu[i].tolist()]),
                            "gate_logits": _json_csv_cell([float(x) for x in gate_logits_cpu[i].tolist()]),
                            "sign_logits": _json_csv_cell([float(x) for x in sign_logits_cpu[i].tolist()]),
                            "sign_soft": _json_csv_cell([float(x) for x in sign_soft_cpu[i].tolist()]),
                            "magnitude": _json_csv_cell([float(x) for x in magnitude_cpu[i].tolist()]),
                            "magnitude_raw": _json_csv_cell([float(x) for x in magnitude_raw_cpu[i].tolist()]),
                            "gate_mean": float(gate_cpu[i].mean()),
                            "news_count": int(float(news_counts_d[i])) if i < len(news_counts_d) else 0,
                            "policy": str(policy_name),
                            "template_id": int(tpl_id),
                        }
                    )
                    sample_idx += 1

                if testing:
                    structured_merged = (
                        dict(structured_events_d[i])
                        if i < len(structured_events_d) and isinstance(structured_events_d[i], dict)
                        else {}
                    )
                    structured_docs = (
                        list(structured_doc_events_d[i])
                        if i < len(structured_doc_events_d) and isinstance(structured_doc_events_d[i], list)
                        else []
                    )
                    record = {
                        "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                        "base_pred_z": [float(x) for x in base_pred_cpu[i].tolist()],
                        "pred": [float(x) for x in pred_denorm],
                        "base_pred": [float(x) for x in base_denorm],
                        "true": [float(x) for x in true_vals],
                        "gate_mean": float(gate_cpu[i].mean()),
                        "news_count": int(float(news_counts_d[i])) if i < len(news_counts_d) else 0,
                        "structured_events": structured_merged,
                        "structured_events_per_doc": structured_docs,
                        "structured_doc_count": int(len(structured_docs)),
                        "structured_doc_nonempty": int(
                            sum(1 for rec in structured_docs if isinstance(rec, dict) and bool(rec.get("has_events", False)))
                        ),
                        "mu_global": mu_global,
                        "sigma_global": sigma_global,
                        "mu": mu_global,
                        "sigma": sigma_global,
                        "policy": str(policy_name),
                        "template_id": int(tpl_id),
                    }
                    with open(ans_json_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if debug_fh is not None:
            debug_fh.close()

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    base_loss_avg = base_loss_sum / max(1, n_samples)
    base_mse_avg = base_se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    base_mae_avg = base_ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    setattr(
        args,
        "_last_residual_eval_diag",
        {
            "gate_mean": gate_sum / max(1, gate_count),
            "gate_frac_gt_0_5": gate_active_sum / max(1, gate_count),
            "sign_acc": sign_correct_sum / max(1.0, sign_valid_sum),
            "pred_mag_mean": mag_pred_sum / max(1, n_elems),
            "true_abs_residual_mean": mag_true_sum / max(1, n_elems),
            "delta_abs_mean": delta_abs_sum / max(1, n_elems),
        },
    )
    return loss_avg, mse_avg, mae_avg, base_loss_avg, base_mse_avg, base_mae_avg


def _log_last_residual_eval_diag(args, live_logger, tag: str):
    if live_logger is None:
        return
    diag = getattr(args, "_last_residual_eval_diag", None)
    if not isinstance(diag, dict) or not diag:
        return
    live_logger.info(
        f"{tag} gate_mean={float(diag.get('gate_mean', 0.0)):.4f} "
        f"gate>0.5={float(diag.get('gate_frac_gt_0_5', 0.0)):.4f} "
        f"sign_acc={float(diag.get('sign_acc', 0.0)):.4f} "
        f"pred_mag={float(diag.get('pred_mag_mean', 0.0)):.4f} "
        f"true_|res|={float(diag.get('true_abs_residual_mean', 0.0)):.4f} "
        f"|delta|={float(diag.get('delta_abs_mean', 0.0)):.4f}"
    )


@torch.no_grad()
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

    raw_texts = in_scope[text_col].fillna("").astype(str).tolist()
    uniq = []
    seen = set()
    for txt in raw_texts:
        s = str(txt).strip()
        if not s:
            continue
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append(s)
    if max_items > 0:
        uniq = uniq[:max_items]

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
            f"news_rows={len(news_df)}, in_scope_rows={len(in_scope)}, unique_docs={len(uniq)}, "
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
        uniq,
        desc="[DELTA][NEWS_API_PREPROCESS]",
        leave=show_progress,
        dynamic_ncols=True,
        mininterval=0.3,
        disable=(not show_progress),
    )
    news_budget = int(args.token_budget * args.token_budget_news_frac)
    total_docs = len(uniq)
    structured_context = _doc_refine_context(args)
    structured_source_kind = _structured_doc_source_kind(args)
    for idx, raw_text in enumerate(pbar, start=1):
        refined_doc = ""
        if run_refine:
            refined_doc = _refine_one_news_doc(
                raw_news_text=raw_text,
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
            f"unique_docs={len(uniq)} "
            f"refine_entries_before={before_n} refine_entries_after={after_n} "
            f"refine_hits_delta={hit1 - hit0} refine_misses_delta={miss1 - miss0} "
            f"structured_entries_before={structured_before_n} structured_entries_after={structured_after_n} "
            f"structured_hits_delta={structured_hit1 - structured_hit0} "
            f"structured_misses_delta={structured_miss1 - structured_miss0}"
        )


# ----------------------------
# setup (new)
# ----------------------------
def setup_env_and_data(args):
    stage = str(getattr(args, "stage", "all")).lower()
    base_backbone_name = str(getattr(args, "base_backbone", "dlinear"))

    def _safe_name(s: str) -> str:
        s = str(s).strip()
        s = s.replace("/", "-").replace("\\", "-")
        s = re.sub(r"\s+", "_", s)
        return s if s else "na"

    # Base part stays concise: taskName + base backbone.
    filename = f"{_safe_name(args.taskName)}_{_safe_name(base_backbone_name)}"
    log_filename = filename + ".log"

    live_logger, live_path, log_jsonl = setup_live_logger(
        save_dir=args.save_dir + "/" + args.taskName, filename=log_filename
    )
    print(f"[live log] {live_path}  (实时查看: tail -f '{live_path}')")
    _log_run_args(args, live_logger)
    _log_enabled_mechanisms(args, live_logger, stage=stage)
    news_api_adapter = build_news_api_adapter(args, live_logger=live_logger)

    ckpt_dir = os.path.join("./checkpoints", args.taskName)
    os.makedirs(ckpt_dir, exist_ok=True)

    # fixed output paths (clearing controlled by main())
    prompt_path = os.path.join(ckpt_dir, f"prompts_{filename}.json")
    ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")
    true_pred_csv_path = os.path.join(ckpt_dir, f"true_pred_{filename}.csv")
    val_residual_debug_csv_path = os.path.join(ckpt_dir, f"val_delta_residual_debug_{filename}.csv")
    test_residual_debug_csv_path = os.path.join(ckpt_dir, f"test_delta_residual_debug_{filename}.csv")

    set_seed(args.seed)
    device = device_from_id(args.gpu)

    def _read(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    train_df = _read(args.train_file)
    val_df = _read(args.val_file)
    test_df = _read(args.test_file)

    global_zstats = _compute_global_zstats_from_train_df(train_df, args)
    live_logger.info(
        "[ZSCORE] global stats from train_df: "
        f"mu_global={global_zstats['mu_global']:.6f}, sigma_global={global_zstats['sigma_global']:.6f}"
    )

    train_df[args.time_col] = pd.to_datetime(train_df[args.time_col], dayfirst=args.dayFirst)
    val_df[args.time_col] = pd.to_datetime(val_df[args.time_col], dayfirst=args.dayFirst)
    test_df[args.time_col] = pd.to_datetime(test_df[args.time_col], dayfirst=args.dayFirst)

    val_loader_df, val_min_target_time = _prepend_split_history(
        train_df,
        val_df,
        time_col=args.time_col,
        history_len=args.history_len,
        id_col=args.id_col,
    )
    test_loader_df, test_min_target_time = _prepend_split_history(
        val_df,
        test_df,
        time_col=args.time_col,
        history_len=args.history_len,
        id_col=args.id_col,
    )

    train_loader = make_loader(
        train_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=True,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )
    val_loader = make_loader(
        val_loader_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
        min_target_time=val_min_target_time if not isinstance(val_min_target_time, dict) else None,
        min_target_time_by_id=val_min_target_time if isinstance(val_min_target_time, dict) else None,
    )
    test_loader = make_loader(
        test_loader_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
        min_target_time=test_min_target_time if not isinstance(test_min_target_time, dict) else None,
        min_target_time_by_id=test_min_target_time if isinstance(test_min_target_time, dict) else None,
    )

    # news
    
    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])

    news_df[args.news_time_col] = pd.to_datetime(news_df[args.news_time_col], dayfirst=args.dayFirst)
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)
        # 去除空的总结后的新闻
        col = args.news_text_col
        news_df = news_df.loc[
            news_df[col].fillna("").astype(str).str.strip().ne("")
        ].reset_index(drop=True)
    setattr(
        args,
        "_news_doc_meta_by_text",
        _build_news_doc_meta_index(
            news_df,
            text_col=str(getattr(args, "news_text_col", "content")),
            time_col=str(getattr(args, "news_time_col", "date")),
        ),
    )
    news_total = int(len(news_df))
    news_time_min = news_df[args.news_time_col].min() if (len(news_df) > 0 and args.news_time_col in news_df.columns) else None
    news_time_max = news_df[args.news_time_col].max() if (len(news_df) > 0 and args.news_time_col in news_df.columns) else None
    live_logger.info(
        f"[NEWS_DATA] total_rows={news_total} time_range={_format_ts_range(news_time_min, news_time_max)}"
    )

    train_series_min, train_series_max = _loader_series_time_range(train_loader, args.time_col)
    val_series_min, val_series_max = _loader_series_time_range(val_loader, args.time_col)
    test_series_min, test_series_max = _loader_series_time_range(test_loader, args.time_col)
    live_logger.info(f"[DATA_RANGE][TRAIN] series={_format_ts_range(train_series_min, train_series_max)}")
    live_logger.info(f"[DATA_RANGE][VAL] series={_format_ts_range(val_series_min, val_series_max)}")
    live_logger.info(f"[DATA_RANGE][TEST] series={_format_ts_range(test_series_min, test_series_max)}")

    train_news_min, train_news_max, train_news_rows = _matched_news_time_range(
        train_loader,
        news_df,
        time_col=args.time_col,
        news_time_col=args.news_time_col,
        window_days=args.news_window_days,
    )
    val_news_min, val_news_max, val_news_rows = _matched_news_time_range(
        val_loader,
        news_df,
        time_col=args.time_col,
        news_time_col=args.news_time_col,
        window_days=args.news_window_days,
    )
    test_news_min, test_news_max, test_news_rows = _matched_news_time_range(
        test_loader,
        news_df,
        time_col=args.time_col,
        news_time_col=args.news_time_col,
        window_days=args.news_window_days,
    )
    live_logger.info(
        f"[NEWS_RANGE][TRAIN] matched={_format_ts_range(train_news_min, train_news_max)} rows={train_news_rows}"
    )
    live_logger.info(
        f"[NEWS_RANGE][VAL] matched={_format_ts_range(val_news_min, val_news_max)} rows={val_news_rows}"
    )
    live_logger.info(
        f"[NEWS_RANGE][TEST] matched={_format_ts_range(test_news_min, test_news_max)} rows={test_news_rows}"
    )

    templates = load_templates(args.template_pool)

    patch_len = int(getattr(args, "patch_len", 4))

    volatility_bin = compute_volatility_bin(
        train_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )
    volatility_bin_val = compute_volatility_bin(
        val_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )
    volatility_bin_test = compute_volatility_bin(
        test_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )

    return {
        "stage": stage,
        "live_logger": live_logger,
        "live_path": live_path,
        "log_jsonl": log_jsonl,
        "ckpt_dir": ckpt_dir,
        "prompt_path": prompt_path,
        "ans_json_path": ans_json_path,
        "true_pred_csv_path": true_pred_csv_path,
        "val_residual_debug_csv_path": val_residual_debug_csv_path,
        "test_residual_debug_csv_path": test_residual_debug_csv_path,
        "device": device,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "news_df": news_df,
        "templates": templates,
        "patch_len": patch_len,
        "volatility_bin": volatility_bin,
        "volatility_bin_val": volatility_bin_val,
        "volatility_bin_test": volatility_bin_test,
        "global_zstats": global_zstats,
        "news_api_adapter": news_api_adapter,
        "prompt_path": prompt_path,
        "test_filename": filename,
    }


# ----------------------------
# stage 1: BASE (new)
# ----------------------------
def train_base_stage(args, bundle):
    live_logger = bundle["live_logger"]
    device = bundle["device"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    templates = bundle["templates"]
    global_zstats = _coerce_global_zstats(bundle.get("global_zstats", None), args, required=True)

    # base epochs
    base_epochs_override = int(getattr(args, "base_epochs", -1))
    if base_epochs_override >= 0:
        base_epochs = base_epochs_override
    else:
        base_frac = float(getattr(args, "residual_base_frac", 0.3))
        base_epochs, _ = split_two_stage_epochs(
            total_epochs=int(args.epochs),
            base_frac=base_frac,
            min_base=int(getattr(args, "residual_min_base_epochs", 1)),
            min_delta=int(getattr(args, "residual_min_delta_epochs", 1)),
        )
        if getattr(args, "residual_base_epochs", None) is not None:
            base_epochs = int(getattr(args, "residual_base_epochs"))
            base_epochs = max(0, min(base_epochs, int(args.epochs) - 1))

    live_logger.info("-----------------------------------------------------")
    live_logger.info(
        f"[BASE] Training pure TS backbone ({getattr(args, 'base_backbone', 'dlinear')}), "
        f"epochs={base_epochs} (no-news)"
    )
    live_logger.info("-----------------------------------------------------")

    base_train_model = build_base_backbone(
        backbone_name=getattr(args, "base_backbone", "dlinear"),
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        hidden_dim=int(getattr(args, "base_hidden_dim", 256)),
        moving_avg=int(getattr(args, "base_moving_avg", 25)),
        dropout=float(getattr(args, "base_dropout", 0.0)),
    )
    base_train_model.to(device)

    base_lr = float(getattr(args, "base_lr", -1.0))
    if base_lr <= 0:
        base_lr = float(args.lr)
    base_wd = float(getattr(args, "base_weight_decay", -1.0))
    if base_wd < 0:
        base_wd = float(args.weight_decay)

    optim_base = AdamW(
        base_train_model.parameters(),
        lr=base_lr,
        weight_decay=base_wd,
    )

    num_batches = len(train_loader)
    total_opt_steps_base = math.ceil((num_batches * max(1, base_epochs)) / max(1, args.grad_accum))
    warmup_steps_base = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps_base)
    warmup_steps_base = min(warmup_steps_base, max(0, total_opt_steps_base - 1))


    if args.scheduler == 1:
        scheduler_base = get_cosine_schedule_with_warmup(
            optim_base,
            num_warmup_steps=warmup_steps_base,
            num_training_steps=total_opt_steps_base,
        )
    else:
        scheduler_base = None

    best_base_metric = float("inf")
    stale_rounds = 0
    loss_window = deque(maxlen=50)
    global_step = 0

    for epoch in range(base_epochs):
        pbar = tqdm(train_loader, desc=f"[BASE] Epoch {epoch+1}/{base_epochs}")

        for _, batch in enumerate(pbar):
            history_z, targets_z, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
            history_z = history_z.to(device)
            targets_z = targets_z.to(device)

            base_train_model.train()
            pred_z = base_train_model(history_z)
            loss = _point_loss(pred_z, targets_z, mode=getattr(args, "base_loss", "smooth_l1")) / args.grad_accum
            loss.backward()

            loss_window.append(float(loss.detach().cpu()))
            if global_step % 10 == 0:
                avg_train_loss = sum(loss_window) / len(loss_window)
                pbar.set_postfix(train_loss=f"{avg_train_loss:.6f}")

            if (global_step + 1) % args.grad_accum == 0:
                optim_base.step()
                if args.scheduler == 1:
                    scheduler_base.step()
                optim_base.zero_grad(set_to_none=True)

            global_step += 1

        val_loss, val_mse, val_mae = evaluate_metrics_backbone(
            base_backbone=base_train_model,
            data_loader=val_loader,
            args=args,
            global_zstats=global_zstats,
            device=device,
            testing=False,
            true_pred_csv_path=None,
            filename=None,
        )

        if args.select_metric == "loss":
            metric_now = val_loss
        elif args.select_metric == "mse":
            metric_now = val_mse
        else:
            metric_now = val_mae

        live_logger.info(
            f"[BASE][EVAL] epoch={epoch+1} "
            f"val_loss(zMSE)={val_loss:.6f} val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
        )

        best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")
        if metric_now < best_base_metric - 1e-6:
            best_base_metric = metric_now
            stale_rounds = 0
            # if os.path.isfile(best_base_path):
            #     os.remove(best_base_path)
            shutil.rmtree(best_base_path, ignore_errors=True)

            save_base_backbone_checkpoint(
                best_base_path,
                base_train_model,
                backbone_name=getattr(args, "base_backbone", "dlinear"),
                history_len=int(args.history_len),
                horizon=int(args.horizon),
                hidden_dim=int(getattr(args, "base_hidden_dim", 256)),
                moving_avg=int(getattr(args, "base_moving_avg", 25)),
                dropout=float(getattr(args, "base_dropout", 0.0)),
                optimizer=optim_base,
                scheduler=scheduler_base,
                epoch=epoch,
                global_step=global_step,
                global_zstats=global_zstats,
            )
            live_logger.info(f"[BASE] New best saved to {best_base_path} ({args.select_metric}={best_base_metric:.6f})")
        else:
            stale_rounds += 1
            live_logger.info(f"[BASE] stale_rounds={stale_rounds}/{args.early_stop_patience} best={best_base_metric:.6f}")

        if stale_rounds >= args.early_stop_patience:
            live_logger.info(f"[BASE] Early stopping triggered at epoch {epoch+1}.")
            break

    # cleanup
    del base_train_model
    gc.collect()
    torch.cuda.empty_cache()

    best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")
    if not os.path.exists(best_base_path):
        raise FileNotFoundError(f"Base checkpoint not found: {best_base_path}")

    return {
        "best_base_path":best_base_path,
        "device": device,
        "live_logger": live_logger,
        "templates":templates,
        "best_base_metric": best_base_metric,
        "global_zstats": global_zstats,
    }



# ----------------------------
# stage 2: DELTA (new)
# ----------------------------
def train_delta_stage(args, bundle, best_base_path: str, best_base_metric):
    live_logger = bundle["live_logger"]
    device = bundle["device"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    news_df = bundle["news_df"]
    policy_kw = []
    templates = bundle["templates"]
    patch_len = bundle["patch_len"]
    volatility_bin = bundle["volatility_bin"]
    volatility_bin_val = bundle["volatility_bin_val"]
    volatility_bin_test = bundle["volatility_bin_test"]
    true_pred_csv_path = bundle["true_pred_csv_path"]
    val_residual_debug_csv_path = bundle.get("val_residual_debug_csv_path")
    test_residual_debug_csv_path = bundle.get("test_residual_debug_csv_path")
    global_zstats_bundle = _coerce_global_zstats(bundle.get("global_zstats", None), args, required=True)
    news_api_adapter = bundle.get("news_api_adapter", None)
    _init_refine_cache(args, live_logger=live_logger)
    _init_structured_cache(args, live_logger=live_logger)
    

    train_cfg = {}

    delta_epochs_override = int(getattr(args, "delta_epochs", -1))
    if delta_epochs_override >= 0:
        delta_epochs = delta_epochs_override
    else:
        base_frac = float(getattr(args, "residual_base_frac", 0.3))
        base_epochs, delta_epochs = split_two_stage_epochs(
            total_epochs=int(args.epochs),
            base_frac=base_frac,
            min_base=int(getattr(args, "residual_min_base_epochs", 1)),
            min_delta=int(getattr(args, "residual_min_delta_epochs", 1)),
        )
        if (args.delta_epochs > 0):
            delta_epochs = args.delta_epochs
        else:
            delta_epochs = args.epochs - base_epochs

    live_logger.info("-----------------------------------------------------")
    live_logger.info(f"[DELTA] Training DELTA: epochs={delta_epochs}, base_ckpt={best_base_path}")
    live_logger.info("-----------------------------------------------------")

    base_backbone, base_meta = load_base_backbone_checkpoint(
        best_base_path,
        device=device,
        is_trainable=False,
    )
    live_logger.info(
        f"[DELTA] Loaded base backbone: {base_meta.get('backbone_name')} "
        f"(L(seq_len)={base_meta.get('history_len')}, H(horizon/pred_len)={base_meta.get('horizon')})"
    )
    # mu_global = mean value of train_df; sigma_global = std value of train_df (with optional z-score clipping)
    # z = (x - mu_global) / sigma_global
    global_zstats = _coerce_global_zstats(base_meta, args, required=False)
    if global_zstats is None:
        global_zstats = global_zstats_bundle
        live_logger.info(
            "[DELTA] base checkpoint has no global z-score stats; "
            f"fallback to train_df stats: mu_global={global_zstats['mu_global']:.6f}, "
            f"sigma_global={global_zstats['sigma_global']:.6f}"
        )
    else:
        live_logger.info(
            "[DELTA] using global z-score stats from base checkpoint: "
            f"mu_global={global_zstats['mu_global']:.6f}, "
            f"sigma_global={global_zstats['sigma_global']:.6f}"
        )

    tokenizer, delta_model = build_delta_model(
        base_model=args.base_model,
        tokenizer_id=args.tokenizer,
        horizon=args.horizon,
        patch_dim=patch_len,
        patch_dropout=args.patch_dropout,
        head_dropout=args.head_dropout,
        head_mlp=args.head_mlp,
        delta_gate_init_bias=float(getattr(args, "delta_gate_init_bias", 0.0)),
        delta_head_init_std=float(getattr(args, "delta_head_init_std", 0.01)),
        delta_internal_gate=bool(int(getattr(args, "delta_internal_gate_in_model", getattr(args, "delta_internal_gate", 1)))) and (not _all_gates_disabled(args)),
        disable_all_gates=_all_gates_disabled(args),
        delta_clip=float(getattr(args, "delta_clip", 3.0)),
        delta_news_tail_tokens=int(getattr(args, "delta_news_tail_tokens", 160)),
        delta_rel_floor=float(getattr(args, "delta_rel_floor", 0.05)),
        delta_structured_feature_dim=(
            int(getattr(args, "delta_structured_feature_dim", 12))
            if int(getattr(args, "delta_structured_enable", 0)) == 1
            else 0
        ),
        delta_model_variant=str(getattr(args, "delta_model_variant", "tiny_news_ts")),
        tiny_news_model_preset=str(getattr(args, "tiny_news_model_preset", "custom")),
        tiny_news_model=str(getattr(args, "tiny_news_model", "")),
        tiny_news_tokenizer=str(getattr(args, "tiny_news_tokenizer", "")),
        tiny_news_hidden_size=int(getattr(args, "tiny_news_hidden_size", 256)),
        tiny_news_text_trainable=bool(int(getattr(args, "tiny_news_text_trainable", 0))),
        tiny_news_loader=str(getattr(args, "tiny_news_loader", "auto")),
        delta_text_direct_enable=bool(int(getattr(args, "delta_text_direct_enable", 0))),
        delta_text_fuse_lambda=float(getattr(args, "delta_text_fuse_lambda", 0.5)),
        delta_text_gate_init_bias=float(getattr(args, "delta_text_gate_init_bias", -2.0)),
        delta_text_clip=float(getattr(args, "delta_text_clip", 1.5)),
        delta_text_max_len=int(getattr(args, "delta_text_max_len", 160)),
        delta_doc_direct_enable=bool(int(getattr(args, "delta_doc_direct_enable", 0))),
        delta_doc_fuse_lambda=float(getattr(args, "delta_doc_fuse_lambda", 0.75)),
        delta_doc_gate_init_bias=float(getattr(args, "delta_doc_gate_init_bias", -2.0)),
        delta_doc_clip=float(getattr(args, "delta_doc_clip", 1.0)),
        delta_doc_max_len=int(getattr(args, "delta_doc_max_len", 96)),
        delta_doc_max_docs=int(getattr(args, "delta_doc_max_docs", 4)),
        delta_alpha_scale=float(getattr(args, "delta_alpha_scale", 0.75)),
        delta_patch_prototypes=int(getattr(args, "delta_patch_prototypes", 0)),
        delta_patch_proto_temp=float(getattr(args, "delta_patch_proto_temp", 1.0)),
        delta_sign_tau=float(getattr(args, "delta_sign_tau", 1.0)),
        delta_sign_mode=_resolve_delta_sign_mode(args),
        delta_mag_max=float(getattr(args, "delta_mag_max", 0.0)),
        doc_candidate_mode=str(getattr(args, "doc_candidate_mode", "beta_only")),
    )
    delta_model.to(device)
    live_logger.info(
        f"[DELTA] model_variant={str(getattr(delta_model, 'model_variant', 'tiny_news_ts')).lower()}"
    )

    # when to run validation set
    delta_val_mode = _normalize_delta_val_mode(getattr(args, "delta_val_mode", "each_epoch"))
    live_logger.info(f"[DELTA] validation mode: {delta_val_mode}")

    # freeze base head
    for p in delta_model.base_head.parameters():
        p.requires_grad = False

    freeze_feature_modules = int(getattr(args, "delta_freeze_feature_modules", 0)) == 1
    if freeze_feature_modules:
        # legacy option: keep feature extractor fixed during delta adaptation
        freeze_modules = [
            "patch_proj",
            "patch_gate",
            "patch_pos",
            "pool_attn",
            "pool_ln",
            "text_ctx_ln",
            "text2q",
        ]
        for name in freeze_modules:
            if hasattr(delta_model, name):
                m = getattr(delta_model, name)
                if hasattr(m, "parameters"):
                    for p in m.parameters():
                        p.requires_grad = False
        if hasattr(delta_model, "pool_q"):
            delta_model.pool_q.requires_grad = False
        if hasattr(delta_model, "layer_w"):
            delta_model.layer_w.requires_grad = False
        live_logger.info("[DELTA] feature modules frozen (legacy mode).")
    else:
        live_logger.info("[DELTA] feature modules remain trainable (recommended).")

    # ensure delta-specific heads remain trainable
    train_modules = [
        "delta_head",
        "delta_gate",
        "delta_fuse",
        "delta_mag_head",
        "delta_text_ln",
        "text_proj",
        "text_delta_head",
        "text_gate",
        "rel_head",
    ]
    for name in train_modules:
        if hasattr(delta_model, name):
            m = getattr(delta_model, name)
            if hasattr(m, "parameters"):
                for p in m.parameters():
                    p.requires_grad = True
    if hasattr(delta_model, "delta_log_scale"):
        delta_model.delta_log_scale.requires_grad = True
    if hasattr(delta_model, "text_log_scale") and delta_model.text_log_scale is not None:
        delta_model.text_log_scale.requires_grad = True

    optim_delta, lr_info = _build_delta_optimizer(delta_model, args)
    live_logger.info(
        "[DELTA] optimizer groups: "
        f"base_lr={lr_info['base_lr']:.3e}, "
        f"head_lr={lr_info['head_lr']:.3e} (n={lr_info['n_head']}), "
        f"other_lr={lr_info['other_lr']:.3e} (n={lr_info['n_other']})"
    )

    total_opt_steps_delta = math.ceil((len(train_loader) * max(1, delta_epochs)) / max(1, args.grad_accum))
    warmup_steps_delta = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps_delta)
    warmup_steps_delta = min(warmup_steps_delta, max(0, total_opt_steps_delta - 1))

    if args.scheduler == 1:
        scheduler_delta = get_cosine_schedule_with_warmup(
            optim_delta,
            num_warmup_steps=warmup_steps_delta,
            num_training_steps=total_opt_steps_delta,
        )
    else:
        scheduler_delta = None

    allowed_tpl_ids = sorted([t["id"] for t in templates.values()])
    tpl_id = allowed_tpl_ids[0]
    policy_name = args.default_policy
    best_tpl_id = tpl_id
    best_policy_name = policy_name

    _prewarm_refine_cache(
        args=args,
        news_df=news_df,
        train_df=bundle.get("train_df"),
        val_df=bundle.get("val_df"),
        test_df=bundle.get("test_df"),
        tokenizer=tokenizer,
        live_logger=live_logger,
        api_adapter=news_api_adapter,
    )

    external_signnet_model = _train_external_signnet(
        args=args,
        base_backbone=base_backbone,
        tokenizer=tokenizer,
        templates=templates,
        tpl_id=tpl_id,
        policy_name=policy_name,
        policy_kw=policy_kw,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        news_df=news_df,
        volatility_bin=volatility_bin,
        volatility_bin_val=volatility_bin_val,
        volatility_bin_test=volatility_bin_test,
        global_zstats=global_zstats,
        device=device,
        live_logger=live_logger,
        api_adapter=news_api_adapter,
    )
    external_sign_enabled = external_signnet_model is not None
    if external_sign_enabled:
        live_logger.info(
            "[DELTA] external signnet is active: DELTA internal sign branch is overridden during train/val/test."
        )

    if math.isfinite(float(best_base_metric)):
        base_ref_metric = float(best_base_metric)
        live_logger.info(
            f"[DELTA] base reference on val: threshold={base_ref_metric:.6f} "
            f"(metric={str(args.select_metric).lower()})"
        )
    else:
        base_ref_loss, base_ref_mse, base_ref_mae = evaluate_metrics_backbone(
            base_backbone=base_backbone,
            data_loader=val_loader,
            args=args,
            global_zstats=global_zstats,
            device=device,
            testing=False,
            true_pred_csv_path=None,
            filename=None,
        )
        if args.select_metric == "loss":
            base_ref_metric = float(base_ref_loss)
        elif args.select_metric == "mse":
            base_ref_metric = float(base_ref_mse)
        else:
            base_ref_metric = float(base_ref_mae)
        live_logger.info(
            f"[DELTA] base reference on val: loss={base_ref_loss:.6f} mse={base_ref_mse:.6f} "
            f"mae={base_ref_mae:.6f}; threshold={base_ref_metric:.6f}"
        )
    best_metric = float("inf")
    stale_rounds = 0
    has_saved_delta = False
    loss_window = deque(maxlen=50)

    val_loss_per_epoch, mse_loss_per_epoch, mae_loss_per_epoch = [], [], []
    global_step = 0
    best_delta_alpha = 1.0
    early_stop_patience = max(0, int(getattr(args, "early_stop_patience", 0))) if delta_val_mode == "each_epoch" else 0
    best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")
    delta_warmup_epochs = int(max(0, getattr(args, "delta_warmup_epochs", 0)))
    delta_curriculum_epochs = int(max(1, getattr(args, "delta_curriculum_epochs", 1)))
    delta_null_warmup_steps = int(max(0, getattr(args, "delta_null_warmup_steps", 0)))
    delta_null_ramp_steps = int(max(0, getattr(args, "delta_null_ramp_steps", 0)))
    live_logger.info(
        "[DELTA] regularization curriculum: "
        f"warmup_epochs={delta_warmup_epochs}, curriculum_epochs={delta_curriculum_epochs}, "
        f"null_warmup_steps={delta_null_warmup_steps}, null_ramp_steps={delta_null_ramp_steps}"
    )
    live_logger.info(
        "[DELTA] residual branch: "
        f"mode={_resolve_delta_residual_mode(args)} "
        f"sign_mode={_resolve_delta_sign_mode(args)} "
        f"denom_floor={float(getattr(args, 'delta_relative_denom_floor', 1.0) or 1.0):.6f} "
        f"ratio_clip={float(getattr(args, 'delta_relative_ratio_clip', 0.0) or 0.0):.6f}"
    )

    def _save_residual_best(epoch_idx: int, metric_now: float | None, tpl_id_now: int, policy_name_now: str):
        metric_value = None if metric_now is None else float(metric_now)
        shutil.rmtree(best_delta_path, ignore_errors=True)

        save_checkpoint(
            best_delta_path,
            tokenizer,
            delta_model,
            base_model_id=args.base_model,
            tokenizer_id=args.tokenizer or args.base_model,
            train_cfg=train_cfg,
            optimizer=optim_delta,
            scheduler=scheduler_delta,
            epoch=int(epoch_idx),
            global_step=global_step,
            extra_meta={
                "mu_global": float(global_zstats["mu_global"]),
                "sigma_global": float(global_zstats["sigma_global"]),
                "delta_residual_mode": _resolve_delta_residual_mode(args),
                "delta_relative_denom_floor": float(getattr(args, "delta_relative_denom_floor", 1.0) or 1.0),
                "delta_relative_ratio_clip": float(getattr(args, "delta_relative_ratio_clip", 0.0) or 0.0),
                "delta_text_direct_enable": int(getattr(args, "delta_text_direct_enable", 0)),
                "delta_text_fuse_lambda": float(getattr(args, "delta_text_fuse_lambda", 0.5)),
                "delta_text_gate_init_bias": float(getattr(args, "delta_text_gate_init_bias", -2.0)),
                "delta_text_clip": float(getattr(args, "delta_text_clip", 1.5)),
                "delta_text_max_len": int(getattr(args, "delta_text_max_len", 160)),
                "delta_doc_direct_enable": int(getattr(args, "delta_doc_direct_enable", 0)),
                "delta_doc_fuse_lambda": float(getattr(args, "delta_doc_fuse_lambda", 0.75)),
                "delta_doc_gate_init_bias": float(getattr(args, "delta_doc_gate_init_bias", -2.0)),
                "delta_doc_clip": float(getattr(args, "delta_doc_clip", 1.0)),
                "delta_doc_max_len": int(getattr(args, "delta_doc_max_len", 96)),
                "delta_doc_max_docs": int(getattr(args, "delta_doc_max_docs", 4)),
                "delta_sign_mode": _resolve_delta_sign_mode(args),
                "delta_sign_external_enable": int(external_sign_enabled),
                "delta_sign_external_tau": float(getattr(args, "delta_sign_external_tau", 1.0)),
            },
        )
        if external_sign_enabled:
            ext_sign_path = os.path.join(best_delta_path, "external_signnet.pt")
            torch.save(
                {
                    "state_dict": external_signnet_model.state_dict(),
                    "history_len": int(max(1, getattr(args, "history_len", 1))),
                    "horizon": int(max(1, getattr(args, "horizon", 1))),
                    "structured_dim": int(max(0, getattr(args, "delta_structured_feature_dim", 12))),
                    "hidden_size": int(max(32, getattr(args, "delta_sign_external_hidden", 256))),
                    "dropout": float(max(0.0, getattr(args, "delta_sign_external_dropout", 0.1))),
                    "tau": float(getattr(args, "delta_sign_external_tau", 1.0)),
                    "decision_bias": float(
                        external_signnet_model.decision_bias.detach().cpu().item()
                    ) if hasattr(external_signnet_model, "decision_bias") else 0.0,
                },
                ext_sign_path,
            )

        with open(os.path.join(f"./checkpoints/{args.taskName}", "residual_pair.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_base": os.path.basename(best_base_path),
                    # "best_delta": f"best_delta_{args.taskName}",
                    # "best_tpl_id": int(tpl_id_now),
                    # "best_policy_name": str(policy_name_now),
                    "best_delta_alpha": float(best_delta_alpha),
                    "select_metric": str(args.select_metric),
                    "best_metric": metric_value,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _dump_val_residual_debug_csv():
        if not val_residual_debug_csv_path:
            return
        evaluate_metrics_residual(
            base_model=base_backbone,
            delta_model=delta_model,
            external_signnet=external_signnet_model,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            testing=False,
            true_pred_csv_path=None,
            news_dropout=False,
            api_adapter=news_api_adapter,
            residual_debug_csv_path=val_residual_debug_csv_path,
            residual_debug_split="val",
        )
        live_logger.info(f"[DELTA][VAL_DEBUG_CSV] updated path={val_residual_debug_csv_path}")

    for epoch in range(delta_epochs):
        pbar = tqdm(train_loader, desc=f"[DELTA] Epoch {epoch+1}/{delta_epochs}")
        hard_reflect_mode = str(getattr(args, "hard_reflection_mode", "off")).lower().strip()
        hard_reflect_buffer = []
        epoch_reg_scale = _epoch_ramp_scale(
            epoch_idx=epoch,
            warmup_epochs=delta_warmup_epochs,
            curriculum_epochs=delta_curriculum_epochs,
        )
        live_logger.info(f"[DELTA][CURRICULUM] epoch={epoch+1} reg_scale={epoch_reg_scale:.3f}")

        for bidx, batch in enumerate(pbar):
            with torch.no_grad():
                history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
                history_z = history_z.to(device)
                base_pred = base_backbone(history_z).to(torch.float32)

            # build delta inputs (with news)
            delta_inputs = build_delta_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                testing=False,
                force_no_news=False,
                news_dropout=True,
                api_adapter=news_api_adapter,
            )
            ts_p = delta_inputs["ts_patches"]
            ts_pm = delta_inputs["ts_patch_mask"]
            targets_z = delta_inputs["targets_z"]
            prompt_texts_d = delta_inputs["prompt_texts"]
            news_counts_d = delta_inputs["news_counts"]
            structured_feats_d = delta_inputs["structured_feats"]
            refined_news_ids_d = delta_inputs["refined_news_ids"]
            refined_news_attn_d = delta_inputs["refined_news_attn"]
            refined_news_doc_ids_d = delta_inputs["refined_news_doc_ids"]
            refined_news_doc_attn_d = delta_inputs["refined_news_doc_attn"]
            refined_news_doc_mask_d = delta_inputs["refined_news_doc_mask"]
            # build base text inputs (no news)
            delta_null_inputs = build_delta_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                testing=False,
                force_no_news=True,
                news_dropout=False,
                api_adapter=news_api_adapter,
            )
            news_counts_b = delta_null_inputs["news_counts"]
            structured_feats_b = delta_null_inputs["structured_feats"]
            refined_news_ids_b = delta_null_inputs["refined_news_ids"]
            refined_news_attn_b = delta_null_inputs["refined_news_attn"]
            refined_news_doc_ids_b = delta_null_inputs["refined_news_doc_ids"]
            refined_news_doc_attn_b = delta_null_inputs["refined_news_doc_attn"]
            refined_news_doc_mask_b = delta_null_inputs["refined_news_doc_mask"]

            ts_p = ts_p.to(device)
            ts_pm = ts_pm.to(device)
            targets_z = targets_z.to(device)
            news_counts_d = news_counts_d.to(device=device, dtype=torch.float32)
            news_counts_b = news_counts_b.to(device=device, dtype=torch.float32)
            structured_feats_d = structured_feats_d.to(device=device, dtype=torch.float32)
            structured_feats_b = structured_feats_b.to(device=device, dtype=torch.float32)
            refined_news_ids_d = refined_news_ids_d.to(device)
            refined_news_attn_d = refined_news_attn_d.to(device)
            refined_news_ids_b = refined_news_ids_b.to(device)
            refined_news_attn_b = refined_news_attn_b.to(device)
            refined_news_doc_ids_d = refined_news_doc_ids_d.to(device)
            refined_news_doc_attn_d = refined_news_doc_attn_d.to(device)
            refined_news_doc_mask_d = refined_news_doc_mask_d.to(device)
            refined_news_doc_ids_b = refined_news_doc_ids_b.to(device)
            refined_news_doc_attn_b = refined_news_doc_attn_b.to(device)
            refined_news_doc_mask_b = refined_news_doc_mask_b.to(device)
            has_news = (news_counts_d > 0).to(dtype=torch.float32)
            history_raw = (
                _history_raw_tensor_from_batch(batch, args, device=device)
                if external_sign_enabled
                else None
            )

            delta_model.train()

            delta_targets = _build_delta_targets(
                targets_z=targets_z,
                base_pred=base_pred,
                mu_global=float(global_zstats["mu_global"]),
                sigma_global=float(global_zstats["sigma_global"]),
                args=args,
            )

            out_delta = delta_model(
                ts_patches=ts_p,
                ts_patch_mask=ts_pm,
                refined_news_input_ids=refined_news_ids_d,
                refined_news_attention_mask=refined_news_attn_d,
                refined_news_doc_input_ids=refined_news_doc_ids_d,
                refined_news_doc_attention_mask=refined_news_doc_attn_d,
                refined_news_doc_mask=refined_news_doc_mask_d,
                targets=None,
                head_mode="delta",
                rel_targets=None,
                rel_lambda=0.0,
                structured_feats=structured_feats_d,
            )
            delta_pred_model_real = out_delta["pred"].to(torch.float32)
            gate_logits_real = out_delta.get("gate_logits", out_delta["rel_logits"]).to(torch.float32)
            gate_real = out_delta.get("gate", torch.sigmoid(gate_logits_real)).to(torch.float32)
            magnitude_real = out_delta.get("magnitude", delta_pred_model_real.abs()).to(torch.float32)
            magnitude_raw_real = out_delta.get("magnitude_raw", magnitude_real).to(torch.float32)
            if external_sign_enabled:
                with torch.no_grad():
                    sign_logits_real, sign_soft_real = _run_external_signnet(
                        signnet_model=external_signnet_model,
                        history_raw=history_raw,
                        base_pred_z=base_pred,
                        structured_feats=structured_feats_d,
                        news_counts=news_counts_d,
                        tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
                    )
                delta_pred_real, gate_real_h = _compose_delta_with_external_sign(
                    gate=gate_real,
                    magnitude=magnitude_real,
                    sign_soft=sign_soft_real,
                    delta_clip=float(getattr(delta_model, "delta_clip", getattr(args, "delta_clip", 0.0))),
                )
            else:
                sign_logits_real = out_delta.get("sign_logits", torch.zeros_like(delta_pred_model_real)).to(torch.float32)
                sign_soft_real = out_delta.get("sign_soft", torch.zeros_like(delta_pred_model_real)).to(torch.float32)
                delta_pred_real = delta_pred_model_real
                gate_real_h = _match_gate_shape(gate_real, delta_pred_real)
            delta_init_real = (sign_soft_real * magnitude_real).to(torch.float32)
            news_available_mask = out_delta.get(
                "news_available_mask",
                has_news.unsqueeze(1),
            ).to(torch.float32)
            usable_news = news_available_mask.reshape(news_available_mask.size(0), -1).max(dim=1).values.clamp(0.0, 1.0)
            news_usefulness_weighting = int(getattr(args, "news_usefulness_weighting", 1)) == 1
            sample_weight = _build_news_usefulness_weights(
                has_news=usable_news,
                news_counts=news_counts_d,
                structured_feats=structured_feats_d,
                enabled=news_usefulness_weighting,
            )

            pred_real_z = _fuse_base_and_delta(
                base_pred_z=base_pred,
                delta_pred=delta_pred_real,
                gate_h=gate_real_h,
                args=args,
                mu_global=float(global_zstats["mu_global"]),
                sigma_global=float(global_zstats["sigma_global"]),
            )
            targets_z_typed = targets_z.to(torch.float32)
            true_residual_z = delta_targets.to(torch.float32)
            abs_residual_target = true_residual_z.abs()
            sign_eps = float(getattr(args, "delta_sign_eps", 0.03))
            gate_eps_raw = getattr(args, "delta_gate_eps", None)
            gate_eps = float(sign_eps if gate_eps_raw is None else gate_eps_raw)
            valid_sign_mask = (abs_residual_target > sign_eps).to(torch.float32)
            sign_target_bin = (true_residual_z > 0).to(torch.float32)
            gate_target = (abs_residual_target > gate_eps).to(torch.float32)
            position_weight = _build_delta_residual_position_weights(abs_residual_target, args)
            sample_pos_weight = sample_weight.unsqueeze(1) * position_weight

            residual_mode = str(getattr(args, "residual_loss", "mae")).lower()
            loss_final = _point_loss(pred_real_z, targets_z_typed, mode=residual_mode)
            err_real = torch.abs(pred_real_z - targets_z_typed).mean(dim=1)
            err_base = torch.abs(base_pred.to(torch.float32) - targets_z_typed).mean(dim=1)

            delta_init_aux_weight = float(
                getattr(args, "delta_init_aux_weight", getattr(args, "delta_aux_lambda", 0.2))
            )
            if delta_init_aux_weight > 0.0:
                delta_aux_mode = str(getattr(args, "delta_aux_loss", residual_mode)).lower()
                loss_delta_aux = _point_loss(delta_init_real, true_residual_z, mode=delta_aux_mode)
            else:
                loss_delta_aux = torch.zeros((), device=device, dtype=torch.float32)

            delta_gate_loss_weight = float(
                getattr(args, "delta_gate_loss_weight", getattr(args, "final_gate_sup_weight", getattr(args, "gate_lambda", 0.2)))
            )
            if delta_gate_loss_weight > 0.0:
                gate_bce = F.binary_cross_entropy_with_logits(
                    gate_logits_real,
                    gate_target,
                    reduction="none",
                )
                loss_gate_sup = _masked_weighted_mean(gate_bce, sample_pos_weight)
            else:
                loss_gate_sup = torch.zeros((), device=device, dtype=torch.float32)

            delta_sign_loss_weight = float(
                getattr(args, "delta_sign_loss_weight", getattr(args, "delta_sign_lambda", 0.1))
            )
            if external_sign_enabled:
                delta_sign_loss_weight = 0.0
                loss_delta_sign = torch.zeros((), device=device, dtype=torch.float32)
            elif delta_sign_loss_weight > 0.0:
                sign_bce = F.binary_cross_entropy_with_logits(
                    sign_logits_real,
                    sign_target_bin,
                    reduction="none",
                )
                loss_delta_sign = _masked_weighted_mean(sign_bce, sample_pos_weight, mask=valid_sign_mask)
            else:
                loss_delta_sign = torch.zeros((), device=device, dtype=torch.float32)

            delta_mag_loss_weight = float(getattr(args, "delta_mag_loss_weight", 0.5))
            if delta_mag_loss_weight > 0.0:
                mag_pred_target = _transform_delta_magnitude_target(magnitude_real, args)
                mag_true_target = _transform_delta_magnitude_target(abs_residual_target, args)
                mag_beta = float(getattr(args, "huber_beta", 0.5))
                mag_loss_raw = F.smooth_l1_loss(
                    mag_pred_target,
                    mag_true_target,
                    beta=max(1e-6, mag_beta),
                    reduction="none",
                )
                loss_delta_mag = _masked_weighted_mean(mag_loss_raw, sample_pos_weight)
            else:
                loss_delta_mag = torch.zeros((), device=device, dtype=torch.float32)

            delta_non_degrade_lambda = float(getattr(args, "delta_non_degrade_lambda", 0.0))
            delta_non_degrade_margin = float(getattr(args, "delta_non_degrade_margin", 0.0))
            if delta_non_degrade_lambda > 0.0:
                loss_non_degrade = torch.relu(err_real - err_base + delta_non_degrade_margin).mean()
            else:
                loss_non_degrade = torch.zeros((), device=device, dtype=torch.float32)

            gate_reg_lambda = float(getattr(args, "delta_gate_reg_lambda", 0.0))
            loss_gate_reg = gate_real.mean()
            gate_null_lambda = float(getattr(args, "gate_null_lambda", 0.0))
            struct_impact_weight = float(getattr(args, "struct_impact_weight", 0.0))

            delta_cf_lambda = float(getattr(args, "delta_cf_lambda", 0.0))
            delta_cf_margin = float(getattr(args, "delta_cf_margin", 0.0))
            delta_null_lambda = float(getattr(args, "delta_null_lambda", 0.05))
            null_step_scale = _step_ramp_scale(
                step_idx=global_step,
                warmup_steps=delta_null_warmup_steps,
                ramp_steps=delta_null_ramp_steps,
            )
            gate_reg_lambda_eff = gate_reg_lambda * epoch_reg_scale
            gate_null_lambda_eff = gate_null_lambda * epoch_reg_scale
            delta_cf_lambda_eff = delta_cf_lambda * epoch_reg_scale
            delta_null_lambda_eff = delta_null_lambda * epoch_reg_scale * null_step_scale
            if _all_gates_disabled(args):
                delta_null_lambda_eff = 0.0
            loss_cf = torch.zeros((), device=device, dtype=torch.float32)
            loss_null = torch.zeros((), device=device, dtype=torch.float32)
            loss_gate_null = torch.zeros((), device=device, dtype=torch.float32)
            gate_targets = None

            if (
                delta_cf_lambda_eff > 0.0
                or delta_null_lambda_eff > 0.0
                or gate_null_lambda_eff > 0.0
            ):
                out_null = delta_model(
                    ts_patches=ts_p,
                    ts_patch_mask=ts_pm,
                    refined_news_input_ids=refined_news_ids_b,
                    refined_news_attention_mask=refined_news_attn_b,
                    refined_news_doc_input_ids=refined_news_doc_ids_b,
                    refined_news_doc_attention_mask=refined_news_doc_attn_b,
                    refined_news_doc_mask=refined_news_doc_mask_b,
                    targets=None,
                    head_mode="delta",
                    rel_targets=None,
                    rel_lambda=0.0,
                    structured_feats=structured_feats_b,
                )
                delta_pred_null_model = out_null["pred"].to(torch.float32)
                gate_logits_null = out_null.get("gate_logits", out_null["rel_logits"]).to(torch.float32)
                gate_null = out_null.get("gate", torch.sigmoid(gate_logits_null)).to(torch.float32)
                magnitude_null = out_null.get("magnitude", delta_pred_null_model.abs()).to(torch.float32)
                if external_sign_enabled:
                    with torch.no_grad():
                        sign_logits_null, sign_soft_null = _run_external_signnet(
                            signnet_model=external_signnet_model,
                            history_raw=history_raw,
                            base_pred_z=base_pred,
                            structured_feats=structured_feats_b,
                            news_counts=news_counts_b,
                            tau=float(getattr(args, "delta_sign_external_tau", 1.0)),
                        )
                    delta_pred_null, gate_null_h = _compose_delta_with_external_sign(
                        gate=gate_null,
                        magnitude=magnitude_null,
                        sign_soft=sign_soft_null,
                        delta_clip=float(getattr(delta_model, "delta_clip", getattr(args, "delta_clip", 0.0))),
                    )
                else:
                    delta_pred_null = delta_pred_null_model
                    gate_null_h = _match_gate_shape(gate_null, delta_pred_null)
                pred_null_z = _fuse_base_and_delta(
                    base_pred_z=base_pred,
                    delta_pred=delta_pred_null,
                    gate_h=gate_null_h,
                    args=args,
                    mu_global=float(global_zstats["mu_global"]),
                    sigma_global=float(global_zstats["sigma_global"]),
                )

                err_null = torch.abs(pred_null_z - targets_z_typed).mean(dim=1)
                loss_cf = torch.relu(delta_cf_margin + err_real - err_null).mean()
                loss_null = delta_pred_null.pow(2).mean() + 0.25 * magnitude_null.pow(2).mean()
                if gate_null_lambda_eff > 0.0:
                    null_targets = torch.zeros_like(gate_logits_null, dtype=torch.float32)
                    loss_gate_null = F.binary_cross_entropy_with_logits(
                        gate_logits_null,
                        null_targets,
                        reduction="mean",
                    )

            loss_struct_consistency, struct_loss_parts = _structured_consistency_losses(
                out_delta=out_delta,
                structured_feats=structured_feats_d,
                sample_weight=sample_weight,
                gate_targets=gate_targets,
            )

            loss_total = (
                loss_final
                + delta_init_aux_weight * loss_delta_aux
                + delta_gate_loss_weight * loss_gate_sup
                + delta_sign_loss_weight * loss_delta_sign
                + delta_mag_loss_weight * loss_delta_mag
                + delta_non_degrade_lambda * loss_non_degrade
                + struct_impact_weight * loss_struct_consistency
                + gate_reg_lambda_eff * loss_gate_reg
                + gate_null_lambda_eff * loss_gate_null
                + delta_cf_lambda_eff * loss_cf
                + delta_null_lambda_eff * loss_null
            )

            if hard_reflect_mode not in {"off", "none"} and len(prompt_texts_d) > 0:
                err_real_batch = torch.abs(pred_real_z - targets_z_typed).mean(dim=1).detach().cpu().numpy()
                if err_real_batch.size > 0:
                    hard_i = int(np.argmax(err_real_batch))
                    hard_reflect_buffer.append(
                        {
                            "epoch": int(epoch + 1),
                            "batch_idx": int(bidx),
                            "error_z_mae": float(err_real_batch[hard_i]),
                            "prompt": str(prompt_texts_d[hard_i]),
                            "target_time": str(batch["target_time"][hard_i]),
                            "has_news": int(float(has_news[hard_i].detach().cpu()) > 0.5),
                        }
                    )

            loss = loss_total / args.grad_accum
            loss.backward()

            loss_window.append(float(loss.detach().cpu()))
            if global_step % 10 == 0:
                avg_train_loss = sum(loss_window) / len(loss_window)
                gate_mean = float(gate_real_h.mean().detach().cpu())
                gate_frac = float((gate_real_h > 0.5).to(torch.float32).mean().detach().cpu())
                sign_acc = float(
                    _masked_binary_accuracy_from_logits(sign_logits_real, sign_target_bin, valid_sign_mask)
                    .detach()
                    .cpu()
                )
                mag_mean = float(magnitude_real.mean().detach().cpu())
                mag_true_mean = float(abs_residual_target.mean().detach().cpu())
                delta_abs_mean = float(delta_pred_real.abs().mean().detach().cpu())
                news_frac = float(usable_news.mean().detach().cpu())
                pbar.set_postfix(
                    train_loss=f"{avg_train_loss:.6f}",
                    gate=gate_mean,
                    g50=gate_frac,
                    final=float(loss_final.detach().cpu()),
                    aux=float(loss_delta_aux.detach().cpu()),
                    gsup=float(loss_gate_sup.detach().cpu()),
                    sgn=sign_acc,
                    mag=mag_mean,
                    true=mag_true_mean,
                    delta=delta_abs_mean,
                    ndg=float(loss_non_degrade.detach().cpu()),
                    news=news_frac,
                    gnull=float(loss_gate_null.detach().cpu()),
                    cf=float(loss_cf.detach().cpu()),
                )
                if global_step % 100 == 0:
                    dh_grad = 0.0
                    dh_grad_terms = []
                    for n, p in delta_model.named_parameters():
                        if n.startswith("delta_head") and p.grad is not None:
                            g = float(p.grad.norm().cpu())
                            if np.isfinite(g):
                                dh_grad_terms.append(g * g)
                    if dh_grad_terms:
                        dh_grad = float(np.sqrt(sum(dh_grad_terms)))
                    doc_attn_entropy = float(out_delta.get("doc_attn_entropy", torch.zeros((), device=device)).detach().cpu())
                    doc_attn_max = float(out_delta.get("doc_attn_max", torch.zeros((), device=device)).detach().cpu())
                    live_logger.info(
                        f"[GRAD] step={global_step} delta_head_grad={dh_grad:.6f} "
                        f"gate={gate_mean:.4f} gate>0.5={gate_frac:.4f} "
                        f"sgn_acc={sign_acc:.4f} mag={mag_mean:.4f} |r|={mag_true_mean:.4f} |d|={delta_abs_mean:.4f} "
                        f"reg={epoch_reg_scale:.3f} null={null_step_scale:.3f} "
                        f"gsup={float(loss_gate_sup.detach().cpu()):.6f} "
                        f"sgn={float(loss_delta_sign.detach().cpu()):.6f} "
                        f"magL={float(loss_delta_mag.detach().cpu()):.6f} "
                        f"ndg={float(loss_non_degrade.detach().cpu()):.6f} "
                        f"news_frac={news_frac:.4f} "
                        f"docH={doc_attn_entropy:.4f} docW={doc_attn_max:.4f} "
                        f"lam(reg/g/s/m/g0/c/n/nd)=("
                        f"{gate_reg_lambda_eff:.4f}/{delta_gate_loss_weight:.4f}/"
                        f"{delta_sign_loss_weight:.4f}/{delta_mag_loss_weight:.4f}/"
                        f"{gate_null_lambda_eff:.4f}/{delta_cf_lambda_eff:.4f}/"
                        f"{delta_null_lambda_eff:.4f}/{delta_non_degrade_lambda:.4f}/"
                        f"{struct_impact_weight:.4f})"
                    )

            if (global_step + 1) % args.grad_accum == 0:
                grad_clip = float(getattr(args, "delta_grad_clip", 0.0))
                if grad_clip > 0:
                    clip_grad_norm_((p for p in delta_model.parameters() if p.requires_grad), grad_clip)
                optim_delta.step()
                if args.scheduler == 1:
                    scheduler_delta.step()
                optim_delta.zero_grad(set_to_none=True)

            global_step += 1

        if delta_val_mode == "each_epoch":
            # end-of-epoch eval (combined)
            val_loss, val_mse, val_mae, base_val_loss, base_val_mse, base_val_mae = evaluate_metrics_residual(
                base_model=base_backbone,
                delta_model=delta_model,
                external_signnet=external_signnet_model,
                tokenizer=tokenizer,
                data_loader=val_loader,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                device=device,
                volatility_bin=volatility_bin_val,
                testing=False,
                true_pred_csv_path=None,
                news_dropout=False,
                api_adapter=news_api_adapter,
            )
            val_loss_per_epoch.append(val_loss)
            mse_loss_per_epoch.append(val_mse)
            mae_loss_per_epoch.append(val_mae)

            live_logger.info(
                f"[DELTA][EVAL] epoch={epoch+1} tpl_id={tpl_id} policy={policy_name} "
                f"val_loss(zMSE)={val_loss:.6f} "
                f"val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
            )
            _log_last_residual_eval_diag(args, live_logger, f"[DELTA][EVAL_DIAG][VAL] epoch={epoch+1}")
            live_logger.info(
                f"[DELTA][BASE_ONLY][VAL] epoch={epoch+1} "
                f"loss(zMAE)={base_val_loss:.6f} mse(raw)={base_val_mse:.6f} mae(raw)={base_val_mae:.6f}"
            )
            metric_now = _select_metric(val_loss, val_mse, val_mae, args.select_metric)
            delta_vs_base = float(metric_now - base_ref_metric)
            live_logger.info(
                f"[DELTA][COMPARE][VAL] epoch={epoch+1} metric={metric_now:.6f} "
                f"delta_vs_base={delta_vs_base:+.6f}"
            )

            should_save = (not has_saved_delta) or (metric_now < best_metric - 1e-6)
            if should_save:
                best_metric = float(metric_now)
                stale_rounds = 0
                has_saved_delta = True
                best_tpl_id = tpl_id
                best_policy_name = policy_name
                best_delta_alpha = 1.0
                _save_residual_best(
                    epoch_idx=epoch,
                    metric_now=best_metric,
                    tpl_id_now=best_tpl_id,
                    policy_name_now=best_policy_name,
                )
                _dump_val_residual_debug_csv()
                live_logger.info(
                    f"[DELTA] New best delta saved: {best_delta_path} "
                    f"({args.select_metric}={best_metric:.6f}, "
                    f"delta_vs_base={best_metric - base_ref_metric:+.6f})"
                )
            else:
                stale_rounds += 1
                if early_stop_patience > 0:
                    live_logger.info(
                        f"[DELTA] stale_rounds={stale_rounds}/{early_stop_patience} "
                        f"best={best_metric:.6f} delta_vs_base={best_metric - base_ref_metric:+.6f}"
                    )

            if early_stop_patience > 0 and stale_rounds >= early_stop_patience:
                live_logger.info(f"[DELTA] Early stopping triggered at epoch {epoch+1}.")
                break

        if hard_reflect_mode not in {"off", "none"} and len(hard_reflect_buffer) > 0:
            top_k = int(max(1, getattr(args, "hard_reflection_topk", 8)))
            hard_reflect_buffer = sorted(
                hard_reflect_buffer,
                key=lambda x: float(x.get("error_z_mae", 0.0)),
                reverse=True,
            )[:top_k]
            reflections = reflect_hard_samples(
                hard_samples=hard_reflect_buffer,
                mode=hard_reflect_mode,
                api_adapter=news_api_adapter,
            )
            live_logger.info(
                f"[DELTA][REFLECT] epoch={epoch+1} mode={hard_reflect_mode} "
                f"hard_samples={len(hard_reflect_buffer)} reflections={len(reflections)}"
            )

        _save_refine_cache(args, live_logger=live_logger, force=False)
        _save_structured_cache(args, live_logger=live_logger, force=False)

        if epoch == 0:
            _log_prompt_stats_if_available(
                live_logger,
                dataStatistic,
                "---------------------trainset and valset prompt statistics--------------------------------",
                "[DELTA][PROMPT_STATS] skipped: DELTA prompt path is disabled in this stage.",
            )

    if not has_saved_delta:
        final_epoch = max(0, int(delta_epochs) - 1)
        if delta_val_mode == "end_only":
            val_loss, val_mse, val_mae, base_val_loss, base_val_mse, base_val_mae = evaluate_metrics_residual(
                base_model=base_backbone,
                delta_model=delta_model,
                external_signnet=external_signnet_model,
                tokenizer=tokenizer,
                data_loader=val_loader,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                device=device,
                volatility_bin=volatility_bin_val,
                testing=False,
                true_pred_csv_path=None,
                news_dropout=False,
                api_adapter=news_api_adapter,
            )
            metric_now = _select_metric(val_loss, val_mse, val_mae, args.select_metric)
            best_metric = float(metric_now)
            has_saved_delta = True
            best_tpl_id = tpl_id
            best_policy_name = policy_name
            _save_residual_best(
                epoch_idx=final_epoch,
                metric_now=best_metric,
                tpl_id_now=best_tpl_id,
                policy_name_now=best_policy_name,
            )
            _dump_val_residual_debug_csv()
            live_logger.info(
                f"[DELTA][VAL] end_only: val_loss(zMSE)={val_loss:.6f} "
                f"val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
            )
            _log_last_residual_eval_diag(args, live_logger, "[DELTA][EVAL_DIAG][VAL] end_only")
            live_logger.info(
                f"[DELTA][BASE_ONLY][VAL] end_only: loss(zMAE)={base_val_loss:.6f} "
                f"mse(raw)={base_val_mse:.6f} mae(raw)={base_val_mae:.6f}"
            )
            live_logger.info(
                f"[DELTA] Saved end_only best delta: {best_delta_path} "
                f"({args.select_metric}={best_metric:.6f})"
            )
        elif delta_val_mode == "none":
            has_saved_delta = True
            best_tpl_id = tpl_id
            best_policy_name = policy_name
            best_metric = float("nan")
            _save_residual_best(
                epoch_idx=final_epoch,
                metric_now=None,
                tpl_id_now=best_tpl_id,
                policy_name_now=best_policy_name,
            )
            live_logger.info(
                f"[DELTA][VAL] skipped (mode={delta_val_mode}); "
                f"saved final checkpoint: {best_delta_path} (epoch={final_epoch+1})"
            )
        else:
            # safety fallback for unexpected state in each_epoch mode
            val_loss, val_mse, val_mae, base_val_loss, base_val_mse, base_val_mae = evaluate_metrics_residual(
                base_model=base_backbone,
                delta_model=delta_model,
                external_signnet=external_signnet_model,
                tokenizer=tokenizer,
                data_loader=val_loader,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                global_zstats=global_zstats,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                device=device,
                volatility_bin=volatility_bin_val,
                testing=False,
                true_pred_csv_path=None,
                news_dropout=False,
                api_adapter=news_api_adapter,
            )
            metric_now = _select_metric(val_loss, val_mse, val_mae, args.select_metric)
            best_metric = float(metric_now)
            has_saved_delta = True
            best_tpl_id = tpl_id
            best_policy_name = policy_name
            _save_residual_best(
                epoch_idx=final_epoch,
                metric_now=best_metric,
                tpl_id_now=best_tpl_id,
                policy_name_now=best_policy_name,
            )
            _dump_val_residual_debug_csv()
            live_logger.info(
                f"[DELTA][VAL] fallback eval: val_loss(zMSE)={val_loss:.6f} "
                f"val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
            )
            _log_last_residual_eval_diag(args, live_logger, "[DELTA][EVAL_DIAG][VAL] fallback")
            live_logger.info(
                f"[DELTA][BASE_ONLY][VAL] fallback: loss(zMAE)={base_val_loss:.6f} "
                f"mse(raw)={base_val_mse:.6f} mae(raw)={base_val_mae:.6f}"
            )

    dataStatistic.clear()
    _save_refine_cache(args, live_logger=live_logger, force=True)
    _save_structured_cache(args, live_logger=live_logger, force=True)

    # TEST (combined with best delta; base computed via adapter_off)
    if test_loader is not None:
        del delta_model
         # ---- free training-time GPU objects ----
        # del delta_model
        # del optim_delta
        # if scheduler_delta is not None:
        #     del scheduler_delta

        gc.collect()
        torch.cuda.empty_cache()

        best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")
        if not os.path.exists(best_delta_path):
            live_logger.info("[DELTA] best delta checkpoint not found; fallback to base-only test.")
            test_loss, test_mse, test_mae = evaluate_metrics_backbone(
                base_backbone=base_backbone,
                data_loader=test_loader,
                args=args,
                global_zstats=global_zstats,
                device=device,
                testing=True,
                true_pred_csv_path=true_pred_csv_path,
                filename=bundle["test_filename"],
            )
            tqdm.write(f"[TEST][FALLBACK-BASE] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
            live_logger.info(
                f"[TEST][FALLBACK-BASE] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
            )
            record_test_results_csv(args, live_logger, test_mse, test_mae)
            draw_pred_true(live_logger, args, true_pred_csv_path)
            _save_refine_cache(args, live_logger=live_logger, force=True)
            _save_structured_cache(args, live_logger=live_logger, force=True)
            return {
                "test_loader": test_loader,
                "device": device,
                "live_logger": live_logger,
                "best_tpl_id": best_tpl_id,
                "best_policy_name": best_policy_name,
                "tpl_id": tpl_id,
                "policy_name": policy_name,
                "templates": templates,
                "news_df": news_df,
                "policy_kw": policy_kw,
                "volatility_bin_test": volatility_bin_test,
                "true_pred_csv_path": true_pred_csv_path,
                "global_zstats": global_zstats,
            }

        tok_d, model_best = load_checkpoint(
            best_delta_path,
            _single_device_map(args),
            False,
            head_mlp=args.head_mlp,
            hd=args.head_dropout,
            pd=args.patch_dropout


        )

        tokenizer = tok_d
        model_best.to(device)
        model_best.eval()

        delta_meta = {}
        delta_meta_path = os.path.join(best_delta_path, "meta.json")
        if os.path.isfile(delta_meta_path):
            try:
                with open(delta_meta_path, "r", encoding="utf-8") as f:
                    delta_meta = json.load(f)
            except Exception:
                delta_meta = {}
        global_zstats_eval = _coerce_global_zstats(delta_meta, args, required=False)
        if global_zstats_eval is None:
            global_zstats_eval = global_zstats
        else:
            live_logger.info(
                "[DELTA] loaded global z-score stats from delta checkpoint meta: "
                f"mu_global={global_zstats_eval['mu_global']:.6f}, "
                f"sigma_global={global_zstats_eval['sigma_global']:.6f}"
            )

        live_logger.info(
            f"Loaded best DELTA model for testing (final = {base_meta.get('backbone_name')} + delta(adapter_on,news))."
        )

        tpl_for_test = tpl_id
        pol_for_test = policy_name
        (
            test_loss,
            test_mse,
            test_mae,
            base_test_loss,
            base_test_mse,
            base_test_mae,
        ) = evaluate_metrics_residual(
            base_model=base_backbone,
            delta_model=model_best,
            external_signnet=external_signnet_model,
            tokenizer=tokenizer,
            data_loader=test_loader,
            templates=templates,
            tpl_id=tpl_for_test,
            args=args,
            global_zstats=global_zstats_eval,
            news_df=news_df,
            policy_name=pol_for_test,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_test,
            testing=True,
            true_pred_csv_path=true_pred_csv_path,
            news_dropout=False,
            filename=bundle["test_filename"],
            api_adapter=news_api_adapter,
            residual_debug_csv_path=test_residual_debug_csv_path,
            residual_debug_split="test",
        )

        _log_prompt_stats_if_available(
            live_logger,
            dataStatistic,
            "---------------------testset prompt statistics--------------------------------",
            "[DELTA][PROMPT_STATS] skipped: DELTA test path did not build model-consumed prompts.",
        )

        tqdm.write(
            f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
        )
        live_logger.info(
            f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
        )
        _log_last_residual_eval_diag(args, live_logger, "[TEST][FINAL_DIAG]")
        live_logger.info(
            f"[TEST][BASE_ONLY] loss(zMAE)={base_test_loss:.6f} mse(raw)={base_test_mse:.6f} "
            f"mae(raw)={base_test_mae:.6f}"
        )
        if test_residual_debug_csv_path:
            live_logger.info(f"[DELTA][TEST_DEBUG_CSV] updated path={test_residual_debug_csv_path}")

        record_test_results_csv(args, live_logger, test_mse, test_mae)
        draw_pred_true(live_logger, args, true_pred_csv_path)
        _save_refine_cache(args, live_logger=live_logger, force=True)
        _save_structured_cache(args, live_logger=live_logger, force=True)

    return {
        "test_loader":test_loader,
        "device": device,
        "live_logger": live_logger,
        "best_tpl_id":best_tpl_id,
        "best_policy_name":best_policy_name,
        "tpl_id":tpl_id,
        "policy_name":policy_name,
        "templates":templates,
        "news_df":news_df,
        "policy_kw":policy_kw,
        "volatility_bin_test": volatility_bin_test,
        "true_pred_csv_path": true_pred_csv_path,
        "global_zstats": global_zstats,
    }


# ----------------------------
# main entry (refactored)
# ----------------------------
def testing_base(test_loader, args, device, live_logger, templates, volatility_bin_test, true_pred_csv_path, global_zstats):
    if test_loader is not None:
        gc.collect()
        torch.cuda.empty_cache()

        best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")

        model_best, base_meta = load_base_backbone_checkpoint(
            best_base_path,
            device=device,
            is_trainable=False,
        )

        live_logger.info(
            f"Loaded best BASE backbone for testing: {base_meta.get('backbone_name')} (final = base(no-news))."
        )
        stats = _coerce_global_zstats(base_meta, args, required=False)
        if stats is None:
            stats = _coerce_global_zstats(global_zstats, args, required=True)

        test_loss, test_mse, test_mae = evaluate_metrics_backbone(
            base_backbone=model_best,
            data_loader=test_loader,
            args=args,
            global_zstats=stats,
            device=device,
            testing=True,
            true_pred_csv_path=true_pred_csv_path,
            filename=getattr(args, "taskName", "base_only"),
        )

        _log_prompt_stats_if_available(
            live_logger,
            dataStatistic,
            "---------------------testset prompt statistics--------------------------------",
            "[BASE][PROMPT_STATS] skipped: no prompts were recorded in this test stage.",
        )

        tqdm.write(f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
        live_logger.info(f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")

        record_test_results_csv(args, live_logger, test_mse, test_mae)
        draw_pred_true(live_logger, args, true_pred_csv_path)

def main(args):
    bundle = setup_env_and_data(args)
    stage = bundle["stage"]

    ckpt_dir = bundle["ckpt_dir"]
    prompt_path = bundle["prompt_path"]
    ans_json_path = bundle["ans_json_path"]
    true_pred_csv_path = bundle["true_pred_csv_path"]
    val_residual_debug_csv_path = bundle.get("val_residual_debug_csv_path")
    test_residual_debug_csv_path = bundle.get("test_residual_debug_csv_path")

    with open(prompt_path, "w", encoding="utf-8"):
        pass
    with open(ans_json_path, "w", encoding="utf-8"):
        pass
    with open(true_pred_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pred", "true"])
    for extra_csv_path in [val_residual_debug_csv_path, test_residual_debug_csv_path]:
        if extra_csv_path and os.path.exists(extra_csv_path):
            os.remove(extra_csv_path)

    # resolve base ckpt path for delta-only / all
    def _resolve_base_ckpt():
        return os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")

    if stage == "base":
        cfg = train_base_stage(args, bundle)

        testing_base(
            bundle["test_loader"],
            args,
            bundle["device"],
            bundle["live_logger"],
            cfg["templates"],
            bundle["volatility_bin_test"],
            bundle["true_pred_csv_path"],
            cfg.get("global_zstats", bundle.get("global_zstats")),
        )
        return

    if stage == "delta":
        best_base_path = _resolve_base_ckpt()
        if not os.path.exists(best_base_path):
            raise FileNotFoundError(f"Base checkpoint not found: {best_base_path}")
        train_delta_stage(args, bundle, best_base_path=best_base_path, best_base_metric=float("inf"))
        return

    # stage == "all"
    if stage == "all":
        cfg_base = train_base_stage(args, bundle)
        train_delta_stage(
            args,
            bundle,
            best_base_path=cfg_base["best_base_path"],
            best_base_metric=float(cfg_base["best_base_metric"]),
        )
        return
