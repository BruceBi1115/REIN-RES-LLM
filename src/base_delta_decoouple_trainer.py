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
import torch.nn.functional as F
from torch.optim import AdamW
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
    format_structured_events_for_prompt,
    reflect_hard_samples,
    build_news_api_adapter,
)
from .delta_case_retrieval import (
    build_case_bank,
    build_case_record,
    build_retrieval_features,
    retrieve_similar_cases,
    save_case_bank,
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


def _refine_doc_cache_key(
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


def _stable_refine_cache_tag(args) -> str:
    news_path = str(getattr(args, "news_path", "") or "").strip()
    try:
        news_path = os.path.abspath(news_path) if news_path else ""
    except Exception:
        pass
    payload = {
        "news_path": news_path,
        "news_text_col": str(getattr(args, "news_text_col", "content") or "content"),
        "refine_mode": str(getattr(args, "news_refine_mode", "local") or "local").lower().strip(),
        "api_model": str(getattr(args, "news_api_model", "") or "").strip(),
        "region": str(getattr(args, "region", "") or "").strip(),
        "description": str(getattr(args, "description", "") or "").strip(),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _refine_cache_path(args) -> str:
    p = str(getattr(args, "news_refine_cache_path", "") or "").strip()
    if p:
        return p
    tag = _stable_refine_cache_tag(args)
    return os.path.join("./checkpoints", "_shared_refine_cache", f"refine_news_cache_{tag}.json")


def _resolve_refine_model_name(args, api_adapter) -> str:
    return str(getattr(api_adapter, "model", getattr(args, "news_api_model", "")))


def _doc_refine_context(args) -> dict:
    return {
        "target_time": "",
        "region": str(getattr(args, "region", "")),
        "description": str(getattr(args, "description", "")),
    }


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
            setattr(args, "_refine_cache_store", cache_store)
            setattr(args, "_refine_cache_dirty", True)
    return refined


def _refine_news_from_doc_cache(
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

    if len(snippets) == 0:
        return ""
    merged = "\n".join([f"- {s}" for s in snippets])
    return _truncate_with_tokenizer(merged, tokenizer, max_tokens=max_tokens)


def _init_refine_cache(args, live_logger=None):
    enabled = int(getattr(args, "news_refine_cache_enable", 1)) == 1
    setattr(args, "_refine_cache_enabled", bool(enabled))
    setattr(args, "_refine_cache_store", {})
    setattr(args, "_refine_cache_dirty", False)
    setattr(args, "_refine_cache_hits", 0)
    setattr(args, "_refine_cache_misses", 0)
    if not enabled:
        if live_logger is not None:
            live_logger.info("[NEWS_REFINE_CACHE] disabled.")
        return

    path = _refine_cache_path(args)
    setattr(args, "_refine_cache_path", path)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    store = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
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
            store = {}
    setattr(args, "_refine_cache_store", store)
    if live_logger is not None:
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
    if not path or not isinstance(store, dict):
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp = path + ".tmp"
    payload = [
        {"cache_key": str(k), "refined_news": str(v)}
        for k, v in sorted(store.items(), key=lambda x: str(x[0]))
        if isinstance(v, str)
    ]
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    setattr(args, "_refine_cache_dirty", False)
    if live_logger is not None:
        hits = int(getattr(args, "_refine_cache_hits", 0))
        misses = int(getattr(args, "_refine_cache_misses", 0))
        live_logger.info(
            f"[NEWS_REFINE_CACHE] saved entries={len(store)} hits={hits} misses={misses}"
        )


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


def _resolve_delta_residual_mode(args) -> str:
    mode = str(getattr(args, "delta_residual_mode", "additive")).lower().strip()
    if mode not in {"additive", "relative"}:
        mode = "additive"
    return mode


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


def _select_metric(loss_v: float, mse_v: float, mae_v: float, select_metric: str) -> float:
    rm = str(select_metric).lower()
    if rm == "loss":
        return float(loss_v)
    if rm == "mse":
        return float(mse_v)
    return float(mae_v)


def _resolve_retrieval_mode(
    args,
    retrieval_enable_override: bool | None = None,
    retrieval_mode_override: str | None = None,
) -> str:
    if retrieval_mode_override is not None:
        mode = str(retrieval_mode_override).lower().strip()
    else:
        mode = str(getattr(args, "case_retrieval_mode", "price_event")).lower().strip()
    if mode not in {"off", "price", "price_event", "random"}:
        mode = "price_event"

    if retrieval_enable_override is None:
        enabled = int(getattr(args, "case_retrieval_enable", 0)) == 1
    else:
        enabled = bool(retrieval_enable_override)
    top_k = int(max(0, getattr(args, "case_retrieval_topk", 0)))
    if (not enabled) or top_k <= 0:
        return "off"
    return mode


def _retrieval_feature_dim(args) -> int:
    return int(max(8, getattr(args, "case_retrieval_feature_dim", 12)))


def _blend_with_retrieval_knn_prior(
    *,
    delta_pred: torch.Tensor,
    retrieval_feats: torch.Tensor | None,
    retrieval_delta_prior: torch.Tensor | None,
    args,
    enabled: bool = True,
) -> torch.Tensor:
    if (not enabled) or int(getattr(args, "case_retrieval_knn_enable", 1)) != 1:
        return delta_pred
    if retrieval_delta_prior is None:
        return delta_pred
    if delta_pred.ndim != 2:
        return delta_pred
    alpha_base = float(getattr(args, "case_retrieval_knn_alpha", 0.35))
    alpha_cap = float(getattr(args, "case_retrieval_knn_alpha_cap", 0.85))
    if alpha_base <= 0.0:
        return delta_pred

    prior = retrieval_delta_prior.to(device=delta_pred.device, dtype=delta_pred.dtype)
    if prior.ndim == 1:
        prior = prior.unsqueeze(0)
    if prior.size(0) != delta_pred.size(0):
        if prior.size(0) == 1:
            prior = prior.expand(delta_pred.size(0), -1)
        else:
            return delta_pred
    if prior.size(1) != delta_pred.size(1):
        h = int(delta_pred.size(1))
        if prior.size(1) > h:
            prior = prior[:, :h]
        else:
            pad = prior.new_zeros((prior.size(0), h - prior.size(1)))
            prior = torch.cat([prior, pad], dim=1)

    if retrieval_feats is not None and retrieval_feats.ndim == 2 and retrieval_feats.size(0) == delta_pred.size(0):
        rf = retrieval_feats.to(device=delta_pred.device, dtype=delta_pred.dtype)
        soft = rf[:, 0].clamp(0.0, 1.0) if rf.size(1) >= 1 else torch.ones((rf.size(0),), device=rf.device, dtype=rf.dtype)
        conf = rf[:, 1].clamp(0.0, 1.0) if rf.size(1) >= 2 else torch.ones((rf.size(0),), device=rf.device, dtype=rf.dtype)
        alpha = (soft * conf * float(alpha_base)).clamp(0.0, float(max(0.0, alpha_cap)))
    else:
        alpha = delta_pred.new_full((delta_pred.size(0),), float(alpha_base))
        alpha = alpha.clamp(0.0, float(max(0.0, alpha_cap)))

    return (1.0 - alpha.unsqueeze(1)) * delta_pred + alpha.unsqueeze(1) * prior


def _summarize_retrieval_stats(
    *,
    sample_mae: list[float],
    rel_labels: list[float],
    retrieval_meta: list[dict],
    strong_news_thresh: float,
) -> dict:
    if len(sample_mae) == 0:
        return {
            "coverage_rate": 0.0,
            "mae_valid": float("nan"),
            "mae_rejected": float("nan"),
            "mae_gap_valid_minus_rejected": float("nan"),
            "mae_strong_news_valid": float("nan"),
            "mae_strong_news_rejected": float("nan"),
            "n_samples": 0,
        }
    mae_arr = np.asarray(sample_mae, dtype=np.float32)
    rel_arr = np.asarray(rel_labels, dtype=np.float32)
    soft_weights = np.asarray(
        [
            float(
                m.get(
                    "retrieval_soft_weight",
                    1.0 if bool(m.get("retrieval_valid", False)) else 0.0,
                )
            )
            for m in retrieval_meta
        ],
        dtype=np.float32,
    )
    valid_mask = soft_weights > 0.05
    strong_mask = rel_arr >= float(strong_news_thresh)

    def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
        if arr.size == 0 or mask.size != arr.size or not bool(mask.any()):
            return float("nan")
        return float(arr[mask].mean())

    mae_valid = _masked_mean(mae_arr, valid_mask)
    mae_rej = _masked_mean(mae_arr, ~valid_mask)
    mae_strong_valid = _masked_mean(mae_arr, valid_mask & strong_mask)
    mae_strong_rej = _masked_mean(mae_arr, (~valid_mask) & strong_mask)
    gap = float(mae_valid - mae_rej) if np.isfinite(mae_valid) and np.isfinite(mae_rej) else float("nan")

    return {
        "coverage_rate": float(valid_mask.mean()),
        "effective_coverage": float(np.clip(soft_weights, 0.0, 1.0).mean()),
        "mae_valid": mae_valid,
        "mae_rejected": mae_rej,
        "mae_gap_valid_minus_rejected": gap,
        "mae_strong_news_valid": mae_strong_valid,
        "mae_strong_news_rejected": mae_strong_rej,
        "n_samples": int(mae_arr.size),
    }


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
    Print full run args at the beginning of logging.
    Sensitive fields are masked.
    """
    if live_logger is None:
        return
    try:
        arg_dict = dict(vars(args))
    except Exception:
        live_logger.info("[CONFIG] failed to read args via vars(args).")
        return

    live_logger.info("-----------------------------------------------------")
    live_logger.info("[CONFIG] Run Arguments")
    for k in sorted(arg_dict.keys()):
        v = _mask_sensitive_arg(k, arg_dict[k])
        if isinstance(v, (list, tuple, dict)):
            try:
                v_str = json.dumps(v, ensure_ascii=False)
            except Exception:
                v_str = str(v)
        else:
            v_str = str(v)
        live_logger.info(f"[CONFIG] {k} = {v_str}")
    live_logger.info("-----------------------------------------------------")


def _log_enabled_mechanisms(args, live_logger, stage: str):
    if live_logger is None:
        return
    stage_norm = str(stage or "").lower().strip()
    delta_active = stage_norm in {"delta", "all"}

    retrieval_mode = _resolve_retrieval_mode(args)
    if retrieval_mode != "off":
        if delta_active:
            live_logger.info(
                "[MECH][CASE_RETRIEVAL] enabled: retrieve similar historical cases and build "
                "auxiliary retrieval features for DELTA correction/gating/relevance."
            )
        else:
            live_logger.info(
                "[MECH][CASE_RETRIEVAL] enabled in args, but current stage is BASE-only; "
                "it will not be executed in this run."
            )
    if int(getattr(args, "delta_text_direct_enable", 0)) == 1:
        if delta_active:
            live_logger.info(
                "[MECH][TEXT_DIRECT] enabled: refined-news text encoder branch is fused into DELTA correction."
            )
        else:
            live_logger.info(
                "[MECH][TEXT_DIRECT] enabled in args, but current stage is BASE-only; "
                "it will not be executed in this run."
            )

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
            or lname.startswith("delta_text_ln")
            or lname.startswith("delta_log_scale")
            or lname.startswith("text_")
            or lname.startswith("rel_head")
            or lname.startswith("retrieval_")
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
    case_bank=None,
    query_base_pred=None,
    retrieval_enable_override: bool | None = None,
    retrieval_mode_override: str | None = None,
    api_adapter=None,
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

    tpl_text = templates[tpl_id]["text"]
    B = len(batch["history_value"])

    targets_z_list = []
    patches_list = []
    patchmask_list = []
    metas = []

    hist_strs = []
    news_str_list = []
    refined_news_list = []
    rel_labels_list = []

    start_dates = []
    end_dates = []
    pred_starts = []
    pred_ends = []

    len_selected_news = []
    retrieval_feature_list = []
    retrieval_delta_prior_list = []
    retrieval_meta_list = []

    retrieval_feat_dim = _retrieval_feature_dim(args)
    retrieval_mode = _resolve_retrieval_mode(
        args,
        retrieval_enable_override=retrieval_enable_override,
        retrieval_mode_override=retrieval_mode_override,
    )
    if force_no_news:
        retrieval_mode = "off"
    top_k = int(max(1, getattr(args, "case_retrieval_topk", 5)))

    if isinstance(query_base_pred, torch.Tensor):
        query_base_pred_arr = query_base_pred.detach().to(torch.float32).cpu().numpy()
    elif query_base_pred is None:
        query_base_pred_arr = None
    else:
        query_base_pred_arr = np.asarray(query_base_pred, dtype=np.float32)

    for i in range(B):
        history = batch["history_value"][i].tolist()
        target = batch["target_value"][i].tolist()
        t_target = batch["target_time"][i]
        series_id = str(batch.get("series_id", ["Not Specified"] * B)[i])

        history_z = _zscore(history, mu_global, sigma_global)
        target_z = _zscore(target, mu_global, sigma_global)
        if query_base_pred_arr is not None and i < len(query_base_pred_arr):
            base_pred_query = np.asarray(query_base_pred_arr[i], dtype=np.float32).reshape(-1)
        else:
            base_pred_query = np.zeros((H,), dtype=np.float32)


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
        structured_events = {}
        if (not force_no_news) and len(selected) > 0:
            raw_news_texts = selected[news_text_col].fillna("").astype(str).tolist()
            refine_context = {
                "target_time": str(t_target),
                "region": str(getattr(args, "region", "")),
                "description": str(getattr(args, "description", "")),
            }
            refined_news = _refine_news_from_doc_cache(
                raw_news_texts=raw_news_texts,
                tokenizer=tokenizer,
                max_tokens=news_budget,
                args=args,
                api_adapter=api_adapter,
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

            pieces = [refined_news] if refined_news else []

            need_structured_for_prompt = int(getattr(args, "delta_include_structured_news", 0)) == 1
            need_structured_for_retrieval = retrieval_mode != "off"
            if need_structured_for_prompt or need_structured_for_retrieval:
                structured_events = extract_structured_events(
                    raw_or_refined_news=refined_news,
                    mode=str(getattr(args, "news_structured_mode", "off")),
                    api_adapter=api_adapter,
                    context=refine_context,
                )
            if need_structured_for_prompt:
                structured_text = format_structured_events_for_prompt(structured_events)
                if structured_text:
                    pieces.append(structured_text)

            news_str = "\n\n".join([p for p in pieces if str(p).strip()])
            if news_dropout:
                news_str = _maybe_news_dropout(news_str, args)

        retrieval_feat = np.zeros((retrieval_feat_dim,), dtype=np.float32)
        retrieval_delta_prior = np.zeros((H,), dtype=np.float32)
        retrieval_meta = {
            "retrieval_valid": False,
            "retrieval_confidence": 0.0,
            "retrieval_soft_weight": 0.0,
            "retrieval_mode": "off",
            "reject_reason": "disabled",
            "direction_agreement": 0.0,
            "event_mismatch_rate": 1.0,
            "topk": 0,
        }
        if retrieval_mode != "off" and isinstance(case_bank, dict) and len(case_bank.get("cases", [])) > 0:
            sample_id = f"{series_id}::{str(t_target)}"
            query_case = build_case_record(
                sample_id=sample_id,
                split="query",
                target_time=str(t_target),
                history_z=history_z,
                base_pred_z=base_pred_query,
                target_z=None,
                refined_news=refined_news,
                structured_events=structured_events,
                metadata={"series_id": series_id},
            )
            # with open("query_case_dump.json", "a", encoding="utf-8") as _f:
            #     _f.write(json.dumps(query_case, ensure_ascii=False) + "\n")
            retrieval_out = retrieve_similar_cases(
                query_case=query_case,
                case_bank=case_bank,
                top_n=top_k,
                mode=retrieval_mode,
                alpha_price=float(getattr(args, "case_retrieval_alpha_price", 0.85)),
                alpha_event=float(getattr(args, "case_retrieval_alpha_event", 0.15)),
                alpha_text=float(getattr(args, "case_retrieval_alpha_text", 0.20)),
                alpha_recency=float(getattr(args, "case_retrieval_alpha_recency", 0.10)),
                alpha_regime=float(getattr(args, "case_retrieval_alpha_regime", 0.05)),
                recency_tau_hours=float(getattr(args, "case_retrieval_recency_tau_hours", 168.0)),
                min_top_score=float(getattr(args, "case_retrieval_min_top_score", 0.12)),
                min_candidates=int(max(1, getattr(args, "case_retrieval_min_candidates", 2))),
                min_direction_agreement=float(getattr(args, "case_retrieval_min_dir_agree", 0.45)),
                max_event_mismatch=float(getattr(args, "case_retrieval_max_event_mismatch", 0.8)),
                knn_temperature=float(getattr(args, "case_retrieval_knn_temperature", 0.20)),
                knn_horizon=int(H),
            )
            retrieval_feat, retrieval_meta = build_retrieval_features(
                query_case=query_case,
                retrieval_output=retrieval_out,
                feature_dim=retrieval_feat_dim,
            )
            prior_arr = np.asarray(
                retrieval_meta.get("knn_delta_prior_z", []), dtype=np.float32
            ).reshape(-1)
            if prior_arr.size > H:
                prior_arr = prior_arr[:H]
            elif prior_arr.size < H:
                pad = np.zeros((H - prior_arr.size,), dtype=np.float32)
                prior_arr = np.concatenate([prior_arr, pad], axis=0)
            if _resolve_delta_residual_mode(args) == "relative":
                prior_arr = _convert_knn_prior_z_to_relative_np(
                    prior_z=prior_arr,
                    base_pred_z=base_pred_query,
                    args=args,
                    mu_global=mu_global,
                    sigma_global=sigma_global,
                )
            retrieval_delta_prior = prior_arr.astype(np.float32, copy=False)
        retrieval_feature_list.append(np.asarray(retrieval_feat, dtype=np.float32))
        retrieval_delta_prior_list.append(np.asarray(retrieval_delta_prior, dtype=np.float32))
        retrieval_meta_list.append(dict(retrieval_meta))


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

        hist_strs.append(history_text(history_z, mu_global, sigma_global))
        news_str_list.append(news_str)
        refined_news_list.append(str(refined_news or ""))

        start_dates.append(start_date)
        end_dates.append(end_date)
        pred_starts.append(prediction_start)
        pred_ends.append(prediction_end)

    # tokenize prompts
    ids_list = []
    prompt_texts = []

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

        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=int(args.max_seq_len),
            return_attention_mask=False,
        )
        ids_list.append(enc["input_ids"])
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
    refined_max_len = int(max(1, getattr(args, "delta_text_max_len", 160)))
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
    refined_news_ids, refined_news_attn = _pad_2d_int(refined_ids_list, pad_id=tokenizer.pad_token_id)
    ts_patches, ts_patch_mask = _pad_patches(patches_list, patchmask_list, patch_len=patch_len)
    targets_z = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets_z_list], dim=0)
    # print("max = ", len_selected_news)
    rel_labels = torch.tensor(rel_labels_list, dtype=torch.float32)
    news_counts = torch.tensor(len_selected_news, dtype=torch.float32)
    retrieval_feats = torch.tensor(np.stack(retrieval_feature_list, axis=0), dtype=torch.float32)
    retrieval_delta_prior = torch.tensor(np.stack(retrieval_delta_prior_list, axis=0), dtype=torch.float32)
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
        retrieval_feats,
        retrieval_delta_prior,
        retrieval_meta_list,
        refined_news_ids,
        refined_news_attn,
    )

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
            _retrieval_feats,
            _retrieval_delta_prior,
            _retrieval_meta,
            _refined_news_ids,
            _refined_news_attn,
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
    case_bank=None,
    retrieval_mode_override: str | None = None,
    retrieval_enable_override: bool | None = None,
    retrieval_gate_only_override: bool | None = None,
    return_retrieval_stats: bool = False,
    api_adapter=None,
):
    """
    Residual evaluation:
      additive mode: final_pred = base_pred + gate * delta_pred
      relative mode: final_pred = base_pred * (1 + gate * delta_ratio)
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

    loss_sum, n_samples = 0.0, 0
    base_loss_sum = 0.0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0
    base_se_sum, base_ae_sum = 0.0, 0.0
    sample_mae = []
    sample_rel_labels = []
    sample_retrieval_meta = []

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{filename}.json")

    retrieval_gate_only = (
        bool(int(getattr(args, "case_retrieval_gate_only", 0)))
        if retrieval_gate_only_override is None
        else bool(retrieval_gate_only_override)
    )
    eval_desc = "[EVAL][RESIDUAL][TEST]" if testing else "[EVAL][RESIDUAL][VAL]"
    eval_loader, use_pbar = _eval_iter(data_loader, args, desc=eval_desc)
    for _, batch in enumerate(eval_loader):
        history_z, _, _ = _z_batch_tensors(batch, args, global_zstats=stats)
        history_z = history_z.to(device)
        base_pred = base_model(history_z).to(torch.float32)  # (B,H)
        base_pred_cpu = base_pred.detach().cpu()

        # build delta (with news)
        (
            ids_d,
            attn_d,
            ts_p,
            ts_pm,
            targets_z,
            metas,
            prompt_texts,
            rel_labels_d,
            news_counts,
            retrieval_feats_d,
            retrieval_delta_prior_d,
            retrieval_meta_d,
            refined_news_ids_d,
            refined_news_attn_d,
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
            force_no_news=False,
            news_dropout=news_dropout,
            case_bank=case_bank,
            query_base_pred=base_pred_cpu,
            retrieval_enable_override=retrieval_enable_override,
            retrieval_mode_override=retrieval_mode_override,
            api_adapter=api_adapter,
        )

        ids_d = ids_d.to(device)
        attn_d = attn_d.to(device)
        ts_p = ts_p.to(device)
        ts_pm = ts_pm.to(device)
        targets_z = targets_z.to(device)
        retrieval_feats_d = retrieval_feats_d.to(device=device, dtype=torch.float32)
        retrieval_delta_prior_d = retrieval_delta_prior_d.to(device=device, dtype=torch.float32)
        refined_news_ids_d = refined_news_ids_d.to(device)
        refined_news_attn_d = refined_news_attn_d.to(device)

        # delta pred: adapter on + with news
        out_delta = delta_model(
            input_ids=ids_d,
            attention_mask=attn_d,
            ts_patches=ts_p,
            ts_patch_mask=ts_pm,
            refined_news_input_ids=refined_news_ids_d,
            refined_news_attention_mask=refined_news_attn_d,
            targets=None,
            head_mode="delta",
            rel_targets=None,
            rel_lambda=0.0,
            retrieval_feats=retrieval_feats_d,
            retrieval_gate_only=retrieval_gate_only,
        )
        delta_corr = out_delta["pred"].to(torch.float32)
        delta_corr = _blend_with_retrieval_knn_prior(
            delta_pred=delta_corr,
            retrieval_feats=retrieval_feats_d,
            retrieval_delta_prior=retrieval_delta_prior_d,
            args=args,
            enabled=True,
        )
        rel_logits = out_delta["rel_logits"].to(torch.float32)
        if int(getattr(args, "news_gate_enable", 1)) == 1:
            gate = _bounded_sigmoid_gate(rel_logits, args)
        else:
            gate = torch.ones_like(rel_logits)

        targets_cpu = batch["target_value"].detach().cpu().numpy()  # raw

        bs = ids_d.size(0)
        targets_z_f = targets_z.to(torch.float32)
        base_pred_f = base_pred.to(torch.float32)
        delta_corr_f = delta_corr.to(torch.float32)
        gate_f = gate.to(torch.float32).unsqueeze(1)
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
        pred_z_cpu = pred_z.detach().cpu().numpy()
        base_pred_cpu = base_pred_f.detach().cpu().numpy()
        gate_cpu = gate_f.detach().cpu().numpy()
        loss_sum += float(loss.detach().cpu()) * bs
        base_loss_sum += float(base_loss.detach().cpu()) * bs
        n_samples += bs
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
            sample_mae.append(float(np.abs(pred - true).mean()))
            sample_rel_labels.append(float(rel_labels_d[i].detach().cpu()))
            sample_retrieval_meta.append(dict(retrieval_meta_d[i]))

            se_sum += float(((pred - true) ** 2).sum())
            ae_sum += float(np.abs(pred - true).sum())
            base_se_sum += float(((base_only - true) ** 2).sum())
            base_ae_sum += float(np.abs(base_only - true).sum())
            n_elems += int(args.horizon)

            if true_pred_csv_path is not None:
                with open(true_pred_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_denorm, true_vals))

            if testing:
                record = {
                    "test_prompt": prompt_texts[i],
                    "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                    "base_pred_z": [float(x) for x in base_pred_cpu[i].tolist()],
                    "pred": [float(x) for x in pred_denorm],
                    "base_pred": [float(x) for x in base_denorm],
                    "true": [float(x) for x in true_vals],
                    "gate_mean": float(gate_cpu[i].mean()),
                    "mu_global": mu_global,
                    "sigma_global": sigma_global,
                    "mu": mu_global,
                    "sigma": sigma_global,
                    "policy": str(policy_name),
                    "template_id": int(tpl_id),
                }
                with open(ans_json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    base_loss_avg = base_loss_sum / max(1, n_samples)
    base_mse_avg = base_se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    base_mae_avg = base_ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    if return_retrieval_stats:
        retrieval_stats = _summarize_retrieval_stats(
            sample_mae=sample_mae,
            rel_labels=sample_rel_labels,
            retrieval_meta=sample_retrieval_meta,
            strong_news_thresh=float(getattr(args, "case_retrieval_strong_news_thresh", 0.6)),
        )
        return (
            loss_avg,
            mse_avg,
            mae_avg,
            base_loss_avg,
            base_mse_avg,
            base_mae_avg,
            retrieval_stats,
        )
    return loss_avg, mse_avg, mae_avg, base_loss_avg, base_mse_avg, base_mae_avg


@torch.no_grad()
def run_retrieval_ablations(
    *,
    base_model,
    delta_model,
    tokenizer,
    data_loader,
    split_name: str,
    templates,
    tpl_id,
    args,
    global_zstats,
    news_df,
    policy_name,
    policy_kw,
    device,
    volatility_bin,
    case_bank,
    live_logger,
    api_adapter=None,
):
    if case_bank is None or len(case_bank.get("cases", [])) == 0:
        if live_logger is not None:
            live_logger.info(f"[ABLATION][{split_name}] skip: no active case bank.")
        return {}

    plans = [
        {"name": "no_retrieval", "enable": False, "mode": "off", "gate_only": False},
        {"name": "price_only", "enable": True, "mode": "price", "gate_only": False},
        {"name": "price_event", "enable": True, "mode": "price_event", "gate_only": False},
        {"name": "random_retrieval", "enable": True, "mode": "random", "gate_only": False},
        {"name": "gate_only", "enable": True, "mode": "price_event", "gate_only": True},
    ]
    results = {}
    for p in plans:
        (
            loss_v,
            mse_v,
            mae_v,
            _base_loss_v,
            _base_mse_v,
            _base_mae_v,
            retrieval_stats,
        ) = evaluate_metrics_residual(
            base_model=base_model,
            delta_model=delta_model,
            tokenizer=tokenizer,
            data_loader=data_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin,
            testing=False,
            true_pred_csv_path=None,
            news_dropout=False,
            filename=None,
            case_bank=case_bank,
            retrieval_mode_override=p["mode"],
            retrieval_enable_override=p["enable"],
            retrieval_gate_only_override=p["gate_only"],
            return_retrieval_stats=True,
            api_adapter=api_adapter,
        )
        metric = _select_metric(loss_v, mse_v, mae_v, args.select_metric)
        results[p["name"]] = {
            "loss": float(loss_v),
            "mse": float(mse_v),
            "mae": float(mae_v),
            "metric": float(metric),
            "retrieval": retrieval_stats,
        }

    base_metric = float(results["no_retrieval"]["metric"])
    if live_logger is not None:
        live_logger.info(f"[ABLATION][{split_name}] metric={str(args.select_metric).lower()}")
        for name in ["no_retrieval", "price_only", "price_event", "random_retrieval", "gate_only"]:
            rec = results[name]
            delta = rec["metric"] - base_metric
            rs = rec["retrieval"]
            live_logger.info(
                f"[ABLATION][{split_name}] {name} "
                f"metric={rec['metric']:.6f} delta_vs_no={delta:+.6f} "
                f"coverage={float(rs.get('coverage_rate', 0.0)):.3f} "
                f"mae_valid={float(rs.get('mae_valid', float('nan'))):.6f} "
                f"mae_rejected={float(rs.get('mae_rejected', float('nan'))):.6f} "
                f"strong_valid={float(rs.get('mae_strong_news_valid', float('nan'))):.6f} "
                f"strong_rej={float(rs.get('mae_strong_news_rejected', float('nan'))):.6f}"
            )
    return results


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
    if not bool(getattr(args, "_refine_cache_enabled", False)):
        return
    if int(getattr(args, "news_refine_prewarm", 1)) != 1:
        if live_logger is not None:
            live_logger.info("[NEWS_REFINE_CACHE] prewarm disabled.")
        return
    if news_df is None or len(news_df) == 0:
        return

    text_col = str(getattr(args, "news_text_col", "content"))
    if text_col not in news_df.columns:
        if live_logger is not None:
            live_logger.info(f"[NEWS_REFINE_CACHE] prewarm skipped: text_col not found: {text_col}")
        return

    max_items = int(getattr(args, "news_refine_prewarm_max_batches", -1))
    before_n = len(getattr(args, "_refine_cache_store", {}))
    hit0 = int(getattr(args, "_refine_cache_hits", 0))
    miss0 = int(getattr(args, "_refine_cache_misses", 0))

    time_col = str(getattr(args, "news_time_col", "date"))
    in_scope = news_df
    if time_col in news_df.columns:
        lo = None
        hi = None
        for df in [train_df, val_df, test_df]:
            if isinstance(df, pd.DataFrame) and (args.time_col in df.columns) and len(df) > 0:
                ts = pd.to_datetime(df[args.time_col], errors="coerce", dayfirst=args.dayFirst)
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
            nts = pd.to_datetime(news_df[time_col], errors="coerce", dayfirst=args.dayFirst)
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

    show_progress = int(getattr(args, "news_refine_show_progress", 1)) == 1
    if live_logger is not None:
        live_logger.info(
            "[NEWS_REFINE_CACHE] prewarm start: "
            f"news_rows={len(news_df)}, in_scope_rows={len(in_scope)}, unique_docs={len(uniq)}, "
            f"show_progress={int(show_progress)}"
        )

    pbar = tqdm(
        uniq,
        desc="[DELTA][REFINE_PREWARM_NEWS]",
        leave=show_progress,
        dynamic_ncols=True,
        mininterval=0.3,
        disable=(not show_progress),
    )
    news_budget = int(args.token_budget * args.token_budget_news_frac)
    total_docs = len(uniq)
    for idx, raw_text in enumerate(pbar, start=1):
        _ = _refine_one_news_doc(
            raw_news_text=raw_text,
            tokenizer=tokenizer,
            max_tokens=news_budget,
            args=args,
            api_adapter=api_adapter,
        )
        if show_progress and (idx == 1 or idx % 10 == 0 or idx == total_docs):
            hit_now = int(getattr(args, "_refine_cache_hits", 0))
            miss_now = int(getattr(args, "_refine_cache_misses", 0))
            pbar.set_postfix(
                {
                    "cache_hit": max(0, hit_now - hit0),
                    "refined": max(0, miss_now - miss0),
                },
                refresh=False,
            )
    _save_refine_cache(args, live_logger=live_logger, force=True)
    after_n = len(getattr(args, "_refine_cache_store", {}))
    hit1 = int(getattr(args, "_refine_cache_hits", 0))
    miss1 = int(getattr(args, "_refine_cache_misses", 0))
    if live_logger is not None:
        live_logger.info(
            "[NEWS_REFINE_CACHE] prewarm done: "
            f"news_rows={len(news_df)}, in_scope_rows={len(in_scope)}, unique_docs={len(uniq)}, "
            f"entries_before={before_n}, entries_after={after_n}, "
            f"hits_delta={hit1 - hit0}, misses_delta={miss1 - miss0}"
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
        val_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )
    test_loader = make_loader(
        test_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )

    # news
    
    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])

    news_df[args.news_time_col] = pd.to_datetime(news_df[args.news_time_col], dayfirst=args.dayFirst)
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)
        print(len(news_df))
        # 去除空的总结后的新闻
        col = args.news_text_col
        news_df = news_df.loc[
            news_df[col].fillna("").astype(str).str.strip().ne("")
        ].reset_index(drop=True)
        print(len(news_df))

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
    global_zstats_bundle = _coerce_global_zstats(bundle.get("global_zstats", None), args, required=True)
    news_api_adapter = bundle.get("news_api_adapter", None)
    _init_refine_cache(args, live_logger=live_logger)
    

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
        delta_internal_gate=bool(int(getattr(args, "delta_internal_gate", 1))),
        delta_clip=float(getattr(args, "delta_clip", 3.0)),
        delta_news_tail_tokens=int(getattr(args, "delta_news_tail_tokens", 160)),
        delta_rel_floor=float(getattr(args, "delta_rel_floor", 0.05)),
        retrieval_feat_dim=int(getattr(args, "case_retrieval_feature_dim", 12)),
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
        "delta_text_ln",
        "text_proj",
        "text_delta_head",
        "text_gate",
        "rel_head",
        "retrieval_proj",
        "retrieval_gate_bias",
        "retrieval_rel_bias",
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

    active_case_bank = None
    retrieval_mode_active = _resolve_retrieval_mode(args)
    if retrieval_mode_active != "off":
        case_loader = make_loader(
            bundle["train_df"],
            args.time_col,
            args.value_col,
            args.history_len,
            args.horizon,
            args.stride,
            args.batch_size,
            shuffle=False,
            id_col=args.id_col,
            dayFirst=args.dayFirst,
        )
        active_case_bank = build_case_bank(
            train_loader=case_loader,
            base_model=base_backbone,
            args=args,
            global_zstats=global_zstats,
            news_df=news_df,
            tokenizer=tokenizer,
            policy_name=policy_name,
            policy_kw=policy_kw,
            live_logger=live_logger,
            api_adapter=news_api_adapter,
        )
        if int(getattr(args, "case_retrieval_save_bank", 1)) == 1:
            case_bank_path = os.path.join(
                bundle["ckpt_dir"], f"case_bank_train_{args.taskName}.json"
            )
            save_case_bank(case_bank_path, active_case_bank)
            live_logger.info(f"[CASE_BANK] saved to {case_bank_path}")
    else:
        live_logger.info("[CASE_BANK] retrieval disabled; skip active case bank build.")

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
    retrieval_gate_only = bool(int(getattr(args, "case_retrieval_gate_only", 0)))
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
            },
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
                base_pred_cpu = base_pred.detach().cpu()

            # build delta inputs (with news)
            (
                ids_d,
                attn_d,
                ts_p,
                ts_pm,
                targets_z,
                metas,
                prompt_texts_d,
                _rel_labels_d,
                news_counts_d,
                retrieval_feats_d,
                retrieval_delta_prior_d,
                retrieval_meta_d,
                refined_news_ids_d,
                refined_news_attn_d,
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
                epoch=epoch,
                record_train_prompt=True,
                testing=False,
                force_no_news=False,
                news_dropout=True,
                prompt_path=bundle["prompt_path"],
                case_bank=active_case_bank,
                query_base_pred=base_pred_cpu,
                api_adapter=news_api_adapter,
            )
            # build base text inputs (no news)
            (
                ids_b,
                attn_b,
                _ts_b,
                _ts_pm_b,
                _targets_b,
                _metas_b,
                _prompt_b,
                _rel_b,
                _news_counts_b,
                retrieval_feats_b,
                retrieval_delta_prior_b,
                _retrieval_meta_b,
                refined_news_ids_b,
                refined_news_attn_b,
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
                epoch=epoch,
                record_train_prompt=False,
                testing=False,
                force_no_news=True,
                news_dropout=False,
                case_bank=active_case_bank,
                query_base_pred=base_pred_cpu,
                retrieval_enable_override=False,
                api_adapter=news_api_adapter,
            )

            ids_d = ids_d.to(device)
            attn_d = attn_d.to(device)
            ids_b = ids_b.to(device)
            attn_b = attn_b.to(device)

            ts_p = ts_p.to(device)
            ts_pm = ts_pm.to(device)
            targets_z = targets_z.to(device)
            news_counts_d = news_counts_d.to(device=device, dtype=torch.float32)
            retrieval_feats_d = retrieval_feats_d.to(device=device, dtype=torch.float32)
            retrieval_delta_prior_d = retrieval_delta_prior_d.to(device=device, dtype=torch.float32)
            retrieval_feats_b = retrieval_feats_b.to(device=device, dtype=torch.float32)
            retrieval_delta_prior_b = retrieval_delta_prior_b.to(device=device, dtype=torch.float32)
            refined_news_ids_d = refined_news_ids_d.to(device)
            refined_news_attn_d = refined_news_attn_d.to(device)
            refined_news_ids_b = refined_news_ids_b.to(device)
            refined_news_attn_b = refined_news_attn_b.to(device)
            has_news = (news_counts_d > 0).to(dtype=torch.float32)

            delta_model.train()

            delta_targets = _build_delta_targets(
                targets_z=targets_z,
                base_pred=base_pred,
                mu_global=float(global_zstats["mu_global"]),
                sigma_global=float(global_zstats["sigma_global"]),
                args=args,
            )

            out_delta = delta_model(
                input_ids=ids_d,
                attention_mask=attn_d,
                ts_patches=ts_p,
                ts_patch_mask=ts_pm,
                refined_news_input_ids=refined_news_ids_d,
                refined_news_attention_mask=refined_news_attn_d,
                targets=None,
                head_mode="delta",
                rel_targets=None,
                rel_lambda=0.0,
                retrieval_feats=retrieval_feats_d,
                retrieval_gate_only=retrieval_gate_only,
            )
            delta_pred_real = out_delta["pred"].to(torch.float32)
            delta_pred_real = _blend_with_retrieval_knn_prior(
                delta_pred=delta_pred_real,
                retrieval_feats=retrieval_feats_d,
                retrieval_delta_prior=retrieval_delta_prior_d,
                args=args,
                enabled=True,
            )
            rel_logits_real = out_delta["rel_logits"].to(torch.float32)

            if int(getattr(args, "news_gate_enable", 1)) == 1:
                gate_real = _bounded_sigmoid_gate(rel_logits_real, args)
            else:
                gate_real = torch.ones_like(rel_logits_real)
            gate_real_h = gate_real.unsqueeze(1)

            pred_real_z = _fuse_base_and_delta(
                base_pred_z=base_pred,
                delta_pred=delta_pred_real,
                gate_h=gate_real_h,
                args=args,
                mu_global=float(global_zstats["mu_global"]),
                sigma_global=float(global_zstats["sigma_global"]),
            )
            targets_z_typed = targets_z.to(torch.float32)

            residual_mode = str(getattr(args, "residual_loss", "mae")).lower()
            loss_final = _point_loss(pred_real_z, targets_z_typed, mode=residual_mode)

            delta_aux_lambda = float(getattr(args, "delta_aux_lambda", 0.0))
            if delta_aux_lambda > 0.0:
                delta_aux_mode = str(getattr(args, "delta_aux_loss", residual_mode)).lower()
                loss_delta_aux = _point_loss(delta_pred_real, delta_targets, mode=delta_aux_mode)
            else:
                loss_delta_aux = torch.zeros((), device=device, dtype=torch.float32)

            gate_reg_lambda = float(getattr(args, "delta_gate_reg_lambda", getattr(args, "gate_lambda", 0.0)))
            loss_gate_reg = gate_real.mean()

            delta_cf_lambda = float(getattr(args, "delta_cf_lambda", 0.0))
            delta_cf_margin = float(getattr(args, "delta_cf_margin", 0.0))
            delta_null_lambda = float(getattr(args, "delta_null_lambda", 0.05))
            null_step_scale = _step_ramp_scale(
                step_idx=global_step,
                warmup_steps=delta_null_warmup_steps,
                ramp_steps=delta_null_ramp_steps,
            )
            gate_reg_lambda_eff = gate_reg_lambda * epoch_reg_scale
            delta_cf_lambda_eff = delta_cf_lambda * epoch_reg_scale
            delta_null_lambda_eff = delta_null_lambda * epoch_reg_scale * null_step_scale
            loss_cf = torch.zeros((), device=device, dtype=torch.float32)
            loss_null = torch.zeros((), device=device, dtype=torch.float32)

            if delta_cf_lambda_eff > 0.0 or delta_null_lambda_eff > 0.0:
                out_null = delta_model(
                    input_ids=ids_b,
                    attention_mask=attn_b,
                    ts_patches=ts_p,
                    ts_patch_mask=ts_pm,
                    refined_news_input_ids=refined_news_ids_b,
                    refined_news_attention_mask=refined_news_attn_b,
                    targets=None,
                    head_mode="delta",
                    rel_targets=None,
                    rel_lambda=0.0,
                    retrieval_feats=retrieval_feats_b,
                    retrieval_gate_only=retrieval_gate_only,
                )
                delta_pred_null = out_null["pred"].to(torch.float32)
                delta_pred_null = _blend_with_retrieval_knn_prior(
                    delta_pred=delta_pred_null,
                    retrieval_feats=retrieval_feats_b,
                    retrieval_delta_prior=retrieval_delta_prior_b,
                    args=args,
                    enabled=True,
                )
                rel_logits_null = out_null["rel_logits"].to(torch.float32)
                if int(getattr(args, "news_gate_enable", 1)) == 1:
                    gate_null = _bounded_sigmoid_gate(rel_logits_null, args)
                else:
                    gate_null = torch.ones_like(rel_logits_null)
                gate_null_h = gate_null.unsqueeze(1)
                pred_null_z = _fuse_base_and_delta(
                    base_pred_z=base_pred,
                    delta_pred=delta_pred_null,
                    gate_h=gate_null_h,
                    args=args,
                    mu_global=float(global_zstats["mu_global"]),
                    sigma_global=float(global_zstats["sigma_global"]),
                )

                err_real = torch.abs(pred_real_z - targets_z_typed).mean(dim=1)
                err_null = torch.abs(pred_null_z - targets_z_typed).mean(dim=1)
                loss_cf = torch.relu(delta_cf_margin + err_real - err_null).mean()
                loss_null = (gate_null_h * delta_pred_null).pow(2).mean()

            loss_total = (
                loss_final
                + delta_aux_lambda * loss_delta_aux
                + gate_reg_lambda_eff * loss_gate_reg
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
                retr_cov = 0.0
                retr_conf = 0.0
                if len(retrieval_meta_d) > 0:
                    retr_cov = float(
                        np.mean([1.0 if bool(m.get("retrieval_valid", False)) else 0.0 for m in retrieval_meta_d])
                    )
                    retr_conf = float(
                        np.mean([float(m.get("retrieval_confidence", 0.0)) for m in retrieval_meta_d])
                    )
                pbar.set_postfix(
                    train_loss=f"{avg_train_loss:.6f}",
                    gate=float(gate_real.mean().detach().cpu()),
                    final=float(loss_final.detach().cpu()),
                    aux=float(loss_delta_aux.detach().cpu()),
                    cf=float(loss_cf.detach().cpu()),
                    rcov=float(retr_cov),
                    rconf=float(retr_conf),
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
                    live_logger.info(
                            f"[GRAD] step={global_step} delta_head_grad={dh_grad:.6f} "
                            f"delta_scale={float(delta_model.delta_log_scale.exp().detach().cpu()):.4f} "
                            f"reg={epoch_reg_scale:.3f} null={null_step_scale:.3f} "
                            f"lam(g/c/n)=("
                            f"{gate_reg_lambda_eff:.4f}/{delta_cf_lambda_eff:.4f}/{delta_null_lambda_eff:.4f})"
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
                case_bank=active_case_bank,
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

        if epoch == 0:
            live_logger.info("---------------------trainset and valset prompt statistics--------------------------------")
            print_prompt_stats(live_logger, dataStatistic)
            live_logger.info("-----------------------------------------------------")

    if not has_saved_delta:
        final_epoch = max(0, int(delta_epochs) - 1)
        if delta_val_mode == "end_only":
            val_loss, val_mse, val_mae, base_val_loss, base_val_mse, base_val_mae = evaluate_metrics_residual(
                base_model=base_backbone,
                delta_model=delta_model,
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
                case_bank=active_case_bank,
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
            live_logger.info(
                f"[DELTA][VAL] end_only: val_loss(zMSE)={val_loss:.6f} "
                f"val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
            )
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
                case_bank=active_case_bank,
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
            live_logger.info(
                f"[DELTA][VAL] fallback eval: val_loss(zMSE)={val_loss:.6f} "
                f"val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
            )
            live_logger.info(
                f"[DELTA][BASE_ONLY][VAL] fallback: loss(zMAE)={base_val_loss:.6f} "
                f"mse(raw)={base_val_mse:.6f} mae(raw)={base_val_mae:.6f}"
            )

    dataStatistic.clear()
    _save_refine_cache(args, live_logger=live_logger, force=True)

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
        if int(getattr(args, "case_retrieval_run_ablations", 1)) == 1:
            ablation_split = str(getattr(args, "case_retrieval_ablation_split", "val")).lower().strip()
            if ablation_split in {"val", "both"}:
                run_retrieval_ablations(
                    base_model=base_backbone,
                    delta_model=model_best,
                    tokenizer=tokenizer,
                    data_loader=val_loader,
                    split_name="VAL",
                    templates=templates,
                    tpl_id=tpl_for_test,
                    args=args,
                    global_zstats=global_zstats_eval,
                    news_df=news_df,
                    policy_name=pol_for_test,
                    policy_kw=policy_kw,
                    device=device,
                    volatility_bin=volatility_bin_val,
                    case_bank=active_case_bank,
                    live_logger=live_logger,
                    api_adapter=news_api_adapter,
                )
            if ablation_split in {"test", "both"}:
                run_retrieval_ablations(
                    base_model=base_backbone,
                    delta_model=model_best,
                    tokenizer=tokenizer,
                    data_loader=test_loader,
                    split_name="TEST",
                    templates=templates,
                    tpl_id=tpl_for_test,
                    args=args,
                    global_zstats=global_zstats_eval,
                    news_df=news_df,
                    policy_name=pol_for_test,
                    policy_kw=policy_kw,
                    device=device,
                    volatility_bin=volatility_bin_test,
                    case_bank=active_case_bank,
                    live_logger=live_logger,
                    api_adapter=news_api_adapter,
                )
        test_loss, test_mse, test_mae, base_test_loss, base_test_mse, base_test_mae = evaluate_metrics_residual(
            base_model=base_backbone,
            delta_model=model_best,
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
            case_bank=active_case_bank,
            api_adapter=news_api_adapter,
        )

        live_logger.info("---------------------testset prompt statistics--------------------------------")
        print_prompt_stats(live_logger, dataStatistic)
        live_logger.info("-----------------------------------------------------")

        tqdm.write(
            f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
        )
        live_logger.info(
            f"[TEST][FINAL] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
        )
        live_logger.info(
            f"[TEST][BASE_ONLY] loss(zMAE)={base_test_loss:.6f} mse(raw)={base_test_mse:.6f} "
            f"mae(raw)={base_test_mae:.6f}"
        )

        record_test_results_csv(args, live_logger, test_mse, test_mae)
        draw_pred_true(live_logger, args, true_pred_csv_path)
        _save_refine_cache(args, live_logger=live_logger, force=True)

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

        live_logger.info("---------------------testset prompt statistics--------------------------------")
        print_prompt_stats(live_logger, dataStatistic)
        live_logger.info("-----------------------------------------------------")

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

    with open(prompt_path, "w", encoding="utf-8"):
        pass
    with open(ans_json_path, "w", encoding="utf-8"):
        pass
    with open(true_pred_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pred", "true"])

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
