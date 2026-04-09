from __future__ import annotations

import csv
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_construction.DataStatistic import DataStatistic
from ..data_construction.prompt import build_prompt
from ..delta_news_hooks import extract_structured_events, format_structured_events_for_prompt, refine_news_text
from ..news_rules import get_candidates, rerank_selected_news_by_utility, select_news
from ..refine_cache_utils import build_refine_context
from ..utils.batch_utils import _batch_time_seq_for_sample
from ..utils.utils import print_prompt_stats
from ..delta.core import _resolve_delta_sign_mode
from ..refine.cache import (
    _align_ts_to_ref_tz,
    _ascii_table,
    _cache_decision_rows,
    _extract_structured_events_from_refined_docs_detailed,
    _merge_refined_news_docs,
    _news_cache_is_read_only,
    _refine_news_docs_aligned_from_doc_cache,
    _selected_news_meta_records,
    _structured_events_to_feature_vec,
)

dataStatistic = DataStatistic()

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
        "sign_logits",
        "sign_soft",
        "state_logits",
        "state_score",
        "magnitude",
        "magnitude_raw",
        "news_count",
        "candidate_news_count",
        "history_range_news_count",
        "selected_news_in_history_range_count",
        "news_max_utility",
        "temporal_text_source",
        "temporal_text_doc_total",
        "temporal_text_doc_nonempty",
        "temporal_text_doc_attached_any_step",
        "temporal_text_doc_pre_history",
        "temporal_text_doc_in_history_range",
        "temporal_text_doc_post_history_pre_target",
        "temporal_text_step_nonempty_count",
        "temporal_text_step_tokenized_count",
        "temporal_text_step_total",
        "temporal_text_all_nonempty_docs_attached",
        "base_residual_abs",
        "delta_helped",
        "direction_correct",
        "confidence_value",
        "regime_route",
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
    is_secret = any(tok in k for tok in ["api_key", "access_token", "auth_token", "refresh_token", "secret", "password"])
    if (not is_secret) and k.endswith("_token") and ("budget" not in k):
        is_secret = True
    if not is_secret:
        return value
    s = str(value or "")
    if not s:
        return ""
    if len(s) <= 8:
        return "***"
    return f"{s[:4]}***{s[-4:]}"

def _pair_rows_for_ascii_table(rows: list[list[str]], *, headers: list[str]) -> tuple[list[str], list[list[str]]]:
    left_headers = [str(headers[0]), str(headers[1])]
    right_headers = [str(headers[0]), str(headers[1])]
    table_headers = left_headers + right_headers
    body = []
    idx = 0
    while idx < len(rows):
        left = [str(x) for x in rows[idx]]
        idx += 1
        if idx < len(rows):
            right = [str(x) for x in rows[idx]]
            idx += 1
        else:
            right = ["", ""]
        body.append(left + right)
    return table_headers, body

def _run_arg_section_name(key: str) -> str:
    k = str(key or "").strip()
    if not k:
        return "Other"

    if k in {
        "taskName",
        "stage",
        "seed",
        "gpu",
        "save_dir",
        "select_metric",
        "early_stop_patience",
        "eval_progress_bar",
        "eval_progress_leave",
        "run_name",
    }:
        return "Basic Setup"

    if k.startswith("base_"):
        return "Base Backbone"

    if k.startswith("delta_temporal_text") or k == "temporal_text_model_id":
        return "Temporal Text Input"

    if (
        k.startswith("delta_sign_external")
        or k in {"delta_sign_mode", "delta_sign_eps", "delta_sign_tau"}
    ):
        return "DELTA / SignNet"

    if (
        k.startswith("delta_structured")
        or k.startswith("cleaned_residual")
        or k == "delta_include_structured_news"
        or k.startswith("news_structured_")
    ):
        return "Structured News"

    if (
        k.startswith("news_")
        or k.startswith("utility_")
        or k.startswith("smart_")
        or k == "default_policy"
    ):
        return "News / Cache / API"

    if k in {
        "train_file",
        "val_file",
        "test_file",
        "time_col",
        "value_col",
        "id_col",
        "freq_min",
        "region",
        "unit",
        "description",
        "dayFirst",
        "history_len",
        "horizon",
        "stride",
        "volatility_bin_tiers",
        "token_budget",
        "token_budget_news_frac",
        "news_topM",
        "news_topK",
        "patch_len",
    }:
        return "Data & Window"

    if (
        k.startswith("delta_")
        or k in {
            "rel_lambda",
            "residual_loss",
            "residual_base_frac",
            "doc_candidate_mode",
            "patch_dropout",
            "head_dropout",
            "head_mlp",
            "news_usefulness_weighting",
        }
    ):
        return "DELTA Core"

    if k in {"batch_size", "lr", "weight_decay"}:
        return "Optimization"

    return "Other"

def _section_order_for_run_args() -> list[str]:
    return [
        "Basic Setup",
        "Data & Window",
        "Base Backbone",
        "DELTA Core",
        "DELTA / SignNet",
        "Temporal Text Input",
        "Structured News",
        "News / Cache / API",
        "Optimization",
        "Other",
    ]

def _ordered_run_arg_sections(grouped_rows: dict[str, list[list[str]]]) -> list[str]:
    ordered = []
    seen = set()
    for section in list(_section_order_for_run_args()) + sorted(grouped_rows.keys()):
        if section in seen:
            continue
        seen.add(section)
        if not grouped_rows.get(section):
            continue
        ordered.append(section)
    return ordered

def _log_run_args(args, live_logger):
    if live_logger is None:
        return
    grouped_rows = {}
    for key in sorted(k for k in vars(args).keys() if not str(k).startswith("_")):
        value = getattr(args, key)
        value = _mask_sensitive_arg(key, value)
        section = _run_arg_section_name(key)
        grouped_rows.setdefault(section, []).append([str(key), str(value)])
    live_logger.info("-----------------------------------------------------")
    live_logger.info("[CONFIG] Parameters")
    for section in _ordered_run_arg_sections(grouped_rows):
        rows = grouped_rows.get(section, [])
        headers, body = _pair_rows_for_ascii_table(rows, headers=["Parameter", "Value"])
        live_logger.info(f"[CONFIG][SECTION] {section}")
        live_logger.info("\n" + _ascii_table(headers, body, max_col_widths=[28, 36, 28, 36]))
    live_logger.info("-----------------------------------------------------")


def _log_cache_decision(args, live_logger):
    if live_logger is None:
        return
    rows = _cache_decision_rows(args)
    headers, body = _pair_rows_for_ascii_table(rows, headers=["Field", "Value"])
    live_logger.info("-----------------------------------------------------")
    live_logger.info("[CONFIG] Cache Decision")
    live_logger.info("\n" + _ascii_table(headers, body, max_col_widths=[18, 42, 18, 42]))
    live_logger.info("-----------------------------------------------------")

def _log_enabled_mechanisms(args, live_logger, stage: str):
    if live_logger is None:
        return
    stage_norm = str(stage or "").lower().strip()
    delta_active = stage_norm in {"delta", "all"}
    news_api_enabled = int(getattr(args, "news_api_enable", 0)) == 1 or any(
        str(getattr(args, attr, "off") or "off").lower().strip() == "api"
        for attr in ["news_refine_mode", "news_structured_mode", "hard_reflection_mode"]
    )
    structured_enabled = int(getattr(args, "delta_structured_enable", 0)) == 1

    live_logger.info(
        f"[MECH][NEWS_API] {'enabled' if news_api_enabled else 'disabled'}: "
        f"refine_mode={getattr(args, 'news_refine_mode', 'na')} "
        f"structured_mode={getattr(args, 'news_structured_mode', 'na')}"
    )
    live_logger.info(
        f"[MECH][STRUCTURED] {'enabled' if structured_enabled else 'disabled'}: "
        f"delta_structured_feature_dim={int(getattr(args, 'delta_structured_feature_dim', 0) or 0)}"
    )
    if delta_active:
        live_logger.info(
            "[MECH][DELTA_PROMPT] skipped in DELTA model path: template prompt tokens are not "
            "consumed by tiny_news_ts; news enters through structured features only."
        )
    sign_mode = _resolve_delta_sign_mode(args)
    if sign_mode == "signnet_binary" and delta_active:
        live_logger.info(
            "[MECH][DELTA_SIGN] external signnet enabled: "
            "an independent sign classifier is pretrained first and then replaces DELTA's internal sign path."
        )
    else:
        live_logger.info(
            f"[MECH][DELTA_SIGN] internal sign path only (mode={sign_mode}); "
            "DELTA training itself is supervised by final prediction loss."
        )

def _format_ts_range(ts_min, ts_max) -> str:
    if ts_min is None or ts_max is None or pd.isna(ts_min) or pd.isna(ts_max):
        return "N/A"
    return f"{pd.Timestamp(ts_min)} -> {pd.Timestamp(ts_max)}"

def _df_series_time_range(df: pd.DataFrame | None, time_col: str):
    if df is None or not isinstance(df, pd.DataFrame) or time_col not in df.columns or len(df) == 0:
        return None, None
    s = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if len(s) == 0:
        return None, None
    return s.min(), s.max()

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

def _split_time_order_issues(
    train_df: pd.DataFrame | None,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    *,
    time_col: str,
) -> list[str]:
    issues = []
    _, train_max = _df_series_time_range(train_df, time_col)
    val_min, val_max = _df_series_time_range(val_df, time_col)
    test_min, _ = _df_series_time_range(test_df, time_col)

    if train_max is not None and val_min is not None and pd.Timestamp(train_max) >= pd.Timestamp(val_min):
        issues.append(
            "[DATA_SPLIT] raw TRAIN/VAL ranges overlap or are out of order: "
            f"train_max={pd.Timestamp(train_max)} val_min={pd.Timestamp(val_min)}"
        )
    if val_max is not None and test_min is not None and pd.Timestamp(val_max) >= pd.Timestamp(test_min):
        issues.append(
            "[DATA_SPLIT] raw VAL/TEST ranges overlap or are out of order: "
            f"val_max={pd.Timestamp(val_max)} test_min={pd.Timestamp(test_min)}"
        )
    return issues

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


def _count_news_rows_in_time_range(news_df, time_col: str, start_time, end_time) -> int:
    if news_df is None or len(news_df) == 0 or time_col not in news_df.columns:
        return 0
    ts_series = news_df[time_col]
    start_ts = _align_ts_to_ref_tz(start_time, ts_series)
    end_ts = _align_ts_to_ref_tz(end_time, ts_series)
    if pd.isna(start_ts) or pd.isna(end_ts):
        return 0
    lo = min(start_ts, end_ts)
    hi = max(start_ts, end_ts)
    return int(((ts_series >= lo) & (ts_series <= hi)).sum())


def _count_selected_news_in_history_range(selected: pd.DataFrame, *, time_col: str, history_times: list[str]) -> int:
    if selected is None or len(selected) == 0 or time_col not in selected.columns or len(history_times) == 0:
        return 0
    hist_series = pd.Series(pd.to_datetime(history_times, errors="coerce"))
    hist_valid = hist_series.dropna()
    if len(hist_valid) == 0:
        return 0
    selected_ts = pd.to_datetime(selected[time_col], errors="coerce")
    if getattr(hist_series.dt, "tz", None) is not None:
        selected_ts = selected_ts.apply(
            lambda x: _align_ts_to_ref_tz(x, hist_series) if not pd.isna(x) else x
        )
    lo = hist_valid.min()
    hi = hist_valid.max()
    return int(((selected_ts >= lo) & (selected_ts <= hi)).sum())


def _summarize_temporal_text_alignment(
    *,
    history_times: list[str],
    target_time,
    news_metas: list[dict],
    news_docs: list[str],
) -> dict[str, int]:
    hist_series = pd.Series(pd.to_datetime(history_times, errors="coerce"))
    hist_valid = hist_series.dropna()
    if len(hist_valid) == 0:
        history_start = pd.NaT
        history_end = pd.NaT
    else:
        history_start = hist_valid.min()
        history_end = hist_valid.max()
    target_ts = _align_ts_to_ref_tz(target_time, hist_series) if len(hist_series) > 0 else pd.to_datetime(target_time, errors="coerce")
    step_times = hist_series.tolist()

    doc_total = 0
    doc_nonempty = 0
    doc_attached_any_step = 0
    doc_pre_history = 0
    doc_in_history_range = 0
    doc_post_history_pre_target = 0

    for meta, doc_txt in zip(list(news_metas or []), list(news_docs or [])):
        doc_total += 1
        clean = str(doc_txt or "").strip()
        if not clean:
            continue
        doc_nonempty += 1
        raw_ts = meta.get("date", "") if isinstance(meta, dict) else ""
        doc_ts = _align_ts_to_ref_tz(raw_ts, hist_series)

        if not pd.isna(doc_ts):
            if not pd.isna(history_start) and doc_ts < history_start:
                doc_pre_history += 1
            elif not pd.isna(history_end) and doc_ts <= history_end:
                doc_in_history_range += 1
            elif not pd.isna(target_ts) and doc_ts < target_ts:
                doc_post_history_pre_target += 1

        attached = False
        for step_ts in step_times:
            if pd.isna(step_ts):
                continue
            if pd.isna(doc_ts) or doc_ts <= step_ts:
                attached = True
                break
        if attached:
            doc_attached_any_step += 1

    return {
        "temporal_text_doc_total": int(doc_total),
        "temporal_text_doc_nonempty": int(doc_nonempty),
        "temporal_text_doc_attached_any_step": int(doc_attached_any_step),
        "temporal_text_doc_pre_history": int(doc_pre_history),
        "temporal_text_doc_in_history_range": int(doc_in_history_range),
        "temporal_text_doc_post_history_pre_target": int(doc_post_history_pre_target),
    }

def _build_temporal_text_series_for_sample(
    *,
    history_times: list[str],
    news_metas: list[dict],
    news_docs: list[str],
    tokenizer,
    max_tokens: int,
    per_step_topk: int,
) -> list[str]:
    L = len(history_times)
    if L <= 0:
        return []

    hist_series = pd.Series(pd.to_datetime(history_times, errors="coerce"))
    step_times = hist_series.tolist()
    topk = int(max(1, per_step_topk))
    doc_items = []
    for meta, doc_txt in zip(list(news_metas or []), list(news_docs or [])):
        clean = str(doc_txt or "").strip()
        if not clean:
            continue
        raw_ts = ""
        if isinstance(meta, dict):
            raw_ts = meta.get("date", "")
        ts = _align_ts_to_ref_tz(raw_ts, hist_series)
        doc_items.append((ts, clean))

    out = []
    for step_ts in step_times:
        if pd.isna(step_ts) or len(doc_items) == 0:
            out.append("")
            continue
        eligible = []
        for doc_ts, clean in doc_items:
            if pd.isna(doc_ts) or doc_ts <= step_ts:
                eligible.append((doc_ts, clean))
        if len(eligible) == 0:
            out.append("")
            continue
        eligible.sort(
            key=lambda item: (
                pd.Timestamp.min if pd.isna(item[0]) else item[0]
            ),
            reverse=True,
        )
        merged = _merge_refined_news_docs(
            [clean for _, clean in eligible[:topk]],
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        out.append(str(merged or "").strip())
    return out

def _tokenize_temporal_text_series(
    temporal_text_series: list[list[str]],
    *,
    tokenizer,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokenizer is None:
        raise ValueError("tokenizer is required for temporal text tokenization")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    ids_3d = []
    max_len = int(max(1, max_length))
    for seq in list(temporal_text_series or []):
        step_ids = []
        for step_text in list(seq or []):
            clean = str(step_text or "").strip()
            if not clean:
                step_ids.append([])
                continue
            try:
                enc = tokenizer(
                    clean,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_len,
                    return_attention_mask=False,
                )
                ids = enc.get("input_ids", []) if hasattr(enc, "get") else []
            except Exception:
                ids = []
            step_ids.append(list(ids or []))
        ids_3d.append(step_ids)
    return _pad_3d_int(ids_3d, pad_id=pad_id)

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
    testing: bool = False,
    force_no_news: bool = False,
    news_dropout: bool = False,
    api_adapter=None,
    temporal_text_tokenizer=None,
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
    need_prompt_context = bool(build_prompt_inputs)

    tpl_text = templates[tpl_id]["text"]
    temporal_tok = temporal_text_tokenizer if temporal_text_tokenizer is not None else tokenizer
    temporal_text_source = str(getattr(args, "delta_temporal_text_source")).lower().strip()
    if temporal_text_source not in {"refined", "raw"}:
        raise ValueError(f"Invalid delta_temporal_text_source: {temporal_text_source}")
    B = len(batch["history_value"])

    targets_z_list = []
    patches_list = []
    patchmask_list = []
    metas = []

    hist_strs = []
    news_str_list = []
    structured_events_list = []
    structured_doc_events_list = []
    structured_feature_list = []
    rel_labels_list = []
    temporal_text_series_list = []
    sample_debug_records = []

    start_dates = []
    end_dates = []
    pred_starts = []
    pred_ends = []

    len_selected_news = []
    news_max_utility_list = []

    for i in range(B):
        history = batch["history_value"][i].tolist()
        target = batch["target_value"][i].tolist()
        t_target = batch["target_time"][i]
        history_times_i = _batch_time_seq_for_sample(batch.get("history_times"), i)
        target_times_i = _batch_time_seq_for_sample(batch.get("target_times"), i)

        history_z = _zscore(history, mu_global, sigma_global)
        target_z = _zscore(target, mu_global, sigma_global)

        p, pm = _make_patches(history_z, patch_len=patch_len, stride=patch_stride)
        news_text_col = args.news_text_col
        cand = pd.DataFrame(columns=[args.news_time_col, news_text_col])
        candidate_news_count = 0
        # news
        if force_no_news or (news_df is None) or (len(news_df) == 0):
            selected = pd.DataFrame(columns=[args.news_time_col, news_text_col])
        else:
            cand = get_candidates(news_df, args.news_time_col, t_target, args.news_window_days, args.news_topM)
            candidate_news_count = int(len(cand))
            policy_k = int(args.news_topK)

            selected = select_news(
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

        len_selected_news.append(len(selected))
        if len(selected) > 0 and "utility_score" in selected.columns:
            try:
                max_utility = float(pd.to_numeric(selected["utility_score"], errors="coerce").max())
            except Exception:
                max_utility = float("nan")
        else:
            max_utility = float("nan")
        news_max_utility_list.append(max_utility if np.isfinite(max_utility) else 0.0)


        # <<<<<<< refine news text and extract structured events for prompt and features >>>>>
        news_str = ""
        refined_news = ""
        refined_news_docs = []
        structured_events = {}
        structured_doc_events = []
        temporal_text_series = [""] * L
        temporal_text_docs = []
        temporal_text_debug = {
            "temporal_text_doc_total": 0,
            "temporal_text_doc_nonempty": 0,
            "temporal_text_doc_attached_any_step": 0,
            "temporal_text_doc_pre_history": 0,
            "temporal_text_doc_in_history_range": 0,
            "temporal_text_doc_post_history_pre_target": 0,
        }
        if (not force_no_news) and len(selected) > 0:
            raw_news_texts = selected[news_text_col].fillna("").astype(str).tolist()
            selected_news_metas = _selected_news_meta_records(
                selected,
                args,
                text_col=news_text_col,
                time_col=args.news_time_col,
            )

            # target time, region, dataset description
            refine_context = build_refine_context(args, target_time=t_target)
            aligned_refined_news_docs = _refine_news_docs_aligned_from_doc_cache(
                raw_news_texts=raw_news_texts,
                news_metas=selected_news_metas,
                tokenizer=tokenizer,
                max_tokens=news_budget,
                args=args,
                api_adapter=api_adapter,
            )
            refined_news_docs = []
            seen_refined_docs = set()
            for refined_item in aligned_refined_news_docs:
                clean = str(refined_item or "").strip()
                if not clean or clean in seen_refined_docs:
                    continue
                seen_refined_docs.add(clean)
                refined_news_docs.append(clean)
            refined_news = _merge_refined_news_docs(
                refined_news_docs,
                tokenizer=tokenizer,
                max_tokens=news_budget,
            )

            if (not refined_news) and (not _news_cache_is_read_only(args)):
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
            doc_cap = 4
            if doc_cap > 0:
                refined_news_docs = refined_news_docs[:doc_cap]

            if int(getattr(args, "delta_temporal_text_enable", 0)) == 1:
                temporal_text_docs = aligned_refined_news_docs if temporal_text_source == "refined" else raw_news_texts
                temporal_text_series = _build_temporal_text_series_for_sample(
                    history_times=history_times_i,
                    news_metas=selected_news_metas,
                    news_docs=temporal_text_docs,
                    tokenizer=temporal_tok,
                    max_tokens=int(max(1, getattr(args, "delta_temporal_text_max_len", 96))),
                    per_step_topk=int(max(1, getattr(args, "delta_temporal_text_per_step_topk", 3))),
                )
                temporal_text_debug = _summarize_temporal_text_alignment(
                    history_times=history_times_i,
                    target_time=t_target,
                    news_metas=selected_news_metas,
                    news_docs=temporal_text_docs,
                )

            pieces = [refined_news] if need_prompt_context and refined_news else []

            need_structured_for_prompt = need_prompt_context and int(getattr(args, "delta_include_structured_news", 0)) == 1
            need_structured_for_delta = int(getattr(args, "delta_structured_enable", 0)) == 1
            if need_structured_for_prompt or need_structured_for_delta:
                structured_events, structured_doc_events = _extract_structured_events_from_refined_docs_detailed(
                    raw_news_texts=raw_news_texts,
                    refined_news_texts=aligned_refined_news_docs,
                    news_metas=selected_news_metas,
                    args=args,
                    api_adapter=api_adapter,
                    context=refine_context,
                )
                if (not structured_events) and refined_news and (not _news_cache_is_read_only(args)):
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
        rel_labels_list.append(0.0)

        start_date = history_times_i[0] if len(history_times_i) > 0 else ""
        end_date = history_times_i[-1] if len(history_times_i) > 0 else ""
        prediction_start = target_times_i[0] if len(target_times_i) > 0 else ""
        prediction_end = target_times_i[-1] if len(target_times_i) > 0 else ""

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
        structured_events_list.append(dict(structured_events) if isinstance(structured_events, dict) else {})
        structured_doc_events_list.append(list(structured_doc_events))
        temporal_text_series_list.append(list(temporal_text_series))
        sample_debug_records.append(
            {
                "history_start": start_date,
                "history_end": end_date,
                "target_start": prediction_start,
                "target_end": prediction_end,
                "candidate_news_count": int(candidate_news_count),
                "history_range_news_count": int(
                    _count_news_rows_in_time_range(
                        news_df,
                        args.news_time_col,
                        start_date,
                        end_date,
                    )
                ),
                "selected_news_in_history_range_count": int(
                    _count_selected_news_in_history_range(
                        selected,
                        time_col=args.news_time_col,
                        history_times=history_times_i,
                    )
                ),
                "temporal_text_source": temporal_text_source if int(getattr(args, "delta_temporal_text_enable", 0)) == 1 else "disabled",
                "temporal_text_doc_total": int(temporal_text_debug["temporal_text_doc_total"]),
                "temporal_text_doc_nonempty": int(temporal_text_debug["temporal_text_doc_nonempty"]),
                "temporal_text_doc_attached_any_step": int(temporal_text_debug["temporal_text_doc_attached_any_step"]),
                "temporal_text_doc_pre_history": int(temporal_text_debug["temporal_text_doc_pre_history"]),
                "temporal_text_doc_in_history_range": int(temporal_text_debug["temporal_text_doc_in_history_range"]),
                "temporal_text_doc_post_history_pre_target": int(temporal_text_debug["temporal_text_doc_post_history_pre_target"]),
                "temporal_text_step_nonempty_count": int(sum(1 for step_text in temporal_text_series if str(step_text or "").strip())),
                "temporal_text_step_tokenized_count": 0,
                "temporal_text_step_total": int(len(temporal_text_series)),
                "temporal_text_all_nonempty_docs_attached": int(
                    int(temporal_text_debug["temporal_text_doc_nonempty"]) == int(temporal_text_debug["temporal_text_doc_attached_any_step"])
                ),
            }
        )

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

    input_ids, attn = _pad_2d_int(ids_list, pad_id=tokenizer.pad_token_id)
    ts_patches, ts_patch_mask = _pad_patches(patches_list, patchmask_list, patch_len=patch_len)
    targets_z = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets_z_list], dim=0)
    # print("max = ", len_selected_news)
    rel_labels = torch.tensor(rel_labels_list, dtype=torch.float32)
    news_counts = torch.tensor(len_selected_news, dtype=torch.float32)
    structured_feats = torch.tensor(np.stack(structured_feature_list, axis=0), dtype=torch.float32)
    if int(getattr(args, "delta_temporal_text_enable", 0)) == 1:
        temporal_text_ids, temporal_text_attn, temporal_text_step_mask = _tokenize_temporal_text_series(
            temporal_text_series_list,
            tokenizer=temporal_tok,
            max_length=int(max(1, getattr(args, "delta_temporal_text_max_len", 96))),
        )
        step_tokenized = (temporal_text_attn.sum(dim=-1) > 0).to(torch.int64)
        for i in range(min(len(sample_debug_records), int(step_tokenized.size(0)))):
            sample_debug_records[i]["temporal_text_step_tokenized_count"] = int(step_tokenized[i].sum().item())
    else:
        temporal_text_ids, temporal_text_attn, temporal_text_step_mask = None, None, None
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
        torch.tensor(news_max_utility_list, dtype=torch.float32),
        structured_events_list,
        structured_doc_events_list,
        structured_feats,
        temporal_text_ids,
        temporal_text_attn,
        temporal_text_step_mask,
        sample_debug_records,
    )

def build_delta_batch_inputs(
    *,
    batch,
    tokenizer,
    temporal_text_tokenizer=None,
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
        news_max_utility,
        structured_events_list,
        structured_doc_events_list,
        structured_feats,
        temporal_text_ids,
        temporal_text_attn,
        temporal_text_step_mask,
        sample_debug_records,
    ) = build_batch_inputs(
        batch=batch,
        tokenizer=tokenizer,
        temporal_text_tokenizer=temporal_text_tokenizer,
        templates=templates,
        tpl_id=tpl_id,
        args=args,
        global_zstats=global_zstats,
        news_df=news_df,
        policy_name=policy_name,
        policy_kw=policy_kw,
        volatility_bin=volatility_bin,
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
        "news_max_utility": news_max_utility,
        "structured_events": structured_events_list,
        "structured_doc_events": structured_doc_events_list,
        "structured_feats": structured_feats,
        "temporal_text_ids": temporal_text_ids,
        "temporal_text_attn": temporal_text_attn,
        "temporal_text_step_mask": temporal_text_step_mask,
        "sample_debug_records": sample_debug_records,
    }

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
            _metas,
            prompt_texts,
            rel_labels,
            _n_selected,
            _news_max_utility,
            _structured_events,
            _structured_doc_events,
            _structured_feats,
            _temporal_text_ids,
            _temporal_text_attn,
            _temporal_text_step_mask,
            _sample_debug_records,
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
        history_z, targets_z, _metas = _z_batch_tensors(batch, args, global_zstats=stats)
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
