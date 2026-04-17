from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data_construction.DataStatistic import DataStatistic
from ..utils.utils import print_prompt_stats

dataStatistic = DataStatistic()
_NORMAL_DIST = NormalDist()


def _single_device_map(args):
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


def _normalize_values(x, center, scale):
    x = np.asarray(x, dtype=np.float32)
    return ((x - float(center)) / float(scale)).tolist()


def _denormalize_values(x_scaled, center, scale):
    x_scaled = np.asarray(x_scaled, dtype=np.float32)
    return (x_scaled * float(scale) + float(center)).tolist()


def _zscore(x, mu, sigma):
    return _normalize_values(x, mu, sigma)


def _inv_zscore(z, mu, sigma):
    return _denormalize_values(z, mu, sigma)


def _with_legacy_scale_aliases(stats: dict, eps: float) -> dict:
    out = dict(stats)
    center = float(out["center_global"])
    scale = max(float(out["scale_global"]), eps)
    out["center_global"] = center
    out["scale_global"] = scale
    out.setdefault("normalization_mode", "zscore")
    out["mu_global"] = center
    out["sigma_global"] = scale
    out["center"] = center
    out["scale"] = scale
    out["mu"] = center
    out["sigma"] = scale
    return out


def _robust_quantile_stats(x, *, q_low: float, q_high: float, eps: float) -> dict:
    x = np.asarray(x, dtype=np.float32)
    if not (0.0 < q_low < q_high < 1.0):
        raise ValueError(f"Invalid quantile range for robust_quantile normalization: low={q_low}, high={q_high}")
    center = float(np.quantile(x, 0.5))
    q_low_value = float(np.quantile(x, q_low))
    q_high_value = float(np.quantile(x, q_high))
    normal_span = float(_NORMAL_DIST.inv_cdf(q_high) - _NORMAL_DIST.inv_cdf(q_low))
    scale = (q_high_value - q_low_value) / normal_span if abs(normal_span) > 0 else 0.0
    if scale < eps:
        mad = float(np.median(np.abs(x - center)))
        scale = 1.4826 * mad
    if scale < eps:
        scale = float(np.std(x))
    if scale < eps:
        scale = eps
    return {
        "normalization_mode": "robust_quantile",
        "center_global": center,
        "scale_global": float(scale),
        "quantile_low": float(q_low),
        "quantile_high": float(q_high),
        "quantile_low_value": q_low_value,
        "quantile_high_value": q_high_value,
    }


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
        "pred_z",
        "base_pred_z",
        "true_residual_z",
        "residual_hat_z",
        "slow_ts",
        "shape_ts",
        "spike_ts",
        "spike_gate_mean",
        "lambda_base",
        "shape_gain",
        "spike_bias",
        "relevance_mass",
        "regime_active",
        "delta_helped",
        "top10pct_residual",
        "regime_days_used",
        "regime_docs_used",
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


def _coerce_global_zstats(global_zstats, args, required: bool = True):
    if isinstance(global_zstats, dict):
        center = global_zstats.get("center_global", global_zstats.get("mu_global", global_zstats.get("center", global_zstats.get("mu"))))
        scale = global_zstats.get("scale_global", global_zstats.get("sigma_global", global_zstats.get("scale", global_zstats.get("sigma"))))
        if center is not None and scale is not None:
            eps = float(getattr(args, "zscore_eps", 1e-6))
            mode = global_zstats.get("normalization_mode", None)
            if mode is None:
                mode = "zscore"
            stats = {
                "normalization_mode": str(mode or "zscore").strip().lower(),
                "center_global": float(center),
                "scale_global": max(float(scale), eps),
            }
            for key in ["quantile_low", "quantile_high", "quantile_low_value", "quantile_high_value"]:
                if key in global_zstats and global_zstats.get(key) is not None:
                    stats[key] = float(global_zstats.get(key))
            return _with_legacy_scale_aliases(stats, eps)
    if required:
        raise ValueError("global_zstats must contain center/scale stats (or legacy mu/sigma stats).")
    return None


def _compute_global_zstats_from_train_df(train_df: pd.DataFrame, args):
    if train_df is None or len(train_df) == 0:
        raise ValueError("Cannot compute global normalization stats from empty train_df.")
    if args.value_col not in train_df.columns:
        raise KeyError(f"value_col not found in train_df: {args.value_col}")
    vals = pd.to_numeric(train_df[args.value_col], errors="coerce").to_numpy(dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Train data has no finite values for global normalization statistics.")
    spike_clip = float(getattr(args, "spike_clip_threshold", 0.0) or 0.0)
    if spike_clip > 0:
        n_before = vals.size
        vals = vals[np.abs(vals) <= spike_clip]
        n_dropped = n_before - vals.size
        if n_dropped > 0:
            import logging
            logging.getLogger(__name__).info(
                f"[SPIKE_CLIP] Dropped {n_dropped}/{n_before} training values "
                f"exceeding |{spike_clip}| for normalization stats."
            )
        if vals.size == 0:
            raise ValueError(
                f"All training values exceed spike_clip_threshold={spike_clip}. "
                "Lower the threshold or disable spike clipping."
            )
    eps = float(getattr(args, "zscore_eps", 1e-6))
    norm_mode = str(getattr(args, "normalization_mode", "robust_quantile") or "robust_quantile").strip().lower()
    if norm_mode == "robust_quantile":
        stats = _robust_quantile_stats(
            vals,
            q_low=float(getattr(args, "norm_quantile_low", 0.25)),
            q_high=float(getattr(args, "norm_quantile_high", 0.75)),
            eps=eps,
        )
    elif norm_mode == "zscore":
        mu, sigma = _zstats(vals, eps=eps)
        stats = {
            "normalization_mode": "zscore",
            "center_global": float(mu),
            "scale_global": float(sigma),
        }
    else:
        raise ValueError(f"Unsupported normalization_mode: {norm_mode}")
    return _with_legacy_scale_aliases(stats, eps)


def _maybe_news_dropout(news_payload, args):
    p = float(getattr(args, "news_dropout", 0.0) or 0.0)
    if p <= 0:
        return news_payload
    if np.random.rand() < p:
        return None
    return news_payload


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


def _run_arg_section_name(key: str) -> str:
    k = str(key or "").strip()
    if not k:
        return "Other"
    if k in {"taskName", "stage", "seed", "gpu", "save_dir", "run_name"}:
        return "Basic Setup"
    if k.endswith("_file") or k in {"time_col", "value_col", "id_col", "news_path"}:
        return "Data"
    if k.startswith("base_") or k == "base_backbone":
        return "Base Backbone"
    if k.startswith("delta_v3_"):
        return "Delta V3"
    if k.startswith("news_api_") or k in {"news_time_col", "news_text_col", "news_tz", "news_window_days"}:
        return "News"
    if k in {"history_len", "horizon", "stride", "batch_size", "epochs", "lr", "weight_decay", "grad_accum"}:
        return "Training"
    return "Other"


def _section_order_for_run_args() -> list[str]:
    return ["Basic Setup", "Data", "Training", "Base Backbone", "News", "Delta V3", "Other"]


def _ordered_run_arg_sections(args) -> OrderedDict[str, list[tuple[str, object]]]:
    grouped: OrderedDict[str, list[tuple[str, object]]] = OrderedDict(
        (name, []) for name in _section_order_for_run_args()
    )
    for key in sorted(vars(args).keys()):
        section = _run_arg_section_name(key)
        grouped.setdefault(section, [])
        grouped[section].append((key, _mask_sensitive_arg(key, getattr(args, key))))
    return grouped


def _log_run_args(args, live_logger) -> None:
    if live_logger is None:
        return
    live_logger.info("[RUN_ARGS] ----------------------------------------")
    for section, pairs in _ordered_run_arg_sections(args).items():
        if not pairs:
            continue
        live_logger.info(f"[RUN_ARGS][{section}]")
        for key, value in pairs:
            live_logger.info(f"  - {key}: {value}")
    live_logger.info("[RUN_ARGS] ----------------------------------------")


def _log_cache_decision(args, live_logger) -> None:
    if live_logger is None:
        return
    live_logger.info(
        "[CACHE] "
        f"legacy_backup_dir=_shared_refine_cache/LEGACY_BACKUP "
        f"regime_cache_dir=_shared_refine_cache/v4 "
        f"delta_v3_regime_bank_path={getattr(args, 'delta_v3_regime_bank_path', '') or '<unset>'}"
    )


def _log_enabled_mechanisms(args, live_logger, stage: str) -> None:
    if live_logger is None:
        return
    live_logger.info(
        "[MECHANISMS] "
        f"stage={stage} "
        f"base_backbone={getattr(args, 'base_backbone', 'mlp')} "
        f"delta_v3_arch={getattr(args, 'delta_v3_arch', 'patchtst_regime_modulation')} "
        f"delta_v3_enabled={int(stage in {'delta', 'all'})}"
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
    for _, group_df in getattr(ds, "groups", []):
        if isinstance(group_df, pd.DataFrame) and time_col in group_df.columns and len(group_df) > 0:
            vals.append(pd.to_datetime(group_df[time_col], errors="coerce"))
    if not vals:
        return None, None
    s = pd.concat(vals, ignore_index=True).dropna()
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
        _, group_df = ds.groups[gi]
        target_idx = int(sidx) + int(getattr(ds, "L", 0))
        if target_idx >= len(group_df) or time_col not in group_df.columns:
            continue
        vals.append(pd.to_datetime(group_df.iloc[target_idx][time_col], errors="coerce"))
    if not vals:
        return None, None
    s = pd.Series(vals).dropna()
    if len(s) == 0:
        return None, None
    return s.min(), s.max()


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
            target_start_by_id = cur.groupby(id_key, sort=False)[time_col].min().dropna().to_dict()
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
    history_len = int(args.history_len)
    horizon = int(args.horizon)
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    center_global = float(stats["center_global"])
    scale_global = float(stats["scale_global"])

    history_z_list = []
    targets_z_list = []
    metas = []

    spike_clip = float(getattr(args, "spike_clip_threshold", 0.0) or 0.0)

    batch_size = len(batch["history_value"])
    for i in range(batch_size):
        history = np.asarray(batch["history_value"][i].tolist(), dtype=np.float32)
        target = np.asarray(batch["target_value"][i].tolist(), dtype=np.float32)

        if spike_clip > 0:
            history = np.clip(history, -spike_clip, spike_clip)
            target = np.clip(target, -spike_clip, spike_clip)

        history_z = np.asarray(_normalize_values(history, center_global, scale_global), dtype=np.float32)
        target_z = np.asarray(_normalize_values(target, center_global, scale_global), dtype=np.float32)

        if history_z.shape[0] > history_len:
            history_z = history_z[-history_len:]
        elif history_z.shape[0] < history_len:
            padded = np.zeros((history_len,), dtype=np.float32)
            padded[-history_z.shape[0] :] = history_z
            history_z = padded

        if target_z.shape[0] > horizon:
            target_z = target_z[:horizon]
        elif target_z.shape[0] < horizon:
            padded = np.zeros((horizon,), dtype=np.float32)
            padded[: target_z.shape[0]] = target_z
            target_z = padded

        history_z_list.append(history_z)
        targets_z_list.append(target_z)
        metas.append(
            {
                "normalization_mode": stats["normalization_mode"],
                "center_global": center_global,
                "scale_global": scale_global,
                "mu_global": center_global,
                "sigma_global": scale_global,
                "center": center_global,
                "scale": scale_global,
                "mu": center_global,
                "sigma": scale_global,
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
    x = np.asarray(seq, dtype=np.float32)
    seq_len = int(x.shape[0])
    patch_len = int(patch_len)
    stride = int(stride)
    if patch_len <= 0:
        raise ValueError("patch_len must be > 0")
    if stride <= 0:
        raise ValueError("patch_stride must be > 0")

    if seq_len < patch_len:
        patches = np.zeros((1, patch_len), dtype=np.float32)
        patches[0, :seq_len] = x
        mask = np.ones((1,), dtype=np.int64)
        return patches, mask

    starts = list(range(0, seq_len - patch_len + 1, stride))
    patches = np.stack([x[s : s + patch_len] for s in starts], axis=0).astype(np.float32)
    mask = np.ones((patches.shape[0],), dtype=np.int64)
    return patches, mask


def history_text(history_z: list[float], center_global: float, scale_global: float, normalization_mode: str = "zscore") -> str:
    hz = np.asarray(history_z, dtype=np.float32)
    slope = float(hz[-1] - hz[-9]) if len(hz) >= 9 else float(hz[-1] - hz[0]) if len(hz) >= 2 else 0.0
    return (
        f"History normalized (mode={normalization_mode}, center={center_global:.4f}, scale={scale_global:.4f}). "
        f"Last {len(hz)} normed values: {', '.join(f'{v:.3f}' for v in hz.tolist())}. "
        f"Recent slope: {slope:.3f}."
    )


def _pad_2d_int(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(seqs)
    max_t = max((len(seq) for seq in seqs), default=0)
    max_t = max(1, max_t)
    input_ids = torch.full((batch_size, max_t), pad_id, dtype=torch.long)
    attn = torch.zeros((batch_size, max_t), dtype=torch.long)
    for i, seq in enumerate(seqs):
        t = len(seq)
        if t <= 0:
            continue
        input_ids[i, :t] = torch.tensor(seq, dtype=torch.long)
        attn[i, :t] = 1
    return input_ids, attn


def _pad_2d_float(seqs: list[list[float]], pad_value: float = 0.0) -> torch.Tensor:
    batch_size = len(seqs)
    max_t = max((len(seq) for seq in seqs), default=0)
    max_t = max(1, max_t)
    out = torch.full((batch_size, max_t), float(pad_value), dtype=torch.float32)
    for i, seq in enumerate(seqs):
        t = min(len(seq), max_t)
        if t <= 0:
            continue
        out[i, :t] = torch.tensor(seq[:t], dtype=torch.float32)
    return out


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
    base_backbone.eval()
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    center_global = float(stats["center_global"])
    scale_global = float(stats["scale_global"])

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
        history_z, targets_z, _ = _z_batch_tensors(batch, args, global_zstats=stats)
        history_z = history_z.to(device)
        targets_z = targets_z.to(device)

        pred_z = base_backbone(history_z).to(torch.float32)
        loss = _point_loss(pred_z, targets_z.to(torch.float32), mode=getattr(args, "base_loss", "smooth_l1"))

        batch_size = pred_z.size(0)
        loss_sum += float(loss.detach().cpu()) * batch_size
        n_samples += batch_size
        if use_pbar:
            eval_loader.set_postfix(zLoss=f"{loss_sum / max(1, n_samples):.6f}")

        pred_z_cpu = pred_z.detach().cpu().numpy()
        targets_cpu = batch["target_value"].detach().cpu().numpy()

        for i in range(batch_size):
            pred_denorm = _denormalize_values(pred_z_cpu[i].tolist(), center_global, scale_global)
            true_vals = targets_cpu[i].reshape(-1).tolist()
            true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

            pred = np.asarray(pred_denorm, dtype=np.float32)
            true = np.asarray(true_vals, dtype=np.float32)
            se_sum += float(((pred - true) ** 2).sum())
            ae_sum += float(np.abs(pred - true).sum())
            n_elems += int(args.horizon)

            if true_pred_csv_path is not None:
                with open(true_pred_csv_path, "a", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerows(zip(pred_denorm, true_vals))

            if ans_json_path is not None:
                record = {
                    "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                    "pred": [float(x) for x in pred_denorm],
                    "true": [float(x) for x in true_vals],
                    "normalization_mode": stats["normalization_mode"],
                    "center_global": center_global,
                    "scale_global": scale_global,
                    "mu_global": center_global,
                    "sigma_global": scale_global,
                    "center": center_global,
                    "scale": scale_global,
                    "mu": center_global,
                    "sigma": scale_global,
                }
                with open(ans_json_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    return loss_avg, mse_avg, mae_avg


def evaluate_metrics_single(*args, **kwargs):
    raise NotImplementedError("Single-model prompt evaluation was removed during the delta_v3 refactor.")
