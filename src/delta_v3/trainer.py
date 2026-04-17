from __future__ import annotations

import csv
import gc
import json
import math
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base.common import (
    _coerce_global_zstats,
    _denormalize_values,
    _eval_iter,
    _open_residual_debug_csv,
    _z_batch_tensors,
)
from ..base_backbone import load_base_backbone_checkpoint
from ..delta.core import _build_delta_optimizer, _build_delta_scheduler, skill_score
from ..delta_news_hooks import build_news_api_adapter
from ..utils.batch_utils import _batch_time_seq_for_sample
from ..utils.utils import draw_pred_true, record_test_results_csv
from .config import (
    DeltaV3Config,
    build_legacy_refined_cache_path,
    build_legacy_regime_bank_path,
    build_refined_cache_path,
)
from .importance import HardResidualSampler
from .model import build_delta_v3_model
from .modulation_heads import RegimePretrainHeads
from .pretrain_v2 import run_regime_self_supervised_pretrain
from .regime_bank import build_regime_bank, load_regime_bank, read_regime_bank_metadata
from .schema_refine_v2 import refine_dataset_news_corpus
from .targets import ResidualTargetDecomposer, compute_residual_calendar_baseline


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(path)
    return out


def _resolve_v3_paths(args, cfg: DeltaV3Config) -> tuple[str, list[str], str, list[str]]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    refined_path = build_refined_cache_path(cfg.cache_dir, cfg.news_path)
    legacy_refined_paths = _dedupe_paths(
        [
            build_legacy_refined_cache_path(cfg.cache_dir, cfg.dataset_key, cfg.schema_variant, cfg.news_path),
            build_legacy_refined_cache_path(cfg.legacy_cache_dir, cfg.dataset_key, cfg.schema_variant, cfg.news_path),
        ]
    )
    legacy_bank_paths = _dedupe_paths(
        [
            cfg.regime_bank_legacy_path,
            build_legacy_regime_bank_path(cfg.legacy_cache_dir, cfg.dataset_key, cfg.news_path),
        ]
    )
    return refined_path, legacy_refined_paths, cfg.regime_bank_path, legacy_bank_paths


def _expected_regime_bank_metadata(args, bundle, cfg: DeltaV3Config) -> dict[str, str | float | int]:
    date_start, date_end = _series_day_bounds(bundle, args)
    return {
        "source_news_file": os.path.basename(str(cfg.news_path or "").strip()),
        "source_news_stem": os.path.splitext(os.path.basename(str(cfg.news_path or "").strip()))[0],
        "schema_variant": str(cfg.schema_variant or "").strip(),
        "dataset_key": str(cfg.dataset_key or "").strip(),
        "encoder_model_id": str(cfg.text_encoder_model_id or "").strip(),
        "max_length": int(cfg.text_encoder_max_length),
        "date_start": str(pd.Timestamp(date_start).date()),
        "date_end": str(pd.Timestamp(date_end).date()),
        "tau_days": float(cfg.regime_tau_days),
        "ema_alpha": float(cfg.regime_ema_alpha),
        "ema_window": int(cfg.regime_ema_window),
    }


def _regime_bank_metadata_matches(
    path: str,
    expected: dict[str, str | float | int],
) -> tuple[bool | None, dict[str, str | float | int]]:
    actual = read_regime_bank_metadata(path)
    if not actual:
        return None, {}
    for key, expected_value in expected.items():
        if key not in actual:
            return None, actual
        actual_value = actual[key]
        if isinstance(expected_value, float):
            try:
                if abs(float(actual_value) - float(expected_value)) > 1e-8:
                    return False, actual
            except Exception:
                return False, actual
            continue
        if str(actual_value) != str(expected_value):
            return False, actual
    return True, actual


def _summarize_refined_corpus(refined_path: str) -> dict[str, float | int]:
    total_docs = 0
    actionable_docs = 0
    parseable_dates = 0
    with open(refined_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = str(line).strip()
            if not text:
                continue
            total_docs += 1
            try:
                row = json.loads(text)
            except Exception:
                continue
            if bool(row.get("is_actionable", False)):
                actionable_docs += 1
            published_at = pd.to_datetime(row.get("published_at", ""), errors="coerce", dayfirst=True)
            if not pd.isna(published_at):
                parseable_dates += 1
    actionable_pct = 100.0 * float(actionable_docs) / float(total_docs) if total_docs > 0 else 0.0
    return {
        "total_docs": int(total_docs),
        "actionable_docs": int(actionable_docs),
        "parseable_dates": int(parseable_dates),
        "actionable_pct": float(actionable_pct),
    }


def _inspect_refined_corpus(refined_path: str) -> dict[str, object]:
    total_docs = 0
    schema_variants: set[str] = set()
    source_news_files: set[str] = set()
    source_news_stems: set[str] = set()
    with open(refined_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = str(line).strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            total_docs += 1
            schema_variant = str(row.get("schema_variant", "") or "").strip()
            if schema_variant:
                schema_variants.add(schema_variant)
            source_news_file = os.path.basename(str(row.get("source_news_file", "") or "").strip())
            if source_news_file:
                source_news_files.add(source_news_file)
            source_news_stem = str(row.get("source_news_stem", "") or "").strip()
            if source_news_stem:
                source_news_stems.add(source_news_stem)
    return {
        "total_docs": int(total_docs),
        "schema_variants": sorted(schema_variants),
        "source_news_files": sorted(source_news_files),
        "source_news_stems": sorted(source_news_stems),
    }


def _refined_corpus_matches(
    refined_path: str,
    cfg: DeltaV3Config,
) -> tuple[bool, dict[str, object]]:
    metadata = _inspect_refined_corpus(refined_path)
    if int(metadata.get("total_docs", 0) or 0) <= 0:
        return False, metadata

    expected_schema_variant = str(cfg.schema_variant or "").strip()
    schema_variants = list(metadata.get("schema_variants", []) or [])
    if len(schema_variants) != 1 or schema_variants[0] != expected_schema_variant:
        return False, metadata

    expected_news_file = os.path.basename(str(cfg.news_path or "").strip())
    expected_news_stem = os.path.splitext(expected_news_file)[0]

    source_news_files = list(metadata.get("source_news_files", []) or [])
    if source_news_files and (len(source_news_files) != 1 or source_news_files[0] != expected_news_file):
        return False, metadata

    source_news_stems = list(metadata.get("source_news_stems", []) or [])
    if source_news_stems and (len(source_news_stems) != 1 or source_news_stems[0] != expected_news_stem):
        return False, metadata

    return True, metadata


def _select_usable_refined_path(candidate_paths: list[str], cfg: DeltaV3Config, live_logger) -> str | None:
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        matches, metadata = _refined_corpus_matches(path, cfg)
        if matches:
            return path
        if live_logger is not None:
            live_logger.info(
                "[DELTA_V3] refined corpus cache mismatch; skipping cache "
                f"path={path} actual_meta={json.dumps(metadata, ensure_ascii=True, sort_keys=True)} "
                f"expected_schema_variant={json.dumps(str(cfg.schema_variant or '').strip())} "
                f"expected_news_file={json.dumps(os.path.basename(str(cfg.news_path or '').strip()))}"
            )
    return None


def _summarize_regime_bank(bank, *, active_mass_threshold: float) -> dict[str, float | int]:
    relevance = np.asarray(getattr(bank, "relevance_mass", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    in_force = np.asarray(getattr(bank, "in_force_doc_count", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)

    bank_days = int(relevance.size)
    covered_days = int((in_force > 0).sum()) if in_force.size > 0 else 0
    active_days = int((relevance > float(active_mass_threshold)).sum()) if relevance.size > 0 else 0
    covered_pct = 100.0 * float(covered_days) / float(bank_days) if bank_days > 0 else 0.0
    active_pct = 100.0 * float(active_days) / float(bank_days) if bank_days > 0 else 0.0

    return {
        "bank_days": bank_days,
        "covered_days": covered_days,
        "covered_pct": float(covered_pct),
        "active_days": active_days,
        "active_pct": float(active_pct),
        "mass_mean": float(relevance.mean()) if relevance.size > 0 else 0.0,
        "mass_median": float(np.median(relevance)) if relevance.size > 0 else 0.0,
        "mass_min": float(relevance.min()) if relevance.size > 0 else 0.0,
        "mass_max": float(relevance.max()) if relevance.size > 0 else 0.0,
    }


def _fail_fast_on_empty_bank(bank, *, bank_path: str, refined_path: str):
    num_dates = int(len(getattr(bank, "dates", [])))
    num_active_days = int(np.asarray(getattr(bank, "relevance_mass", np.zeros((0,), dtype=np.float32)) > 0).sum())
    if num_dates > 0 and num_active_days > 0:
        return
    raise RuntimeError(
        "delta_v3 regime bank is empty. "
        f"bank_path={bank_path} refined_path={refined_path} dates={num_dates} active_days={num_active_days}. "
        "Delete stale v4 cache files and rebuild with delta_v3_refined_bank_build=1."
    )


def _series_day_bounds(bundle, args) -> tuple[pd.Timestamp, pd.Timestamp]:
    all_times = pd.concat(
        [
            pd.to_datetime(bundle["train_df"][args.time_col], errors="coerce"),
            pd.to_datetime(bundle["val_df"][args.time_col], errors="coerce"),
            pd.to_datetime(bundle["test_df"][args.time_col], errors="coerce"),
        ],
        axis=0,
    ).dropna()
    if len(all_times) <= 0:
        raise ValueError("Cannot build regime bank without any valid series timestamps.")
    return pd.Timestamp(all_times.min()).normalize(), pd.Timestamp(all_times.max()).normalize()


def _prepare_regime_bank(args, bundle, cfg: DeltaV3Config):
    live_logger = bundle["live_logger"]
    refined_path, legacy_refined_paths, bank_path, legacy_bank_paths = _resolve_v3_paths(args, cfg)
    refined_candidate_paths = _dedupe_paths([refined_path] + legacy_refined_paths)
    reusable_refined_path = _select_usable_refined_path(refined_candidate_paths, cfg, live_logger)
    refined_debug_path = reusable_refined_path or refined_path
    expected_bank_meta = _expected_regime_bank_metadata(args, bundle, cfg)
    if os.path.exists(bank_path) and not cfg.refined_bank_build:
        meta_matches, actual_meta = _regime_bank_metadata_matches(bank_path, expected_bank_meta)
        if meta_matches is not False:
            bank = load_regime_bank(bank_path)
            _fail_fast_on_empty_bank(bank, bank_path=bank_path, refined_path=refined_debug_path)
            return bank, refined_debug_path, bank_path
        live_logger.info(
            "[DELTA_V3] preferred regime bank cache metadata mismatch; ignoring cache "
            f"path={bank_path} actual_meta={json.dumps(actual_meta, ensure_ascii=True, sort_keys=True)} "
            f"expected_meta={json.dumps(expected_bank_meta, ensure_ascii=True, sort_keys=True)}"
        )

    if not cfg.refined_bank_build:
        for legacy_bank_path in legacy_bank_paths:
            if bank_path == legacy_bank_path or not os.path.exists(legacy_bank_path):
                continue
            meta_matches, actual_meta = _regime_bank_metadata_matches(legacy_bank_path, expected_bank_meta)
            if meta_matches is False:
                live_logger.info(
                    "[DELTA_V3] legacy regime bank cache metadata mismatch; skipping cache "
                    f"path={legacy_bank_path} actual_meta={json.dumps(actual_meta, ensure_ascii=True, sort_keys=True)} "
                    f"expected_meta={json.dumps(expected_bank_meta, ensure_ascii=True, sort_keys=True)}"
                )
                continue
            live_logger.info(
                "[DELTA_V3] using legacy regime bank cache "
                f"legacy_path={legacy_bank_path} preferred_path={bank_path}"
            )
            bank = load_regime_bank(legacy_bank_path)
            _fail_fast_on_empty_bank(bank, bank_path=legacy_bank_path, refined_path=refined_debug_path)
            return bank, refined_debug_path, legacy_bank_path

    actual_refined_path = reusable_refined_path
    if actual_refined_path is None and not cfg.refined_bank_build:
        raise FileNotFoundError(
            f"delta_v3 regime bank missing: {bank_path}. "
            f"refined_checked={refined_candidate_paths}. "
            f"legacy_checked={legacy_bank_paths}. "
            "Provide a compatible refined_{news_file}.jsonl cache or set --delta_v3_refined_bank_build 1 to rebuild from --news_path."
        )

    if actual_refined_path is None:
        live_logger.info(f"[DELTA_V3] building regime refine corpus at {refined_path}")
        api_adapter = build_news_api_adapter(args, live_logger=live_logger)
        if api_adapter is None:
            raise RuntimeError(
                "delta_v3 regime bank build requested but no API key was discovered. "
                "Set OPENAI_API_KEY or point --news_api_key_path to a readable key file "
                "(default .secrets/api_key.txt)."
            )

        refine_dataset_news_corpus(
            news_path=args.news_path,
            schema_variant=cfg.schema_variant,
            api_adapter=api_adapter,
            cache_path=refined_path,
        )
        actual_refined_path = refined_path
    else:
        live_logger.info(
            "[DELTA_V3] reusing refined corpus cache and building regime bank locally "
            f"refined_path={actual_refined_path} bank_path={bank_path}"
        )

    refined_stats = _summarize_refined_corpus(actual_refined_path)
    live_logger.info(
        "[DELTA_V3] regime refine stats "
        f"total_docs={refined_stats['total_docs']} "
        f"actionable_docs={refined_stats['actionable_docs']} "
        f"actionable_pct={float(refined_stats['actionable_pct']):.2f}% "
        f"parseable_dates={refined_stats['parseable_dates']}"
    )
    if refined_stats["total_docs"] <= 0:
        raise RuntimeError(f"delta_v3 regime refine corpus is empty: {actual_refined_path}")
    if refined_stats["actionable_docs"] <= 0:
        raise RuntimeError(
            "delta_v3 regime refine corpus produced zero actionable articles. "
            f"refined_path={actual_refined_path}."
        )

    date_start, date_end = _series_day_bounds(bundle, args)
    live_logger.info(f"[DELTA_V3] building regime bank at {bank_path}")
    build_regime_bank(
        refined_jsonl=actual_refined_path,
        out_path=bank_path,
        encoder_model_id=cfg.text_encoder_model_id,
        max_length=cfg.text_encoder_max_length,
        date_start=date_start,
        date_end=date_end,
        source_news_path=cfg.news_path,
        schema_variant=cfg.schema_variant,
        dataset_key=cfg.dataset_key,
        tau_days=cfg.regime_tau_days,
        ema_alpha=cfg.regime_ema_alpha,
        ema_window=cfg.regime_ema_window,
    )
    bank = load_regime_bank(bank_path)
    _fail_fast_on_empty_bank(bank, bank_path=bank_path, refined_path=actual_refined_path)
    return bank, actual_refined_path, bank_path


def _sample_history_times(batch) -> list[list[str]]:
    batch_size = len(batch["history_value"])
    return [_batch_time_seq_for_sample(batch.get("history_times"), i) for i in range(batch_size)]


def _sample_target_times(batch) -> list[list[str]]:
    batch_size = len(batch["target_value"])
    return [_batch_time_seq_for_sample(batch.get("target_times"), i) for i in range(batch_size)]


def _build_regime_pack(batch, bank, device, *, active_mass_threshold: float) -> tuple[dict[str, torch.Tensor], list[int], list[int]]:
    regime_vec_rows = []
    topic_rows = []
    text_rows = []
    relevance_rows = []
    doc_count_rows = []
    active_flags = []
    in_force_docs = []

    batch_size = len(batch["target_time"])
    for i in range(batch_size):
        target_time = batch["target_time"][i]
        regime_vec, topic_tag_mass, text_emb, relevance_mass, in_force_doc_count = bank.lookup(target_time)
        regime_vec_rows.append(regime_vec)
        topic_rows.append(topic_tag_mass)
        text_rows.append(text_emb)
        relevance_rows.append(relevance_mass)
        doc_count_rows.append(in_force_doc_count)
        active_flags.append(int(float(relevance_mass[0]) > float(active_mass_threshold)))
        in_force_docs.append(int(in_force_doc_count[0]))

    regime_pack = {
        "regime_vec": torch.tensor(np.stack(regime_vec_rows, axis=0), dtype=torch.float32, device=device),
        "topic_tag_mass": torch.tensor(np.stack(topic_rows, axis=0), dtype=torch.float32, device=device),
        "text_emb": torch.tensor(np.stack(text_rows, axis=0), dtype=torch.float32, device=device),
        "relevance_mass": torch.tensor(np.stack(relevance_rows, axis=0), dtype=torch.float32, device=device),
        "in_force_doc_count": torch.tensor(np.stack(doc_count_rows, axis=0), dtype=torch.long, device=device),
    }
    return regime_pack, active_flags, in_force_docs


def _blank_regime_pack(regime_pack: dict[str, torch.Tensor], blank_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in regime_pack.items()}
    batch_size = int(next(iter(out.values())).shape[0]) if out else 0
    if blank_mask is None:
        blank_mask = torch.ones((batch_size,), dtype=torch.bool, device=next(iter(out.values())).device)
    blank_mask = blank_mask.to(torch.bool)
    for key in ["regime_vec", "topic_tag_mass", "text_emb", "relevance_mass", "in_force_doc_count"]:
        if key in out:
            out[key][blank_mask] = 0
    return out


def _base_forward(base_backbone, history_z: torch.Tensor):
    out = base_backbone(history_z, return_hidden=True)
    if isinstance(out, tuple):
        base_pred_z, base_hidden = out
    else:
        base_pred_z = out
        base_hidden = out
    return base_pred_z.to(torch.float32), base_hidden.to(torch.float32)


def _collect_train_residual_magnitudes(train_eval_loader, base_backbone, args, global_zstats, device) -> np.ndarray:
    magnitudes: list[float] = []
    base_backbone.eval()
    with torch.no_grad():
        for batch in train_eval_loader:
            history_z, targets_z, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
            history_z = history_z.to(device)
            pred_z, _ = _base_forward(base_backbone, history_z)
            residual = (targets_z.to(device) - pred_z).detach().cpu().numpy()
            magnitudes.extend(np.abs(residual).mean(axis=1).tolist())
    return np.asarray(magnitudes, dtype=np.float32)


def _normalize_day_shape(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    centered = arr - float(arr.mean())
    scale = float(centered.std())
    if scale < 1e-6:
        scale = 1.0
    return centered / scale


def _build_daily_target_map(bundle, args) -> dict[str, dict[str, float]]:
    full_df = pd.concat([bundle["train_df"], bundle["val_df"], bundle["test_df"]], axis=0, ignore_index=True)
    series_df = full_df[[args.time_col, args.value_col]].copy()
    series_df[args.time_col] = pd.to_datetime(series_df[args.time_col], errors="coerce")
    series_df[args.value_col] = pd.to_numeric(series_df[args.value_col], errors="coerce")
    series_df = series_df.dropna(subset=[args.time_col, args.value_col]).sort_values(args.time_col).reset_index(drop=True)
    series_df["date_key"] = series_df[args.time_col].dt.normalize()

    days: list[pd.Timestamp] = []
    day_values: list[np.ndarray] = []
    for day, group in series_df.groupby("date_key", sort=True):
        values = group.sort_values(args.time_col)[args.value_col].to_numpy(dtype=np.float32)
        if values.size <= 0:
            continue
        days.append(pd.Timestamp(day))
        day_values.append(values)

    label_map: dict[str, dict[str, float]] = {}
    for idx, day in enumerate(days):
        if idx <= 0:
            continue
        current = day_values[idx]
        prev_days = day_values[max(0, idx - 5) : idx]
        if not prev_days:
            continue
        prev_concat = np.concatenate(prev_days, axis=0).astype(np.float32)
        prev_std = max(float(prev_concat.std()), 1e-6)
        current_std = max(float(np.asarray(current, dtype=np.float32).std()), 1e-6)
        vol_ratio = float(math.log(current_std / prev_std))
        prev_mean = float(prev_concat.mean())
        spike_flag = float(np.any(np.abs(np.asarray(current, dtype=np.float32) - prev_mean) > 3.0 * prev_std))

        prev_shapes = [_normalize_day_shape(vals) for vals in day_values[max(0, idx - 30) : idx] if len(vals) == len(current)]
        if prev_shapes:
            avg_shape = np.mean(np.stack(prev_shapes, axis=0), axis=0)
            shape_dev = float(np.mean((_normalize_day_shape(current) - avg_shape) ** 2))
        else:
            shape_dev = 0.0

        label_map[str(day.date())] = {
            "vol_ratio": vol_ratio,
            "spike_flag": spike_flag,
            "shape_dev": shape_dev,
        }
    return label_map


def _build_pretrain_payload(df: pd.DataFrame, bank, label_map: dict[str, dict[str, float]], args) -> dict[str, torch.Tensor]:
    dates = (
        pd.to_datetime(df[args.time_col], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    regime_vec_rows = []
    topic_rows = []
    text_rows = []
    relevance_rows = []
    vol_targets = []
    spike_targets = []
    shape_targets = []
    for day in dates:
        day_key = str(pd.Timestamp(day).date())
        labels = label_map.get(day_key)
        if labels is None:
            continue
        regime_vec, topic_tag_mass, text_emb, relevance_mass, _ = bank.lookup(day_key)
        regime_vec_rows.append(regime_vec)
        topic_rows.append(topic_tag_mass)
        text_rows.append(text_emb)
        relevance_rows.append(relevance_mass)
        vol_targets.append(float(labels["vol_ratio"]))
        spike_targets.append(float(labels["spike_flag"]))
        shape_targets.append(float(labels["shape_dev"]))

    if not regime_vec_rows:
        return {}
    return {
        "regime_vec": torch.tensor(np.stack(regime_vec_rows, axis=0), dtype=torch.float32),
        "topic_tag_mass": torch.tensor(np.stack(topic_rows, axis=0), dtype=torch.float32),
        "text_emb": torch.tensor(np.stack(text_rows, axis=0), dtype=torch.float32),
        "relevance_mass": torch.tensor(np.stack(relevance_rows, axis=0), dtype=torch.float32),
        "vol_target": torch.tensor(vol_targets, dtype=torch.float32),
        "spike_target": torch.tensor(spike_targets, dtype=torch.float32),
        "shape_target": torch.tensor(shape_targets, dtype=torch.float32),
    }


def _write_debug_rows(
    debug_writer,
    *,
    split: str,
    start_sample_idx: int,
    batch,
    pred_z_cpu,
    base_pred_z_cpu,
    residual_hat_cpu,
    out,
    true_residual_z_cpu,
    delta_helped_flags,
    top10_flags,
    regime_day_used,
    regime_doc_used,
):
    if debug_writer is None:
        return
    batch_size = len(batch["target_time"])
    history_times_all = _sample_history_times(batch)
    target_times_all = _sample_target_times(batch)
    for i in range(batch_size):
        debug_writer.writerow(
            {
                "split": split,
                "sample_idx": start_sample_idx + i,
                "series_id": batch["series_id"][i],
                "target_time": batch["target_time"][i],
                "history_start": history_times_all[i][0] if history_times_all[i] else "",
                "history_end": history_times_all[i][-1] if history_times_all[i] else "",
                "target_start": target_times_all[i][0] if target_times_all[i] else "",
                "target_end": target_times_all[i][-1] if target_times_all[i] else "",
                "history_times": json.dumps(history_times_all[i], ensure_ascii=False),
                "target_times": json.dumps(target_times_all[i], ensure_ascii=False),
                "pred_z": json.dumps([float(x) for x in pred_z_cpu[i].tolist()], ensure_ascii=False),
                "base_pred_z": json.dumps([float(x) for x in base_pred_z_cpu[i].tolist()], ensure_ascii=False),
                "true_residual_z": json.dumps([float(x) for x in true_residual_z_cpu[i].tolist()], ensure_ascii=False),
                "residual_hat_z": json.dumps([float(x) for x in residual_hat_cpu[i].tolist()], ensure_ascii=False),
                "slow_ts": float(out["slow_ts"][i].detach().cpu()),
                "shape_ts": json.dumps([float(x) for x in out["shape_ts"][i].detach().cpu().tolist()], ensure_ascii=False),
                "spike_ts": json.dumps([float(x) for x in out["spike_ts"][i].detach().cpu().tolist()], ensure_ascii=False),
                "spike_gate_mean": float(out["spike_gate"][i].detach().cpu().mean()),
                "lambda_base": float(out["lambda_base"][i].detach().cpu()),
                "shape_gain": float(out["shape_gain"][i].detach().cpu()),
                "spike_bias": float(out["spike_bias"][i].detach().cpu()),
                "relevance_mass": float(out["relevance_mass"][i].detach().cpu()),
                "regime_active": int(out["active_mask"][i].detach().cpu() > 0),
                "delta_helped": int(delta_helped_flags[i]),
                "top10pct_residual": int(top10_flags[i]),
                "regime_days_used": int(regime_day_used[i]),
                "regime_docs_used": int(regime_doc_used[i]),
            }
        )


def _tensor_denorm(z: torch.Tensor, center: float, scale: float) -> torch.Tensor:
    return z.to(torch.float32) * float(scale) + float(center)


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    err = target - pred
    return torch.maximum(float(q) * err, (float(q) - 1.0) * err).mean()


def _main_point_loss(
    pred_z: torch.Tensor,
    targets_z: torch.Tensor,
    targets_raw: torch.Tensor,
    *,
    cfg: DeltaV3Config,
    center_global: float,
    scale_global: float,
    price_winsor_bounds: tuple[float, float] | None,
) -> torch.Tensor:
    if cfg.schema_variant == "price":
        pred_raw = _tensor_denorm(pred_z, center_global, scale_global)
        true_raw = targets_raw.to(pred_raw.device, dtype=torch.float32)
        if price_winsor_bounds is None:
            raise ValueError("price_winsor_bounds is required for price training.")
        low, high = price_winsor_bounds
        pred_w = pred_raw.clamp(min=float(low), max=float(high))
        true_w = true_raw.clamp(min=float(low), max=float(high))
        quantile = (
            _pinball_loss(pred_raw, true_raw, 0.1)
            + _pinball_loss(pred_raw, true_raw, 0.5)
            + _pinball_loss(pred_raw, true_raw, 0.9)
        ) / 3.0
        raw_loss = F.smooth_l1_loss(pred_w, true_w) + 0.1 * quantile
        # Rescale raw-space price loss to z-space magnitude so that auxiliary
        # losses (consistency, counterfactual, inactive_residual) which live in
        # z-space have comparable gradient scale. Without this, point_loss
        # gradients dominate by O(scale_global) ~ 30-100x, causing news path
        # regularization terms to have near-zero effective influence.
        return raw_loss / max(float(scale_global), 1e-6)
    return F.smooth_l1_loss(pred_z, targets_z)


def _subset_mean(values: list[float], mask: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0 or mask.size <= 0:
        return float("nan")
    valid = mask.astype(bool)
    if not valid.any():
        return float("nan")
    return float(arr[valid].mean())


def _histogram(values: list[float], bins: int = 8) -> dict[str, list[float] | list[int]]:
    if not values:
        return {"edges": [], "counts": []}
    arr = np.asarray(values, dtype=np.float32)
    counts, edges = np.histogram(arr, bins=int(max(2, bins)))
    return {
        "edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def evaluate_delta_v3(
    model,
    *,
    base_backbone,
    decomposer=None,
    data_loader,
    bank,
    shuffled_bank,
    args,
    global_zstats,
    device,
    cfg: DeltaV3Config,
    price_winsor_bounds: tuple[float, float] | None,
    debug_csv_path: str | None = None,
    split_name: str = "val",
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    filename: str | None = None,
):
    del filename
    model.eval()
    base_backbone.eval()
    stats = _coerce_global_zstats(global_zstats, args, required=True)
    center_global = float(stats["center_global"])
    scale_global = float(stats["scale_global"])

    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0
    base_se_sum, base_ae_sum = 0.0, 0.0
    sample_final_abs = []
    sample_base_abs = []
    sample_blank_abs = []
    sample_perm_abs = []
    sample_final_abs_z = []
    sample_blank_abs_z = []
    sample_perm_abs_z = []
    sample_active = []
    sample_lambda = []
    sample_lambda_ts = []
    sample_lambda_news_delta = []
    sample_shape_gain = []
    sample_spike_bias = []
    sample_relevance = []
    sample_spike_hit = []
    sample_spike_target_hit = []
    regime_days_all = []
    regime_docs_all = []

    fh, debug_writer = _open_residual_debug_csv(debug_csv_path)
    sample_cursor = 0
    try:
        eval_desc = "[EVAL][DELTA_V3][TEST]" if testing else "[EVAL][DELTA_V3][VAL]"
        eval_loader, use_pbar = _eval_iter(data_loader, args, desc=eval_desc)
        for _, batch in enumerate(eval_loader):
            history_z, targets_z, _ = _z_batch_tensors(batch, args, global_zstats=stats)
            history_times = _sample_history_times(batch)
            target_times = _sample_target_times(batch)

            history_z = history_z.to(device)
            targets_z = targets_z.to(device)
            targets_raw = batch["target_value"].to(device, dtype=torch.float32)
            _spike_clip = float(getattr(args, "spike_clip_threshold", 0.0) or 0.0)
            if _spike_clip > 0:
                targets_raw = targets_raw.clamp(-_spike_clip, _spike_clip)
            base_pred_z, base_hidden = _base_forward(base_backbone, history_z)
            regime_pack, regime_active_used, in_force_docs_used = _build_regime_pack(
                batch,
                bank,
                device,
                active_mass_threshold=cfg.active_mass_threshold,
            )

            with torch.no_grad():
                out = model(
                    history_z=history_z,
                    history_times=history_times,
                    base_pred_z=base_pred_z,
                    base_hidden=base_hidden if model.cfg.use_base_hidden else None,
                    regime_pack=regime_pack,
                )
                pred_z = out["pred_z"]
                loss = _main_point_loss(
                    pred_z,
                    targets_z,
                    targets_raw,
                    cfg=cfg,
                    center_global=center_global,
                    scale_global=scale_global,
                    price_winsor_bounds=price_winsor_bounds,
                )
                target_parts = None
                if decomposer is not None:
                    residual_target = targets_z - base_pred_z
                    target_parts = decomposer.decompose(residual_target, target_times)

                blank_pack = _blank_regime_pack(regime_pack)
                blank_out = model(
                    history_z=history_z,
                    history_times=history_times,
                    base_pred_z=base_pred_z,
                    base_hidden=base_hidden if model.cfg.use_base_hidden else None,
                    regime_pack=blank_pack,
                )

                perm_out = None
                if shuffled_bank is not None:
                    perm_pack, _, _ = _build_regime_pack(
                        batch,
                        shuffled_bank,
                        device,
                        active_mass_threshold=cfg.active_mass_threshold,
                    )
                    perm_out = model(
                        history_z=history_z,
                        history_times=history_times,
                        base_pred_z=base_pred_z,
                        base_hidden=base_hidden if model.cfg.use_base_hidden else None,
                        regime_pack=perm_pack,
                    )

            batch_size = pred_z.size(0)
            loss_sum += float(loss.detach().cpu()) * batch_size
            n_samples += batch_size
            if use_pbar:
                eval_loader.set_postfix(loss=f"{loss_sum / max(1, n_samples):.6f}")

            pred_z_cpu = pred_z.detach().cpu().numpy()
            base_pred_z_cpu = base_pred_z.detach().cpu().numpy()
            blank_pred_z_cpu = blank_out["pred_z"].detach().cpu().numpy()
            perm_pred_z_cpu = perm_out["pred_z"].detach().cpu().numpy() if perm_out is not None else None
            residual_hat_cpu = out["residual_hat"].detach().cpu().numpy()
            targets_cpu = batch["target_value"].detach().cpu().numpy()
            if _spike_clip > 0:
                targets_cpu = np.clip(targets_cpu, -_spike_clip, _spike_clip)
            targets_z_cpu = targets_z.detach().cpu().numpy()
            true_residual_z_cpu = targets_z_cpu - base_pred_z_cpu
            active_flags_batch = (
                regime_pack["relevance_mass"].detach().cpu().view(-1).numpy() > float(cfg.active_mass_threshold)
            )

            for i in range(batch_size):
                pred_denorm = _denormalize_values(pred_z_cpu[i].tolist(), center_global, scale_global)
                base_denorm = _denormalize_values(base_pred_z_cpu[i].tolist(), center_global, scale_global)
                blank_denorm = _denormalize_values(blank_pred_z_cpu[i].tolist(), center_global, scale_global)
                perm_denorm = (
                    _denormalize_values(perm_pred_z_cpu[i].tolist(), center_global, scale_global)
                    if perm_pred_z_cpu is not None
                    else None
                )
                true_vals = targets_cpu[i].reshape(-1).tolist()
                true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

                pred = np.asarray(pred_denorm, dtype=np.float32)
                base_only = np.asarray(base_denorm, dtype=np.float32)
                blank_pred = np.asarray(blank_denorm, dtype=np.float32)
                true = np.asarray(true_vals, dtype=np.float32)

                se_sum += float(((pred - true) ** 2).sum())
                ae_sum += float(np.abs(pred - true).sum())
                base_se_sum += float(((base_only - true) ** 2).sum())
                base_ae_sum += float(np.abs(base_only - true).sum())
                n_elems += int(args.horizon)

                sample_final_abs.append(float(np.abs(pred - true).mean()))
                sample_base_abs.append(float(np.abs(base_only - true).mean()))
                sample_blank_abs.append(float(np.abs(blank_pred - true).mean()))
                sample_perm_abs.append(float(np.abs(np.asarray(perm_denorm, dtype=np.float32) - true).mean()) if perm_denorm is not None else float("nan"))
                sample_final_abs_z.append(float(np.abs(pred_z_cpu[i] - targets_z_cpu[i]).mean()))
                sample_blank_abs_z.append(float(np.abs(blank_pred_z_cpu[i] - targets_z_cpu[i]).mean()))
                sample_perm_abs_z.append(
                    float(np.abs(perm_pred_z_cpu[i] - targets_z_cpu[i]).mean()) if perm_pred_z_cpu is not None else float("nan")
                )
                sample_active.append(bool(active_flags_batch[i]))
                sample_lambda.append(float(out["lambda_base"][i].detach().cpu()))
                sample_lambda_ts.append(float(out["lambda_ts"][i].detach().cpu()))
                sample_lambda_news_delta.append(float(out["lambda_news_delta"][i].detach().cpu()))
                sample_shape_gain.append(float(out["shape_gain"][i].detach().cpu()))
                sample_spike_bias.append(float(out["spike_bias"][i].detach().cpu()))
                sample_relevance.append(float(out["relevance_mass"][i].detach().cpu()))
                sample_spike_hit.append(float(out["spike_gate_hard"][i].detach().cpu().mean()))
                if target_parts is not None:
                    sample_spike_target_hit.append(float(target_parts["spike_mask"][i].detach().cpu().mean()))
                regime_days_all.append(int(regime_active_used[i]))
                regime_docs_all.append(int(in_force_docs_used[i]))

                if true_pred_csv_path is not None:
                    with open(true_pred_csv_path, "a", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerows(zip(pred_denorm, true_vals))

            top10_flags = [0] * batch_size
            _write_debug_rows(
                debug_writer,
                split=split_name,
                start_sample_idx=sample_cursor,
                batch=batch,
                pred_z_cpu=pred_z_cpu,
                base_pred_z_cpu=base_pred_z_cpu,
                residual_hat_cpu=residual_hat_cpu,
                out=out,
                true_residual_z_cpu=true_residual_z_cpu,
                delta_helped_flags=[final < base for final, base in zip(sample_final_abs[-batch_size:], sample_base_abs[-batch_size:])],
                top10_flags=top10_flags,
                regime_day_used=regime_active_used,
                regime_doc_used=in_force_docs_used,
            )
            sample_cursor += batch_size

        loss_avg = loss_sum / max(1, n_samples)
        mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
        mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
        base_mse_avg = base_se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
        base_mae_avg = base_ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")

        sample_base_abs_arr = np.asarray(sample_base_abs, dtype=np.float32)
        if sample_base_abs_arr.size > 0:
            threshold = float(np.quantile(sample_base_abs_arr, 0.90))
            top_mask = sample_base_abs_arr >= threshold
        else:
            top_mask = np.zeros((0,), dtype=bool)
        sample_final_abs_arr = np.asarray(sample_final_abs, dtype=np.float32)
        sample_final_abs_z_arr = np.asarray(sample_final_abs_z, dtype=np.float32)
        sample_blank_abs_z_arr = np.asarray(sample_blank_abs_z, dtype=np.float32)
        sample_perm_abs_z_arr = np.asarray(sample_perm_abs_z, dtype=np.float32)
        helped_mask = sample_final_abs_arr < sample_base_abs_arr if sample_base_abs_arr.size > 0 else np.zeros((0,), dtype=bool)
        active_mask = np.asarray(sample_active, dtype=bool)
        inactive_mask = ~active_mask if active_mask.size > 0 else np.zeros((0,), dtype=bool)

        inactive_blank_gap_pct = float("nan")
        inactive_normal = _subset_mean(sample_final_abs, inactive_mask)
        inactive_blank = _subset_mean(sample_blank_abs, inactive_mask)
        if math.isfinite(inactive_normal) and math.isfinite(inactive_blank) and abs(inactive_blank) > 1e-6:
            inactive_blank_gap_pct = 100.0 * (inactive_normal - inactive_blank) / inactive_blank

        lambda_arr = np.asarray(sample_lambda, dtype=np.float32)
        relevance_arr = np.asarray(sample_relevance, dtype=np.float32)
        lambda_saturation_threshold = float(cfg.lambda_max) * 0.98
        lambda_saturation_pct = float((lambda_arr >= lambda_saturation_threshold).mean()) if lambda_arr.size > 0 else 0.0
        active_blank_gain_z = _subset_mean((sample_blank_abs_z_arr - sample_final_abs_z_arr).tolist(), active_mask)
        active_permuted_gain_z = _subset_mean((sample_perm_abs_z_arr - sample_final_abs_z_arr).tolist(), active_mask)

        diag = {
            "base_mse": float(base_mse_avg),
            "base_mae": float(base_mae_avg),
            "skill_score_mse": float(skill_score(mse_avg, base_mse_avg)),
            "skill_score_mae": float(skill_score(mae_avg, base_mae_avg)),
            "delta_helped_rate": float(helped_mask.mean()) if helped_mask.size > 0 else 0.0,
            "delta_helped_rate_top10pct_residual": float(helped_mask[top_mask].mean()) if top_mask.size > 0 and top_mask.any() else 0.0,
            "top10pct_residual_mae": float(sample_final_abs_arr[top_mask].mean()) if top_mask.size > 0 and top_mask.any() else float("nan"),
            "spike_gate_hit_rate": float(np.mean(sample_spike_hit)) if sample_spike_hit else 0.0,
            "spike_target_hit_rate": float(np.mean(sample_spike_target_hit)) if sample_spike_target_hit else 0.0,
            "active_subset_mae": _subset_mean(sample_final_abs, active_mask),
            "blank_active_subset_mae": _subset_mean(sample_blank_abs, active_mask),
            "inactive_subset_mae": inactive_normal,
            "blank_inactive_subset_mae": inactive_blank,
            "permuted_active_subset_mae": _subset_mean(sample_perm_abs, active_mask),
            "inactive_blank_gap_pct": inactive_blank_gap_pct,
            "active_blank_gain_z": active_blank_gain_z,
            "active_permuted_gain_z": active_permuted_gain_z,
            "lambda_base_mean": float(np.mean(sample_lambda)) if sample_lambda else 0.0,
            "lambda_base_active_mean": _subset_mean(sample_lambda, active_mask),
            "lambda_base_inactive_mean": _subset_mean(sample_lambda, inactive_mask),
            "lambda_ts_mean": float(np.mean(sample_lambda_ts)) if sample_lambda_ts else 0.0,
            "lambda_news_delta_mean": float(np.mean(sample_lambda_news_delta)) if sample_lambda_news_delta else 0.0,
            "lambda_saturation_pct": lambda_saturation_pct,
            "shape_gain_mean": float(np.mean(sample_shape_gain)) if sample_shape_gain else 1.0,
            "shape_gain_active_mean": _subset_mean(sample_shape_gain, active_mask),
            "shape_gain_inactive_mean": _subset_mean(sample_shape_gain, inactive_mask),
            "spike_bias_mean": float(np.mean(sample_spike_bias)) if sample_spike_bias else 0.0,
            "spike_bias_active_mean": _subset_mean(sample_spike_bias, active_mask),
            "spike_bias_inactive_mean": _subset_mean(sample_spike_bias, inactive_mask),
            "relevance_mass_mean": float(relevance_arr.mean()) if relevance_arr.size > 0 else 0.0,
            "relevance_mass_median": float(np.median(relevance_arr)) if relevance_arr.size > 0 else 0.0,
            "relevance_mass_max": float(relevance_arr.max()) if relevance_arr.size > 0 else 0.0,
            "active_mass_mean": float(relevance_arr.mean()) if relevance_arr.size > 0 else 0.0,
            "active_mass_median": float(np.median(relevance_arr)) if relevance_arr.size > 0 else 0.0,
            "active_mass_max": float(relevance_arr.max()) if relevance_arr.size > 0 else 0.0,
            "regime_active_pct": float(active_mask.mean()) if active_mask.size > 0 else 0.0,
            "regime_days_mean": float(np.mean(regime_days_all)) if regime_days_all else 0.0,
            "regime_docs_mean": float(np.mean(regime_docs_all)) if regime_docs_all else 0.0,
            "lambda_hist_active": _histogram([v for v, a in zip(sample_lambda, sample_active) if a]),
            "lambda_hist_inactive": _histogram([v for v, a in zip(sample_lambda, sample_active) if not a]),
            "shape_gain_hist_active": _histogram([v for v, a in zip(sample_shape_gain, sample_active) if a]),
            "shape_gain_hist_inactive": _histogram([v for v, a in zip(sample_shape_gain, sample_active) if not a]),
            "spike_bias_hist_active": _histogram([v for v, a in zip(sample_spike_bias, sample_active) if a]),
            "spike_bias_hist_inactive": _histogram([v for v, a in zip(sample_spike_bias, sample_active) if not a]),
        }
        setattr(args, "_last_residual_eval_diag", diag)

        return loss_avg, mse_avg, mae_avg, base_mse_avg, base_mae_avg
    finally:
        if fh is not None:
            fh.close()


def train_delta_v3_stage(args, bundle, best_base_path: str, best_base_metric):
    del best_base_metric
    live_logger = bundle["live_logger"]
    device = bundle["device"]
    cfg = DeltaV3Config.from_args(args)

    # load base model
    base_backbone, base_meta = load_base_backbone_checkpoint(best_base_path, device=device, is_trainable=False)
    live_logger.info(
        f"[DELTA_V3] Loaded frozen base backbone={base_meta.get('backbone_name')} from {best_base_path}"
    )
    bundle_stats = _coerce_global_zstats(bundle.get("global_zstats"), args, required=False)
    ckpt_stats = _coerce_global_zstats(base_meta, args, required=False)
    stats = ckpt_stats if ckpt_stats is not None else _coerce_global_zstats(bundle.get("global_zstats"), args, required=True)
    if ckpt_stats is not None and bundle_stats is not None:
        mode_mismatch = str(ckpt_stats.get("normalization_mode")) != str(bundle_stats.get("normalization_mode"))
        scale_diff = abs(float(ckpt_stats["scale_global"]) - float(bundle_stats["scale_global"]))
        center_diff = abs(float(ckpt_stats["center_global"]) - float(bundle_stats["center_global"]))
        if mode_mismatch or center_diff > 1e-6 or scale_diff > 1e-6:
            live_logger.warning(
                "[NORMALIZE] delta stage is using normalization stats from base checkpoint "
                f"(mode={ckpt_stats.get('normalization_mode')} center={ckpt_stats['center_global']:.6f} "
                f"scale={ckpt_stats['scale_global']:.6f}) instead of current bundle stats "
                f"(mode={bundle_stats.get('normalization_mode')} center={bundle_stats['center_global']:.6f} "
                f"scale={bundle_stats['scale_global']:.6f})."
            )

    # load regime bank
    bank, refined_path, bank_path = _prepare_regime_bank(args, bundle, cfg)
    # create shuffled bank for permutation ablation
    # A shuffled bank is created by shuffling the date-to-regime mapping
    # Preserves the overall regime distribution but breaks the temporal alignment between regimes and the target series.
    shuffled_bank = bank.shuffled(cfg.eval_permutation_seed)
    live_logger.info(
        f"[DELTA_V3] regime bank ready path={bank_path} refined={refined_path} dates={len(bank.dates)} "
        f"regime_dim={bank.regime_dim} topic_dim={bank.topic_dim} text_dim={bank.text_dim}"
    )
    refined_stats = _summarize_refined_corpus(refined_path) if os.path.exists(refined_path) else None
    bank_stats = _summarize_regime_bank(bank, active_mass_threshold=cfg.active_mass_threshold)
    if refined_stats is not None:
        live_logger.info(
            "[DELTA_V3][BANK_COVERAGE] "
            f"actionable_news={refined_stats['actionable_docs']}/{refined_stats['total_docs']} "
            f"({float(refined_stats['actionable_pct']):.2f}% refined) "
            f"covered_days={bank_stats['covered_days']}/{bank_stats['bank_days']} "
            f"({float(bank_stats['covered_pct']):.2f}% bank; in_force_doc_count>0) "
            f"active_days={bank_stats['active_days']}/{bank_stats['bank_days']} "
            f"({float(bank_stats['active_pct']):.2f}% bank; mass>{float(cfg.active_mass_threshold):.4f})"
        )
    else:
        live_logger.info(
            "[DELTA_V3][BANK_COVERAGE] "
            f"actionable_news=N/A refined={refined_path} "
            f"covered_days={bank_stats['covered_days']}/{bank_stats['bank_days']} "
            f"({float(bank_stats['covered_pct']):.2f}% bank; in_force_doc_count>0) "
            f"active_days={bank_stats['active_days']}/{bank_stats['bank_days']} "
            f"({float(bank_stats['active_pct']):.2f}% bank; mass>{float(cfg.active_mass_threshold):.4f})"
        )
    live_logger.info(
        "[DELTA_V3][BANK_MASS] "
        f"mean={float(bank_stats['mass_mean']):.4f} "
        f"median={float(bank_stats['mass_median']):.4f} "
        f"min={float(bank_stats['mass_min']):.4f} "
        f"max={float(bank_stats['mass_max']):.4f}"
    )

    train_vals = pd.to_numeric(bundle["train_df"][args.value_col], errors="coerce").to_numpy(dtype=np.float32)
    train_vals = train_vals[np.isfinite(train_vals)]
    # Apply winsorization to the target values to mitigate the influence of extreme outliers.
    price_winsor_bounds = None
    if cfg.schema_variant == "price" and train_vals.size > 0:
        price_winsor_bounds = (
            float(np.quantile(train_vals, cfg.price_winsor_low)),
            float(np.quantile(train_vals, cfg.price_winsor_high)),
        )
        live_logger.info(
            f"[DELTA_V3] price winsor bounds low={price_winsor_bounds[0]:.4f} high={price_winsor_bounds[1]:.4f}"
        )

    # compute residual magnitudes for hard sampling
    residual_magnitudes = _collect_train_residual_magnitudes(
        bundle["train_eval_loader"],
        base_backbone,
        args,
        stats,
        device,
    )
    # let delta model focus more on samples with large residuals, 
    # which are likely underfit by the base model and have more room for improvement.
    sampler_helper = HardResidualSampler(
        residual_magnitudes=residual_magnitudes,
        top_pct=cfg.hard_residual_pct,
        hard_frac=cfg.hard_residual_frac,
    )
    live_logger.info(
        f"[DELTA_V3] hard residual sampler hit_rate={sampler_helper.hard_hit_rate:.4f} top_pct={cfg.hard_residual_pct:.2f}"
    )
    
    # compute residual calendar baseline for target decomposition
    dow_hod_mean, spike_sigma, spike_threshold_abs = compute_residual_calendar_baseline(
        bundle["train_eval_loader"],
        base_backbone,
        args,
        stats,
        device,
        spike_target_pct=cfg.spike_target_pct,
    )
    # The residual target decomposer breaks down the prediction target into three components: 
    # a slow-moving baseline that captures predictable calendar effects, 
    # a shape component that captures systematic intraday patterns, 
    # and a spike component that captures unpredictable extreme movements. 
    # By explicitly modeling these distinct components, the delta model can learn specialized mechanisms for each.
    decomposer = ResidualTargetDecomposer(
        dow_hod_mean=dow_hod_mean,
        spike_sigma=spike_sigma,
        spike_threshold_abs=spike_threshold_abs,
        spike_k=cfg.spike_k,
        spike_target_pct=cfg.spike_target_pct,
    )
    live_logger.info(
        f"[DELTA_V3] residual baseline sigma={spike_sigma:.6f} "
        f"spike_abs_threshold={spike_threshold_abs:.6f} spike_k={cfg.spike_k:.2f} "
        f"spike_target_pct={cfg.spike_target_pct:.2f}"
    )


    # Pretrain the regime attention and gating heads with self-supervised regime signals derived from the target series.
    model = build_delta_v3_model(cfg).to(device)
    daily_label_map = _build_daily_target_map(bundle, args)
    pretrain_train_payload = _build_pretrain_payload(bundle["train_df"], bank, daily_label_map, args)
    pretrain_val_payload = _build_pretrain_payload(bundle["val_df"], bank, daily_label_map, args)
    pretrain_helper = RegimePretrainHeads(hidden_size=cfg.hidden_size)
    pretrain_diag = run_regime_self_supervised_pretrain(
        model,
        pretrain_helper,
        pretrain_train_payload,
        pretrain_val_payload,
        epochs=cfg.pretrain_epochs,
        lr=cfg.pretrain_lr,
        device=device,
        scheduler_name=cfg.scheduler,
        warmup_pct=cfg.pretrain_warmup_pct,
        min_lr_ratio=cfg.min_lr_ratio,
        live_logger=live_logger,
    )
    if math.isfinite(float(pretrain_diag.get("best_loss", float("nan")))):
        model.initialize_from_pretrain(pretrain_helper)
    live_logger.info(f"[DELTA_V3][PRETRAIN_V2] best_loss={pretrain_diag.get('best_loss', float('nan')):.4f}")


    # Main delta training loop
    train_dataset = bundle["train_loader"].dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler_helper.build(len(train_dataset)),
        drop_last=False,
    )
    optimizer = _build_delta_optimizer(model, args)
    total_steps = len(train_loader) * int(args.epochs)
    scheduler = _build_delta_scheduler(
        optimizer,
        total_steps=total_steps,
        scheduler_name=cfg.scheduler,
        warmup_pct=cfg.warmup_pct,
        min_lr_ratio=cfg.min_lr_ratio,
    )
    warmup_steps = min(
        max(0, int(total_steps * float(max(0.0, cfg.warmup_pct)))),
        max(0, total_steps - 1),
    )
    live_logger.info(
        f"[DELTA_V3][LR] scheduler={cfg.scheduler} total_steps={total_steps} "
        f"warmup_steps={warmup_steps} min_lr={float(args.lr) * float(cfg.min_lr_ratio):.2e}"
    )

    best_metric = float("inf")
    best_fallback_metric = float("inf")  # Plan B safety net: tracks lowest val_mae when gate never passes
    stale_rounds = 0
    best_ckpt_dir = os.path.join("./checkpoints", args.taskName, f"best_delta_v3_{args.taskName}")
    os.makedirs(os.path.dirname(best_ckpt_dir), exist_ok=True)

    center_global = float(stats["center_global"])
    scale_global = float(stats["scale_global"])

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"[DELTA_V3] Epoch {epoch + 1}/{int(args.epochs)}")
        for batch in pbar:
            history_z, targets_z, _ = _z_batch_tensors(batch, args, global_zstats=stats)
            history_times = _sample_history_times(batch)
            target_times = _sample_target_times(batch)
            history_z = history_z.to(device)
            targets_z = targets_z.to(device)
            targets_raw = batch["target_value"].to(device, dtype=torch.float32)
            _spike_clip = float(getattr(args, "spike_clip_threshold", 0.0) or 0.0)
            if _spike_clip > 0:
                targets_raw = targets_raw.clamp(-_spike_clip, _spike_clip)

            with torch.no_grad():
                base_pred_z, base_hidden = _base_forward(base_backbone, history_z)

            # get regime pack
            regime_pack, _, _ = _build_regime_pack(
                batch,
                bank,
                device,
                active_mass_threshold=cfg.active_mass_threshold,
            )
            original_active = regime_pack["relevance_mass"].view(-1) > float(cfg.active_mass_threshold)
            # dropout, randomly blank out the regime signals
            if cfg.news_blank_prob > 0:
                blank_mask = torch.rand((history_z.size(0),), device=device) < float(cfg.news_blank_prob)
                train_regime_pack = _blank_regime_pack(regime_pack, blank_mask=blank_mask)
            else:
                train_regime_pack = regime_pack
            # forward pass
            out = model(
                history_z=history_z,
                history_times=history_times,
                base_pred_z=base_pred_z,
                base_hidden=base_hidden if cfg.use_base_hidden else None,
                regime_pack=train_regime_pack,
            )
            # delta prediction: the sum of the base prediction and the residual correction predicted by the delta model
            pred_z = out["pred_z"]
            # real residual target
            residual_target = targets_z - base_pred_z
            # decompose the residual target into slow, shape, and spike components for auxiliary losses
            target_parts = decomposer.decompose(residual_target, target_times)
            # 最终预测 pred_z 和真实目标之间的主损失，基于平滑L1损失，价格模式下还加入了分位数损失以更好地捕捉预测分布。
            point_loss = _main_point_loss(
                pred_z,
                targets_z,
                targets_raw,
                cfg=cfg,
                center_global=center_global,
                scale_global=scale_global,
                price_winsor_bounds=price_winsor_bounds,
            )
            # 要求模型的 slow_ts 学到 residual 里的慢变化部分
            slow_loss = F.mse_loss(out["slow_ts"], target_parts["slow"])
            # 要求模型的 shape_ts 学到 residual 里的常规形状变化部分
            shape_loss = F.smooth_l1_loss(out["shape_ts"], target_parts["shape"])
            # 标出哪些位置是真正的 spike
            spike_mask = target_parts["spike_mask"]
            # 先逐点计算模型预测的 spike 和真实 spike 的误差
            # 这里用的是 out["spike_gate"] * out["spike_ts"]，意思是“尖峰幅度乘上尖峰开关”
            spike_raw = F.smooth_l1_loss(out["spike_gate"] * out["spike_ts"], target_parts["spike"], reduction="none")
            # 只在 spike_mask=1 的位置上统计 spike 误差
            # 非 spike 位置不让它参与这个损失
            spike_loss = (spike_raw * spike_mask).sum() / spike_mask.sum().clamp_min(1.0)

            # 单独训练“spike 开关”要不要打开。
            # 把前面分解出来的真实 spike 位置，当成开关的监督标签
            # 有 spike 的位置是 1，没有 spike 的位置是 0
            gate_target = spike_mask
            # 算这一批里 spike 占比有多少; 因为 spike 通常很少，所以正样本比例会很低
            gate_pos_rate = gate_target.mean().detach().clamp_min(1e-3)
            # 给正样本更高权重; 这样 BCE 不会因为“0 太多、1 太少”而学成总是预测没 spike
            gate_pos_weight = ((1.0 - gate_pos_rate) / gate_pos_rate).clamp(1.0, 20.0)
            # 逐点的 spike 分类损失; 让 out["spike_gate_logits"] 学会判断每个位置是否该开 spike
            spike_gate_loss = F.binary_cross_entropy_with_logits(
                out["spike_gate_logits"],
                gate_target,
                pos_weight=gate_pos_weight,
            )
            # 额外的“比例约束” loss，鼓励 spike_gate 的平均开关率接近真实 spike 的平均发生率
            spike_gate_rate_loss = F.smooth_l1_loss(
                out["spike_gate"].mean(dim=-1),
                gate_target.mean(dim=-1),
            )

            # 防止模型在“其实没有有效新闻”的样本上，仍然被新闻分支扰动。
            # 也就是：没新闻价值时，新闻开着和关着，结果应该几乎一样。
            consistency_loss = torch.zeros((), dtype=torch.float32, device=device)
            if cfg.consistency_weight > 0 and bool((~original_active).any()):
                blank_all_pack = _blank_regime_pack(regime_pack)
                with torch.no_grad():
                    out_blank_all = model(
                        history_z=history_z,
                        history_times=history_times,
                        base_pred_z=base_pred_z,
                        base_hidden=base_hidden if cfg.use_base_hidden else None,
                        regime_pack=blank_all_pack,
                    )
                inactive_mask = ~original_active
                consistency_loss = F.mse_loss(pred_z[inactive_mask], out_blank_all["pred_z"][inactive_mask])

            # “反事实约束”：对于那些本来新闻是活跃的样本，模型用“真实新闻”时，应该比用“没有新闻”或“乱配新闻”更好
            counterfactual_loss = torch.zeros((), dtype=torch.float32, device=device)
            if cfg.counterfactual_weight > 0 and bool(original_active.any()):
                cf_mask = original_active.clone()
                if cfg.news_blank_prob > 0:
                    cf_mask = cf_mask & (~blank_mask)
                if bool(cf_mask.any()):
                    real_out = out
                    if cfg.news_blank_prob > 0 and bool(blank_mask.any()):
                        real_out = model(
                            history_z=history_z,
                            history_times=history_times,
                            base_pred_z=base_pred_z,
                            base_hidden=base_hidden if cfg.use_base_hidden else None,
                            regime_pack=regime_pack,
                        )
                    blank_all_pack = _blank_regime_pack(regime_pack)
                    with torch.no_grad():
                        blank_all_out = model(
                            history_z=history_z,
                            history_times=history_times,
                            base_pred_z=base_pred_z,
                            base_hidden=base_hidden if cfg.use_base_hidden else None,
                            regime_pack=blank_all_pack,
                        )
                        perm_pack, _, _ = _build_regime_pack(
                            batch,
                            shuffled_bank,
                            device,
                            active_mass_threshold=cfg.active_mass_threshold,
                        )
                        perm_out = model(
                            history_z=history_z,
                            history_times=history_times,
                            base_pred_z=base_pred_z,
                            base_hidden=base_hidden if cfg.use_base_hidden else None,
                            regime_pack=perm_pack,
                        )
                    margin = float(cfg.counterfactual_margin)
                    real_err = (real_out["pred_z"] - targets_z).abs().mean(dim=-1)
                    blank_err = (blank_all_out["pred_z"] - targets_z).abs().mean(dim=-1)
                    perm_err = (perm_out["pred_z"] - targets_z).abs().mean(dim=-1)
                    counterfactual_loss = (
                        F.relu(real_err[cf_mask] - blank_err[cf_mask] + margin).mean()
                        + F.relu(real_err[cf_mask] - perm_err[cf_mask] + margin).mean()
                    )
            # L2 正则化，防止 spike_bias 过大。因为 spike_bias 是直接加在预测上的，如果它过大可能会导致不稳定。
            spike_bias_reg = out["spike_bias"].pow(2).mean()
            # 对于那些本来没有有效新闻的样本，如果模型在这些样本上预测了很大的残差（无论是正的还是负的），都应该被惩罚。
            inactive_residual_loss = torch.zeros((), dtype=torch.float32, device=device)
            if cfg.inactive_residual_weight > 0 and bool((~original_active).any()):
                inactive_residual_loss = out["residual_hat"][~original_active].pow(2).mean()
            # 最终总损失是主损失加上各种辅助损失的加权和。每个辅助损失都有一个权重系数，可以通过 cfg 来调整它们的重要性。
            loss = (
                point_loss
                + cfg.slow_weight * slow_loss
                + cfg.shape_weight * shape_loss
                + cfg.spike_weight * spike_loss
                + cfg.spike_gate_loss_weight * (spike_gate_loss + spike_gate_rate_loss)
                + cfg.consistency_weight * consistency_loss
                + cfg.counterfactual_weight * counterfactual_loss
                + cfg.inactive_residual_weight * inactive_residual_loss
                + cfg.spike_bias_l2 * spike_bias_reg
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss_sum += float(loss.detach().cpu())
            train_steps += 1
            pbar.set_postfix(
                loss=f"{train_loss_sum / max(1, train_steps):.5f}",
                lambda_base=f"{float(out['lambda_base'].detach().cpu().mean()):.3f}",
                active=f"{float((regime_pack['relevance_mass'] > cfg.active_mass_threshold).detach().cpu().float().mean()):.3f}",
                spike=f"{float(out['spike_gate_hard'].detach().cpu().mean()):.3f}",
            )

        val_loss, val_mse, val_mae, base_val_mse, base_val_mae = evaluate_delta_v3(
            model,
            base_backbone=base_backbone,
            decomposer=decomposer,
            data_loader=bundle["val_loader"],
            bank=bank,
            shuffled_bank=shuffled_bank,
            args=args,
            global_zstats=stats,
            device=device,
            cfg=cfg,
            price_winsor_bounds=price_winsor_bounds,
            debug_csv_path=bundle.get("val_residual_debug_csv_path"),
            split_name="val",
        )
        diag = getattr(args, "_last_residual_eval_diag", {}) or {}
        raw_metric = (
            float(diag.get("top10pct_residual_mae", float("inf")))
            if cfg.select_metric == "top10pct_mae"
            else float(val_mae)
        )
        # Plan B selection: binary hard gate + val_mae ordering, no soft penalties.
        #   eps1 : required raw-MAE margin of full-val delta vs base.
        #   eps2 : required raw-MAE gain of real-news over blank/permuted on the active subset.
        # `selection_counterfactual_gain_min` is reused as eps2 in raw-MAE units now
        # (previously interpreted as z-score); the lambda-saturation knob is intentionally
        # dropped from selection and should be controlled via capacity caps instead.
        eps1 = 0.0
        eps2 = float(cfg.selection_counterfactual_gain_min)
        active_mae_val = float(diag.get("active_subset_mae", float("nan")))
        blank_active_val = float(diag.get("blank_active_subset_mae", float("nan")))
        perm_active_val = float(diag.get("permuted_active_subset_mae", float("nan")))
        beats_base = math.isfinite(base_val_mae) and float(val_mae) < float(base_val_mae) - eps1
        blank_ok = (
            math.isfinite(active_mae_val)
            and math.isfinite(blank_active_val)
            and active_mae_val < blank_active_val - eps2
        )
        perm_ok = (
            math.isfinite(active_mae_val)
            and math.isfinite(perm_active_val)
            and active_mae_val < perm_active_val - eps2
        )
        news_ok = blank_ok or perm_ok
        gate_pass = beats_base and news_ok
        metric_now = raw_metric if gate_pass else float("inf")
        if gate_pass:
            gate_reason = "pass"
        elif not beats_base and not news_ok:
            gate_reason = "fail:base+news"
        elif not beats_base:
            gate_reason = "fail:base"
        else:
            gate_reason = "fail:news"
        live_logger.info(
            f"[DELTA_V3][VAL] epoch={epoch + 1} "
            f"val_loss={val_loss:.6f} val_mse={val_mse:.6f} val_mae={val_mae:.6f} "
            f"base_mae={base_val_mae:.6f} active_mae={float(diag.get('active_subset_mae', float('nan'))):.6f} "
            f"blank_active={float(diag.get('blank_active_subset_mae', float('nan'))):.6f} "
            f"blank_gain_z={float(diag.get('active_blank_gain_z', float('nan'))):.4f} "
            f"perm_gain_z={float(diag.get('active_permuted_gain_z', float('nan'))):.4f} "
            f"inactive_gap_pct={float(diag.get('inactive_blank_gap_pct', float('nan'))):.3f} "
            f"perm_active={float(diag.get('permuted_active_subset_mae', float('nan'))):.6f} "
            f"gate={gate_reason} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"lambda={float(diag.get('lambda_base_mean', 0.0)):.4f} "
            f"lambda_sat={float(diag.get('lambda_saturation_pct', 0.0)):.3f} "
            f"shape_gain={float(diag.get('shape_gain_mean', 1.0)):.4f} "
            f"spike_bias={float(diag.get('spike_bias_mean', 0.0)):.4f} "
            f"active_mass_mean={float(diag.get('active_mass_mean', 0.0)):.4f} "
            f"active_mass_median={float(diag.get('active_mass_median', 0.0)):.4f} "
            f"active_mass_max={float(diag.get('active_mass_max', 0.0)):.4f} "
            f"spike_hit={float(diag.get('spike_gate_hit_rate', 0.0)):.4f} "
            f"spike_tgt={float(diag.get('spike_target_hit_rate', 0.0)):.4f}"
        )
        live_logger.info(
            "[DELTA_V3][KNOB_HIST] "
            f"epoch={epoch + 1} "
            f"lambda_active={json.dumps(diag.get('lambda_hist_active', {}), ensure_ascii=False)} "
            f"lambda_inactive={json.dumps(diag.get('lambda_hist_inactive', {}), ensure_ascii=False)} "
            f"shape_active={json.dumps(diag.get('shape_gain_hist_active', {}), ensure_ascii=False)} "
            f"shape_inactive={json.dumps(diag.get('shape_gain_hist_inactive', {}), ensure_ascii=False)} "
            f"spike_active={json.dumps(diag.get('spike_bias_hist_active', {}), ensure_ascii=False)} "
            f"spike_inactive={json.dumps(diag.get('spike_bias_hist_inactive', {}), ensure_ascii=False)}"
        )

        primary_improved = metric_now < best_metric - 1e-6
        # Fallback only fires while no gate-pass epoch has ever been recorded, so a
        # gate-pass ckpt (once it appears) cannot be overwritten by a lower-val_mae
        # but gate-failing epoch.
        fallback_candidate = float(raw_metric)
        fallback_improved = (
            not math.isfinite(best_metric)
            and fallback_candidate < best_fallback_metric - 1e-6
        )
        if primary_improved:
            best_metric = metric_now
            stale_rounds = 0
            os.makedirs(best_ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                },
                os.path.join(best_ckpt_dir, "model.pt"),
            )
            with open(os.path.join(best_ckpt_dir, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump({"cfg": asdict(cfg), "best_metric": best_metric}, handle, ensure_ascii=False, indent=2)
            live_logger.info(f"[DELTA_V3] New best checkpoint saved to {best_ckpt_dir} (metric={best_metric:.6f} gate=pass)")
        elif fallback_improved:
            best_fallback_metric = fallback_candidate
            stale_rounds = 0
            os.makedirs(best_ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "epoch": epoch,
                },
                os.path.join(best_ckpt_dir, "model.pt"),
            )
            with open(os.path.join(best_ckpt_dir, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {"cfg": asdict(cfg), "best_metric": best_fallback_metric, "fallback": True},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            live_logger.info(
                f"[DELTA_V3] Fallback checkpoint saved to {best_ckpt_dir} "
                f"(val_mae={best_fallback_metric:.6f} gate={gate_reason})"
            )
        else:
            stale_rounds += 1
            best_display = best_metric if math.isfinite(best_metric) else best_fallback_metric
            live_logger.info(
                f"[DELTA_V3] stale_rounds={stale_rounds}/{args.early_stop_patience} "
                f"best={best_display:.6f} gate={gate_reason}"
            )

        if stale_rounds >= args.early_stop_patience:
            live_logger.info(f"[DELTA_V3] Early stopping triggered at epoch {epoch + 1}.")
            break

    checkpoint = torch.load(os.path.join(best_ckpt_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)

    test_loss, test_mse, test_mae, base_test_mse, base_test_mae = evaluate_delta_v3(
        model,
        base_backbone=base_backbone,
        decomposer=decomposer,
        data_loader=bundle["test_loader"],
        bank=bank,
        shuffled_bank=shuffled_bank,
        args=args,
        global_zstats=stats,
        device=device,
        cfg=cfg,
        price_winsor_bounds=price_winsor_bounds,
        debug_csv_path=bundle.get("test_residual_debug_csv_path"),
        split_name="test",
        testing=True,
        true_pred_csv_path=bundle.get("true_pred_csv_path"),
        filename=getattr(args, "taskName", "delta_v3"),
    )
    diag = getattr(args, "_last_residual_eval_diag", {}) or {}
    live_logger.info(
        f"[TEST][FINAL] loss(main)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
    )
    live_logger.info(
        f"[TEST][BASE_ONLY] mse(raw)={base_test_mse:.6f} mae(raw)={base_test_mae:.6f}"
    )
    live_logger.info(
        "[TEST][COUNTERFACTUAL] "
        f"active_mae={float(diag.get('active_subset_mae', float('nan'))):.6f} "
        f"blank_active={float(diag.get('blank_active_subset_mae', float('nan'))):.6f} "
        f"inactive_mae={float(diag.get('inactive_subset_mae', float('nan'))):.6f} "
        f"blank_inactive={float(diag.get('blank_inactive_subset_mae', float('nan'))):.6f} "
        f"perm_active={float(diag.get('permuted_active_subset_mae', float('nan'))):.6f}"
    )
    live_logger.info(
        "[TEST][ACTIVE_MASS] "
        f"mean={float(diag.get('active_mass_mean', 0.0)):.4f} "
        f"median={float(diag.get('active_mass_median', 0.0)):.4f} "
        f"max={float(diag.get('active_mass_max', 0.0)):.4f}"
    )
    record_test_results_csv(args, live_logger, test_mse, test_mae, base_mse=base_test_mse, base_mae=base_test_mae)
    draw_pred_true(live_logger, args, bundle.get("true_pred_csv_path"))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_delta_path": best_ckpt_dir,
        "best_metric": best_metric,
    }
