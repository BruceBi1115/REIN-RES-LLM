from __future__ import annotations

import csv
import gc
import math
import os
import re
import shutil
from collections import deque

import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ..base_backbone import build_base_backbone, load_base_backbone_checkpoint, save_base_backbone_checkpoint
from ..data_construction.data import make_loader
from ..news_rules import load_news
from ..utils.logger import setup_live_logger
from ..utils.residual_utils import split_two_stage_epochs
from ..utils.utils import (
    build_experiment_task_name,
    compute_volatility_bin,
    device_from_id,
    draw_pred_true,
    record_test_results_csv,
    set_seed,
)
from .common import (
    _coerce_global_zstats,
    _compute_global_zstats_from_train_df,
    _df_series_time_range,
    _format_ts_range,
    _log_cache_decision,
    _log_enabled_mechanisms,
    _log_prompt_stats_if_available,
    _log_run_args,
    _point_loss,
    _prepend_split_history,
    _split_time_order_issues,
    _z_batch_tensors,
    dataStatistic,
    evaluate_metrics_backbone,
)
from ..delta.stage import train_delta_stage


def _parse_series_time_values(series, *, dayfirst: bool):
    if isinstance(series, pd.Series) and pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    text = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")

    exact_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    if dayfirst:
        exact_formats.extend([
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
        ])
    else:
        exact_formats.extend([
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
        ])

    for fmt in exact_formats:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format=fmt, errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format="ISO8601", errors="coerce")

    return parsed


def _parse_split_time_values(raw_splits: dict[str, pd.Series], *, time_col: str, dayfirst: bool):
    parsed = {
        split_name: _parse_series_time_values(series, dayfirst=dayfirst)
        for split_name, series in raw_splits.items()
    }
    nat_counts = {split_name: int(series.isna().sum()) for split_name, series in parsed.items()}
    issues = _split_time_order_issues(
        pd.DataFrame({time_col: parsed["train"]}),
        pd.DataFrame({time_col: parsed["val"]}),
        pd.DataFrame({time_col: parsed["test"]}),
        time_col=time_col,
    )
    return parsed, nat_counts, issues


def setup_env_and_data(args):
    if not hasattr(args, "_raw_task_name"):
        args._raw_task_name = str(getattr(args, "taskName", "task1") or "task1").strip()
    args.taskName = build_experiment_task_name(args)
    stage = str(getattr(args, "stage", "all")).lower()

    def _safe_name(s: str) -> str:
        s = str(s).strip()
        s = s.replace("/", "-").replace("\\", "-")
        s = re.sub(r"\s+", "_", s)
        return s if s else "na"

    filename = _safe_name(args.taskName)
    log_filename = filename + ".log"
    live_logger, live_path, log_jsonl = setup_live_logger(
        save_dir=os.path.join(args.save_dir, args.taskName),
        filename=log_filename,
    )
    print(f"[live log] {live_path}  (实时查看: tail -f '{live_path}')")
    _log_run_args(args, live_logger)
    _log_cache_decision(args, live_logger)
    _log_enabled_mechanisms(args, live_logger, stage=stage)

    ckpt_dir = os.path.join("./checkpoints", args.taskName)
    os.makedirs(ckpt_dir, exist_ok=True)
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
    norm_msg = (
        "[NORMALIZE] train_df stats: "
        f"mode={global_zstats['normalization_mode']} "
        f"center_global={global_zstats['center_global']:.6f} "
        f"scale_global={global_zstats['scale_global']:.6f}"
    )
    if global_zstats["normalization_mode"] == "robust_quantile":
        norm_msg += (
            f" q_low={global_zstats.get('quantile_low', float('nan')):.2f}"
            f" q_high={global_zstats.get('quantile_high', float('nan')):.2f}"
            f" q_low_value={global_zstats.get('quantile_low_value', float('nan')):.6f}"
            f" q_high_value={global_zstats.get('quantile_high_value', float('nan')):.6f}"
        )
    live_logger.info(norm_msg)
    spike_clip = float(getattr(args, "spike_clip_threshold", 0.0) or 0.0)
    if spike_clip > 0:
        live_logger.info(f"[SPIKE_CLIP] Active: raw values clipped to [-{spike_clip}, {spike_clip}]")

    raw_time_splits = {
        "train": train_df[args.time_col].copy(),
        "val": val_df[args.time_col].copy(),
        "test": test_df[args.time_col].copy(),
    }
    parsed_splits, nat_counts, split_issues = _parse_split_time_values(
        raw_time_splits,
        time_col=args.time_col,
        dayfirst=bool(args.dayFirst),
    )

    alternate_dayfirst = not bool(args.dayFirst)
    alternate_parsed_splits, alternate_nat_counts, alternate_split_issues = _parse_split_time_values(
        raw_time_splits,
        time_col=args.time_col,
        dayfirst=alternate_dayfirst,
    )
    primary_nat_total = sum(nat_counts.values())
    alternate_nat_total = sum(alternate_nat_counts.values())
    should_flip_dayfirst = False
    if split_issues and not alternate_split_issues and alternate_nat_total <= primary_nat_total:
        should_flip_dayfirst = True
    elif not split_issues and not alternate_split_issues and alternate_nat_total < primary_nat_total:
        should_flip_dayfirst = True

    if should_flip_dayfirst:
        live_logger.warning(
            "[TIME_PARSE] configured dayFirst=%s produced split-order issues or extra NaT rows; "
            "automatically switching to dayFirst=%s. primary_nat=%s alternate_nat=%s"
            % (
                bool(args.dayFirst),
                alternate_dayfirst,
                nat_counts,
                alternate_nat_counts,
            )
        )
        args.dayFirst = alternate_dayfirst
        parsed_splits = alternate_parsed_splits
        nat_counts = alternate_nat_counts
        split_issues = alternate_split_issues

    train_df[args.time_col] = parsed_splits["train"]
    val_df[args.time_col] = parsed_splits["val"]
    test_df[args.time_col] = parsed_splits["test"]
    live_logger.info(f"[TIME_PARSE] using dayFirst={bool(args.dayFirst)} nat_counts={nat_counts}")

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

    loader_kwargs = dict(
        time_col=args.time_col,
        value_col=args.value_col,
        L=args.history_len,
        H=args.horizon,
        stride=args.stride,
        batch_size=args.batch_size,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )
    train_loader = make_loader(train_df, shuffle=True, **loader_kwargs)
    train_eval_loader = make_loader(train_df, shuffle=False, **loader_kwargs)
    val_loader = make_loader(
        val_loader_df,
        shuffle=False,
        min_target_time=val_min_target_time if not isinstance(val_min_target_time, dict) else None,
        min_target_time_by_id=val_min_target_time if isinstance(val_min_target_time, dict) else None,
        **loader_kwargs,
    )
    test_loader = make_loader(
        test_loader_df,
        shuffle=False,
        min_target_time=test_min_target_time if not isinstance(test_min_target_time, dict) else None,
        min_target_time_by_id=test_min_target_time if isinstance(test_min_target_time, dict) else None,
        **loader_kwargs,
    )

    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)
        text_col = str(getattr(args, "news_text_col", "content"))
        if text_col in news_df.columns:
            news_df = news_df.loc[news_df[text_col].fillna("").astype(str).str.strip().ne("")].reset_index(drop=True)
    news_time_min = news_df[args.news_time_col].min() if len(news_df) > 0 else None
    news_time_max = news_df[args.news_time_col].max() if len(news_df) > 0 else None
    live_logger.info(
        f"[NEWS_DATA] total_rows={int(len(news_df))} time_range={_format_ts_range(news_time_min, news_time_max)}"
    )

    for issue in split_issues:
        live_logger.warning(issue)

    train_raw_min, train_raw_max = _df_series_time_range(train_df, args.time_col)
    val_raw_min, val_raw_max = _df_series_time_range(val_df, args.time_col)
    test_raw_min, test_raw_max = _df_series_time_range(test_df, args.time_col)
    live_logger.info(f"[DATA_RANGE][TRAIN] series={_format_ts_range(train_raw_min, train_raw_max)}")
    live_logger.info(f"[DATA_RANGE][VAL] series={_format_ts_range(val_raw_min, val_raw_max)}")
    live_logger.info(f"[DATA_RANGE][TEST] series={_format_ts_range(test_raw_min, test_raw_max)}")

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
        "ans_json_path": ans_json_path,
        "true_pred_csv_path": true_pred_csv_path,
        "val_residual_debug_csv_path": val_residual_debug_csv_path,
        "test_residual_debug_csv_path": test_residual_debug_csv_path,
        "device": device,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "news_df": news_df,
        "patch_len": patch_len,
        "volatility_bin": volatility_bin,
        "volatility_bin_val": volatility_bin_val,
        "volatility_bin_test": volatility_bin_test,
        "global_zstats": global_zstats,
        "test_filename": filename,
    }


def train_base_stage(args, bundle):
    live_logger = bundle["live_logger"]
    device = bundle["device"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    global_zstats = _coerce_global_zstats(bundle.get("global_zstats", None), args, required=True)

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
        f"epochs={base_epochs} (no external news fusion)"
    )
    live_logger.info("-----------------------------------------------------")

    base_train_model = build_base_backbone(
        backbone_name=getattr(args, "base_backbone", "dlinear"),
        history_len=int(args.history_len),
        horizon=int(args.horizon),
        hidden_dim=int(getattr(args, "base_hidden_dim", 256)),
        moving_avg=int(getattr(args, "base_moving_avg", 25)),
        dropout=float(getattr(args, "base_dropout", 0.0)),
    ).to(device)

    base_lr = float(getattr(args, "base_lr", -1.0))
    if base_lr <= 0:
        base_lr = float(args.lr)
    base_wd = float(getattr(args, "base_weight_decay", -1.0))
    if base_wd < 0:
        base_wd = float(args.weight_decay)

    optim_base = AdamW(base_train_model.parameters(), lr=base_lr, weight_decay=base_wd)

    num_batches = len(train_loader)
    total_opt_steps_base = math.ceil((num_batches * max(1, base_epochs)) / max(1, args.grad_accum))
    warmup_steps_base = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps_base)
    warmup_steps_base = min(warmup_steps_base, max(0, total_opt_steps_base - 1))

    scheduler_base = None
    if args.scheduler == 1:
        scheduler_base = get_cosine_schedule_with_warmup(
            optim_base,
            num_warmup_steps=warmup_steps_base,
            num_training_steps=total_opt_steps_base,
        )

    best_base_metric = float("inf")
    stale_rounds = 0
    loss_window = deque(maxlen=50)
    global_step = 0

    for epoch in range(base_epochs):
        pbar = tqdm(train_loader, desc=f"[BASE] Epoch {epoch + 1}/{base_epochs}")
        for _, batch in enumerate(pbar):
            history_z, targets_z, _ = _z_batch_tensors(batch, args, global_zstats=global_zstats)
            history_z = history_z.to(device)
            targets_z = targets_z.to(device)

            base_train_model.train()
            pred_z = base_train_model(history_z)
            loss = _point_loss(pred_z, targets_z, mode=getattr(args, "base_loss", "smooth_l1")) / args.grad_accum
            loss.backward()

            loss_window.append(float(loss.detach().cpu()))
            if global_step % 10 == 0 and loss_window:
                pbar.set_postfix(train_loss=f"{sum(loss_window) / len(loss_window):.6f}")

            if (global_step + 1) % args.grad_accum == 0:
                optim_base.step()
                if scheduler_base is not None:
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
            f"[BASE][EVAL] epoch={epoch + 1} "
            f"val_loss(normMSE)={val_loss:.6f} val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
        )

        best_base_path = os.path.join("./checkpoints", args.taskName, f"best_base_{args.taskName}")
        if metric_now < best_base_metric - 1e-6:
            best_base_metric = metric_now
            stale_rounds = 0
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
            live_logger.info(f"[BASE] Early stopping triggered at epoch {epoch + 1}.")
            break

    del base_train_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_base_path = os.path.join("./checkpoints", args.taskName, f"best_base_{args.taskName}")
    if not os.path.exists(best_base_path):
        raise FileNotFoundError(f"Base checkpoint not found: {best_base_path}")

    return {
        "best_base_path": best_base_path,
        "device": device,
        "live_logger": live_logger,
        "best_base_metric": best_base_metric,
        "global_zstats": global_zstats,
    }


def testing_base(test_loader, args, device, live_logger, true_pred_csv_path, global_zstats):
    if test_loader is None:
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_base_path = os.path.join("./checkpoints", args.taskName, f"best_base_{args.taskName}")
    model_best, base_meta = load_base_backbone_checkpoint(best_base_path, device=device, is_trainable=False)
    live_logger.info(f"Loaded best BASE backbone for testing: {base_meta.get('backbone_name')} (final = base-only).")
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
    tqdm.write(f"[TEST][FINAL] loss(normMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
    live_logger.info(f"[TEST][FINAL] loss(normMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
    record_test_results_csv(args, live_logger, test_mse, test_mae)
    draw_pred_true(live_logger, args, true_pred_csv_path)


def main(args):
    bundle = setup_env_and_data(args)
    stage = bundle["stage"]

    ans_json_path = bundle["ans_json_path"]
    true_pred_csv_path = bundle["true_pred_csv_path"]
    val_residual_debug_csv_path = bundle.get("val_residual_debug_csv_path")
    test_residual_debug_csv_path = bundle.get("test_residual_debug_csv_path")

    with open(ans_json_path, "w", encoding="utf-8"):
        pass
    with open(true_pred_csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pred", "true"])
    for extra_csv_path in [val_residual_debug_csv_path, test_residual_debug_csv_path]:
        if extra_csv_path and os.path.exists(extra_csv_path):
            os.remove(extra_csv_path)

    def _resolve_base_ckpt():
        return os.path.join("./checkpoints", args.taskName, f"best_base_{args.taskName}")

    if stage == "base":
        cfg = train_base_stage(args, bundle)
        testing_base(
            bundle["test_loader"],
            args,
            bundle["device"],
            bundle["live_logger"],
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

    if stage == "all":
        cfg_base = train_base_stage(args, bundle)
        train_delta_stage(
            args,
            bundle,
            best_base_path=cfg_base["best_base_path"],
            best_base_metric=float(cfg_base["best_base_metric"]),
        )
        return

    raise ValueError(f"Unsupported stage: {stage}")
