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
from ..data_construction.prompt import load_templates
from ..delta_news_hooks import build_news_api_adapter
from ..news_rules import load_news
from ..refine.cache import (
    _build_news_doc_meta_index,
    _news_cache_mode,
    _prime_news_api_key_state,
    _resolve_news_doc_cache_mode,
    _validate_news_identity_source,
)
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
    _matched_news_time_range,
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
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], format="ISO8601", errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], dayfirst=dayfirst, errors="coerce")

    return parsed


def setup_env_and_data(args):
    if not hasattr(args, "_raw_task_name"):
        args._raw_task_name = str(getattr(args, "taskName", "task1") or "task1").strip()
    args.taskName = build_experiment_task_name(args)

    stage = str(getattr(args, "stage", "all")).lower()
    base_backbone_name = str(getattr(args, "base_backbone", "dlinear"))

    def _safe_name(s: str) -> str:
        s = str(s).strip()
        s = s.replace("/", "-").replace("\\", "-")
        s = re.sub(r"\s+", "_", s)
        return s if s else "na"

    filename = _safe_name(args.taskName)
    log_filename = filename + ".log"

    live_logger, live_path, log_jsonl = setup_live_logger(
        save_dir=args.save_dir + "/" + args.taskName, filename=log_filename
    )
    print(f"[live log] {live_path}  (实时查看: tail -f '{live_path}')")
    _resolve_news_doc_cache_mode(args)
    _prime_news_api_key_state(args)
    _log_run_args(args, live_logger)
    _log_cache_decision(args, live_logger)
    _log_enabled_mechanisms(args, live_logger, stage=stage)
    if bool(getattr(args, "_require_news_api_adapter", False)) and not bool(getattr(args, "_news_api_key_detected", False)):
        raise RuntimeError(
            "API mode is required to build refined news cache, but no API key was found. "
            "Provide OPENAI_API_KEY or --news_api_key_path."
        )
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

    train_df[args.time_col] = _parse_series_time_values(train_df[args.time_col], dayfirst=args.dayFirst)
    val_df[args.time_col] = _parse_series_time_values(val_df[args.time_col], dayfirst=args.dayFirst)
    test_df[args.time_col] = _parse_series_time_values(test_df[args.time_col], dayfirst=args.dayFirst)

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
        if _news_cache_mode(args) != "disabled":
            _validate_news_identity_source(
                news_df,
                args,
                time_col=str(getattr(args, "news_time_col", "date")),
                text_col=str(getattr(args, "news_text_col", "content")),
            )
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

    for issue in _split_time_order_issues(train_df, val_df, test_df, time_col=args.time_col):
        live_logger.warning(issue)

    train_raw_min, train_raw_max = _df_series_time_range(train_df, args.time_col)
    val_raw_min, val_raw_max = _df_series_time_range(val_df, args.time_col)
    test_raw_min, test_raw_max = _df_series_time_range(test_df, args.time_col)
    live_logger.info(f"[DATA_RANGE][TRAIN] series={_format_ts_range(train_raw_min, train_raw_max)}")
    live_logger.info(f"[DATA_RANGE][VAL] series={_format_ts_range(val_raw_min, val_raw_max)}")
    live_logger.info(f"[DATA_RANGE][TEST] series={_format_ts_range(test_raw_min, test_raw_max)}")

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
