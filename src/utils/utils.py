from __future__ import annotations

import csv
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_id(gpu_id: int):
    print("torch version:", torch.__version__)
    print("compiled with CUDA:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print(f"Using GPU id: {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    print("Using CPU")
    return torch.device("cpu")


def count_tokens(tokenizer, text: str):
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_volatility_bin(df, time_col="", value_col="", window=48, bins=10, dayfirst=True):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=dayfirst)
    df = df.sort_values(time_col)

    recent = df[value_col].iloc[-window:]
    if len(recent) < 2:
        return 0

    vol = recent.std()
    all_std = df[value_col].rolling(window).std().dropna()
    if len(all_std) == 0:
        return 0

    thresholds = np.quantile(all_std, np.linspace(0, 1, bins + 1)[1:-1])
    bin_id = np.digitize(vol, thresholds, right=True)
    return int(min(bin_id, bins - 1))


def draw_pred_true(live_logger, args, true_pred_csv_path):
    try:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        df = pd.read_csv(true_pred_csv_path)
        plt.figure()
        plt.plot(df["true"], label="True Values")
        plt.plot(df["pred"], label="Predicted Values")
        plt.xlabel("Sample Index")
        plt.ylabel("Values")
        plt.title("Predicted and True Values")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(ckpt_dir, f"PredVsTrue_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        live_logger.info(f"Saved Pred vs True plot to {fig_path}")
    except Exception as exc:
        live_logger.error(f"Failed to draw Pred vs True plot: {exc}")


def _sanitize_task_component(value, *, default: str = "na", strip_extension: bool = False) -> str:
    text = "" if value is None else str(value).strip()
    if strip_extension:
        text = os.path.splitext(os.path.basename(text))[0]
    if not text:
        text = default
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def _extract_task_name_parts(raw_task_name: str) -> tuple[str, str]:
    text = "" if raw_task_name is None else str(raw_task_name).strip()
    match = re.match(r"^(?P<base>.+?)__lr(?P<lr>[^_]+)__ga[^_]+__sch[^_]+(?P<suffix>.*)$", text)
    if match:
        base_name = f"{match.group('base')}{match.group('suffix')}" or "task1"
        return base_name, match.group("lr").strip()
    return text or "task1", ""


def build_experiment_task_name(args) -> str:
    raw_task_name = str(
        getattr(args, "_raw_task_name", None) or getattr(args, "taskName", "task1") or "task1"
    ).strip()
    task_base_name, task_lr = _extract_task_name_parts(raw_task_name)
    parts = [
        _sanitize_task_component(task_base_name, default="task1"),
        _sanitize_task_component(getattr(args, "base_backbone", "mlp"), default="mlp"),
        _sanitize_task_component(getattr(args, "news_path", ""), default="no-news", strip_extension=True),
        _sanitize_task_component(getattr(args, "stride", 1), default="1"),
        _sanitize_task_component(getattr(args, "horizon", 1), default="1"),
        _sanitize_task_component(task_lr or getattr(args, "lr", "1e-4"), default="1e-4"),
        _sanitize_task_component(getattr(args, "delta_v3_active_mass_threshold", "0.7"), default="0.7"),
        _sanitize_task_component(
            getattr(args, "delta_v3_text_encoder_model_id", "intfloat-e5-small-v2"),
            default="intfloat-e5-small-v2",
        ),
        _sanitize_task_component(getattr(args, "delta_v3_pretrain_lr", "1e-3"), default="1e-3"),
    ]
    return "_".join(parts)


def record_test_results_csv(args, live_logger, mse, mae, base_mse=None, base_mae=None):
    try:
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "test_results.csv")
        task = build_experiment_task_name(args)
        residual_diag = getattr(args, "_last_residual_eval_diag", {}) or {}

        row = {
            "Task": task,
            "MSE": float(mse),
            "MAE": float(mae),
            "Base_MSE": float(base_mse) if base_mse is not None else float(residual_diag.get("base_mse", np.nan)),
            "Base_MAE": float(base_mae) if base_mae is not None else float(residual_diag.get("base_mae", np.nan)),
            "Skill_Score_MSE": float(residual_diag.get("skill_score_mse", np.nan)),
            "Skill_Score_MAE": float(residual_diag.get("skill_score_mae", np.nan)),
            "Delta_Helped_Rate": float(residual_diag.get("delta_helped_rate", np.nan)),
            "Delta_Helped_Rate_Top10Pct": float(residual_diag.get("delta_helped_rate_top10pct_residual", np.nan)),
            "Top10Pct_Residual_MAE": float(residual_diag.get("top10pct_residual_mae", np.nan)),
            "Regime_Active_Pct": float(residual_diag.get("regime_active_pct", np.nan)),
            "Regime_Days_Mean": float(residual_diag.get("regime_days_mean", np.nan)),
            "Regime_Docs_Mean": float(residual_diag.get("regime_docs_mean", np.nan)),
            "Active_Subset_MSE": float(residual_diag.get("active_subset_mse", np.nan)),
            "Counterfactual_Blank_Active_MSE": float(residual_diag.get("blank_active_subset_mse", np.nan)),
            "Counterfactual_Blank_Active_MAE": float(residual_diag.get("blank_active_subset_mae", np.nan)),
            "Counterfactual_Blank_Inactive_MAE": float(residual_diag.get("blank_inactive_subset_mae", np.nan)),
            "Counterfactual_Permuted_Active_MSE": float(residual_diag.get("permuted_active_subset_mse", np.nan)),
            "Counterfactual_Permuted_Active_MAE": float(residual_diag.get("permuted_active_subset_mae", np.nan)),
            "Inactive_Blank_Gap_Pct": float(residual_diag.get("inactive_blank_gap_pct", np.nan)),
            "Lambda_Base_Mean": float(residual_diag.get("lambda_base_mean", np.nan)),
            "Shape_Gain_Mean": float(residual_diag.get("shape_gain_mean", np.nan)),
            "Spike_Bias_Mean": float(residual_diag.get("spike_bias_mean", np.nan)),
            "Relevance_Mass_Mean": float(residual_diag.get("relevance_mass_mean", np.nan)),
            "Spike_Gate_Hit_Rate": float(residual_diag.get("spike_gate_hit_rate", np.nan)),
            "Spike_Target_Hit_Rate": float(residual_diag.get("spike_target_hit_rate", np.nan)),
        }
        new_row = pd.DataFrame([row])
        ordered_cols = list(row.keys())
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for col in ordered_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[ordered_cols]
            df = df[df["Task"] != task]
            out_df = pd.concat([df, new_row], ignore_index=True)
        else:
            out_df = new_row
        out_df = out_df[ordered_cols]
        out_df.to_csv(csv_path, index=False)
        live_logger.info(f"[RESULT_TASK] {task}")
        live_logger.info(f"Saved test results to {csv_path} (upsert by Task)")
    except Exception as exc:
        live_logger.error(f"Failed to save test results to CSV: {exc}")


def print_prompt_stats(live_logger, dataStatistic):
    live_logger.info(f"Prompt number: {dataStatistic.prompt_num}")
    live_logger.info(f"News number total: {dataStatistic.news_num_total}")
    live_logger.info(f"Max news number per prompt: {dataStatistic.max_news_num_per_prompt}")
    live_logger.info(f"Min news number per prompt: {dataStatistic.min_news_num_per_prompt}")
    live_logger.info(f"Mean news number per prompt: {dataStatistic.mean_news_num_per_prompt}")
    live_logger.info(f"Prompt with max news number: {dataStatistic.the_prompt_with_most_news_num}")
    live_logger.info(f"Prompt with max news length: {dataStatistic.prompt_with_max_news_len}")
    live_logger.info(f"Prompt with max total length: {dataStatistic.prompt_with_max_total_len}")


def draw_metric_trend(args, live_logger, val_loss_per_epoch, mse_loss_per_epoch, mae_loss_per_epoch):
    ckpt_dir = os.path.join("./checkpoints", args.taskName)
    os.makedirs(ckpt_dir, exist_ok=True)
    epochs = list(range(1, len(val_loss_per_epoch) + 1))

    plt.figure()
    plt.plot(epochs, val_loss_per_epoch, label="Val Loss (z-MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ckpt_dir, f"ValLoss_{args.taskName}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, mse_loss_per_epoch, label="Val MSE (raw)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Validation MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ckpt_dir, f"ValMSE_{args.taskName}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, mae_loss_per_epoch, label="Val MAE (raw)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Validation MAE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ckpt_dir, f"ValMAE_{args.taskName}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    live_logger.info(f"Saved metric curves under {ckpt_dir}")
