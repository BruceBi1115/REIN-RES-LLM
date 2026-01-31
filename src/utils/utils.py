import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_from_id(gpu_id: int):
    print("torch version:", torch.__version__)
    print("compiled with CUDA:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print(f'Using GPU id: {gpu_id}')
        return torch.device(f'cuda:{gpu_id}')
    else:
        print('Using CPU')
        return torch.device('cpu')

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
        p = f"./checkpoints/{args.taskName}"
        os.makedirs(p, exist_ok=True)
        df = pd.read_csv(true_pred_csv_path)
        plt.figure()
        plt.plot(df["true"], label="True Values")
        plt.plot(df["pred"], label="Predicted Values")
        plt.xlabel("Sample Index")
        plt.ylabel("Values")
        plt.title("Predicted and True Values (final prediction)")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"PredVsTrue_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        live_logger.info(f"Saved Pred vs True plot to {fig_path}")
    except Exception as e:
        live_logger.error(f"Failed to draw Pred vs True plot: {e}")


def record_test_results_csv(args, live_logger, mse, mae):
    try:
        p = f"./results"
        os.makedirs(p, exist_ok=True)
        csv_path = os.path.join(p, f"test_results.csv")
        task = f"{args.taskName}_{args.stage}_news_{args.news_path}_{args.patch_dropout}_{args.head_dropout}_{args.news_dropout}_{args.delta_null_lambda}_{args.delta_margin_lambda}_{args.delta_adv_margin}_basefrac_{args.residual_base_frac}_epochs_{args.epochs}_lookback_{args.news_window_days}_topK_{args.news_topK}"
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Task", "MSE", "MAE"])
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([task, mse, mae])
        live_logger.info(f"Saved test results to {csv_path}")
    except Exception as e:
        live_logger.error(f"Failed to save test results to CSV: {e}")


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
    p = f"./checkpoints/{args.taskName}"
    os.makedirs(p, exist_ok=True)
    epochs = list(range(1, len(val_loss_per_epoch) + 1))

    plt.figure()
    plt.plot(epochs, val_loss_per_epoch, label="Val Loss (z-MSE, final pred)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss (final prediction z-space MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(p, f"ValLoss_{args.taskName}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, mse_loss_per_epoch, label="Val MSE (raw, final pred)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Validation MSE (final prediction raw scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(p, f"ValMSE_{args.taskName}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, mae_loss_per_epoch, label="Val MAE (raw, final pred)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Validation MAE (final prediction raw scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(p, f"ValMAE_{args.taskName}.png"), dpi=200, bbox_inches="tight")
    plt.close()

    live_logger.info(f"Saved metric curves under {p}")

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