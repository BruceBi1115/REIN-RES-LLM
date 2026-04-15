import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecasting With Delta V3")

    parser.add_argument("--taskName", type=str, default="task1")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--stage", type=str, default="all", choices=["base", "delta", "all"])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    parser.add_argument("--dataset_key", type=str, default="dataset")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--time_col", type=str, default="date")
    parser.add_argument("--value_col", type=str, default="value")
    parser.add_argument("--id_col", type=str, default="")
    parser.add_argument("--unit", type=str, default="")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--region", type=str, default="")
    parser.add_argument("--freq_min", type=str, default="")
    parser.add_argument("--dayFirst", action="store_true", default=False)

    parser.add_argument("--history_len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--scheduler", type=int, default=1, choices=[0, 1])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--select_metric", type=str, default="mae", choices=["loss", "mse", "mae"])
    parser.add_argument("--eval_progress_bar", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eval_progress_leave", type=int, default=0, choices=[0, 1])
    parser.add_argument("--normalization_mode", type=str, default="robust_quantile", choices=["robust_quantile", "zscore"])
    parser.add_argument("--norm_quantile_low", type=float, default=0.25)
    parser.add_argument("--norm_quantile_high", type=float, default=0.75)
    parser.add_argument("--zscore_eps", type=float, default=1e-6)
    parser.add_argument("--spike_clip_threshold", type=float, default=0.0,
                        help="Clip raw values whose absolute value exceeds this threshold. 0 = disabled.")
    parser.add_argument("--volatility_bin_tiers", type=int, default=10)
    parser.add_argument("--residual_base_frac", type=float, default=0.3)
    parser.add_argument("--residual_min_base_epochs", type=int, default=1)
    parser.add_argument("--residual_min_delta_epochs", type=int, default=1)
    parser.add_argument("--residual_base_epochs", type=int, default=-1)

    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--patch_stride", type=int, default=4)

    parser.add_argument("--base_backbone", type=str, default="mlp", choices=["mlp", "dlinear"])
    parser.add_argument("--base_hidden_dim", type=int, default=256)
    parser.add_argument("--base_moving_avg", type=int, default=25)
    parser.add_argument("--base_dropout", type=float, default=0.0)
    parser.add_argument("--base_loss", type=str, default="smooth_l1", choices=["smooth_l1", "mse", "mae"])
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--base_weight_decay", type=float, default=1e-5)

    parser.add_argument("--news_path", type=str, default="")
    parser.add_argument("--news_time_col", type=str, default="date")
    parser.add_argument("--news_text_col", type=str, default="content")
    parser.add_argument("--news_tz", type=str, default="")
    parser.add_argument("--news_window_days", type=int, default=1)
    parser.add_argument("--news_api_enable", type=int, default=0, choices=[0, 1])
    parser.add_argument("--news_api_model", type=str, default="gpt-5.1")
    parser.add_argument("--news_api_key_path", type=str, default=".secrets/api_key.txt")
    parser.add_argument("--news_api_base_url", type=str, default="")
    parser.add_argument("--news_api_timeout_sec", type=float, default=30.0)
    parser.add_argument("--news_api_max_retries", type=int, default=2)

    parser.add_argument("--delta_v3_regime_bank_path", type=str, default="")
    parser.add_argument("--delta_v3_regime_bank_build", type=int, default=0, choices=[0, 1])
    parser.add_argument("--delta_v3_schema_variant", type=str, default="load", choices=["load", "price", "gas_demand"])
    parser.add_argument("--delta_v3_text_encoder_model_id", type=str, default="intfloat/e5-small-v2")
    parser.add_argument("--delta_v3_text_encoder_max_length", type=int, default=256)
    parser.add_argument("--delta_v3_regime_tau_days", type=float, default=5.0)
    parser.add_argument("--delta_v3_regime_ema_alpha", type=float, default=0.5)
    parser.add_argument("--delta_v3_regime_ema_window", type=int, default=5)

    parser.add_argument(
        "--delta_v3_arch",
        type=str,
        default="patchtst_regime_modulation",
        choices=["patchtst_regime_modulation"],
    )
    parser.add_argument("--delta_v3_hidden_size", type=int, default=128)
    parser.add_argument("--delta_v3_num_layers", type=int, default=2)
    parser.add_argument("--delta_v3_num_heads", type=int, default=4)
    parser.add_argument("--delta_v3_patch_len", type=int, default=8)
    parser.add_argument("--delta_v3_patch_stride", type=int, default=4)
    parser.add_argument("--delta_v3_dropout", type=float, default=0.1)
    parser.add_argument("--delta_v3_use_base_hidden", type=int, default=1, choices=[0, 1])

    parser.add_argument("--delta_v3_slow_weight", type=float, default=1.0)
    parser.add_argument("--delta_v3_shape_weight", type=float, default=1.0)
    parser.add_argument("--delta_v3_spike_weight", type=float, default=1.0)
    parser.add_argument("--delta_v3_spike_gate_threshold", type=float, default=0.8)
    parser.add_argument("--delta_v3_spike_k", type=float, default=3.0)
    parser.add_argument("--delta_v3_spike_target_pct", type=float, default=0.10)
    parser.add_argument("--delta_v3_spike_gate_loss_weight", type=float, default=0.25)

    parser.add_argument("--delta_v3_news_blank_prob", type=float, default=0.3)
    parser.add_argument("--delta_v3_consistency_weight", type=float, default=0.05)
    parser.add_argument("--delta_v3_counterfactual_weight", type=float, default=0.1)
    parser.add_argument("--delta_v3_counterfactual_margin", type=float, default=0.02)
    parser.add_argument("--delta_v3_inactive_residual_weight", type=float, default=0.1)
    parser.add_argument("--delta_v3_spike_bias_l2", type=float, default=1e-3)
    parser.add_argument("--delta_v3_active_mass_threshold", type=float, default=0.7)
    parser.add_argument("--delta_v3_lambda_min", type=float, default=0.05)
    parser.add_argument("--delta_v3_lambda_ts_cap", type=float, default=0.45)
    parser.add_argument("--delta_v3_lambda_news_cap", type=float, default=0.20)
    parser.add_argument("--delta_v3_lambda_max", type=float, default=0.60)
    parser.add_argument("--delta_v3_shape_gain_cap", type=float, default=0.30)
    parser.add_argument("--delta_v3_spike_bias_cap", type=float, default=0.75)
    parser.add_argument("--delta_v3_selection_counterfactual_gain_min", type=float, default=0.01)
    parser.add_argument("--delta_v3_selection_lambda_saturation_max_pct", type=float, default=0.35)
    parser.add_argument("--delta_v3_hard_residual_frac", type=float, default=0.6)
    parser.add_argument("--delta_v3_hard_residual_pct", type=float, default=0.10)

    parser.add_argument("--delta_v3_pretrain_epochs", type=int, default=12)
    parser.add_argument("--delta_v3_pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--delta_v3_price_winsor_low", type=float, default=0.005)
    parser.add_argument("--delta_v3_price_winsor_high", type=float, default=0.995)

    parser.add_argument("--delta_v3_grad_clip", type=float, default=1.0)
    parser.add_argument("--delta_v3_eval_permutation_seed", type=int, default=2024)
    parser.add_argument("--delta_v3_select_metric", type=str, default="mae", choices=["mae", "top10pct_mae"])

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from src.base_delta_decoouple_trainer import main as main_train

    main_train(args)


if __name__ == "__main__":
    main()
