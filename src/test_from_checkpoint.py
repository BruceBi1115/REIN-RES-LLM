from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run import build_parser
from src.base.common import _coerce_global_zstats, evaluate_metrics_backbone
from src.base.stage import setup_env_and_data
from src.base_backbone import load_base_backbone_checkpoint
from src.delta_v3.config import DeltaV3Config
from src.delta_v3.model import build_delta_v3_model
from src.delta_v3.targets import ResidualTargetDecomposer, compute_residual_calendar_baseline
from src.delta_v3.trainer import _prepare_regime_bank, evaluate_delta_v3
from src.utils.utils import draw_pred_true, record_test_results_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Test saved base/delta_v3 checkpoints without retraining."
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Training checkpoint root containing best_base_* and best_delta_v3_* subdirectories.",
    )
    parser.add_argument(
        "--base_checkpoint_path",
        type=str,
        default="",
        help="Path to a saved base checkpoint directory, or its base_backbone.pt/meta.json file.",
    )
    parser.add_argument(
        "--delta_checkpoint_path",
        type=str,
        default="",
        help="Path to a saved delta checkpoint directory, or its model.pt/meta.json file.",
    )
    return parser


def _normalize_ckpt_dir(path: str, *, expected_file: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if p.is_file():
        return str(p.parent)
    if p.is_dir():
        return str(p)
    return raw


def _find_child_ckpt(root: str, *, prefix: str, required_file: str) -> str:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {root}")
    if root_path.name.startswith(prefix) and (root_path / required_file).exists():
        return str(root_path)
    matches = [
        str(child)
        for child in root_path.iterdir()
        if child.is_dir() and child.name.startswith(prefix) and (child / required_file).exists()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No {prefix}* checkpoint with {required_file} found under checkpoint_dir={root}"
        )
    raise RuntimeError(
        f"Multiple {prefix}* checkpoints found under checkpoint_dir={root}: {matches}. "
        "Pass the exact --base_checkpoint_path/--delta_checkpoint_path."
    )


def _resolve_checkpoint_paths(args) -> tuple[str, str]:
    base_path = _normalize_ckpt_dir(
        getattr(args, "base_checkpoint_path", ""), expected_file="base_backbone.pt"
    )
    delta_path = _normalize_ckpt_dir(
        getattr(args, "delta_checkpoint_path", ""), expected_file="model.pt"
    )
    root = str(getattr(args, "checkpoint_dir", "") or "").strip()
    if root:
        if not base_path:
            base_path = _find_child_ckpt(root, prefix="best_base_", required_file="base_backbone.pt")
        if not delta_path and str(getattr(args, "stage", "")).lower() == "all":
            delta_path = _find_child_ckpt(root, prefix="best_delta_v3_", required_file="model.pt")
    return base_path, delta_path


def _require_checkpoint_paths(stage: str, base_path: str, delta_path: str) -> None:
    if stage not in {"base", "all"}:
        raise ValueError("--stage must be either 'base' or 'all' for checkpoint testing.")
    if not base_path:
        raise ValueError("--base_checkpoint_path or --checkpoint_dir is required.")
    if not os.path.exists(os.path.join(base_path, "base_backbone.pt")):
        raise FileNotFoundError(f"Base model file missing: {os.path.join(base_path, 'base_backbone.pt')}")
    if not os.path.exists(os.path.join(base_path, "meta.json")):
        raise FileNotFoundError(f"Base meta file missing: {os.path.join(base_path, 'meta.json')}")
    if stage == "all":
        if not delta_path:
            raise ValueError("--delta_checkpoint_path or --checkpoint_dir is required when --stage all.")
        if not os.path.exists(os.path.join(delta_path, "model.pt")):
            raise FileNotFoundError(f"Delta model file missing: {os.path.join(delta_path, 'model.pt')}")
        if not os.path.exists(os.path.join(delta_path, "meta.json")):
            raise FileNotFoundError(f"Delta meta file missing: {os.path.join(delta_path, 'meta.json')}")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_delta_cfg(delta_path: str, args) -> DeltaV3Config:
    meta = _load_json(os.path.join(delta_path, "meta.json"))
    cfg_payload = meta.get("cfg")
    if not isinstance(cfg_payload, dict):
        ckpt = torch.load(os.path.join(delta_path, "model.pt"), map_location="cpu")
        cfg_payload = ckpt.get("cfg", {})
    if not isinstance(cfg_payload, dict) or not cfg_payload:
        raise RuntimeError(f"Delta checkpoint does not contain cfg metadata: {delta_path}")

    default_cfg = DeltaV3Config.from_args(args)
    merged = {field.name: getattr(default_cfg, field.name) for field in fields(DeltaV3Config)}
    merged.update({k: v for k, v in cfg_payload.items() if k in merged})
    return DeltaV3Config(**merged)


def _sync_args_from_base_meta(args, base_path: str) -> dict:
    meta = _load_json(os.path.join(base_path, "meta.json"))
    args.base_backbone = str(meta.get("backbone_name", getattr(args, "base_backbone", "dlinear")))
    args.history_len = int(meta.get("history_len", getattr(args, "history_len", 48)))
    args.horizon = int(meta.get("horizon", getattr(args, "horizon", 48)))
    args.base_hidden_dim = int(meta.get("hidden_dim", getattr(args, "base_hidden_dim", 256)))
    args.base_moving_avg = int(meta.get("moving_avg", getattr(args, "base_moving_avg", 25)))
    args.base_dropout = float(meta.get("dropout", getattr(args, "base_dropout", 0.0)))
    args.patch_len = int(meta.get("patch_len", getattr(args, "patch_len", 4)))
    args.patch_stride = int(meta.get("patch_stride", getattr(args, "patch_stride", 4)))
    if "normalization_mode" in meta:
        args.normalization_mode = str(meta["normalization_mode"])
    if "quantile_low" in meta:
        args.norm_quantile_low = float(meta["quantile_low"])
    if "quantile_high" in meta:
        args.norm_quantile_high = float(meta["quantile_high"])
    return meta


def _sync_args_from_delta_cfg(args, cfg: DeltaV3Config) -> None:
    direct_arg_names = {
        "dataset_key",
        "news_path",
        "history_len",
        "horizon",
    }
    prefixed_arg_names = {
        "schema_variant",
        "regime_bank_path",
        "refined_bank_build",
        "text_encoder_model_id",
        "text_encoder_max_length",
        "regime_tau_days",
        "regime_ema_alpha",
        "regime_ema_window",
        "arch",
        "hidden_size",
        "num_layers",
        "num_heads",
        "patch_len",
        "patch_stride",
        "dropout",
        "use_base_hidden",
        "slow_weight",
        "shape_weight",
        "spike_weight",
        "spike_gate_threshold",
        "spike_k",
        "spike_target_pct",
        "spike_gate_loss_weight",
        "news_blank_prob",
        "consistency_weight",
        "counterfactual_weight",
        "counterfactual_margin",
        "inactive_residual_weight",
        "spike_bias_l2",
        "active_mass_threshold",
        "lambda_min",
        "lambda_ts_cap",
        "lambda_news_cap",
        "lambda_max",
        "shape_gain_cap",
        "shape_gain_l2_weight",
        "hard_gate_mass_threshold",
        "direction_weight",
        "residual_history_channel",
        "spike_bias_cap",
        "selection_counterfactual_gain_min",
        "selection_lambda_saturation_max_pct",
        "hard_residual_frac",
        "hard_residual_pct",
        "pretrain_epochs",
        "pretrain_lr",
        "scheduler",
        "warmup_pct",
        "min_lr_ratio",
        "pretrain_warmup_pct",
        "price_winsor_low",
        "price_winsor_high",
        "grad_clip",
        "eval_permutation_seed",
        "select_metric",
    }

    for name in direct_arg_names:
        setattr(args, name, getattr(cfg, name))

    for name in prefixed_arg_names:
        value = getattr(cfg, name)
        if isinstance(value, bool):
            value = int(value)
        setattr(args, f"delta_v3_{name}", value)


def _reset_output_files(bundle) -> None:
    with open(bundle["ans_json_path"], "w", encoding="utf-8"):
        pass
    with open(bundle["true_pred_csv_path"], "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pred", "true"])
    for key in ["val_residual_debug_csv_path", "test_residual_debug_csv_path"]:
        path = bundle.get(key)
        if path and os.path.exists(path):
            os.remove(path)


def _price_winsor_bounds(args, bundle, cfg: DeltaV3Config):
    if cfg.schema_variant != "price":
        return None
    train_vals = np.asarray(bundle["train_df"][args.value_col], dtype=np.float32)
    train_vals = train_vals[np.isfinite(train_vals)]
    if train_vals.size <= 0:
        return None
    return (
        float(np.quantile(train_vals, cfg.price_winsor_low)),
        float(np.quantile(train_vals, cfg.price_winsor_high)),
    )


def _run_base_test(args, bundle, base_path: str, base_meta: dict) -> tuple[float, float, float]:
    live_logger = bundle["live_logger"]
    device = bundle["device"]
    model, loaded_meta = load_base_backbone_checkpoint(base_path, device=device, is_trainable=False)
    stats = _coerce_global_zstats(loaded_meta, args, required=False)
    if stats is None:
        stats = _coerce_global_zstats(base_meta, args, required=False)
    if stats is None:
        stats = _coerce_global_zstats(bundle.get("global_zstats"), args, required=True)

    live_logger.info(f"[CHECKPOINT_TEST] Loaded base checkpoint: {base_path}")
    test_loss, test_mse, test_mae = evaluate_metrics_backbone(
        base_backbone=model,
        data_loader=bundle["test_loader"],
        args=args,
        global_zstats=stats,
        device=device,
        testing=True,
        true_pred_csv_path=bundle["true_pred_csv_path"],
        filename=getattr(args, "taskName", "base_checkpoint_test"),
    )
    live_logger.info(
        f"[TEST][FINAL] loss(normMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}"
    )
    record_test_results_csv(args, live_logger, test_mse, test_mae)
    draw_pred_true(live_logger, args, bundle["true_pred_csv_path"])
    return test_loss, test_mse, test_mae


def _run_delta_test(args, bundle, base_path: str, delta_path: str, cfg: DeltaV3Config):
    live_logger = bundle["live_logger"]
    device = bundle["device"]
    base_backbone, base_meta = load_base_backbone_checkpoint(base_path, device=device, is_trainable=False)
    stats = _coerce_global_zstats(base_meta, args, required=False)
    if stats is None:
        stats = _coerce_global_zstats(bundle.get("global_zstats"), args, required=True)

    bank, refined_path, bank_path = _prepare_regime_bank(args, bundle, cfg)
    shuffled_bank = bank.shuffled(cfg.eval_permutation_seed)
    live_logger.info(
        f"[CHECKPOINT_TEST] Loaded base checkpoint: {base_path}"
    )
    live_logger.info(
        f"[CHECKPOINT_TEST] Loaded delta checkpoint: {delta_path}"
    )
    live_logger.info(
        f"[DELTA_V3] regime bank ready path={bank_path} refined={refined_path} dates={len(bank.dates)} "
        f"regime_dim={bank.regime_dim} topic_dim={bank.topic_dim} text_dim={bank.text_dim}"
    )

    dow_hod_mean, spike_sigma, spike_threshold_abs = compute_residual_calendar_baseline(
        bundle["train_eval_loader"],
        base_backbone,
        args,
        stats,
        device,
        spike_target_pct=cfg.spike_target_pct,
    )
    decomposer = ResidualTargetDecomposer(
        dow_hod_mean=dow_hod_mean,
        spike_sigma=spike_sigma,
        spike_threshold_abs=spike_threshold_abs,
        spike_k=cfg.spike_k,
        spike_target_pct=cfg.spike_target_pct,
    )

    model = build_delta_v3_model(cfg).to(device)
    ckpt = torch.load(os.path.join(delta_path, "model.pt"), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

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
        price_winsor_bounds=_price_winsor_bounds(args, bundle, cfg),
        debug_csv_path=bundle.get("test_residual_debug_csv_path"),
        split_name="test",
        testing=True,
        true_pred_csv_path=bundle.get("true_pred_csv_path"),
        filename=getattr(args, "taskName", "delta_checkpoint_test"),
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
        f"active_subset_mse={float(diag.get('active_subset_mse', float('nan'))):.6f} "
        f"active_mae={float(diag.get('active_subset_mae', float('nan'))):.6f} "
        f"blank_active_subset_mse={float(diag.get('blank_active_subset_mse', float('nan'))):.6f} "
        f"blank_active={float(diag.get('blank_active_subset_mae', float('nan'))):.6f} "
        f"inactive_mae={float(diag.get('inactive_subset_mae', float('nan'))):.6f} "
        f"blank_inactive={float(diag.get('blank_inactive_subset_mae', float('nan'))):.6f} "
        f"permuted_active_subset_mse={float(diag.get('permuted_active_subset_mse', float('nan'))):.6f} "
        f"perm_active={float(diag.get('permuted_active_subset_mae', float('nan'))):.6f}"
    )
    record_test_results_csv(args, live_logger, test_mse, test_mae, base_mse=base_test_mse, base_mae=base_test_mae)
    draw_pred_true(live_logger, args, bundle.get("true_pred_csv_path"))
    return test_loss, test_mse, test_mae


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.stage = str(getattr(args, "stage", "base")).lower()

    try:
        base_path, delta_path = _resolve_checkpoint_paths(args)
        _require_checkpoint_paths(args.stage, base_path, delta_path)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from None

    base_meta = _sync_args_from_base_meta(args, base_path)
    delta_cfg = None
    if args.stage == "all":
        delta_cfg = _load_delta_cfg(delta_path, args)
        _sync_args_from_delta_cfg(args, delta_cfg)
        if int(base_meta.get("history_len", args.history_len)) != int(delta_cfg.history_len):
            raise ValueError("Base checkpoint history_len does not match delta checkpoint cfg.")
        if int(base_meta.get("horizon", args.horizon)) != int(delta_cfg.horizon):
            raise ValueError("Base checkpoint horizon does not match delta checkpoint cfg.")

    bundle = setup_env_and_data(args)
    _reset_output_files(bundle)

    if args.stage == "base":
        _run_base_test(args, bundle, base_path, base_meta)
        return

    assert delta_cfg is not None
    _run_delta_test(args, bundle, base_path, delta_path, delta_cfg)


if __name__ == "__main__":
    main()
