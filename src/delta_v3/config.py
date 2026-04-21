from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_SHARED_REFINE_CACHE_DIR = os.path.join("_shared_refine_cache", "v4")
LEGACY_SHARED_REFINE_CACHE_DIR = os.path.join("checkpoints", "_shared_refine_cache", "v4")
LEGACY_BACKUP_DIR = os.path.join("_shared_refine_cache", "LEGACY_BACKUP")


def _sanitize_cache_component(value: str, *, fallback: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    clean = re.sub(r"_+", "_", clean).strip("._-")
    return clean or str(fallback)


def news_dataset_cache_tag(news_path: str) -> str:
    path = str(news_path or "").strip()
    if not path:
        return ""
    stem = os.path.splitext(os.path.basename(path))[0]
    return _sanitize_cache_component(stem, fallback="news")


def _news_cache_filename(news_path: str) -> str:
    return _sanitize_cache_component(news_dataset_cache_tag(news_path), fallback="news")


def append_cache_tag_to_path(path: str, cache_tag: str) -> str:
    raw_path = str(path or "").strip()
    tag = _sanitize_cache_component(cache_tag, fallback="news") if str(cache_tag or "").strip() else ""
    if not raw_path or not tag:
        return raw_path

    directory, filename = os.path.split(raw_path)
    stem, ext = os.path.splitext(filename)
    if stem.endswith(f"__{tag}") or stem.endswith(f"_{tag}"):
        return raw_path
    return os.path.join(directory, f"{stem}__{tag}{ext}")


def build_refined_cache_path(cache_dir: str, news_path: str) -> str:
    news_key = _news_cache_filename(news_path)
    return os.path.join(cache_dir, f"refined_{news_key}.jsonl")


def build_regime_bank_path(cache_dir: str, news_path: str) -> str:
    news_key = _news_cache_filename(news_path)
    return os.path.join(cache_dir, f"regime_bank_{news_key}.npz")


def build_legacy_refined_cache_path(cache_dir: str, dataset_key: str, schema_variant: str, news_path: str) -> str:
    base_path = os.path.join(cache_dir, f"refined_{dataset_key}_{schema_variant}_regime_v2.jsonl")
    return append_cache_tag_to_path(base_path, news_dataset_cache_tag(news_path))


def build_legacy_regime_bank_path(cache_dir: str, dataset_key: str, news_path: str) -> str:
    base_path = os.path.join(cache_dir, f"regime_bank_{dataset_key}.npz")
    return append_cache_tag_to_path(base_path, news_dataset_cache_tag(news_path))


@dataclass(slots=True)
class DeltaV3Config:
    dataset_key: str
    news_path: str
    news_cache_tag: str
    cache_dir: str
    legacy_cache_dir: str
    history_len: int
    horizon: int
    schema_variant: str
    regime_bank_path: str
    regime_bank_legacy_path: str
    refined_bank_build: bool
    text_encoder_model_id: str
    text_encoder_max_length: int
    regime_tau_days: float
    regime_ema_alpha: float
    regime_ema_window: int
    arch: str
    hidden_size: int
    num_layers: int
    num_heads: int
    patch_len: int
    patch_stride: int
    dropout: float
    use_base_hidden: bool
    slow_weight: float
    shape_weight: float
    spike_weight: float
    spike_gate_threshold: float
    spike_k: float
    spike_target_pct: float
    spike_gate_loss_weight: float
    news_blank_prob: float
    consistency_weight: float
    counterfactual_weight: float
    counterfactual_margin: float
    inactive_residual_weight: float
    spike_bias_l2: float
    active_mass_threshold: float
    lambda_min: float
    lambda_ts_cap: float
    lambda_news_cap: float
    lambda_max: float
    shape_gain_cap: float
    shape_gain_l2_weight: float
    hard_gate_mass_threshold: float
    direction_weight: float
    residual_history_channel: bool
    spike_bias_cap: float
    selection_counterfactual_gain_min: float
    selection_lambda_saturation_max_pct: float
    hard_residual_frac: float
    hard_residual_pct: float
    pretrain_epochs: int
    pretrain_lr: float
    scheduler: str
    warmup_pct: float
    min_lr_ratio: float
    pretrain_warmup_pct: float
    price_winsor_low: float
    price_winsor_high: float
    grad_clip: float
    select_metric: str
    eval_permutation_seed: int

    @classmethod
    def from_args(cls, args) -> "DeltaV3Config":
        dataset_key = str(getattr(args, "dataset_key", "dataset") or "dataset").strip()
        news_path = str(getattr(args, "news_path", "") or "").strip()
        news_cache_tag = news_dataset_cache_tag(news_path)
        override_bank_path = str(getattr(args, "delta_v3_regime_bank_path", "") or "").strip()
        regime_bank_path = override_bank_path or build_regime_bank_path(DEFAULT_SHARED_REFINE_CACHE_DIR, news_path)
        legacy_bank_path = override_bank_path or build_legacy_regime_bank_path(
            DEFAULT_SHARED_REFINE_CACHE_DIR,
            dataset_key,
            news_path,
        )
        refined_bank_build = getattr(args, "delta_v3_refined_bank_build", 0)
        return cls(
            dataset_key=dataset_key,
            news_path=news_path,
            news_cache_tag=news_cache_tag,
            cache_dir=DEFAULT_SHARED_REFINE_CACHE_DIR,
            legacy_cache_dir=LEGACY_SHARED_REFINE_CACHE_DIR,
            history_len=int(getattr(args, "history_len", 48) or 48),
            horizon=int(getattr(args, "horizon", 48) or 48),
            schema_variant=str(getattr(args, "delta_v3_schema_variant", "load") or "load").strip(),
            regime_bank_path=regime_bank_path,
            regime_bank_legacy_path=legacy_bank_path,
            refined_bank_build=bool(int(refined_bank_build or 0)),
            text_encoder_model_id=str(
                getattr(args, "delta_v3_text_encoder_model_id", "intfloat/e5-small-v2") or "intfloat/e5-small-v2"
            ).strip(),
            text_encoder_max_length=int(getattr(args, "delta_v3_text_encoder_max_length", 256) or 256),
            regime_tau_days=float(getattr(args, "delta_v3_regime_tau_days", 5.0) or 5.0),
            regime_ema_alpha=float(getattr(args, "delta_v3_regime_ema_alpha", 0.5) or 0.5),
            regime_ema_window=int(getattr(args, "delta_v3_regime_ema_window", 5) or 5),
            arch=str(getattr(args, "delta_v3_arch", "patchtst_regime_modulation") or "patchtst_regime_modulation").strip(),
            hidden_size=int(getattr(args, "delta_v3_hidden_size", 128) or 128),
            num_layers=int(getattr(args, "delta_v3_num_layers", 2) or 2),
            num_heads=int(getattr(args, "delta_v3_num_heads", 4) or 4),
            patch_len=int(getattr(args, "delta_v3_patch_len", 8) or 8),
            patch_stride=int(getattr(args, "delta_v3_patch_stride", 4) or 4),
            dropout=float(getattr(args, "delta_v3_dropout", 0.1) or 0.1),
            use_base_hidden=bool(int(getattr(args, "delta_v3_use_base_hidden", 1) or 0)),
            slow_weight=float(getattr(args, "delta_v3_slow_weight", 1.0) or 1.0),
            shape_weight=float(getattr(args, "delta_v3_shape_weight", 1.0) or 1.0),
            spike_weight=float(getattr(args, "delta_v3_spike_weight", 1.0) or 1.0),
            spike_gate_threshold=float(getattr(args, "delta_v3_spike_gate_threshold", 0.8) or 0.8),
            spike_k=float(getattr(args, "delta_v3_spike_k", 3.0) or 3.0),
            spike_target_pct=float(getattr(args, "delta_v3_spike_target_pct", 0.10) or 0.10),
            spike_gate_loss_weight=float(getattr(args, "delta_v3_spike_gate_loss_weight", 0.25) or 0.25),
            news_blank_prob=float(getattr(args, "delta_v3_news_blank_prob", 0.3) or 0.3),
            consistency_weight=float(getattr(args, "delta_v3_consistency_weight", 0.05) or 0.05),
            counterfactual_weight=float(getattr(args, "delta_v3_counterfactual_weight", 0.1) or 0.1),
            counterfactual_margin=float(getattr(args, "delta_v3_counterfactual_margin", 0.02) or 0.02),
            inactive_residual_weight=float(
                getattr(args, "delta_v3_inactive_residual_weight", 0.1) or 0.1
            ),
            spike_bias_l2=float(getattr(args, "delta_v3_spike_bias_l2", 1e-3) or 1e-3),
            active_mass_threshold=float(getattr(args, "delta_v3_active_mass_threshold", 0.7) or 0.7),
            lambda_min=float(getattr(args, "delta_v3_lambda_min", 0.05) or 0.05),
            lambda_ts_cap=float(getattr(args, "delta_v3_lambda_ts_cap", 0.45) or 0.45),
            lambda_news_cap=float(getattr(args, "delta_v3_lambda_news_cap", 0.20) or 0.20),
            lambda_max=float(getattr(args, "delta_v3_lambda_max", 0.60) or 0.60),
            shape_gain_cap=float(getattr(args, "delta_v3_shape_gain_cap", 0.50) or 0.50),
            shape_gain_l2_weight=float(
                getattr(args, "delta_v3_shape_gain_l2_weight", 0.01) or 0.0
            ),
            hard_gate_mass_threshold=float(
                getattr(args, "delta_v3_hard_gate_mass_threshold", 0.0) or 0.0
            ),
            direction_weight=float(
                getattr(args, "delta_v3_direction_weight", 0.05) or 0.0
            ),
            residual_history_channel=bool(
                int(getattr(args, "delta_v3_residual_history_channel", 1) or 0)
            ),
            spike_bias_cap=float(getattr(args, "delta_v3_spike_bias_cap", 0.75) or 0.75),
            selection_counterfactual_gain_min=float(
                getattr(args, "delta_v3_selection_counterfactual_gain_min", 0.01) or 0.01
            ),
            selection_lambda_saturation_max_pct=float(
                getattr(args, "delta_v3_selection_lambda_saturation_max_pct", 0.35) or 0.35
            ),
            hard_residual_frac=float(getattr(args, "delta_v3_hard_residual_frac", 0.6) or 0.6),
            hard_residual_pct=float(getattr(args, "delta_v3_hard_residual_pct", 0.10) or 0.10),
            pretrain_epochs=int(getattr(args, "delta_v3_pretrain_epochs", 12) or 12),
            pretrain_lr=float(getattr(args, "delta_v3_pretrain_lr", 1e-3) or 1e-3),
            scheduler=str(getattr(args, "delta_v3_scheduler", "warmup_cosine") or "warmup_cosine").strip(),
            warmup_pct=float(getattr(args, "delta_v3_warmup_pct", 0.05) or 0.05),
            min_lr_ratio=float(getattr(args, "delta_v3_min_lr_ratio", 0.05) or 0.05),
            pretrain_warmup_pct=float(getattr(args, "delta_v3_pretrain_warmup_pct", 0.10) or 0.10),
            price_winsor_low=float(getattr(args, "delta_v3_price_winsor_low", 0.005) or 0.005),
            price_winsor_high=float(getattr(args, "delta_v3_price_winsor_high", 0.995) or 0.995),
            grad_clip=float(getattr(args, "delta_v3_grad_clip", 1.0) or 1.0),
            select_metric=str(getattr(args, "delta_v3_select_metric", "mae") or "mae").strip(),
            eval_permutation_seed=int(getattr(args, "delta_v3_eval_permutation_seed", 2024) or 2024),
        )
