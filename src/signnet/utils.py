from __future__ import annotations

def _normalize_signnet_select_metric(raw_metric) -> str:
    metric = str(raw_metric or "acc").lower().strip()
    if metric in {"balanced_acc", "balanced", "balanced_accuracy", "bacc"}:
        return "balanced_acc"
    if metric == "loss":
        return "loss"
    return "acc"

def _normalize_external_signnet_variant(raw_variant) -> str:
    variant = str(raw_variant or "mlp").lower().strip()
    if variant != "dual_stream_tcn":
        return "mlp"
    return "dual_stream_tcn"

def _use_external_signnet_temporal_text(args) -> bool:
    return _normalize_external_signnet_variant(getattr(args, "delta_sign_external_variant", "mlp")) == "dual_stream_tcn"
