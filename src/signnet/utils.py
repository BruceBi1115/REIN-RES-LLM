from __future__ import annotations


def _normalize_signnet_select_metric(raw_metric) -> str:
    metric = str(raw_metric or "acc").lower().strip()
    if metric in {"balanced_acc", "balanced", "balanced_accuracy", "bacc"}:
        return "balanced_acc"
    if metric == "loss":
        return "loss"
    return "acc"
