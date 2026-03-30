from __future__ import annotations

__all__ = ["evaluate_metrics_residual", "train_delta_stage"]


def __getattr__(name: str):
    if name in __all__:
        from .stage import evaluate_metrics_residual, train_delta_stage

        exports = {
            "evaluate_metrics_residual": evaluate_metrics_residual,
            "train_delta_stage": train_delta_stage,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
