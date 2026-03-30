from __future__ import annotations

__all__ = ["_train_external_signnet"]


def __getattr__(name: str):
    if name == "_train_external_signnet":
        from .training import _train_external_signnet

        return _train_external_signnet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
