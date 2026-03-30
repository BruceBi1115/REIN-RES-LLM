from __future__ import annotations

__all__ = ["main", "setup_env_and_data", "testing_base", "train_base_stage"]


def __getattr__(name: str):
    if name in __all__:
        from .stage import main, setup_env_and_data, testing_base, train_base_stage

        exports = {
            "main": main,
            "setup_env_and_data": setup_env_and_data,
            "testing_base": testing_base,
            "train_base_stage": train_base_stage,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
