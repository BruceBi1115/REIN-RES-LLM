from __future__ import annotations

import numpy as np

from .impact_kernel import (
    DUR_VALUES,
    HALF_LIFE_VALUES,
    LAG_VALUES,
    SHAPE_VALUES,
    default_kernel_params,
    build_unit_kernel,
    quantize_amp_to_bin,
    sanitize_kernel_params,
)


def fit_kernel_params_from_residual(
    raw_residual: np.ndarray | list[float],
    amp_table: list[float],
    rel_norm_thresh: float = 0.05,
    rel_improve_min_ratio: float = 0.0,
    rel_improve_min_abs: float = 0.0,
    a_max: float = 2.0,
    force_rel0: bool = False,
) -> dict:
    """
    Fit discrete kernel parameters by grid-search + amplitude projection.
    """
    r = np.asarray(raw_residual, dtype=np.float32).reshape(-1)
    if r.size == 0:
        return default_kernel_params()

    rel_norm = float(np.linalg.norm(r) / np.sqrt(float(max(1, r.size))))
    if force_rel0 or rel_norm < float(rel_norm_thresh):
        return default_kernel_params()

    sign = "UP" if float(r.mean()) > 0.0 else "DOWN"
    y = r if sign == "UP" else -r
    y = y.astype(np.float32)

    best_err = float("inf")
    best = {
        "shape": "SPIKE",
        "lag": 0,
        "half_life": 1,
        "dur": 0,
        "amp": 0.0,
    }
    a_max = float(max(0.0, a_max))
    H = int(r.size)

    for shape in SHAPE_VALUES:
        for lag in LAG_VALUES:
            for hl in HALF_LIFE_VALUES:
                for dur in DUR_VALUES:
                    k = build_unit_kernel(H, shape=shape, lag=int(lag), half_life=int(hl), dur=int(dur))
                    denom = float(np.dot(k, k))
                    if denom <= 1e-8:
                        continue
                    proj = float(np.dot(y, k) / denom)
                    a_star = float(np.clip(proj, 0.0, a_max))
                    err = float(np.dot(y - a_star * k, y - a_star * k))
                    if err < best_err:
                        best_err = err
                        best = {
                            "shape": str(shape),
                            "lag": int(lag),
                            "half_life": int(hl),
                            "dur": int(dur),
                            "amp": float(a_star),
                        }

    baseline_err = float(np.dot(y, y))
    best_err = float(max(0.0, best_err))
    improve = float(max(0.0, baseline_err - best_err))
    min_ratio = float(max(0.0, rel_improve_min_ratio))
    min_abs = float(max(0.0, rel_improve_min_abs))
    improve_req = float(max(min_abs, min_ratio * max(1e-8, baseline_err)))
    if improve < improve_req:
        return default_kernel_params()

    amp_bin = quantize_amp_to_bin(float(best["amp"]), amp_table)
    out = {
        "rel": 1,
        "sign": sign,
        "shape": best["shape"],
        "lag": int(best["lag"]),
        "half_life": int(best["half_life"]),
        "dur": int(best["dur"]),
        "amp_bin": int(amp_bin),
    }
    return sanitize_kernel_params(out)
