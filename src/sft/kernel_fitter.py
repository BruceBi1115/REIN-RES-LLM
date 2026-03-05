from __future__ import annotations

import numpy as np

from .impact_kernel import (
    SHAPE_VALUES,
    build_unit_kernel,
    default_kernel_params,
    quantize_amp_to_bin,
    sanitize_kernel_params,
)


def _arg_int(args, key: str, default: int) -> int:
    if args is None:
        return int(default)
    try:
        return int(getattr(args, key, default))
    except Exception:
        return int(default)


def _coarse_values_between(lo_value: int, hi_value: int, stride: int) -> list[int]:
    lo = int(max(0, lo_value))
    hi = int(max(lo, hi_value))
    step = int(max(1, stride))
    vals = list(range(lo, hi + 1, step))
    if not vals:
        vals = [lo]
    if vals[-1] != hi:
        vals.append(hi)
    return [int(v) for v in vals]


def _coarse_values(max_value: int, stride: int) -> list[int]:
    return _coarse_values_between(0, max_value, stride)


def _coarse_half_life_values_between(lo_hl: int, hi_hl: int) -> list[int]:
    lo = int(max(1, lo_hl))
    hi = int(max(lo, hi_hl))
    vals = [lo]
    v = 1
    while v < hi:
        v = int(v * 2)
        if lo <= v <= hi:
            vals.append(v)
    # Ensure edges are always represented even if they are not powers of two.
    vals.extend([lo, hi])
    return sorted(set(int(x) for x in vals if lo <= int(x) <= hi))


def _coarse_half_life_values(max_hl: int) -> list[int]:
    return _coarse_half_life_values_between(1, max_hl)


def _refine_half_life_values(
    center_hl: int,
    coarse_hl_values: list[int],
    max_hl: int,
    refine_radius: int,
) -> list[int]:
    hi = int(max(1, max_hl))
    center = int(max(1, min(hi, int(center_hl))))
    coarse = sorted(set(int(x) for x in coarse_hl_values if 1 <= int(x) <= hi))
    if not coarse:
        coarse = [1]

    arr = np.asarray(coarse, dtype=np.int32)
    idx = int(np.argmin(np.abs(arr - center)))

    vals = {center}
    for j in range(max(0, idx - 2), min(len(coarse), idx + 3)):
        vals.add(int(coarse[j]))

    hl_delta = int(max(2, refine_radius))
    lo = int(max(1, center - hl_delta))
    hi_local = int(min(hi, center + hl_delta))
    vals.update(range(lo, hi_local + 1))
    return sorted(vals)


def _project_amp_and_err(y: np.ndarray, k: np.ndarray, a_max: float) -> tuple[float, float]:
    denom = float(np.dot(k, k))
    if denom <= 1e-8:
        return float("inf"), 0.0
    proj = float(np.dot(y, k) / denom)
    a_star = float(np.clip(proj, 0.0, a_max))
    err = float(np.dot(y - a_star * k, y - a_star * k))
    return err, a_star


def _clamp_int(v, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _sanitize_range_pair(v, lo: int, hi: int) -> tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            a = int(v[0])
            b = int(v[1])
            lo_i = _clamp_int(min(a, b), lo, hi)
            hi_i = _clamp_int(max(a, b), lo, hi)
            return lo_i, hi_i
        except Exception:
            pass
    return int(lo), int(hi)


def _sanitize_search_priors(
    priors: dict | None,
    max_lag: int,
    max_dur: int,
    max_hl: int,
) -> dict | None:
    if not isinstance(priors, dict):
        return None
    out = {}
    try:
        causal = int(priors.get("causal", 1))
    except Exception:
        causal = 1
    out["causal"] = 1 if causal == 1 else 0

    sign = str(priors.get("sign", "UNCERTAIN")).upper().strip()
    out["sign"] = sign if sign in {"UP", "DOWN", "UNCERTAIN"} else "UNCERTAIN"

    shape_raw = priors.get("shape_candidates", [])
    shape_candidates = []
    if isinstance(shape_raw, (list, tuple)):
        for x in shape_raw:
            sx = str(x).upper().strip()
            if sx in SHAPE_VALUES and sx not in shape_candidates:
                shape_candidates.append(sx)
    out["shape_candidates"] = shape_candidates if shape_candidates else list(SHAPE_VALUES)

    out["delay_steps"] = _sanitize_range_pair(priors.get("delay_steps", [0, max_lag]), 0, int(max_lag))
    out["duration_steps"] = _sanitize_range_pair(priors.get("duration_steps", [0, max_dur]), 0, int(max_dur))
    out["half_life_steps"] = _sanitize_range_pair(priors.get("half_life_steps", [1, max_hl]), 1, int(max_hl))
    return out


def fit_kernel_params_from_residual(
    raw_residual: np.ndarray | list[float],
    amp_table: list[float],
    rel_norm_thresh: float = 0.05,
    rel_improve_min_ratio: float = 0.0,
    rel_improve_min_abs: float = 0.0,
    a_max: float = 2.0,
    force_rel0: bool = False,
    args=None,
    max_lag: int | None = None,
    max_dur: int | None = None,
    max_hl: int | None = None,
    coarse_stride: int | None = None,
    refine_radius: int | None = None,
    priors: dict | None = None,
) -> dict:
    """
    Fit discrete kernel parameters by coarse-to-fine search + amplitude projection.
    """
    r = np.asarray(raw_residual, dtype=np.float32).reshape(-1)
    if r.size == 0:
        return default_kernel_params()

    H = int(r.size)
    if max_lag is None:
        max_lag_cfg = _arg_int(args, "kernel_max_lag", 96)
        max_lag = int(min(max(0, H - 1), max(0, max_lag_cfg)))
    else:
        max_lag = int(max(0, max_lag))
    if max_dur is None:
        max_dur_cfg = _arg_int(args, "kernel_max_dur", 96)
        max_dur = int(min(max(0, H - 1), max(0, max_dur_cfg)))
    else:
        max_dur = int(max(0, max_dur))
    if max_hl is None:
        max_hl = int(max(1, _arg_int(args, "kernel_max_hl", 256)))
    else:
        max_hl = int(max(1, max_hl))
    if coarse_stride is None:
        coarse_stride = int(max(1, _arg_int(args, "kernel_coarse_stride", 4)))
    else:
        coarse_stride = int(max(1, coarse_stride))
    if refine_radius is None:
        refine_radius = int(max(0, _arg_int(args, "kernel_refine_radius", 4)))
    else:
        refine_radius = int(max(0, refine_radius))

    search_priors = _sanitize_search_priors(
        priors=priors,
        max_lag=max_lag,
        max_dur=max_dur,
        max_hl=max_hl,
    )
    if isinstance(search_priors, dict) and int(search_priors.get("causal", 1)) == 0:
        return default_kernel_params()

    rel_norm = float(np.linalg.norm(r) / np.sqrt(float(max(1, r.size))))
    if force_rel0 or rel_norm < float(rel_norm_thresh):
        return default_kernel_params()

    base_sign = "UP" if float(r.mean()) > 0.0 else "DOWN"
    sign_candidates = [base_sign]
    if isinstance(search_priors, dict):
        p_sign = str(search_priors.get("sign", "UNCERTAIN")).upper().strip()
        if p_sign in {"UP", "DOWN"}:
            sign_candidates = [p_sign]
        elif p_sign == "UNCERTAIN":
            sign_candidates = ["UP", "DOWN"]

    shape_candidates = list(SHAPE_VALUES)
    lag_lo, lag_hi = 0, int(max_lag)
    dur_lo, dur_hi = 0, int(max_dur)
    hl_lo, hl_hi = 1, int(max_hl)
    if isinstance(search_priors, dict):
        shape_candidates = list(search_priors.get("shape_candidates", shape_candidates))
        lag_lo, lag_hi = search_priors.get("delay_steps", (lag_lo, lag_hi))
        dur_lo, dur_hi = search_priors.get("duration_steps", (dur_lo, dur_hi))
        hl_lo, hl_hi = search_priors.get("half_life_steps", (hl_lo, hl_hi))

    lag_lo = _clamp_int(lag_lo, 0, int(max_lag))
    lag_hi = _clamp_int(lag_hi, lag_lo, int(max_lag))
    dur_lo = _clamp_int(dur_lo, 0, int(max_dur))
    dur_hi = _clamp_int(dur_hi, dur_lo, int(max_dur))
    hl_lo = _clamp_int(hl_lo, 1, int(max_hl))
    hl_hi = _clamp_int(hl_hi, hl_lo, int(max_hl))

    coarse_lag = _coarse_values_between(lag_lo, lag_hi, coarse_stride)
    coarse_dur = _coarse_values_between(dur_lo, dur_hi, coarse_stride)
    coarse_hl = _coarse_half_life_values_between(hl_lo, hl_hi)

    if __debug__ and isinstance(search_priors, dict):
        assert len(coarse_lag) > 0 and min(coarse_lag) >= lag_lo and max(coarse_lag) <= lag_hi
        assert len(coarse_dur) > 0 and min(coarse_dur) >= dur_lo and max(coarse_dur) <= dur_hi
        assert len(coarse_hl) > 0 and min(coarse_hl) >= hl_lo and max(coarse_hl) <= hl_hi
        assert set(shape_candidates).issubset(set(SHAPE_VALUES))

    best_err = float("inf")
    best = {
        "sign": str(sign_candidates[0]),
        "shape": str(shape_candidates[0] if len(shape_candidates) > 0 else "SPIKE"),
        "lag": 0,
        "half_life": 1,
        "dur": 0,
        "amp": 0.0,
    }
    a_max = float(max(0.0, a_max))

    # Stage 1: coarse search.
    for sign in sign_candidates:
        y = r if str(sign).upper() == "UP" else -r
        y = y.astype(np.float32)
        for shape in shape_candidates:
            for lag in coarse_lag:
                for hl in coarse_hl:
                    for dur in coarse_dur:
                        k = build_unit_kernel(
                            H,
                            shape=shape,
                            lag=int(lag),
                            half_life=int(hl),
                            dur=int(dur),
                            max_lag=max_lag,
                            max_dur=max_dur,
                            max_hl=max_hl,
                        )
                        err, a_star = _project_amp_and_err(y, k, a_max=a_max)
                        if err < best_err:
                            best_err = err
                            best = {
                                "sign": str(sign),
                                "shape": str(shape),
                                "lag": int(lag),
                                "half_life": int(hl),
                                "dur": int(dur),
                                "amp": float(a_star),
                            }

    # Stage 2: local refinement around coarse optimum.
    y_ref = r if str(best["sign"]).upper() == "UP" else -r
    y_ref = y_ref.astype(np.float32)

    lag_ref_lo = int(max(lag_lo, int(best["lag"]) - refine_radius))
    lag_ref_hi = int(min(lag_hi, int(best["lag"]) + refine_radius))
    dur_ref_lo = int(max(dur_lo, int(best["dur"]) - refine_radius))
    dur_ref_hi = int(min(dur_hi, int(best["dur"]) + refine_radius))
    refine_lag = list(range(lag_ref_lo, lag_ref_hi + 1))
    refine_dur = list(range(dur_ref_lo, dur_ref_hi + 1))
    refine_hl = _refine_half_life_values(
        center_hl=int(best["half_life"]),
        coarse_hl_values=coarse_hl,
        max_hl=max_hl,
        refine_radius=refine_radius,
    )
    refine_hl = [int(x) for x in refine_hl if int(hl_lo) <= int(x) <= int(hl_hi)]
    if len(refine_hl) == 0:
        refine_hl = [int(_clamp_int(best["half_life"], hl_lo, hl_hi))]

    for lag in refine_lag:
        for hl in refine_hl:
            for dur in refine_dur:
                k = build_unit_kernel(
                    H,
                    shape=str(best["shape"]),
                    lag=int(lag),
                    half_life=int(hl),
                    dur=int(dur),
                    max_lag=max_lag,
                    max_dur=max_dur,
                    max_hl=max_hl,
                )
                err, a_star = _project_amp_and_err(y_ref, k, a_max=a_max)
                if err < best_err:
                    best_err = err
                    best = {
                        "sign": str(best["sign"]),
                        "shape": str(best["shape"]),
                        "lag": int(lag),
                        "half_life": int(hl),
                        "dur": int(dur),
                        "amp": float(a_star),
                    }

    baseline_err = float(np.dot(r, r))
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
        "sign": str(best["sign"]),
        "shape": best["shape"],
        "lag": int(best["lag"]),
        "half_life": int(best["half_life"]),
        "dur": int(best["dur"]),
        "amp_bin": int(amp_bin),
    }
    return sanitize_kernel_params(
        out,
        horizon=H,
        max_lag=max_lag,
        max_dur=max_dur,
        max_hl=max_hl,
        amp_table=amp_table,
    )
