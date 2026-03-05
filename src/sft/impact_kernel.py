from __future__ import annotations

import json
import math
import re
from typing import Iterable

import numpy as np


REL_VALUES = (0, 1)
SIGN_VALUES = ("UP", "DOWN")
SHAPE_VALUES = ("STEP_DECAY", "RAMP_DECAY")
# Legacy default grids kept for backward compatibility.
LAG_VALUES = tuple(range(7))
HALF_LIFE_VALUES = (1, 2, 4, 8, 16, 32)
DUR_VALUES = tuple(range(7))
AMP_VALUES = tuple(range(21))

DEFAULT_LAG_DUR_CAP = 96
DEFAULT_MAX_HALF_LIFE = 256

TOKEN_ORDER = (
    "REL",
    "SIGN",
    "SHAPE",
    "LAG",
    "HL",
    "DUR",
    "AMP",
)


def default_kernel_params() -> dict:
    return {
        "rel": 0,
        "sign": "UP",
        "shape": "STEP_DECAY",
        "lag": 0,
        "half_life": 1,
        "dur": 0,
        "amp_bin": 0,
    }


def _clamp_int(v, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _resolve_lag_dur_cap(
    horizon: int | None = None,
    max_value: int | None = None,
    default_cap: int = DEFAULT_LAG_DUR_CAP,
) -> int:
    if max_value is not None:
        return int(max(0, int(max_value)))
    if horizon is None:
        return int(max(0, int(default_cap)))
    h = int(max(1, int(horizon)))
    return int(max(0, min(h - 1, int(default_cap))))


def _resolve_half_life_cap(max_hl: int | None = None) -> int:
    if max_hl is None:
        return int(DEFAULT_MAX_HALF_LIFE)
    return int(max(1, int(max_hl)))


def _resolve_amp_upper(
    p: dict,
    amp_table: list[float] | None = None,
    max_amp_bin: int | None = None,
) -> int:
    if isinstance(amp_table, list):
        if len(amp_table) == 0:
            return 0
        return int(len(amp_table) - 1)
    if max_amp_bin is not None:
        return int(max(0, int(max_amp_bin)))
    return int(max(0, int(p.get("amp_bin", 0))))


def sanitize_kernel_params(
    params: dict | None,
    *,
    horizon: int | None = None,
    max_lag: int | None = None,
    max_dur: int | None = None,
    max_hl: int | None = None,
    amp_table: list[float] | None = None,
    max_amp_bin: int | None = None,
) -> dict:
    p = dict(default_kernel_params())
    if isinstance(params, dict):
        p.update(params)

    rel = int(p.get("rel", 0))
    p["rel"] = 1 if rel == 1 else 0

    sign = str(p.get("sign", "UP")).upper().strip()
    p["sign"] = sign if sign in SIGN_VALUES else "UP"

    shape = str(p.get("shape", "SPIKE")).upper().strip()
    p["shape"] = shape if shape in SHAPE_VALUES else "SPIKE"

    lag_cap = _resolve_lag_dur_cap(horizon=horizon, max_value=max_lag)
    dur_cap = _resolve_lag_dur_cap(horizon=horizon, max_value=max_dur)
    hl_cap = _resolve_half_life_cap(max_hl=max_hl)
    amp_cap = _resolve_amp_upper(p, amp_table=amp_table, max_amp_bin=max_amp_bin)

    p["lag"] = _clamp_int(p.get("lag", 0), 0, lag_cap)
    p["half_life"] = _clamp_int(p.get("half_life", 1), 1, hl_cap)
    p["dur"] = _clamp_int(p.get("dur", 0), 0, dur_cap)
    p["amp_bin"] = _clamp_int(p.get("amp_bin", 0), 0, amp_cap)
    return p


def half_life_to_lambda(half_life: int) -> float:
    """
    Convert half-life to exponential decay lambda in:
      exp(-t / lambda)

    By definition exp(-half_life / lambda) = 0.5
    => lambda = half_life / ln(2)
    """
    hl = float(max(1, int(half_life)))
    return hl / math.log(2.0)


def build_unit_kernel(
    horizon: int,
    shape: str,
    lag: int,
    half_life: int,
    dur: int,
    max_lag: int | None = None,
    max_dur: int | None = None,
    max_hl: int | None = None,
) -> np.ndarray:
    H = int(max(1, horizon))
    shape_u = str(shape).upper().strip()
    lag_cap = _resolve_lag_dur_cap(horizon=H, max_value=max_lag)
    dur_cap = _resolve_lag_dur_cap(horizon=H, max_value=max_dur)
    hl_cap = _resolve_half_life_cap(max_hl=max_hl)

    lag_i = _clamp_int(lag, 0, lag_cap)
    hl_i = _clamp_int(half_life, 1, hl_cap)
    dur_i = _clamp_int(dur, 0, dur_cap)
    lam = float(max(1e-6, half_life_to_lambda(hl_i)))

    tau = np.arange(H, dtype=np.float32)
    t = np.maximum(0.0, tau - float(lag_i))

    if shape_u == "SPIKE":
        k = np.exp(-t / lam)
    elif shape_u == "STEP_DECAY":
        if dur_i <= 0:
            k = np.exp(-(t - float(dur_i)) / lam)
        else:
            k = np.where(t < float(dur_i), 1.0, np.exp(-(t - float(dur_i)) / lam))
    elif shape_u == "RAMP_DECAY":
        if dur_i >= 1:
            ramp = np.clip(t / float(dur_i), 0.0, 1.0)
            decay = np.exp(-(t - float(dur_i)) / lam)
            k = np.where(t < float(dur_i), ramp, decay)
        else:
            k = np.exp(-(t - float(dur_i)) / lam)
    else:
        k = np.exp(-t / lam)
    return np.asarray(k, dtype=np.float32)


def build_amp_table_from_residuals(
    residuals: Iterable[Iterable[float] | np.ndarray],
    n_bins: int = 21,
) -> list[float]:
    n_bins = int(max(2, n_bins))
    vals = []
    for r in residuals:
        arr = np.asarray(r, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        vals.append(np.abs(arr))
    if not vals:
        return [float(i) / float(max(1, n_bins - 1)) for i in range(n_bins)]

    flat = np.concatenate(vals, axis=0)
    if flat.size == 0:
        return [float(i) / float(max(1, n_bins - 1)) for i in range(n_bins)]

    q = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
    table = np.quantile(flat, q).astype(np.float32)
    table = np.maximum.accumulate(table)
    table[0] = 0.0
    return [float(x) for x in table.tolist()]


def save_amp_table(path: str, amp_table: list[float]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"amp_table": [float(x) for x in amp_table]}, f, ensure_ascii=False, indent=2)


def load_amp_table(path: str, expected_bins: int = 21) -> list[float]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    arr = obj.get("amp_table", obj if isinstance(obj, list) else [])
    if not isinstance(arr, list):
        arr = []
    vals = []
    for x in arr:
        try:
            vals.append(float(x))
        except Exception:
            continue
    if not vals:
        vals = [float(i) / float(max(1, expected_bins - 1)) for i in range(expected_bins)]
    if len(vals) < expected_bins:
        vals.extend([float(vals[-1])] * (expected_bins - len(vals)))
    elif len(vals) > expected_bins:
        vals = vals[:expected_bins]
    vals = np.maximum.accumulate(np.asarray(vals, dtype=np.float32)).tolist()
    vals[0] = 0.0
    return [float(x) for x in vals]


def amp_from_bin(amp_bin: int, amp_table: list[float]) -> float:
    if not amp_table:
        return 0.0
    idx = _clamp_int(amp_bin, 0, len(amp_table) - 1)
    return float(amp_table[idx])


def quantize_amp_to_bin(amp: float, amp_table: list[float]) -> int:
    if not amp_table:
        return 0
    a = float(max(0.0, amp))
    arr = np.asarray(amp_table, dtype=np.float32)
    idx = int(np.argmin(np.abs(arr - a)))
    return _clamp_int(idx, 0, len(amp_table) - 1)


def params_to_delta(
    params: dict | None,
    horizon: int,
    amp_table: list[float],
    clip_low: float = -1.0,
    clip_high: float = 1.0,
    max_lag: int | None = None,
    max_dur: int | None = None,
    max_hl: int | None = None,
) -> np.ndarray:
    p = sanitize_kernel_params(
        params,
        horizon=horizon,
        max_lag=max_lag,
        max_dur=max_dur,
        max_hl=max_hl,
        amp_table=amp_table,
    )
    H = int(max(1, horizon))
    if int(p["rel"]) == 0:
        return np.zeros((H,), dtype=np.float32)

    sign = 1.0 if str(p["sign"]).upper() == "UP" else -1.0
    amp = amp_from_bin(int(p["amp_bin"]), amp_table)
    k = build_unit_kernel(
        horizon=H,
        shape=str(p["shape"]),
        lag=int(p["lag"]),
        half_life=int(p["half_life"]),
        dur=int(p["dur"]),
        max_lag=max_lag,
        max_dur=max_dur,
        max_hl=max_hl,
    )
    delta = sign * float(amp) * k
    delta = np.clip(delta, float(clip_low), float(clip_high))
    return delta.astype(np.float32)


def format_param_tokens(
    params: dict | None,
    *,
    horizon: int | None = None,
    max_lag: int | None = None,
    max_dur: int | None = None,
    max_hl: int | None = None,
    amp_table: list[float] | None = None,
    max_amp_bin: int | None = None,
) -> str:
    p = sanitize_kernel_params(
        params,
        horizon=horizon,
        max_lag=max_lag,
        max_dur=max_dur,
        max_hl=max_hl,
        amp_table=amp_table,
        max_amp_bin=max_amp_bin,
    )
    return (
        f"<REL_{int(p['rel'])}> "
        f"<SIGN_{str(p['sign']).upper()}> "
        f"<SHAPE_{str(p['shape']).upper()}> "
        f"<LAG_{int(p['lag'])}> "
        f"<HL_{int(p['half_life'])}> "
        f"<DUR_{int(p['dur'])}> "
        f"<AMP_{int(p['amp_bin'])}>"
    )


def parse_param_tokens(
    text: str | None,
    *,
    horizon: int | None = None,
    max_lag: int | None = None,
    max_dur: int | None = None,
    max_hl: int | None = None,
    amp_table: list[float] | None = None,
    max_amp_bin: int | None = None,
) -> dict:
    s = str(text or "").upper()
    p = default_kernel_params()
    has_rel = False
    has_amp = False

    # Be tolerant to malformed wrappers like "REL_1>" or "<REL_1".
    m_rel = re.search(r"(?:<)?REL_(0|1)(?:>)?", s)
    if m_rel:
        has_rel = True
        p["rel"] = int(m_rel.group(1))

    m_sign = re.search(r"(?:<)?SIGN_(UP|DOWN)(?:>)?", s)
    if m_sign:
        p["sign"] = str(m_sign.group(1))

    m_shape = re.search(r"(?:<)?SHAPE_(SPIKE|STEP_DECAY|RAMP_DECAY)(?:>)?", s)
    if m_shape:
        p["shape"] = str(m_shape.group(1))

    m_lag = re.search(r"(?:<)?LAG_(\d+)(?:>)?", s)
    if m_lag:
        p["lag"] = int(m_lag.group(1))

    m_hl = re.search(r"(?:<)?HL_(\d+)(?:>)?", s)
    if m_hl:
        p["half_life"] = int(m_hl.group(1))

    m_dur = re.search(r"(?:<)?DUR_(\d+)(?:>)?", s)
    if m_dur:
        p["dur"] = int(m_dur.group(1))

    m_amp = re.search(r"(?:<)?AMP_(\d+)(?:>)?", s)
    if m_amp:
        has_amp = True
        p["amp_bin"] = int(m_amp.group(1))

    # Robust fallback: when generation contains AMP but misses REL token,
    # treat non-zero amplitude as relevant.
    if (not has_rel) and has_amp and int(p.get("amp_bin", 0)) > 0:
        p["rel"] = 1

    return sanitize_kernel_params(
        p,
        horizon=horizon,
        max_lag=max_lag,
        max_dur=max_dur,
        max_hl=max_hl,
        amp_table=amp_table,
        max_amp_bin=max_amp_bin,
    )
