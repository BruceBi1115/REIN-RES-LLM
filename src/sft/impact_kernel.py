from __future__ import annotations

import json
import math
import re
from typing import Iterable

import numpy as np


REL_VALUES = (0, 1)
SIGN_VALUES = ("UP", "DOWN")
SHAPE_VALUES = ("SPIKE", "STEP_DECAY", "RAMP_DECAY")
LAG_VALUES = tuple(range(7))
HALF_LIFE_VALUES = (1, 2, 4, 8, 16, 32)
DUR_VALUES = tuple(range(7))
AMP_VALUES = tuple(range(21))

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
        "shape": "SPIKE",
        "lag": 0,
        "half_life": 1,
        "dur": 0,
        "amp_bin": 0,
    }


def _clamp_int(v, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def sanitize_kernel_params(params: dict | None) -> dict:
    p = dict(default_kernel_params())
    if isinstance(params, dict):
        p.update(params)

    rel = int(p.get("rel", 0))
    p["rel"] = 1 if rel == 1 else 0

    sign = str(p.get("sign", "UP")).upper().strip()
    p["sign"] = sign if sign in SIGN_VALUES else "UP"

    shape = str(p.get("shape", "SPIKE")).upper().strip()
    p["shape"] = shape if shape in SHAPE_VALUES else "SPIKE"

    p["lag"] = _clamp_int(p.get("lag", 0), 0, 6)
    hl = int(p.get("half_life", 1))
    p["half_life"] = hl if hl in HALF_LIFE_VALUES else 1
    p["dur"] = _clamp_int(p.get("dur", 0), 0, 6)
    p["amp_bin"] = _clamp_int(p.get("amp_bin", 0), 0, 20)
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
) -> np.ndarray:
    H = int(max(1, horizon))
    shape_u = str(shape).upper().strip()
    lag_i = _clamp_int(lag, 0, 6)
    hl_i = int(half_life) if int(half_life) in HALF_LIFE_VALUES else 1
    dur_i = _clamp_int(dur, 0, 6)
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
    return _clamp_int(idx, 0, min(len(amp_table) - 1, 20))


def params_to_delta(
    params: dict | None,
    horizon: int,
    amp_table: list[float],
    clip_low: float = -1.0,
    clip_high: float = 1.0,
) -> np.ndarray:
    p = sanitize_kernel_params(params)
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
    )
    delta = sign * float(amp) * k
    delta = np.clip(delta, float(clip_low), float(clip_high))
    return delta.astype(np.float32)


def format_param_tokens(params: dict | None) -> str:
    p = sanitize_kernel_params(params)
    return (
        f"<REL_{int(p['rel'])}> "
        f"<SIGN_{str(p['sign']).upper()}> "
        f"<SHAPE_{str(p['shape']).upper()}> "
        f"<LAG_{int(p['lag'])}> "
        f"<HL_{int(p['half_life'])}> "
        f"<DUR_{int(p['dur'])}> "
        f"<AMP_{int(p['amp_bin'])}>"
    )


def parse_param_tokens(text: str | None) -> dict:
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
        p["lag"] = _clamp_int(int(m_lag.group(1)), 0, 6)

    m_hl = re.search(r"(?:<)?HL_(\d+)(?:>)?", s)
    if m_hl:
        hl = int(m_hl.group(1))
        p["half_life"] = hl if hl in HALF_LIFE_VALUES else 1

    m_dur = re.search(r"(?:<)?DUR_(\d+)(?:>)?", s)
    if m_dur:
        p["dur"] = _clamp_int(int(m_dur.group(1)), 0, 6)

    m_amp = re.search(r"(?:<)?AMP_(\d+)(?:>)?", s)
    if m_amp:
        has_amp = True
        p["amp_bin"] = _clamp_int(int(m_amp.group(1)), 0, 20)

    # Robust fallback: when generation contains AMP but misses REL token,
    # treat non-zero amplitude as relevant.
    if (not has_rel) and has_amp and int(p.get("amp_bin", 0)) > 0:
        p["rel"] = 1

    return sanitize_kernel_params(p)
