from __future__ import annotations

import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from ..data_construction.data import make_loader
from ..news_rules import get_candidates, rerank_selected_news_by_utility, select_news
from .impact_kernel import (
    build_amp_table_from_residuals,
    default_kernel_params,
    format_param_tokens,
    sanitize_kernel_params,
)
from .kernel_fitter import fit_kernel_params_from_residual

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


KERNEL_API_MODEL_FIXED = "gpt-5.1"
KERNEL_API_TYPE_PRIORS_V1 = "priors_v1"
KERNEL_API_TYPE_RELSIGN_V1 = "relsign_v1"
KERNEL_PRIOR_ALLOWED_SHAPES = ("SPIKE", "STEP_DECAY", "RAMP_DECAY")


def _infer_dataset_context(task_name: str) -> str:
    t = str(task_name or "").lower()
    if "price" in t:
        return "Australian NSW electricity spot price (AUD/MWh)"
    if "demand" in t:
        return "Australian NSW electricity demand (MW)"
    return "Australian electricity market variable"


def _zscore_np(values: list[float], mu: float, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    s = float(max(float(sigma), 1e-6))
    return ((arr - float(mu)) / s).astype(np.float32)


def _extract_texts(rows: pd.DataFrame, text_col: str) -> list[str]:
    if rows is None or len(rows) == 0 or text_col not in rows.columns:
        return []
    out = []
    for v in rows[text_col].tolist():
        txt = str(v or "").strip()
        if txt:
            out.append(txt)
    return out


def _summary(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "first": 0.0,
            "last": 0.0,
            "trend": 0.0,
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "first": float(x[0]),
        "last": float(x[-1]),
        "trend": float(x[-1] - x[0]),
    }


def _stats_line(name: str, st: dict[str, float]) -> str:
    return (
        f"{name}: "
        f"mean={st['mean']:.4f}, std={st['std']:.4f}, "
        f"min={st['min']:.4f}, max={st['max']:.4f}, "
        f"first={st['first']:.4f}, last={st['last']:.4f}, trend={st['trend']:.4f}"
    )


def _kernel_rel_norm(raw_residual: np.ndarray | list[float]) -> float:
    r = np.asarray(raw_residual, dtype=np.float32).reshape(-1)
    if r.size == 0:
        return 0.0
    return float(np.linalg.norm(r) / np.sqrt(float(max(1, r.size))))


def _is_uncertain_for_api(sample: dict, params: dict, rel_thresh: float, args) -> bool:
    if int(sample.get("has_news", 0)) != 1:
        return False
    rel_norm = _kernel_rel_norm(sample.get("raw_residual", []))
    band = float(max(0.0, getattr(args, "kernel_api_uncertain_band", 0.02)))
    low_amp_bin = int(max(0, getattr(args, "kernel_api_low_amp_bin", 2)))
    near_boundary = abs(rel_norm - float(rel_thresh)) <= band
    weak_positive = int(params.get("rel", 0)) == 1 and int(params.get("amp_bin", 0)) <= low_amp_bin
    return bool(near_boundary or weak_positive)


def _truncate_news_texts(news_texts: list[str], max_items: int = 6, max_chars_each: int = 480) -> list[str]:
    out = []
    for t in (news_texts or [])[: max(0, int(max_items))]:
        s = str(t or "").strip()
        if not s:
            continue
        out.append(s[: max(64, int(max_chars_each))])
    return out


def _extract_json(text: str) -> dict | None:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m is None:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _extract_usage_tokens(resp) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        p = _safe_int(usage.get("prompt_tokens", 0), 0)
        c = _safe_int(usage.get("completion_tokens", 0), 0)
        return max(0, p), max(0, c)
    p = _safe_int(getattr(usage, "prompt_tokens", 0), 0)
    c = _safe_int(getattr(usage, "completion_tokens", 0), 0)
    return max(0, p), max(0, c)


def _one_line(text: str, max_chars: int = 240) -> str:
    s = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    m = int(max(32, max_chars))
    if len(s) <= m:
        return s
    return s[: m - 3] + "..."


def _cost_usd_est(prompt_tokens: int, completion_tokens: int, price_in_per_1m: float, price_out_per_1m: float) -> float:
    p = max(0.0, float(price_in_per_1m))
    c = max(0.0, float(price_out_per_1m))
    return (float(max(0, prompt_tokens)) / 1_000_000.0) * p + (float(max(0, completion_tokens)) / 1_000_000.0) * c


def _token_param_is_unsupported_error(err: str) -> bool:
    s = str(err or "").lower()
    mentions_token_arg = ("max_tokens" in s) or ("max_completion_tokens" in s)
    if not mentions_token_arg:
        return False
    return (
        ("unsupported parameter" in s)
        or ("unknown parameter" in s)
        or ("unrecognized request argument" in s)
        or ("not supported" in s)
        or ("invalid_request_error" in s)
    )


def _hash_hex(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _clamp_int(v, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _infer_news_count_from_prompt(prompt: str) -> int:
    s = str(prompt or "")
    if not s:
        return 0
    marker = "[Selected News]"
    i = s.find(marker)
    if i < 0:
        return 0
    tail = s[i + len(marker) :]
    j = tail.find("\n\n[Task]")
    block = tail[:j] if j >= 0 else tail
    block = str(block).strip()
    if not block or block == "(none)":
        return 0
    numbered = 0
    for line in block.splitlines():
        if re.match(r"^\s*\d+\.\s+", str(line)):
            numbered += 1
    if numbered > 0:
        return int(numbered)
    return int(sum(1 for line in block.splitlines() if str(line).strip()))


def _infer_news_count_from_sample(sample: dict) -> int:
    s = sample if isinstance(sample, dict) else {}
    if "news_count" in s:
        try:
            return int(max(0, int(s.get("news_count", 0))))
        except Exception:
            pass
    raw_news = s.get("news_texts", None)
    if isinstance(raw_news, list):
        return int(len(raw_news))
    return _infer_news_count_from_prompt(str(s.get("prompt", "")))


def _save_news_density_artifacts(samples: list[dict], task_name: str, live_logger=None) -> dict:
    counts = np.asarray([_infer_news_count_from_sample(s) for s in (samples or [])], dtype=np.int32)
    n = int(counts.size)
    if n <= 0:
        return {}

    indices = np.arange(n, dtype=np.int32)
    max_news = int(np.max(counts))
    min_news = int(np.min(counts))
    avg_news = float(np.mean(counts))
    std_news = float(np.std(counts))

    max_ids = np.where(counts == max_news)[0].tolist()
    min_ids = np.where(counts == min_news)[0].tolist()
    target_times = [str((s or {}).get("target_time", "")) for s in (samples or [])]

    ckpt_dir = os.path.join("./checkpoints", str(task_name or "task"))
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(ckpt_dir, "kernel_prompt_news_density.csv")
    json_path = os.path.join(ckpt_dir, "kernel_prompt_news_density_summary.json")
    png_path = os.path.join(ckpt_dir, "kernel_prompt_news_density.png")

    df = pd.DataFrame(
        {
            "sample_index": indices.tolist(),
            "target_time": target_times,
            "news_count": counts.tolist(),
        }
    )
    df.to_csv(csv_path, index=False)

    summary = {
        "sample_count": n,
        "min_news_per_prompt": min_news,
        "max_news_per_prompt": max_news,
        "avg_news_per_prompt": avg_news,
        "std_news_per_prompt": std_news,
        "max_news_indices": [int(x) for x in max_ids],
        "min_news_indices": [int(x) for x in min_ids],
        "max_news_target_times": [target_times[int(i)] for i in max_ids],
        "min_news_target_times": [target_times[int(i)] for i in min_ids],
        "csv_file": os.path.basename(csv_path),
        "plot_file": os.path.basename(png_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_saved = False
    plot_error = ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(indices, counts, color="#1f77b4", linewidth=1.5)
        ax.scatter(indices, counts, color="#1f77b4", s=12, alpha=0.65)
        ax.axhline(avg_news, color="#d62728", linestyle="--", linewidth=1.2, label=f"avg={avg_news:.2f}")
        ax.set_title("News Density Per Prompt")
        ax.set_xlabel("sample_index")
        ax.set_ylabel("news_count")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        plot_saved = True
    except Exception as e:
        plot_error = str(e)

    if live_logger is not None:
        live_logger.info(
            f"[KERNEL_SFT][NEWS_DENSITY] samples={n} min={min_news} max={max_news} avg={avg_news:.3f} std={std_news:.3f}"
        )
        live_logger.info(
            f"[KERNEL_SFT][NEWS_DENSITY] csv={csv_path} summary={json_path} "
            + (f"plot={png_path}" if plot_saved else f"plot=skipped ({plot_error})")
        )
    summary["plot_saved"] = int(plot_saved)
    if plot_error:
        summary["plot_error"] = plot_error
    return summary


def save_kernel_prompt_news_density(samples: list[dict], task_name: str, live_logger=None) -> dict:
    return _save_news_density_artifacts(samples=samples, task_name=task_name, live_logger=live_logger)


def _sanitize_range_pair(v, lo: int, hi: int) -> tuple[int, int] | None:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    try:
        a = int(v[0])
        b = int(v[1])
    except Exception:
        return None
    lo_v = _clamp_int(min(a, b), lo, hi)
    hi_v = _clamp_int(max(a, b), lo, hi)
    if hi_v < lo_v:
        lo_v, hi_v = hi_v, lo_v
    return int(lo_v), int(hi_v)


def _sanitize_kernel_priors(obj: dict | None, horizon: int) -> dict | None:
    if not isinstance(obj, dict):
        return None
    H = int(max(1, horizon))
    step_hi = int(max(0, H - 1))
    hl_hi = int(max(1, H - 1))

    try:
        causal = int(obj.get("causal", -1))
    except Exception:
        return None
    if causal not in (0, 1):
        return None

    sign = str(obj.get("sign", "")).upper().strip()
    if sign not in {"UP", "DOWN", "UNCERTAIN"}:
        return None

    raw_shapes = obj.get("shape_candidates", None)
    if not isinstance(raw_shapes, (list, tuple)):
        return None
    shape_candidates = []
    for s in raw_shapes:
        su = str(s).upper().strip()
        if su in KERNEL_PRIOR_ALLOWED_SHAPES and su not in shape_candidates:
            shape_candidates.append(su)
    if len(shape_candidates) == 0:
        return None

    delay_steps = _sanitize_range_pair(obj.get("delay_steps", None), 0, step_hi)
    duration_steps = _sanitize_range_pair(obj.get("duration_steps", None), 0, step_hi)
    half_life_steps = _sanitize_range_pair(obj.get("half_life_steps", None), 1, hl_hi)
    if delay_steps is None or duration_steps is None or half_life_steps is None:
        return None

    strength = str(obj.get("strength", "")).lower().strip()
    if strength not in {"weak", "medium", "strong"}:
        return None

    rationale_short = str(obj.get("rationale_short", "")).strip()
    if len(rationale_short) == 0:
        return None
    words = rationale_short.split()
    if len(words) > 30:
        rationale_short = " ".join(words[:30])

    out = {
        "causal": int(causal),
        "sign": sign,
        "shape_candidates": shape_candidates,
        "delay_steps": [int(delay_steps[0]), int(delay_steps[1])],
        "duration_steps": [int(duration_steps[0]), int(duration_steps[1])],
        "half_life_steps": [int(half_life_steps[0]), int(half_life_steps[1])],
        "strength": strength,
        "rationale_short": rationale_short,
    }
    if __debug__:
        assert out["causal"] in (0, 1)
        assert out["sign"] in {"UP", "DOWN", "UNCERTAIN"}
        assert len(out["shape_candidates"]) > 0
        assert all(x in KERNEL_PRIOR_ALLOWED_SHAPES for x in out["shape_candidates"])
        assert 0 <= out["delay_steps"][0] <= out["delay_steps"][1] <= step_hi
        assert 0 <= out["duration_steps"][0] <= out["duration_steps"][1] <= step_hi
        assert 1 <= out["half_life_steps"][0] <= out["half_life_steps"][1] <= hl_hi
    return out


def _is_high_value_for_priors(sample: dict, rel_thresh: float, args) -> bool:
    if int(sample.get("has_news", 0)) != 1:
        return False
    r = np.asarray(sample.get("raw_residual", []), dtype=np.float32).reshape(-1)
    if r.size == 0:
        return False
    rel_norm = _kernel_rel_norm(r)
    cfg = float(getattr(args, "kernel_api_prior_rel_norm_thresh", -1.0))
    if not np.isfinite(cfg) or cfg <= 0.0:
        cfg = float(max(0.0, rel_thresh))
    peak_cfg = float(getattr(args, "kernel_api_prior_peak_thresh", max(1.0, cfg * 2.0)))
    if not np.isfinite(peak_cfg) or peak_cfg <= 0.0:
        peak_cfg = float(max(1.0, cfg * 2.0))
    peak_abs = float(np.max(np.abs(r)))
    return bool(rel_norm >= cfg or peak_abs >= peak_cfg)


def _call_kernel_rel_sign_api(
    client,
    model: str,
    dataset_context: str,
    sample: dict,
    horizon: int,
    temperature: float = 0.1,
) -> tuple[dict | None, dict]:
    meta = {
        "ok": False,
        "type": "relsign",
        "target_time": str(sample.get("target_time", "")),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "raw": "",
        "parsed": "",
        "error": "",
    }
    news_list = _truncate_news_texts(sample.get("news_texts", []))
    news_block = "\n".join([f"{i+1}. {x}" for i, x in enumerate(news_list)]) if news_list else "(none)"
    r = np.asarray(sample.get("raw_residual", []), dtype=np.float32).reshape(-1)
    h = np.asarray(sample.get("history_z", []), dtype=np.float32).reshape(-1)
    b = np.asarray(sample.get("base_pred_z", []), dtype=np.float32).reshape(-1)
    user_prompt = (
        f"Dataset: {dataset_context}\n"
        f"Target time: {sample.get('target_time', '')}\n"
        f"Horizon: {int(horizon)}\n\n"
        f"News:\n{news_block}\n\n"
        f"History z summary: mean={float(h.mean() if h.size else 0.0):.4f}, std={float(h.std() if h.size else 0.0):.4f}\n"
        f"Base pred z summary: mean={float(b.mean() if b.size else 0.0):.4f}, std={float(b.std() if b.size else 0.0):.4f}\n"
        f"Residual z summary: mean={float(r.mean() if r.size else 0.0):.4f}, std={float(r.std() if r.size else 0.0):.4f}, "
        f"l2norm/sqrtH={_kernel_rel_norm(r):.4f}\n\n"
        "Return JSON only: {\"rel\": 0 or 1, \"sign\": \"UP\" or \"DOWN\", \"reason\": \"short\"}.\n"
        "If news has no reliable incremental effect over base prediction, set rel=0."
    )
    try:
        resp = client.chat.completions.create(
            model=str(model),
            temperature=float(temperature),
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Australian electricity market analyst. "
                        "Judge whether selected news adds reliable incremental signal for forecast correction."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        meta["error"] = _one_line(f"request_error:{type(e).__name__}:{e}", 260)
        return None, meta
    pt, ct = _extract_usage_tokens(resp)
    meta["prompt_tokens"] = int(pt)
    meta["completion_tokens"] = int(ct)
    txt = ""
    try:
        txt = str(resp.choices[0].message.content or "")
    except Exception:
        txt = ""
    meta["raw"] = _one_line(txt, 280)
    obj = _extract_json(txt)
    if not isinstance(obj, dict):
        meta["error"] = "json_parse_failed"
        return None, meta
    rel = obj.get("rel", None)
    sign = str(obj.get("sign", "")).upper().strip()
    out = {}
    try:
        rel_i = int(rel)
        if rel_i in (0, 1):
            out["rel"] = rel_i
    except Exception:
        pass
    if sign in {"UP", "DOWN"}:
        out["sign"] = sign
    if out:
        meta["ok"] = True
        meta["parsed"] = _one_line(json.dumps(out, ensure_ascii=False), 220)
        return out, meta
    meta["error"] = "schema_invalid"
    meta["parsed"] = _one_line(json.dumps(obj, ensure_ascii=False), 220)
    return None, meta


def _call_kernel_priors_api(
    client,
    dataset_context: str,
    sample: dict,
    horizon: int,
    temperature: float = 0.1,
    max_output_tokens: int = 320,
    timeout_sec: float = 25.0,
    max_retries: int = 2,
) -> tuple[dict | None, dict]:
    meta = {
        "ok": False,
        "type": "priors",
        "target_time": str(sample.get("target_time", "")),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "raw": "",
        "parsed": "",
        "error": "",
        "attempts": 0,
    }
    H = int(max(1, horizon))
    h = np.asarray(sample.get("history_z", []), dtype=np.float32).reshape(-1)
    b = np.asarray(sample.get("base_pred_z", []), dtype=np.float32).reshape(-1)
    r = np.asarray(sample.get("raw_residual", []), dtype=np.float32).reshape(-1)

    hst = _summary(h)
    bst = _summary(b)
    rst = _summary(r)
    news_list = _truncate_news_texts(sample.get("news_texts", []), max_items=8, max_chars_each=360)
    news_block = "\n".join([f"{i+1}. {x}" for i, x in enumerate(news_list)]) if news_list else "(none)"
    target_time = str(sample.get("target_time", ""))

    system_prompt = (
        "You are a power-market news-to-impact prior analyzer. "
        "Return JSON only. No markdown. No extra text. "
        "If uncertain, use sign=UNCERTAIN or causal=0."
    )
    user_prompt = (
        f"Market context: {dataset_context}\n"
        "Region: NSW / NEM\n"
        "Time granularity: half-hourly\n"
        f"Target timestamp: {target_time}\n"
        f"Horizon steps H: {int(H)}\n\n"
        f"Selected news:\n{news_block}\n\n"
        f"History z summary: mean={hst['mean']:.4f}, std={hst['std']:.4f}, trend={hst['trend']:.4f}, "
        f"last={hst['last']:.4f}\n"
        f"Base pred z summary: mean={bst['mean']:.4f}, std={bst['std']:.4f}, trend={bst['trend']:.4f}, "
        f"last={bst['last']:.4f}\n"
        f"Residual z summary: mean={rst['mean']:.4f}, std={rst['std']:.4f}, trend={rst['trend']:.4f}, "
        f"max={rst['max']:.4f}, min={rst['min']:.4f}, l2norm/sqrtH={_kernel_rel_norm(r):.4f}\n\n"
        "Return exactly one JSON object with keys:\n"
        "{\n"
        '  "causal": 0 or 1,\n'
        '  "sign": "UP"|"DOWN"|"UNCERTAIN",\n'
        '  "shape_candidates": subset of ["SPIKE","STEP_DECAY","RAMP_DECAY"],\n'
        '  "delay_steps": [lo, hi],\n'
        '  "duration_steps": [lo, hi],\n'
        '  "half_life_steps": [lo, hi],\n'
        '  "strength": "weak"|"medium"|"strong",\n'
        '  "rationale_short": "<<=30 words>"\n'
        "}\n"
        "This is causal prior extraction, not direct price forecasting."
    )

    attempts = int(max(1, max_retries))
    for t in range(attempts):
        meta["attempts"] = int(t + 1)
        try:
            token_limit = int(max(64, max_output_tokens))
            model_name = str(KERNEL_API_MODEL_FIXED)
            prefer_completion_limit = model_name.lower().startswith("gpt-5")
            token_param_order = (
                ["max_completion_tokens", "max_tokens"] if prefer_completion_limit else ["max_tokens", "max_completion_tokens"]
            )
            resp = None
            last_token_param_error = None
            for token_param_name in token_param_order:
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        temperature=float(max(0.0, min(0.3, temperature))),
                        timeout=float(max(5.0, timeout_sec)),
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        **{token_param_name: token_limit},
                    )
                    break
                except Exception as e_inner:
                    e_txt = str(e_inner)
                    if _token_param_is_unsupported_error(e_txt):
                        last_token_param_error = e_inner
                        continue
                    raise
            if resp is None:
                if last_token_param_error is not None:
                    raise last_token_param_error
                raise RuntimeError("priors_api_no_response")
            pt, ct = _extract_usage_tokens(resp)
            meta["prompt_tokens"] = int(meta["prompt_tokens"]) + int(pt)
            meta["completion_tokens"] = int(meta["completion_tokens"]) + int(ct)
            txt = ""
            try:
                txt = str(resp.choices[0].message.content or "")
            except Exception:
                txt = ""
            meta["raw"] = _one_line(txt, 280)
            obj = _extract_json(txt)
            priors = _sanitize_kernel_priors(obj, horizon=H)
            if priors is not None:
                meta["ok"] = True
                meta["parsed"] = _one_line(json.dumps(priors, ensure_ascii=False), 240)
                return priors, meta
            meta["error"] = "sanitize_failed_or_schema_invalid"
            if isinstance(obj, dict):
                meta["parsed"] = _one_line(json.dumps(obj, ensure_ascii=False), 240)
        except Exception as e:
            meta["error"] = _one_line(f"request_error:{type(e).__name__}:{e}", 260)
        if t + 1 < attempts:
            time.sleep(float(0.6 * (2**t)))
    if not str(meta.get("error", "")):
        meta["error"] = "no_valid_response"
    return None, meta


def _api_cache_key(sample: dict, dataset_context: str, horizon: int, api_type: str = KERNEL_API_TYPE_RELSIGN_V1) -> str:
    news_texts = [str(x) for x in (sample.get("news_texts", []) or [])[:8]]
    h = np.asarray(sample.get("history_z", []), dtype=np.float32).reshape(-1)
    b = np.asarray(sample.get("base_pred_z", []), dtype=np.float32).reshape(-1)
    r = np.asarray(sample.get("raw_residual", []), dtype=np.float32).reshape(-1)
    summary_payload = {
        "history": _summary(h),
        "base_pred": _summary(b),
        "residual": _summary(r),
    }
    payload = {
        "api_type": str(api_type),
        "dataset_context": str(dataset_context),
        "horizon": int(horizon),
        "target_time": str(sample.get("target_time", "")),
        "news_hash": _hash_hex("\n".join(news_texts)),
        "summary_hash": _hash_hex(json.dumps(summary_payload, ensure_ascii=False, sort_keys=True)),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _merge_api_rel_sign(auto_params: dict, api_obj: dict | None) -> tuple[dict, bool]:
    p = dict(auto_params or {})
    changed = False
    if not isinstance(api_obj, dict):
        return sanitize_kernel_params(p), changed
    if int(api_obj.get("rel", 1)) == 0:
        # Conservative override: only down-label REL to 0.
        return default_kernel_params(), True
    if int(p.get("rel", 0)) == 1:
        sign = str(api_obj.get("sign", "")).upper().strip()
        if sign in {"UP", "DOWN"} and sign != str(p.get("sign", "UP")).upper().strip():
            p["sign"] = sign
            changed = True
    return sanitize_kernel_params(p), changed


def _select_news_texts(
    news_df,
    target_time: str,
    args,
    policy_name: str,
) -> list[str]:
    if news_df is None or len(news_df) == 0:
        return []

    news_text_col = str(getattr(args, "news_text_col", "content"))
    if policy_name == "no_sum":
        news_text_col = "no_sum"
    elif policy_name == "sum_v0":
        news_text_col = "sum_v0"

    cand = get_candidates(
        news_df,
        args.news_time_col,
        target_time,
        args.news_window_days,
        args.news_topM,
    )
    use_text_col = news_text_col if news_text_col in cand.columns else str(getattr(args, "news_text_col", "content"))
    try:
        selected, _ = select_news(cand, policy_name, use_text_col, [], int(args.news_topK), args=args)
    except Exception:
        selected = cand.head(int(args.news_topK))
    if len(selected) > int(args.news_topK):
        selected = selected.head(int(args.news_topK))
    if int(getattr(args, "utility_rerank_enable", 1)) == 1 and len(selected) > 0:
        try:
            selected = rerank_selected_news_by_utility(
                selected=selected,
                target_time=target_time,
                time_col=args.news_time_col,
                text_col=use_text_col,
                policy_kw=[],
                args=args,
            )
        except Exception:
            pass
    return _extract_texts(selected, text_col=use_text_col)


def build_kernel_prompt(
    dataset_context: str,
    horizon: int,
    target_time: str,
    history_z: np.ndarray,
    base_pred_z: np.ndarray,
    news_texts: list[str],
) -> str:
    hst = _summary(history_z)
    bst = _summary(base_pred_z)
    if news_texts:
        news_block = "\n".join([f"{i+1}. {str(t)}" for i, t in enumerate(news_texts)])
    else:
        news_block = "(none)"
    return (
        f"[Dataset Context]\n{dataset_context}\n\n"
        f"[Target Time]\n{target_time}\n\n"
        f"[History Summary in z-space]\n{_stats_line('history_z', hst)}\n\n"
        f"[Base Prediction Summary in z-space]\n{_stats_line('base_pred_z', bst)}\n\n"
        f"[Selected News]\n{news_block}\n\n"
        f"[Task]\nPredict impact-kernel parameter tokens for horizon={int(horizon)}.\n"
        "Output tokens only, exactly in this order:\n"
        "<REL_0|REL_1> <SIGN_UP|SIGN_DOWN> <SHAPE_SPIKE|SHAPE_STEP_DECAY|SHAPE_RAMP_DECAY> "
        "<LAG_int> <HL_int> <DUR_int> <AMP_int>\n"
        "No JSON. No extra text."
    )


def build_kernel_sft_samples(
    train_df,
    news_df,
    base_backbone,
    args,
    global_zstats,
    device,
    live_logger=None,
):
    if train_df is None or len(train_df) == 0:
        return [], [float(i) / 20.0 for i in range(21)]

    stats = global_zstats or {}
    mu_global = float(stats.get("mu_global", stats.get("mu", 0.0)))
    sigma_global = float(stats.get("sigma_global", stats.get("sigma", 1.0)))
    sigma_global = max(sigma_global, float(getattr(args, "zscore_eps", 1e-6)))

    H = int(args.horizon)
    L = int(args.history_len)
    policy_name = str(getattr(args, "default_policy", "smart"))
    dataset_context = _infer_dataset_context(getattr(args, "taskName", ""))
    rel_thresh = float(getattr(args, "kernel_rel_norm_thresh", 0.05))
    rel_improve_ratio = float(getattr(args, "kernel_rel_improve_ratio", 0.0))
    rel_improve_abs = float(getattr(args, "kernel_rel_improve_abs", 0.0))
    a_max = float(getattr(args, "kernel_a_max", 2.0))
    api_enable = int(getattr(args, "kernel_api_enable", 0)) == 1
    api_model = str(KERNEL_API_MODEL_FIXED)
    requested_model = str(getattr(args, "kernel_api_model", KERNEL_API_MODEL_FIXED) or "").strip()
    api_temperature = float(getattr(args, "kernel_api_temperature", 0.1))
    api_max_calls = int(getattr(args, "kernel_api_max_calls", 200))
    api_log_every = int(max(1, getattr(args, "kernel_api_log_every", 10)))
    api_type_cfg = str(getattr(args, "kernel_api_type", "both")).lower().strip()
    if api_type_cfg not in {"priors", "relsign", "both"}:
        api_type_cfg = "both"
    api_use_priors = api_type_cfg in {"priors", "both"}
    api_use_relsign = api_type_cfg in {"relsign", "both"}
    api_cache_name = str(getattr(args, "kernel_api_cache_file", "sft_kernel_api_cache.json") or "sft_kernel_api_cache.json")
    api_client = None
    api_cache = {}
    api_cache_hits = 0
    api_calls = 0
    api_relsign_applied = 0
    api_priors_applied = 0
    api_priors_force_rel0 = 0
    api_ok_calls = 0
    api_fail_calls = 0
    api_last_error = ""
    api_fail_reason_counts = {}
    api_prompt_tokens = 0
    api_completion_tokens = 0
    api_cost_usd = 0.0
    api_price_in_per_1m = float(getattr(args, "kernel_api_price_in_per_1m", 5.0))
    api_price_out_per_1m = float(getattr(args, "kernel_api_price_out_per_1m", 15.0))
    if (not np.isfinite(api_price_in_per_1m)) or api_price_in_per_1m < 0.0:
        api_price_in_per_1m = 0.0
    if (not np.isfinite(api_price_out_per_1m)) or api_price_out_per_1m < 0.0:
        api_price_out_per_1m = 0.0
    api_examples_max = int(max(0, getattr(args, "kernel_api_log_examples", 3)))
    api_examples_ok = []
    api_examples_fail = []
    api_live_fail_log_max = int(max(0, getattr(args, "kernel_api_live_fail_log_max", 3)))
    api_live_fail_logged = 0
    api_path = os.path.join("./checkpoints", str(getattr(args, "taskName", "task")), api_cache_name)

    def _record_api_meta(meta: dict | None):
        nonlocal api_prompt_tokens, api_completion_tokens, api_cost_usd, api_examples_ok, api_examples_fail
        nonlocal api_ok_calls, api_fail_calls, api_last_error, api_fail_reason_counts, api_live_fail_logged
        if not isinstance(meta, dict):
            return
        pt = int(max(0, _safe_int(meta.get("prompt_tokens", 0), 0)))
        ct = int(max(0, _safe_int(meta.get("completion_tokens", 0), 0)))
        api_prompt_tokens += pt
        api_completion_tokens += ct
        cst = _cost_usd_est(pt, ct, api_price_in_per_1m, api_price_out_per_1m)
        api_cost_usd += float(cst)
        item = {
            "type": str(meta.get("type", "")),
            "target_time": str(meta.get("target_time", "")),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_usd_est": float(cst),
            "parsed": _one_line(str(meta.get("parsed", "")), 220),
            "raw": _one_line(str(meta.get("raw", "")), 220),
            "error": _one_line(str(meta.get("error", "")), 220),
            "attempts": int(max(1, _safe_int(meta.get("attempts", 1), 1))),
        }
        if bool(meta.get("ok", False)):
            api_ok_calls += 1
            if api_examples_max > 0 and len(api_examples_ok) < api_examples_max:
                api_examples_ok.append(item)
        else:
            api_fail_calls += 1
            reason = str(item.get("error", "")).strip() or "unknown_error"
            api_last_error = reason
            api_fail_reason_counts[reason] = int(api_fail_reason_counts.get(reason, 0)) + 1
            if api_examples_max > 0 and len(api_examples_fail) < api_examples_max:
                api_examples_fail.append(item)
            if live_logger is not None and api_live_fail_logged < api_live_fail_log_max:
                api_live_fail_logged += 1
                live_logger.info(
                    f"[KERNEL_API][FAIL_SAMPLE][{api_live_fail_logged}] type={item['type']} "
                    f"target_time={item['target_time']} attempts={item['attempts']} "
                    f"prompt_tokens={item['prompt_tokens']} completion_tokens={item['completion_tokens']} "
                    f"cost_usd_est={item['cost_usd_est']:.6f} error={item['error']} raw={item['raw']}"
                )

    if api_enable:
        if OpenAI is None:
            if live_logger is not None:
                live_logger.info("[KERNEL_API] openai package unavailable; disable API relabel.")
            api_enable = False
        else:
            api_key = ""
            api_key_file = os.path.join(".secrets", "api_key.txt")
            if os.path.isfile(api_key_file):
                try:
                    with open(api_key_file, "r", encoding="utf-8") as f:
                        api_key = str(f.read()).strip()
                except Exception:
                    api_key = ""
            if not api_key:
                api_key = str(getattr(args, "kernel_api_key", "") or os.getenv("OPENAI_API_KEY", "")).strip()
            if not api_key:
                if live_logger is not None:
                    live_logger.info("[KERNEL_API] kernel_api_enable=1 but no API key found; disable API relabel.")
                api_enable = False
            else:
                try:
                    api_client = OpenAI(api_key=api_key)
                    os.makedirs(os.path.dirname(api_path), exist_ok=True)
                    if os.path.isfile(api_path):
                        with open(api_path, "r", encoding="utf-8") as f:
                            loaded = json.load(f)
                        if isinstance(loaded, dict):
                            api_cache = loaded
                    if live_logger is not None:
                        live_logger.info(
                            f"[KERNEL_API] enabled: model={api_model}, api_type={api_type_cfg}, "
                            f"max_calls={api_max_calls}, cache={api_path} (n={len(api_cache)}), "
                            f"price_in_per_1m={api_price_in_per_1m:.4f}, price_out_per_1m={api_price_out_per_1m:.4f}"
                        )
                    if requested_model and requested_model != KERNEL_API_MODEL_FIXED and live_logger is not None:
                        live_logger.info(
                            f"[KERNEL_API] kernel_api_model={requested_model} is ignored; "
                            f"use fixed model={KERNEL_API_MODEL_FIXED}."
                        )
                except Exception as e:
                    if live_logger is not None:
                        live_logger.info(f"[KERNEL_API] init failed; disable API relabel: {e}")
                    api_enable = False
                    api_client = None
    if __debug__ and (not api_enable):
        assert api_client is None

    loader = make_loader(
        train_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        batch_size=1,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
        parse_time=False,
    )

    base_backbone.eval()
    tmp = []
    with torch.no_grad():
        for batch in loader:
            history = batch["history_value"][0].tolist()
            target = batch["target_value"][0].tolist()
            target_time = str(batch["target_time"][0])

            history_z = _zscore_np(history, mu_global, sigma_global)
            target_z = _zscore_np(target, mu_global, sigma_global)
            if history_z.shape[0] > L:
                history_z = history_z[-L:]
            if target_z.shape[0] > H:
                target_z = target_z[:H]

            h_t = torch.tensor(history_z, dtype=torch.float32, device=device).unsqueeze(0)
            base_pred = base_backbone(h_t).to(torch.float32).detach().cpu().numpy().reshape(-1)[:H]
            raw_residual = (target_z - base_pred).astype(np.float32)

            news_texts = _select_news_texts(
                news_df=news_df,
                target_time=target_time,
                args=args,
                policy_name=policy_name,
            )

            tmp.append(
                {
                    "target_time": target_time,
                    "history_z": history_z.astype(np.float32),
                    "base_pred_z": base_pred.astype(np.float32),
                    "raw_residual": raw_residual.astype(np.float32),
                    "news_texts": news_texts,
                    "has_news": 1 if len(news_texts) > 0 else 0,
                }
            )

    amp_source = [x["raw_residual"] for x in tmp if int(x["has_news"]) == 1]
    if len(amp_source) == 0:
        amp_source = [x["raw_residual"] for x in tmp]
    amp_bins = int(getattr(args, "kernel_amp_bins", 21))
    amp_table = build_amp_table_from_residuals(amp_source, n_bins=amp_bins)
    max_lag = int(getattr(args, "kernel_max_lag", 96))
    max_dur = int(getattr(args, "kernel_max_dur", 96))
    max_hl = int(getattr(args, "kernel_max_hl", 256))

    if __debug__ and len(tmp) > 0:
        k1 = _api_cache_key(
            tmp[0],
            dataset_context=dataset_context,
            horizon=H,
            api_type=KERNEL_API_TYPE_PRIORS_V1,
        )
        k2 = _api_cache_key(
            tmp[0],
            dataset_context=dataset_context,
            horizon=H,
            api_type=KERNEL_API_TYPE_RELSIGN_V1,
        )
        assert k1 != k2

    out = []
    for s in tmp:
        priors = None
        force_rel0 = int(s["has_news"]) == 0
        if api_enable and api_client is not None and api_use_priors and (not force_rel0) and _is_high_value_for_priors(s, rel_thresh, args):
            key_priors = _api_cache_key(
                s,
                dataset_context=dataset_context,
                horizon=H,
                api_type=KERNEL_API_TYPE_PRIORS_V1,
            )
            cached_priors = api_cache.get(key_priors)
            if isinstance(cached_priors, dict):
                priors = _sanitize_kernel_priors(cached_priors, horizon=H)
                if priors is not None:
                    api_cache_hits += 1
            if priors is None and (api_max_calls < 0 or api_calls < api_max_calls):
                priors = _call_kernel_priors_api(
                    client=api_client,
                    dataset_context=dataset_context,
                    sample=s,
                    horizon=H,
                    temperature=api_temperature,
                    max_output_tokens=int(getattr(args, "kernel_api_prior_max_output_tokens", 640)),
                    timeout_sec=float(getattr(args, "kernel_api_prior_timeout_sec", 25.0)),
                    max_retries=int(getattr(args, "kernel_api_prior_max_retries", 4)),
                )
                if isinstance(priors, tuple) and len(priors) == 2:
                    priors, api_meta = priors
                else:
                    api_meta = None
                api_calls += 1
                _record_api_meta(api_meta)
                if isinstance(priors, dict):
                    api_cache[key_priors] = priors
                    api_priors_applied += 1
                if live_logger is not None and api_calls % api_log_every == 0:
                    fail_top = ",".join(
                        [f"{k}:{v}" for k, v in sorted(api_fail_reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]]
                    )
                    live_logger.info(
                        f"[KERNEL_API] progress: calls={api_calls}, cache_hits={api_cache_hits}, "
                        f"priors_applied={api_priors_applied}, relsign_applied={api_relsign_applied}, "
                        f"ok_calls={api_ok_calls}, fail_calls={api_fail_calls}, "
                        f"prompt_tokens={api_prompt_tokens}, completion_tokens={api_completion_tokens}, "
                        f"cost_usd_est={api_cost_usd:.6f}, last_error={_one_line(api_last_error, 120)}, "
                        f"fail_top={_one_line(fail_top, 160)}"
                    )
        if isinstance(priors, dict) and int(priors.get("causal", 1)) == 0:
            force_rel0 = True
            api_priors_force_rel0 += 1

        params = fit_kernel_params_from_residual(
            raw_residual=s["raw_residual"],
            amp_table=amp_table,
            rel_norm_thresh=rel_thresh,
            rel_improve_min_ratio=rel_improve_ratio,
            rel_improve_min_abs=rel_improve_abs,
            a_max=a_max,
            force_rel0=force_rel0,
            args=args,
            priors=priors,
        )
        if (
            api_enable
            and api_client is not None
            and api_use_relsign
            and (not force_rel0)
            and _is_uncertain_for_api(s, params, rel_thresh, args)
        ):
            api_obj = None
            key = _api_cache_key(
                s,
                dataset_context=dataset_context,
                horizon=H,
                api_type=KERNEL_API_TYPE_RELSIGN_V1,
            )
            if key in api_cache and isinstance(api_cache.get(key), dict):
                api_obj = api_cache.get(key)
                api_cache_hits += 1
            elif api_max_calls < 0 or api_calls < api_max_calls:
                api_obj, api_meta = _call_kernel_rel_sign_api(
                    client=api_client,
                    model=api_model,
                    dataset_context=dataset_context,
                    sample=s,
                    horizon=H,
                    temperature=api_temperature,
                )
                api_calls += 1
                _record_api_meta(api_meta)
                if isinstance(api_obj, dict):
                    api_cache[key] = api_obj
                if live_logger is not None and api_calls % api_log_every == 0:
                    fail_top = ",".join(
                        [f"{k}:{v}" for k, v in sorted(api_fail_reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]]
                    )
                    live_logger.info(
                        f"[KERNEL_API] progress: calls={api_calls}, cache_hits={api_cache_hits}, "
                        f"priors_applied={api_priors_applied}, relsign_applied={api_relsign_applied}, "
                        f"ok_calls={api_ok_calls}, fail_calls={api_fail_calls}, "
                        f"prompt_tokens={api_prompt_tokens}, completion_tokens={api_completion_tokens}, "
                        f"cost_usd_est={api_cost_usd:.6f}, last_error={_one_line(api_last_error, 120)}, "
                        f"fail_top={_one_line(fail_top, 160)}"
                    )
            params, changed = _merge_api_rel_sign(params, api_obj)
            if changed:
                api_relsign_applied += 1

        prompt = build_kernel_prompt(
            dataset_context=dataset_context,
            horizon=H,
            target_time=s["target_time"],
            history_z=s["history_z"],
            base_pred_z=s["base_pred_z"],
            news_texts=s["news_texts"],
        )
        label = format_param_tokens(
            params,
            horizon=H,
            max_lag=max_lag,
            max_dur=max_dur,
            max_hl=max_hl,
            amp_table=amp_table,
        )
        out.append(
            {
                "prompt": prompt,
                "label_tokens": label,
                "target_time": s["target_time"],
                "news_count": int(len(s["news_texts"])),
                "params": params,
                "has_news": int(s["has_news"]),
            }
        )

    if api_enable and isinstance(api_cache, dict):
        try:
            with open(api_path, "w", encoding="utf-8") as f:
                json.dump(api_cache, f, ensure_ascii=False, indent=2)
            if live_logger is not None:
                live_logger.info(
                    f"[KERNEL_API] cache saved: {api_path} (n={len(api_cache)}), "
                    f"calls={api_calls}, cache_hits={api_cache_hits}, "
                    f"priors_applied={api_priors_applied}, priors_force_rel0={api_priors_force_rel0}, "
                    f"relsign_applied={api_relsign_applied}, ok_calls={api_ok_calls}, fail_calls={api_fail_calls}, "
                    f"prompt_tokens={api_prompt_tokens}, completion_tokens={api_completion_tokens}, "
                    f"cost_usd_est={api_cost_usd:.6f}, "
                    f"fail_reason_counts={json.dumps(api_fail_reason_counts, ensure_ascii=False)}"
                )
                for i, ex in enumerate(api_examples_ok, start=1):
                    live_logger.info(
                        f"[KERNEL_API][EXAMPLE][OK][{i}] type={ex['type']} target_time={ex['target_time']} "
                        f"attempts={ex['attempts']} prompt_tokens={ex['prompt_tokens']} "
                        f"completion_tokens={ex['completion_tokens']} cost_usd_est={ex['cost_usd_est']:.6f} "
                        f"parsed={ex['parsed']} raw={ex['raw']}"
                    )
                for i, ex in enumerate(api_examples_fail, start=1):
                    live_logger.info(
                        f"[KERNEL_API][EXAMPLE][FAIL][{i}] type={ex['type']} target_time={ex['target_time']} "
                        f"attempts={ex['attempts']} prompt_tokens={ex['prompt_tokens']} "
                        f"completion_tokens={ex['completion_tokens']} cost_usd_est={ex['cost_usd_est']:.6f} "
                        f"error={ex['error']} raw={ex['raw']}"
                    )
        except Exception as e:
            if live_logger is not None:
                live_logger.info(f"[KERNEL_API] failed to save cache: {e}")

    if live_logger is not None:
        live_logger.info(
            f"[KERNEL_SFT] built samples={len(out)}, amp_table_bins={len(amp_table)}, "
            f"rel_thresh={rel_thresh:.4f}, rel_improve_ratio={rel_improve_ratio:.4f}, "
            f"rel_improve_abs={rel_improve_abs:.6f}"
        )
    return out, amp_table


def to_kernel_sft_format(samples, tokenizer, args):
    if samples is None or len(samples) == 0:
        return []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = int(tokenizer.pad_token_id)

    sft_max_len = int(getattr(args, "sft_max_seq_len", 256))
    if sft_max_len <= 0:
        max_len = int(getattr(args, "max_seq_len", 768))
    else:
        max_len = sft_max_len

    ds = []
    for s in samples:
        prefix = str(s.get("prompt", "")) + "\n\n[Output Tokens]\n"
        out_text = str(s.get("label_tokens", "")).strip()
        pref_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
        out_ids = tokenizer(out_text, add_special_tokens=False, truncation=False)["input_ids"]

        if len(out_ids) >= max_len:
            out_ids = out_ids[: max_len - 1]
        keep_prefix = max_len - len(out_ids)
        keep_prefix = max(1, keep_prefix)
        if len(pref_ids) > keep_prefix:
            pref_ids = pref_ids[-keep_prefix:]

        ids = pref_ids + out_ids
        labels = ([-100] * len(pref_ids)) + out_ids
        attn = [1] * len(ids)

        if len(ids) < max_len:
            n_pad = max_len - len(ids)
            ids += [pad_id] * n_pad
            labels += [-100] * n_pad
            attn += [0] * n_pad
        else:
            ids = ids[:max_len]
            labels = labels[:max_len]
            attn = attn[:max_len]

        ds.append(
            {
                "input_ids": ids,
                "attention_mask": attn,
                "labels": labels,
            }
        )
    return ds


def build_kernel_prompts_from_batch(
    batch,
    base_pred_z: torch.Tensor,
    args,
    global_zstats,
    news_df,
    policy_name: str,
):
    stats = global_zstats or {}
    mu_global = float(stats.get("mu_global", stats.get("mu", 0.0)))
    sigma_global = float(stats.get("sigma_global", stats.get("sigma", 1.0)))
    sigma_global = max(sigma_global, float(getattr(args, "zscore_eps", 1e-6)))

    dataset_context = _infer_dataset_context(getattr(args, "taskName", ""))
    H = int(args.horizon)
    L = int(args.history_len)

    base_np = base_pred_z.detach().to(torch.float32).cpu().numpy()
    B = len(batch["history_value"])
    prompts = []
    news_counts = []
    for i in range(B):
        history = batch["history_value"][i].tolist()
        target_time = str(batch["target_time"][i])
        history_z = _zscore_np(history, mu_global, sigma_global)
        if history_z.shape[0] > L:
            history_z = history_z[-L:]
        news_texts = _select_news_texts(
            news_df=news_df,
            target_time=target_time,
            args=args,
            policy_name=policy_name,
        )
        prompt = build_kernel_prompt(
            dataset_context=dataset_context,
            horizon=H,
            target_time=target_time,
            history_z=history_z,
            base_pred_z=base_np[i][:H],
            news_texts=news_texts,
        )
        prompts.append(prompt)
        news_counts.append(int(len(news_texts)))
    return prompts, news_counts
