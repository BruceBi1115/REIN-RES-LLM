from __future__ import annotations

import hashlib
import json
import os
import re

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


def _call_kernel_rel_sign_api(
    client,
    model: str,
    dataset_context: str,
    sample: dict,
    horizon: int,
    temperature: float = 0.1,
) -> dict | None:
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
    txt = ""
    try:
        txt = str(resp.choices[0].message.content or "")
    except Exception:
        txt = ""
    obj = _extract_json(txt)
    if not isinstance(obj, dict):
        return None
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
    return out if out else None


def _api_cache_key(sample: dict, dataset_context: str, horizon: int) -> str:
    payload = {
        "dataset_context": str(dataset_context),
        "horizon": int(horizon),
        "target_time": str(sample.get("target_time", "")),
        "news_texts": [str(x) for x in (sample.get("news_texts", []) or [])[:8]],
        "history_z": [round(float(x), 4) for x in np.asarray(sample.get("history_z", []), dtype=np.float32).reshape(-1).tolist()],
        "base_pred_z": [round(float(x), 4) for x in np.asarray(sample.get("base_pred_z", []), dtype=np.float32).reshape(-1).tolist()],
        "raw_residual": [round(float(x), 4) for x in np.asarray(sample.get("raw_residual", []), dtype=np.float32).reshape(-1).tolist()],
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
        "<LAG_0..LAG_6> <HL_1|HL_2|HL_4|HL_8|HL_16|HL_32> <DUR_0..DUR_6> <AMP_0..AMP_20>\n"
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
    api_model = str(getattr(args, "kernel_api_model", "gpt-4o"))
    api_temperature = float(getattr(args, "kernel_api_temperature", 0.1))
    api_max_calls = int(getattr(args, "kernel_api_max_calls", 200))
    api_log_every = int(max(1, getattr(args, "kernel_api_log_every", 10)))
    api_cache_name = str(getattr(args, "kernel_api_cache_file", "sft_kernel_api_cache.json") or "sft_kernel_api_cache.json")
    api_client = None
    api_cache = {}
    api_cache_hits = 0
    api_calls = 0
    api_applied = 0
    api_path = os.path.join("./checkpoints", str(getattr(args, "taskName", "task")), api_cache_name)

    if api_enable:
        if OpenAI is None:
            if live_logger is not None:
                live_logger.info("[KERNEL_API] openai package unavailable; disable API relabel.")
            api_enable = False
        else:
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
                            f"[KERNEL_API] enabled: model={api_model}, max_calls={api_max_calls}, cache={api_path} (n={len(api_cache)})"
                        )
                except Exception as e:
                    if live_logger is not None:
                        live_logger.info(f"[KERNEL_API] init failed; disable API relabel: {e}")
                    api_enable = False
                    api_client = None

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

    out = []
    for s in tmp:
        params = fit_kernel_params_from_residual(
            raw_residual=s["raw_residual"],
            amp_table=amp_table,
            rel_norm_thresh=rel_thresh,
            rel_improve_min_ratio=rel_improve_ratio,
            rel_improve_min_abs=rel_improve_abs,
            a_max=a_max,
            force_rel0=(int(s["has_news"]) == 0),
        )
        if api_enable and api_client is not None and _is_uncertain_for_api(s, params, rel_thresh, args):
            api_obj = None
            key = _api_cache_key(s, dataset_context=dataset_context, horizon=H)
            if key in api_cache and isinstance(api_cache.get(key), dict):
                api_obj = api_cache.get(key)
                api_cache_hits += 1
            elif api_max_calls < 0 or api_calls < api_max_calls:
                try:
                    api_obj = _call_kernel_rel_sign_api(
                        client=api_client,
                        model=api_model,
                        dataset_context=dataset_context,
                        sample=s,
                        horizon=H,
                        temperature=api_temperature,
                    )
                except Exception:
                    api_obj = None
                api_calls += 1
                if isinstance(api_obj, dict):
                    api_cache[key] = api_obj
                if live_logger is not None and api_calls % api_log_every == 0:
                    live_logger.info(f"[KERNEL_API] progress: calls={api_calls}, cache_hits={api_cache_hits}, applied={api_applied}")
            params, changed = _merge_api_rel_sign(params, api_obj)
            if changed:
                api_applied += 1

        prompt = build_kernel_prompt(
            dataset_context=dataset_context,
            horizon=H,
            target_time=s["target_time"],
            history_z=s["history_z"],
            base_pred_z=s["base_pred_z"],
            news_texts=s["news_texts"],
        )
        label = format_param_tokens(params)
        out.append(
            {
                "prompt": prompt,
                "label_tokens": label,
                "target_time": s["target_time"],
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
                    f"calls={api_calls}, cache_hits={api_cache_hits}, applied={api_applied}"
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
