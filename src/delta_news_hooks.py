from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import numpy as np


def _truncate_with_tokenizer(text: str, tokenizer, max_tokens: int) -> str:
    if not text:
        return ""
    max_tokens = int(max(1, max_tokens))
    if tokenizer is None:
        return text[: max_tokens * 4]
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False,
    )
    return tokenizer.decode(enc["input_ids"], skip_special_tokens=True).strip()


def refine_news_text(
    raw_news_texts: list[str],
    tokenizer,
    max_tokens: int,
    mode: str = "local",
    api_adapter: Any = None,
    context: dict | None = None,
) -> str:
    clean = [str(x).strip() for x in raw_news_texts if str(x).strip()]
    if len(clean) == 0:
        return ""

    use_mode = str(mode or "local").lower().strip()
    if use_mode == "api" and api_adapter is not None and hasattr(api_adapter, "refine_news"):
        try:
            out = api_adapter.refine_news(clean, context=context or {})
            if isinstance(out, str) and out.strip():
                return _truncate_with_tokenizer(out.strip(), tokenizer, max_tokens=max_tokens)
        except Exception:
            pass

    joined = "\n".join([f"- {item}" for item in clean])
    return _truncate_with_tokenizer(joined, tokenizer, max_tokens=max_tokens)


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if not (v == v) or v == float("inf") or v == float("-inf"):
        v = float(default)
    return float(max(0.0, min(1.0, v)))


def _parse_json_obj(text: str) -> dict:
    s = str(text or "").strip()
    if not s:
        return {}
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_json_list(text: str) -> list:
    s = str(text or "").strip()
    if not s:
        return []
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _compact_text(text: Any, max_chars: int = 260) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = re.sub(r"^[-*•\d\.\)\s]+", "", s)
    if len(s) > int(max(1, max_chars)):
        s = s[: int(max(1, max_chars))].rstrip(" ,;:")
    return s


def _norm_direction_label(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"1", "+1", "up", "rise", "rising", "positive", "bullish", "upward"}:
        return "upward"
    if s in {"-1", "down", "fall", "falling", "negative", "bearish", "downward"}:
        return "downward"
    if s in {"mixed", "both"}:
        return "mixed"
    if s in {"0", "neutral", "uncertain", "flat"}:
        return "neutral"
    try:
        v = float(s)
        if v > 0.15:
            return "upward"
        if v < -0.15:
            return "downward"
        return "neutral"
    except Exception:
        return "neutral"


def _norm_confidence_label(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"low", "medium", "high"}:
        return s
    try:
        v = float(s)
    except Exception:
        return "medium"
    if v >= 0.67:
        return "high"
    if v <= 0.33:
        return "low"
    return "medium"


class OpenAINewsApiAdapter:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.1",
        base_url: str | None = None,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
        live_logger: Any = None,
    ):
        from openai import OpenAI

        cfg = {"api_key": str(api_key).strip()}
        if str(base_url or "").strip():
            cfg["base_url"] = str(base_url).strip()
        self.client = OpenAI(**cfg)
        self.model = str(model or "gpt-5.1").strip() or "gpt-5.1"
        self.timeout_sec = float(max(1.0, timeout_sec))
        self.max_retries = int(max(0, max_retries))
        self.live_logger = live_logger
        self._refine_example_logged = False

    def _log_refine_example(self, system_prompt: str, user_prompt: str, raw_output: str, clean_news: list[str]):
        if self.live_logger is None or self._refine_example_logged:
            return
        self.live_logger.info(
            "[NEWS_API][REFINE_EXAMPLE] "
            f"model={self.model} "
            f"news_items={len(clean_news)}\n"
            "[SYSTEM]\n"
            f"{system_prompt}\n"
            "[USER]\n"
            f"{user_prompt}\n"
            "[OUTPUT]\n"
            f"{str(raw_output or '').strip() or '<EMPTY>'}"
        )
        self._refine_example_logged = True

    @staticmethod
    def _extract_text_from_responses(resp) -> str:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        parts = []
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            for item in output:
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content", None)
                if not isinstance(content, list):
                    continue
                for c in content:
                    t = None
                    if isinstance(c, dict):
                        t = c.get("text", None) or c.get("output_text", None)
                    else:
                        t = getattr(c, "text", None) or getattr(c, "output_text", None)
                    if t:
                        parts.append(str(t))
        return "\n".join(parts).strip()

    def _chat_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        sys_text = str(system_prompt)
        usr_text = str(user_prompt)
        messages = [{"role": "system", "content": sys_text}, {"role": "user", "content": usr_text}]
        last_err = None
        token_cap = int(max(1, max_tokens))
        for attempt in range(self.max_retries + 1):
            try:
                if hasattr(self.client, "responses"):
                    resp = self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "system", "content": [{"type": "input_text", "text": sys_text}]},
                            {"role": "user", "content": [{"type": "input_text", "text": usr_text}]},
                        ],
                        max_output_tokens=token_cap,
                        timeout=self.timeout_sec,
                    )
                    out = self._extract_text_from_responses(resp)
                    if out:
                        return out
            except Exception as e:
                last_err = e
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=token_cap,
                    timeout=self.timeout_sec,
                )
                txt = ""
                if getattr(resp, "choices", None):
                    msg = resp.choices[0].message
                    txt = getattr(msg, "content", "") or ""
                if isinstance(txt, list):
                    txt = "".join([str(x) for x in txt])
                out = str(txt).strip()
                if out:
                    return out
            except Exception as e:
                last_err = e
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=token_cap,
                    timeout=self.timeout_sec,
                )
                txt = ""
                if getattr(resp, "choices", None):
                    msg = resp.choices[0].message
                    txt = getattr(msg, "content", "") or ""
                if isinstance(txt, list):
                    txt = "".join([str(x) for x in txt])
                out = str(txt).strip()
                if out:
                    return out
            except Exception as e:
                last_err = e
            if attempt < self.max_retries:
                time.sleep(0.6 * (attempt + 1))
        if last_err is not None:
            raise last_err
        return ""

    def refine_news(self, clean_news: list[str], context: dict | None = None) -> str:
        context = context or {}
        region = str(context.get("region", "")).strip()
        dataset_desc = str(
            context.get("description", "")
            or context.get("dataset_description", "")
            or ""
        ).strip()
        topic = dataset_desc or "the target time-series forecasting task"
        joined = "\n".join([f"{i+1}. {str(x).strip()}" for i, x in enumerate(clean_news) if str(x).strip()])
        if not joined:
            return ""
        system = (
            "You summarize news for exogenous impact forecasting.\n"
            "Return JSON only with keys: direction, confidence, summary, key_drivers.\n"
            "Rules:\n"
            "- direction in {upward,downward,neutral,mixed}\n"
            "- confidence in {low,medium,high}\n"
            "- summary: one short natural-language sentence, <=70 words\n"
            "- key_drivers: short phrase list, <=25 words\n"
            "- do not include markdown, bullets, or extra keys."
        )
        user = (
            f"Region: {region}\n"
            f"DatasetTopic: {topic}\n"
            "Task: compress the raw news into a short, fixed-template impact summary for this dataset topic.\n\n"
            f"News:\n{joined}"
        )
        raw = self._chat_text(system, user, max_tokens=300)
        self._log_refine_example(system, user, raw, clean_news)
        obj = _parse_json_obj(raw)
        if obj:
            direction = _norm_direction_label(obj.get("direction", "neutral"))
            confidence = _norm_confidence_label(obj.get("confidence", "medium"))
            summary = _compact_text(obj.get("summary", ""), max_chars=320)
            drivers = _compact_text(obj.get("key_drivers", ""), max_chars=180)
            if not summary:
                summary = "Recent news contains limited high-confidence directional signals."
            if not drivers:
                drivers = "market sentiment, demand-supply conditions"
            return (
                f"Near-term impact direction: {direction} ({confidence} confidence). "
                f"{summary} Key drivers: {drivers}."
            )

        # Fallback to a short single-paragraph template even when model output is not strict JSON.
        fallback = _compact_text(raw, max_chars=360)
        
        if not fallback:
            fallback = "Recent news contains limited high-confidence directional signals."
        return (
            "Near-term impact direction: neutral (medium confidence). "
            f"{fallback}"
        )

    def extract_events(self, text: str, context: dict | None = None) -> dict:
        context = context or {}
        region = str(context.get("region", "")).strip()
        dataset_desc = str(
            context.get("description", "")
            or context.get("dataset_description", "")
            or ""
        ).strip()
        topic = dataset_desc or "the target time-series forecasting task"
        system = (
            "You extract structured event signals for exogenous impact forecasting. "
            "Return JSON only with keys: relevance,direction,strength,persistence,confidence,event_type."
        )
        user = (
            f"Region: {region}\n"
            f"DatasetTopic: {topic}\n"
            "Schema constraints:\n"
            "- relevance/strength/persistence/confidence in [0,1]\n"
            "- direction in {-1,0,1} (or up/down/uncertain)\n"
            "- event_type short domain-appropriate string\n\n"
            "Task: extract structured event labels that describe how this text may affect the dataset topic.\n\n"
            f"Text:\n{text}\n\n"
            "Return JSON only."
        )
        raw = self._chat_text(system, user, max_tokens=220)
        obj = _parse_json_obj(raw)
        if not obj:
            return {}

        direction_raw = obj.get("direction", 0)
        if isinstance(direction_raw, str):
            d = direction_raw.strip().lower()
            if d in {"up", "rise", "positive", "bullish"}:
                direction = 1
            elif d in {"down", "fall", "negative", "bearish"}:
                direction = -1
            else:
                direction = 0
        else:
            try:
                dv = float(direction_raw)
            except Exception:
                dv = 0.0
            direction = 1 if dv > 0.15 else (-1 if dv < -0.15 else 0)

        event_type = str(obj.get("event_type", "general")).strip().lower() or "general"
        return {
            "relevance": _clamp01(obj.get("relevance", 0.5), default=0.5),
            "direction": int(direction),
            "strength": _clamp01(obj.get("strength", 0.4), default=0.4),
            "persistence": _clamp01(obj.get("persistence", 0.5), default=0.5),
            "confidence": _clamp01(obj.get("confidence", 0.5), default=0.5),
            "event_type": event_type,
        }

    def reflect_hard_samples(self, hard_samples: list[dict]) -> list[dict]:
        if not isinstance(hard_samples, list) or len(hard_samples) == 0:
            return []
        keep = []
        for rec in hard_samples[:16]:
            if not isinstance(rec, dict):
                continue
            keep.append(
                {
                    "sample_id": str(rec.get("sample_id", "")),
                    "error_z_mae": float(rec.get("error_z_mae", 0.0) or 0.0),
                    "target_time": str(rec.get("target_time", "")),
                    "news": str(rec.get("news", ""))[:1200],
                }
            )
        if len(keep) == 0:
            return []
        system = (
            "You are a forecasting error analyst. "
            "Return JSON array only; each item has sample_id, failure_reason, fix_hint."
        )
        user = "Analyze these hard samples:\n" + json.dumps(keep, ensure_ascii=False)
        raw = self._chat_text(system, user, max_tokens=500)
        out = _parse_json_list(raw)
        if not isinstance(out, list):
            return []
        return [x for x in out if isinstance(x, dict)]


def _read_api_key_from_path(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            return str(f.read()).strip()
    except Exception:
        return ""


def discover_news_api_key(args) -> tuple[str, str]:
    key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if key:
        return key, "env:OPENAI_API_KEY"

    key_path = str(getattr(args, "news_api_key_path", ".secrets/api_key.txt"))
    key = _read_api_key_from_path(key_path)
    if key:
        return key, key_path

    for fallback in ["api_key.txt", ".secrets/api_key.txt"]:
        key = _read_api_key_from_path(fallback)
        if key:
            return key, fallback
    return "", ""


def build_news_api_adapter(args, live_logger=None):
    refine_mode = str(getattr(args, "news_refine_mode", "local")).lower().strip()
    structured_mode = str(getattr(args, "news_structured_mode", "off")).lower().strip()
    hard_reflect_mode = str(getattr(args, "hard_reflection_mode", "off")).lower().strip()
    need_api = (refine_mode == "api") or (structured_mode == "api") or (hard_reflect_mode == "api")
    if not need_api:
        return None

    key, key_source = discover_news_api_key(args)
    key_path = str(getattr(args, "news_api_key_path", ".secrets/api_key.txt"))
    strict_required = bool(getattr(args, "_require_news_api_adapter", False))
    cache_mode = str(getattr(args, "_news_doc_cache_mode", "") or "").strip()
    if not key:
        if strict_required:
            raise RuntimeError(
                "API mode is required to build refined news cache, but no API key was found. "
                f"Checked OPENAI_API_KEY, {key_path}, api_key.txt, and .secrets/api_key.txt."
            )
        if live_logger is not None:
            if cache_mode == "read_only" and ((refine_mode == "api") or (structured_mode == "api")):
                live_logger.info(
                    "[NEWS_API] API mode configured, but read_only cache mode does not require an adapter; "
                    "no API key found."
                )
            else:
                live_logger.info("[NEWS_API] API mode requested but no API key found; fallback to local/heuristic behavior.")
        return None

    model = str(getattr(args, "news_api_model", "gpt-5.1") or "gpt-5.1")
    base_url = str(getattr(args, "news_api_base_url", "") or "")
    timeout_sec = float(getattr(args, "news_api_timeout_sec", 30.0) or 30.0)
    max_retries = int(getattr(args, "news_api_max_retries", 2) or 2)
    try:
        adapter = OpenAINewsApiAdapter(
            api_key=key,
            model=model,
            base_url=base_url,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            live_logger=live_logger,
        )
    except Exception as exc:
        if strict_required:
            raise RuntimeError(f"Failed to initialize OpenAI adapter for cache build: {exc}") from exc
        if live_logger is not None:
            live_logger.info("[NEWS_API] failed to initialize OpenAI adapter; fallback to local/heuristic behavior.")
        return None

    if live_logger is not None:
        live_logger.info(
            f"[NEWS_API] enabled model={model} key_source={key_source or key_path} "
            f"refine_mode={refine_mode} structured_mode={structured_mode}"
        )
    return adapter


def extract_structured_events(
    raw_or_refined_news: str,
    mode: str = "off",
    api_adapter: Any = None,
    context: dict | None = None,
) -> dict:
    use_mode = str(mode or "off").lower().strip()
    txt = str(raw_or_refined_news or "").strip()
    if not txt:
        return {}

    if use_mode == "api" and api_adapter is not None and hasattr(api_adapter, "extract_events"):
        try:
            out = api_adapter.extract_events(txt, context=context or {})
            if isinstance(out, dict):
                return out
        except Exception:
            pass

    if use_mode != "heuristic":
        return {}

    lo = txt.lower()
    pos_kw = ["rise", "up", "increase", "higher", "strong", "growth", "surge"]
    neg_kw = ["fall", "down", "decrease", "lower", "weak", "drop", "decline"]

    pos = sum(lo.count(k) for k in pos_kw)
    neg = sum(lo.count(k) for k in neg_kw)
    direction = 1 if pos > neg else (-1 if neg > pos else 0)
    strength = min(1.0, abs(pos - neg) / 6.0)
    relevance = min(1.0, max(pos + neg, 1) / 8.0)
    confidence = min(1.0, 0.4 + 0.6 * strength)
    persistence = 0.5

    if any(k in lo for k in ["policy", "government", "regulation"]):
        event_type = "policy"
    elif any(k in lo for k in ["weather", "storm", "temperature", "rain", "heat"]):
        event_type = "weather"
    elif any(k in lo for k in ["gas", "coal", "oil", "fuel"]):
        event_type = "fuel"
    else:
        event_type = "general"

    return {
        "relevance": float(relevance),
        "direction": int(direction),
        "strength": float(strength),
        "persistence": float(persistence),
        "confidence": float(confidence),
        "event_type": event_type,
    }


def merge_structured_events(events_list: list[dict] | None) -> dict:
    items = [x for x in (events_list or []) if isinstance(x, dict) and len(x) > 0]
    if len(items) == 0:
        return {}
    if len(items) == 1:
        return dict(items[0])

    weights = []
    relevance_vals = []
    strength_vals = []
    persistence_vals = []
    confidence_vals = []
    direction_num = 0.0
    direction_den = 0.0
    event_type_scores = {}

    for ev in items:
        relevance = _clamp01(ev.get("relevance", 0.0), default=0.0)
        strength = _clamp01(ev.get("strength", 0.0), default=0.0)
        persistence = _clamp01(ev.get("persistence", 0.5), default=0.5)
        confidence = _clamp01(ev.get("confidence", 0.5), default=0.5)
        direction_raw = ev.get("direction", 0)
        try:
            direction = float(direction_raw)
        except Exception:
            direction = 0.0
        if direction > 0.15:
            direction = 1.0
        elif direction < -0.15:
            direction = -1.0
        else:
            direction = 0.0

        w = max(1e-6, relevance * (0.5 + 0.5 * confidence))
        direction_w = w * max(0.25, strength)
        weights.append(w)
        relevance_vals.append(relevance)
        strength_vals.append(strength)
        persistence_vals.append(persistence)
        confidence_vals.append(confidence)
        direction_num += direction_w * direction
        direction_den += direction_w

        event_type = str(ev.get("event_type", "general")).strip().lower() or "general"
        event_type_scores[event_type] = float(event_type_scores.get(event_type, 0.0)) + w

    w_arr = np.asarray(weights, dtype=np.float32)
    w_sum = float(w_arr.sum())
    if w_sum <= 1e-8:
        return {}

    direction_score = float(direction_num / max(direction_den, 1e-6))
    direction = 1 if direction_score > 0.15 else (-1 if direction_score < -0.15 else 0)
    event_type = max(event_type_scores.items(), key=lambda kv: float(kv[1]))[0] if event_type_scores else "general"

    return {
        "relevance": float(max(relevance_vals) if relevance_vals else 0.0),
        "direction": int(direction),
        "strength": float(np.average(np.asarray(strength_vals, dtype=np.float32), weights=w_arr)),
        "persistence": float(np.average(np.asarray(persistence_vals, dtype=np.float32), weights=w_arr)),
        "confidence": float(np.average(np.asarray(confidence_vals, dtype=np.float32), weights=w_arr)),
        "event_type": event_type,
    }


def format_structured_events_for_prompt(events: dict) -> str:
    if not isinstance(events, dict) or len(events) == 0:
        return ""
    keys = ["relevance", "direction", "strength", "persistence", "confidence", "event_type"]
    parts = []
    for k in keys:
        if k in events and events[k] is not None:
            parts.append(f"{k}={events[k]}")
    if not parts:
        return ""
    return "[Structured Event] " + ", ".join(parts)


def reflect_hard_samples(
    hard_samples: list[dict],
    mode: str = "off",
    api_adapter: Any = None,
) -> list[dict]:
    use_mode = str(mode or "off").lower().strip()
    if use_mode == "api" and api_adapter is not None and hasattr(api_adapter, "reflect_hard_samples"):
        try:
            out = api_adapter.reflect_hard_samples(hard_samples)
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return []
