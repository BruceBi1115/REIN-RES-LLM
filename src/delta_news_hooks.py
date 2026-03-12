from __future__ import annotations

import json
import os
import re
import time
from typing import Any


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


class OpenAINewsApiAdapter:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.1",
        base_url: str | None = None,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        cfg = {"api_key": str(api_key).strip()}
        if str(base_url or "").strip():
            cfg["base_url"] = str(base_url).strip()
        self.client = OpenAI(**cfg)
        self.model = str(model or "gpt-5.1").strip() or "gpt-5.1"
        self.timeout_sec = float(max(1.0, timeout_sec))
        self.max_retries = int(max(0, max_retries))

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
        target_time = str(context.get("target_time", "")).strip()
        joined = "\n".join([f"{i+1}. {str(x).strip()}" for i, x in enumerate(clean_news) if str(x).strip()])
        if not joined:
            return ""
        system = (
            "You are a power-market news refiner. "
            "Compress noisy headlines into concise, factual bullet points for time-series forecasting."
        )
        user = (
            f"Region: {region}\n"
            f"TargetTime: {target_time}\n"
            "Task: keep only price-relevant facts, remove fluff/duplicates, preserve direction cues.\n"
            "Output plain text bullets only.\n\n"
            f"News:\n{joined}"
        )
        return self._chat_text(system, user, max_tokens=500)

    def extract_events(self, text: str, context: dict | None = None) -> dict:
        context = context or {}
        region = str(context.get("region", "")).strip()
        target_time = str(context.get("target_time", "")).strip()
        system = (
            "You extract structured power-market event signals. "
            "Return JSON only with keys: relevance,direction,strength,persistence,confidence,event_type."
        )
        user = (
            f"Region: {region}\n"
            f"TargetTime: {target_time}\n"
            "Schema constraints:\n"
            "- relevance/strength/persistence/confidence in [0,1]\n"
            "- direction in {-1,0,1} (or up/down/uncertain)\n"
            "- event_type short string\n\n"
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


def build_news_api_adapter(args, live_logger=None):
    refine_mode = str(getattr(args, "news_refine_mode", "local")).lower().strip()
    structured_mode = str(getattr(args, "news_structured_mode", "off")).lower().strip()
    hard_reflect_mode = str(getattr(args, "hard_reflection_mode", "off")).lower().strip()
    need_api = (refine_mode == "api") or (structured_mode == "api") or (hard_reflect_mode == "api")
    if not need_api:
        return None

    key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    key_path = str(getattr(args, "news_api_key_path", ".secrets/api_key.txt"))
    if not key:
        key = _read_api_key_from_path(key_path)
    if not key:
        for fallback in ["api_key.txt", ".secrets/api_key.txt"]:
            key = _read_api_key_from_path(fallback)
            if key:
                break
    if not key:
        if live_logger is not None:
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
        )
    except Exception:
        if live_logger is not None:
            live_logger.info("[NEWS_API] failed to initialize OpenAI adapter; fallback to local/heuristic behavior.")
        return None

    if live_logger is not None:
        live_logger.info(
            f"[NEWS_API] enabled model={model} key_path={key_path} "
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
