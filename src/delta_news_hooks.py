from __future__ import annotations

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

