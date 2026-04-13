from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any

from ..delta_news_hooks import OpenAINewsApiAdapter, build_news_api_adapter

SCHEMA_VERSION = "v4_regime_v2"
TOPIC_TAGS = (
    "supply_tight",
    "supply_surplus",
    "heatwave",
    "cold_snap",
    "holiday",
    "outage",
    "fuel_shock",
    "renewable_surge",
    "renewable_drought",
    "interconnector_limit",
    "policy_active",
    "market_intervention",
    "retrospective",
    "routine",
    "other",
)
REGIME_KEYS = (
    "tightness",
    "demand_outlook",
    "renewable_surplus",
    "volatility_tone",
    "policy_in_effect",
)


def _compact_summary(text: str, max_words: int = 60) -> str:
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    words = clean.split()
    return " ".join(words[: max(1, max_words)]).strip()


def _doc_identity(doc: dict[str, Any]) -> str:
    if doc.get("id") not in (None, ""):
        return str(doc["id"])
    basis = "|".join(
        [
            str(doc.get("title", "")),
            str(doc.get("date", "")),
            str(doc.get("url", "")),
        ]
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = str(text or "").strip()
    if not payload:
        return {}
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", payload)
        payload = re.sub(r"\n?```$", "", payload).strip()
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _clamp(value: Any, *, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not (out == out) or out == float("inf") or out == float("-inf"):
        out = float(default)
    return float(max(lo, min(hi, out)))


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "actionable"}


def _normalize_topic_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        items = re.split(r"[,;/|]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = []
    clean: list[str] = []
    for item in items:
        tag = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        if tag in TOPIC_TAGS and tag not in clean:
            clean.append(tag)
    return clean[:5]


def _default_record(doc: dict[str, Any], schema_variant: str) -> dict[str, Any]:
    summary_basis = " ".join(
        part for part in [str(doc.get("title", "")), str(doc.get("content", ""))[:240]] if part.strip()
    ).strip()
    return {
        "doc_key": _doc_identity(doc),
        "published_at": str(doc.get("date", "")),
        "title": str(doc.get("title", "")),
        "schema_version": SCHEMA_VERSION,
        "schema_variant": str(schema_variant),
        "is_actionable": False,
        "topic_tags": ["routine"],
        "regime_vec": {
            "tightness": 0.0,
            "demand_outlook": 0.0,
            "renewable_surplus": 0.0,
            "volatility_tone": 0.0,
            "policy_in_effect": 0.0,
        },
        "horizon_days": 0,
        "confidence": 0.0,
        "summary": _compact_summary(summary_basis),
    }


def _build_prompts(doc: dict[str, Any], schema_variant: str) -> tuple[str, str]:
    title = str(doc.get("title", "")).strip()
    published_at = str(doc.get("date", "")).strip()
    content = str(doc.get("content", "") or doc.get("summary", "") or "").strip()
    body = _compact_summary(content, max_words=220)
    target_name = "NSW electricity load" if schema_variant == "load" else "NSW electricity price"

    system_prompt = (
        f"You are refining generic energy news for {target_name} forecasting into strict JSON. "
        "Return JSON only, no markdown. "
        "The key question is whether the article describes a currently in-force market condition that can still matter within the next 14 days. "
        "If the article is retrospective, routine commentary, historical recap, or not currently actionable, set is_actionable=false. "
        "Use only the closed topic vocabulary: "
        + ", ".join(TOPIC_TAGS)
        + ". "
        "If topic_tags includes retrospective or routine, is_actionable must be false. "
        "Return fields exactly: "
        "{is_actionable:boolean, topic_tags:string[], regime_vec:{tightness:float, demand_outlook:float, renewable_surplus:float, volatility_tone:float, policy_in_effect:float}, horizon_days:int, confidence:float, summary:string}. "
        "tightness, demand_outlook, renewable_surplus are in [-1,1]. "
        "volatility_tone is in [0,1]. "
        "policy_in_effect is 0 or 1. "
        "horizon_days is an integer in [0,14]. "
        "summary must stay under 60 words and be factual."
    )

    user_prompt = (
        f"published_at: {published_at}\n"
        f"title: {title}\n"
        f"content: {body}\n"
        "Return one JSON object."
    )
    return system_prompt, user_prompt


def refine_one_news_doc(
    doc: dict[str, Any],
    *,
    schema_variant: str,
    api_adapter: OpenAINewsApiAdapter | None,
) -> dict[str, Any]:
    record = _default_record(doc, schema_variant=schema_variant)
    if api_adapter is None:
        return record

    system_prompt, user_prompt = _build_prompts(doc, schema_variant)
    try:
        raw_output = api_adapter.chat_json(system_prompt, user_prompt, max_tokens=260)
    except Exception:
        return record

    payload = _extract_json_object(raw_output)
    if not payload:
        return record

    topic_tags = _normalize_topic_tags(payload.get("topic_tags", []))
    is_actionable = _normalize_bool(payload.get("is_actionable", False))
    if "retrospective" in topic_tags or "routine" in topic_tags:
        is_actionable = False
    if not topic_tags:
        topic_tags = ["routine"] if not is_actionable else ["other"]

    horizon_days = int(round(_clamp(payload.get("horizon_days", 0), lo=0.0, hi=14.0, default=0.0)))
    confidence = _clamp(payload.get("confidence", 0.0), lo=0.0, hi=1.0, default=0.0)

    regime_payload = payload.get("regime_vec", {}) if isinstance(payload.get("regime_vec"), dict) else {}
    regime_vec = {
        "tightness": _clamp(regime_payload.get("tightness", 0.0), lo=-1.0, hi=1.0, default=0.0),
        "demand_outlook": _clamp(regime_payload.get("demand_outlook", 0.0), lo=-1.0, hi=1.0, default=0.0),
        "renewable_surplus": _clamp(regime_payload.get("renewable_surplus", 0.0), lo=-1.0, hi=1.0, default=0.0),
        "volatility_tone": _clamp(regime_payload.get("volatility_tone", 0.0), lo=0.0, hi=1.0, default=0.0),
        "policy_in_effect": float(round(_clamp(regime_payload.get("policy_in_effect", 0.0), lo=0.0, hi=1.0, default=0.0))),
    }

    if not is_actionable:
        horizon_days = 0
        confidence = 0.0
        regime_vec = {key: 0.0 for key in REGIME_KEYS}

    return record | {
        "is_actionable": bool(is_actionable),
        "topic_tags": topic_tags,
        "regime_vec": regime_vec,
        "horizon_days": int(horizon_days),
        "confidence": float(confidence),
        "summary": _compact_summary(payload.get("summary", record["summary"]), max_words=60),
    }


def refine_dataset_news_corpus(
    news_path: str,
    schema_variant: str,
    api_adapter: OpenAINewsApiAdapter | None,
    cache_path: str,
    *,
    max_items: int | None = None,
) -> None:
    with open(news_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON array in {news_path}, got {type(payload).__name__}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    limit = len(payload) if max_items is None or int(max_items) <= 0 else min(len(payload), int(max_items))
    with open(cache_path, "w", encoding="utf-8") as handle:
        for doc in payload[:limit]:
            refined = refine_one_news_doc(doc, schema_variant=schema_variant, api_adapter=api_adapter)
            handle.write(json.dumps(refined, ensure_ascii=False) + "\n")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine daily news corpus into regime-oriented JSONL")
    parser.add_argument("--news_path", type=str, required=True)
    parser.add_argument("--schema_variant", type=str, required=True, choices=["load", "price"])
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--news_api_model", type=str, default="gpt-5.1")
    parser.add_argument("--news_api_key_path", type=str, default=".secrets/api_key.txt")
    parser.add_argument("--news_api_base_url", type=str, default="")
    parser.add_argument("--news_api_timeout_sec", type=float, default=30.0)
    parser.add_argument("--news_api_max_retries", type=int, default=2)
    parser.add_argument("--max_items", type=int, default=0)
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    api_adapter = build_news_api_adapter(args)
    refine_dataset_news_corpus(
        news_path=args.news_path,
        schema_variant=args.schema_variant,
        api_adapter=api_adapter,
        cache_path=args.out,
        max_items=None if int(args.max_items) <= 0 else int(args.max_items),
    )


if __name__ == "__main__":
    main()
