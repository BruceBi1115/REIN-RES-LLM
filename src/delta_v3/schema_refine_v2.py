from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any

from tqdm import tqdm

from ..delta_news_hooks import OpenAINewsApiAdapter, build_news_api_adapter

SCHEMA_VERSION = "v4_regime_v2"
SCHEMA_VARIANTS = ("load", "price", "gas_demand", "traffic")
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


def _is_bitcoin_price_corpus(corpus_hint: str) -> bool:
    hint = os.path.basename(str(corpus_hint or "")).strip().lower()
    return any(token in hint for token in ("bitcoin", "btc", "crypto"))


def _build_prompts(doc: dict[str, Any], schema_variant: str, corpus_hint: str = "") -> tuple[str, str]:
    title = str(doc.get("title", "")).strip()
    published_at = str(doc.get("date", "")).strip()
    content = str(doc.get("content", "") or doc.get("summary", "") or "").strip()
    body = _compact_summary(content, max_words=220)
    if schema_variant == "price":
        if _is_bitcoin_price_corpus(corpus_hint):
            system_prompt = (
                "You are refining generic news for Bitcoin hourly price forecasting into JSON. "
                "Return JSON only, no markdown. "
                "Favor recall over precision for is_actionable, and be slightly loose on borderline cases. "
                "If an article has any plausible near-term pathway to move BTC spot level, flows, positioning, or volatility within the next 14 days, keep it actionable even if the pathway is indirect, sentiment-driven, or only moderate. "
                "Treat the following as potentially actionable if they are current, imminent, recently announced and still relevant, or likely to shape trader expectations within the next 14 days: "
                "ETF approvals or ETF flow updates, exchange or custody outages, regulatory or enforcement actions, stablecoin or crypto-liquidity stress, hacks or exploits, miner disruption or miner selling pressure, treasury or institutional accumulation, exchange reserve shifts, large liquidations, derivatives or funding stress, major macro rates or dollar shocks, and geopolitical or risk-asset shocks that plausibly spill into BTC. "
                "Broader macro and geopolitical articles can still be actionable for BTC if they plausibly affect crypto risk appetite, liquidity, safe-haven demand, or volatility. "
                "However, clearly unrelated human-interest, local civic, lifestyle, or sector-specific news with no plausible BTC or macro-risk transmission path should be non-actionable even if it is current. "
                "Set is_actionable=false only if the article is clearly unrelated to BTC and broad risk conditions, purely retrospective, generic commentary with no fresh signal, or stale corporate or project news without a near-term trading path. "
                "For borderline cases, prefer is_actionable=true with lower confidence rather than defaulting to false. "
                "Use only the closed topic vocabulary: "
                + ", ".join(TOPIC_TAGS)
                + ". "
                "If an article is clearly BTC-relevant but the legacy topic set is awkward, keep is_actionable=true and include other instead of routine. "
                "Reserve routine for articles with no clear BTC or macro trading signal. "
                "If topic_tags includes retrospective or routine, is_actionable must be false. "
                "Map BTC drivers into the existing tags and regime fields as follows: "
                "ETF inflows, treasury buying, exchange outflows, supply squeeze, or bullish positioning pressure -> supply_tight; "
                "ETF outflows, miner selling, large holder distribution, or exchange inflows -> supply_surplus; "
                "exchange outages, custody disruptions, settlement problems, or mining/network disruptions -> outage; "
                "stablecoin stress, funding stress, liquidation cascades, sharp rates or dollar shocks, or macro liquidity tightening -> fuel_shock or other; "
                "regulation, ETF approval, enforcement, or trading restrictions already in force -> policy_active or market_intervention and policy_in_effect=1; "
                "broad risk-on or liquidity-relief conditions that support BTC absorption and calm stress -> renewable_surge or other; "
                "risk-off, deleveraging, or liquidity drain that pressures BTC -> renewable_drought or other. "
                "Interpret regime_vec for BTC as: "
                "tightness = near-term BTC spot supply scarcity, squeeze pressure, or upward price pressure; "
                "demand_outlook = near-term investor demand, ETF or treasury flow, or adoption pressure; "
                "renewable_surplus = use this slot as broad liquidity and risk-relief tone that cushions BTC, negative when tightening or risk-off conditions pressure BTC; "
                "volatility_tone = event, leverage, liquidation, regulatory, macro, or shock-driven BTC volatility risk; "
                "policy_in_effect = 1 only when a policy, approval, restriction, or intervention is already active. "
                "For indirect but still relevant signals, use smaller-magnitude regime values instead of zeroing them out. "
                "Return fields exactly: "
                "{is_actionable:boolean, topic_tags:string[], regime_vec:{tightness:float, demand_outlook:float, renewable_surplus:float, volatility_tone:float, policy_in_effect:float}, horizon_days:int, confidence:float, summary:string}. "
                "tightness, demand_outlook, renewable_surplus are in [-1,1]. "
                "volatility_tone is in [0,1]. "
                "policy_in_effect is 0 or 1. "
                "horizon_days is an integer in [0,14]. "
                "summary must stay under 60 words and be factual."
            )
        else:
            system_prompt = (
                "You are refining generic energy news for NSW electricity price forecasting into JSON. "
                "Return JSON only, no markdown. "
                "Favor recall over precision for is_actionable. "
                "If an article has any plausible near-term pathway to move NSW wholesale price level or volatility, keep it actionable even if the impact is indirect, probabilistic, or only moderate. "
                "For price, keep near-term operational and market-microstructure drivers, not just broad macro supply-demand themes. "
                "Treat the following as potentially actionable if they are current, imminent, recently announced and still relevant, or likely to shape trader expectations within the next 14 days: "
                "generator outages, deratings, planned maintenance, forced outages, unit returns, rebidding or bidding behavior, interconnector flows or constraints, transmission limits, fuel supply or spot-price shocks, heatwaves and cooling-demand peaks, renewable drought or renewable surges, reserve scarcity, market notices, operational advisories, and market interventions. "
                "Do not filter out an article just because it is operational, plant-specific, scheduled rather than forced, framed as a market notice or advisory, or about another connected NEM region that can move NSW prices through interconnectors, shared weather, or fuel markets. "
                "Routine maintenance notices, forecast updates, renewable output outlooks, and interconnector advisories can still be actionable if they plausibly affect price or volatility. "
                "Set is_actionable=false only if the article is clearly retrospective, purely corporate or project news without a near-term operational cue, generic commentary with no concrete signal, or not plausibly relevant to NSW wholesale price within 14 days. "
                "For borderline cases, prefer is_actionable=true with lower confidence rather than defaulting to false. "
                "Use only the closed topic vocabulary: "
                + ", ".join(TOPIC_TAGS)
                + ". "
                "If an article is clearly price-relevant but no topic fits well, keep is_actionable=true and include other instead of routine. "
                "Reserve routine for articles with no clear operational or market signal. "
                "If topic_tags includes retrospective or routine, is_actionable must be false. "
                "Map price drivers into the existing tags and regime fields as follows: "
                "outage, derating, maintenance, unit trip, tight reserve margin -> outage or supply_tight; "
                "fuel shortage or fuel-cost shock -> fuel_shock; "
                "interstate flow changes, interconnector outages, binding transmission constraints -> interconnector_limit; "
                "heat-driven demand peaks -> heatwave; cold-driven demand peaks -> cold_snap; "
                "high wind, solar, hydro, or excess imports suppressing price -> renewable_surge or supply_surplus; "
                "low wind, low solar, drought, or import scarcity lifting price -> renewable_drought or supply_tight; "
                "rebidding, scarcity pricing, spike risk, or abrupt operational uncertainty should raise volatility_tone; "
                "administered pricing, market suspension, or intervention already in force -> market_intervention or policy_active and policy_in_effect=1. "
                "Interpret regime_vec for price as: "
                "tightness = near-term NSW supply-demand tightness and scarcity pressure on price; "
                "demand_outlook = near-term load and cooling-demand pressure; "
                "renewable_surplus = expected renewable abundance or import relief that suppresses price, negative when renewable scarcity lifts price; "
                "volatility_tone = spike, scarcity, rebidding, outage, or constraint-driven price volatility risk; "
                "policy_in_effect = 1 only when a policy or market intervention is already active. "
                "For indirect but still relevant signals, use smaller-magnitude regime values instead of zeroing them out. "
                "Return fields exactly: "
                "{is_actionable:boolean, topic_tags:string[], regime_vec:{tightness:float, demand_outlook:float, renewable_surplus:float, volatility_tone:float, policy_in_effect:float}, horizon_days:int, confidence:float, summary:string}. "
                "tightness, demand_outlook, renewable_surplus are in [-1,1]. "
                "volatility_tone is in [0,1]. "
                "policy_in_effect is 0 or 1. "
                "horizon_days is an integer in [0,14]. "
                "summary must stay under 60 words and be factual."
            )
    elif schema_variant == "gas_demand":
        system_prompt = (
            "You are refining generic energy news for Netherlands gas demand forecasting into strict JSON. "
            "Return JSON only, no markdown. "
            "Keep articles that describe a current or near-term condition that can change Dutch or Northwest European gas demand, "
            "gas supply tightness, storage stress, balancing conditions, or gas-for-power burn within the next 14 days. "
            "Treat the following as potentially actionable if they are current, imminent, or still in force: "
            "cold snaps, warm spells, heating demand shifts, storage depletion or refill stress, LNG terminal disruptions, "
            "pipeline outages or constraints, import or export bottlenecks, supply security alerts, balancing-market stress, "
            "fuel shortages, market intervention already in force, and power-system conditions that materially alter gas burn. "
            "Do not filter out an article just because it is framed as security of supply, storage, infrastructure maintenance, "
            "terminal availability, or system balancing, as long as it can plausibly affect gas demand or tightness within 14 days. "
            "Set is_actionable=false only if the article is retrospective, routine commentary, historical recap, corporate or board news, "
            "long-horizon hydrogen/CCS infrastructure news without near-term gas-market impact, or not plausibly relevant to Dutch gas demand "
            "or Dutch/NW European gas tightness within 14 days. "
            "Use only the closed topic vocabulary: "
            + ", ".join(TOPIC_TAGS)
            + ". "
            "If an article is clearly gas-demand relevant but no topic fits well, keep is_actionable=true and include other instead of routine. "
            "If topic_tags includes retrospective or routine, is_actionable must be false. "
            "Map gas-demand drivers into the existing tags and regime fields as follows: "
            "cold-driven heating demand or cold weather risk -> cold_snap; "
            "mild weather, weak heating demand, or demand relief -> supply_surplus or other; "
            "storage depletion, low inventories, import risk, or supply scarcity -> supply_tight; "
            "LNG, pipeline, terminal, compressor, or field disruptions -> outage, interconnector_limit, or fuel_shock depending on context; "
            "high wind/solar or other substitution that reduces gas burn -> renewable_surge or supply_surplus; "
            "low wind/solar or power-market stress that raises gas burn -> renewable_drought or supply_tight; "
            "government emergency actions, storage mandates, or market intervention already active -> policy_active or market_intervention and policy_in_effect=1. "
            "Interpret regime_vec for gas demand as: "
            "tightness = near-term Dutch/NW European gas market tightness and scarcity pressure; "
            "demand_outlook = near-term Dutch gas demand pressure, especially weather-sensitive heating demand and gas burn; "
            "renewable_surplus = conditions that reduce gas demand or gas-for-power burn, negative when weak renewables or substitution constraints increase gas usage; "
            "volatility_tone = uncertainty and volatility risk from storage stress, outages, balancing tension, or supply shocks; "
            "policy_in_effect = 1 only when a policy or intervention is already active. "
            "Return fields exactly: "
            "{is_actionable:boolean, topic_tags:string[], regime_vec:{tightness:float, demand_outlook:float, renewable_surplus:float, volatility_tone:float, policy_in_effect:float}, horizon_days:int, confidence:float, summary:string}. "
            "tightness, demand_outlook, renewable_surplus are in [-1,1]. "
            "volatility_tone is in [0,1]. "
            "policy_in_effect is 0 or 1. "
            "horizon_days is an integer in [0,14]. "
            "summary must stay under 60 words and be factual."
        )
    elif schema_variant == "traffic":
        system_prompt = (
            "You are refining generic news for hourly road traffic count forecasting into strict JSON. "
            "Return JSON only, no markdown. "
            "The goal is to keep articles that describe a current or near-term condition that can plausibly change "
            "road traffic volume, congestion, route availability, commuter behavior, freight flow, or hourly traffic "
            "volatility within the next 14 days. "
            "Treat the following as potentially actionable if they are current, imminent, recently announced and still relevant, or likely to shape travel behavior within 14 days: "
            "public holidays, school holidays, long weekends, major sports or entertainment events, festivals, protest activity, road closures, lane reductions, tunnel or bridge restrictions, major crashes with lingering disruption, rail strikes or transit outages that spill travelers onto roads, airport disruptions, port disruptions, severe weather, heatwaves, cold snaps, flooding, evacuations, fuel shortages, fuel-price shocks, major logistics or freight bottlenecks, congestion charging, and traffic-control interventions already in force. "
            "Do not filter out an article just because it is operational, local, event-specific, or framed as a traffic notice, transport advisory, or disruption update. "
            "Set is_actionable=false only if the article is clearly retrospective, generic commentary with no live signal, corporate or infrastructure planning news without near-term travel impact, or not plausibly relevant to road traffic counts within 14 days. "
            "Use only the closed topic vocabulary: "
            + ", ".join(TOPIC_TAGS)
            + ". "
            "If an article is clearly traffic-relevant but the topic vocabulary is awkward, keep is_actionable=true and include other instead of routine. "
            "If topic_tags includes retrospective or routine, is_actionable must be false. "
            "Map traffic drivers into the existing tags and regime fields as follows: "
            "public holidays, school holidays, long weekends, vacation travel, or event-driven travel surges -> holiday; "
            "road closures, crashes with ongoing disruption, rail or transit outages, strikes, or infrastructure failures -> outage; "
            "bridge, tunnel, motorway, border, corridor, or lane-capacity restrictions -> interconnector_limit; "
            "heat-driven travel changes, heatwave warnings, or fire-weather disruptions -> heatwave; "
            "snow, ice, or cold-weather commuting disruption -> cold_snap; "
            "fuel shortages, strong fuel-cost shocks, or freight cost shocks -> fuel_shock; "
            "traffic control, emergency restrictions, congestion charging, or government intervention already active -> policy_active or market_intervention and policy_in_effect=1; "
            "route reopening, transit relief, remote-work or modal-shift relief, or unusually weak travel demand -> supply_surplus or renewable_surge; "
            "network scarcity, strong commuter demand, freight bottlenecks, or event-driven congestion pressure -> supply_tight or renewable_drought. "
            "Interpret regime_vec for traffic as: "
            "tightness = near-term road-network tightness, congestion pressure, and capacity scarcity; "
            "demand_outlook = near-term traffic demand pressure from commuting, freight, holidays, and major events; "
            "renewable_surplus = use this slot as mobility relief and alternate-capacity availability that reduces road counts or congestion, negative when relief is weak and road usage is pushed higher; "
            "volatility_tone = disruption and uncertainty risk for hourly traffic from outages, closures, weather, protests, strikes, or major events; "
            "policy_in_effect = 1 only when a policy, restriction, emergency measure, or traffic intervention is already active. "
            "Return fields exactly: "
            "{is_actionable:boolean, topic_tags:string[], regime_vec:{tightness:float, demand_outlook:float, renewable_surplus:float, volatility_tone:float, policy_in_effect:float}, horizon_days:int, confidence:float, summary:string}. "
            "tightness, demand_outlook, renewable_surplus are in [-1,1]. "
            "volatility_tone is in [0,1]. "
            "policy_in_effect is 0 or 1. "
            "horizon_days is an integer in [0,14]. "
            "summary must stay under 60 words and be factual."
        )
    else:
        system_prompt = (
            "You are refining generic energy news for NSW electricity load forecasting into strict JSON. "
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
    corpus_hint: str = "",
    api_adapter: OpenAINewsApiAdapter | None,
) -> dict[str, Any]:
    record = _default_record(doc, schema_variant=schema_variant)
    if api_adapter is None:
        return record

    system_prompt, user_prompt = _build_prompts(doc, schema_variant, corpus_hint=corpus_hint)
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
    docs = payload[:limit]
    progress = tqdm(
        docs,
        total=limit,
        desc=f"[DELTA_V3][REFINE][{str(schema_variant).upper()}]",
        dynamic_ncols=True,
        leave=True,
        disable=limit <= 0,
    )
    actionable_count = 0
    if limit > 0:
        progress.set_postfix_str("actionable=0")
    with open(cache_path, "w", encoding="utf-8") as handle:
        for doc in progress:
            refined = refine_one_news_doc(
                doc,
                schema_variant=schema_variant,
                corpus_hint=news_path,
                api_adapter=api_adapter,
            )
            refined["source_news_file"] = os.path.basename(str(news_path or "").strip())
            refined["source_news_stem"] = os.path.splitext(os.path.basename(str(news_path or "").strip()))[0]
            if bool(refined.get("is_actionable", False)):
                actionable_count += 1
            progress.set_postfix_str(f"actionable={actionable_count}")
            handle.write(json.dumps(refined, ensure_ascii=False) + "\n")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine daily news corpus into regime-oriented JSONL")
    parser.add_argument("--news_path", type=str, required=True)
    parser.add_argument("--schema_variant", type=str, required=True, choices=list(SCHEMA_VARIANTS))
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
