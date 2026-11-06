#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.delta_v3.regime_bank import build_regime_bank


def _zero_like(payload, key: str) -> np.ndarray:
    return np.zeros_like(np.asarray(payload[key]))


def _metadata(payload, mode: str, source: str) -> np.ndarray:
    raw = payload["metadata_json"] if "metadata_json" in payload.files else None
    metadata = {}
    if raw is not None:
        try:
            metadata = json.loads(str(raw.item() if raw.shape == () else raw.reshape(-1)[0]))
        except Exception:
            metadata = {}
    metadata["ablation_mode"] = mode
    metadata["ablation_source_bank"] = os.path.basename(source)
    return np.asarray(json.dumps(metadata, ensure_ascii=True))


def _metadata_from_dict(metadata: dict, mode: str, source: str) -> np.ndarray:
    out = dict(metadata or {})
    out["ablation_mode"] = mode
    out["ablation_source_bank"] = os.path.basename(source)
    return np.asarray(json.dumps(out, ensure_ascii=True))


def _read_metadata(payload) -> dict:
    raw = payload["metadata_json"] if "metadata_json" in payload.files else None
    if raw is None:
        return {}
    try:
        return json.loads(str(raw.item() if raw.shape == () else raw.reshape(-1)[0]))
    except Exception:
        return {}


def build_feature_ablation_bank(source: str, output: str, mode: str) -> None:
    with np.load(source, allow_pickle=False) as payload:
        required = {"dates", "regime_vec", "topic_tag_mass", "text_emb", "relevance_mass", "in_force_doc_count"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise RuntimeError(f"{source} is missing required fields: {missing}")

        out = {key: np.asarray(payload[key]).copy() for key in payload.files}

        if mode == "no_regime_vec":
            out["regime_vec"] = _zero_like(payload, "regime_vec")
        elif mode == "no_topic_mass":
            out["topic_tag_mass"] = _zero_like(payload, "topic_tag_mass")
        elif mode == "no_text_emb":
            out["text_emb"] = _zero_like(payload, "text_emb")
        elif mode == "no_relevance_mass":
            out["relevance_mass"] = _zero_like(payload, "relevance_mass")
        elif mode == "no_in_force_doc_count":
            out["in_force_doc_count"] = _zero_like(payload, "in_force_doc_count")
        elif mode == "embedding_only":
            out["regime_vec"] = _zero_like(payload, "regime_vec")
            out["topic_tag_mass"] = _zero_like(payload, "topic_tag_mass")
        elif mode == "regime_vec_only":
            out["topic_tag_mass"] = _zero_like(payload, "topic_tag_mass")
            out["text_emb"] = _zero_like(payload, "text_emb")
        elif mode == "topic_only":
            out["regime_vec"] = _zero_like(payload, "regime_vec")
            out["text_emb"] = _zero_like(payload, "text_emb")
        elif mode == "blank_news_features":
            out["regime_vec"] = _zero_like(payload, "regime_vec")
            out["topic_tag_mass"] = _zero_like(payload, "topic_tag_mass")
            out["text_emb"] = _zero_like(payload, "text_emb")
            out["relevance_mass"] = _zero_like(payload, "relevance_mass")
            out["in_force_doc_count"] = _zero_like(payload, "in_force_doc_count")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        out["metadata_json"] = _metadata(payload, mode, source)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.savez_compressed(output, **out)
    print(output)


def build_no_confidence_weighting_bank(source: str, refined_jsonl: str, output: str) -> None:
    if not refined_jsonl:
        raise ValueError("--refined-jsonl is required for no_confidence_weighting.")
    with np.load(source, allow_pickle=False) as payload:
        metadata = _read_metadata(payload)

    required_meta = ["date_start", "date_end", "encoder_model_id", "max_length"]
    missing_meta = [key for key in required_meta if not str(metadata.get(key, "")).strip()]
    if missing_meta:
        raise RuntimeError(f"{source} metadata is missing fields needed to rebuild bank: {missing_meta}")

    build_regime_bank(
        refined_jsonl=refined_jsonl,
        out_path=output,
        encoder_model_id=str(metadata.get("encoder_model_id", "")),
        max_length=int(metadata.get("max_length", 256)),
        date_start=metadata.get("date_start", ""),
        date_end=metadata.get("date_end", ""),
        source_news_path=str(metadata.get("source_news_file", "")),
        schema_variant=str(metadata.get("schema_variant", "")),
        dataset_key=str(metadata.get("dataset_key", "")),
        tau_days=float(metadata.get("tau_days", 5.0)),
        ema_alpha=float(metadata.get("ema_alpha", 0.5)),
        ema_window=int(metadata.get("ema_window", 5)),
        freq_min=int(metadata.get("freq_min", 0) or 0),
        confidence_weighting=False,
    )

    with np.load(output, allow_pickle=False) as payload:
        out = {key: np.asarray(payload[key]).copy() for key in payload.files}
    with np.load(output, allow_pickle=False) as payload:
        output_metadata = _read_metadata(payload)
    out["metadata_json"] = _metadata_from_dict(output_metadata, "no_confidence_weighting", source)
    np.savez_compressed(output, **out)
    print(output)


def build_ablation_bank(source: str, output: str, mode: str, refined_jsonl: str = "") -> None:
    if mode == "no_confidence_weighting":
        build_no_confidence_weighting_bank(source, refined_jsonl, output)
        return
    build_feature_ablation_bank(source, output, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create modified regime banks for feature ablations.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--refined-jsonl",
        default="",
        help="Required for no_confidence_weighting because aggregation must be rebuilt from article-level rows.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "no_regime_vec",
            "no_topic_mass",
            "no_text_emb",
            "no_relevance_mass",
            "no_in_force_doc_count",
            "embedding_only",
            "regime_vec_only",
            "topic_only",
            "blank_news_features",
            "no_confidence_weighting",
        ],
    )
    args = parser.parse_args()
    build_ablation_bank(args.source, args.output, args.mode, refined_jsonl=args.refined_jsonl)


if __name__ == "__main__":
    main()
