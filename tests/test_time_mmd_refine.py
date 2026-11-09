"""Offline routing, cache migration, and effect-duration regression tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from run import build_parser
from src.time_mmd_prompts import TIME_MMD_PROFILES, TIME_MMD_SCHEMA_VERSION, TIME_MMD_VARIANTS
from src.delta_v3 import config, regime_bank, schema_refine_v2 as refine, trainer


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = {
    "title": "Time-MMD context",
    "date": "01-01-2024 12:00:00 AM",
    "content": "2023-12-25: A persistent wet circulation pattern is forecast for the contiguous US.",
}


def make_args(variant, news_path="context.json"):
    return build_parser().parse_args([
        "--train_file", "train.csv", "--val_file", "val.csv", "--test_file", "test.csv",
        "--dataset_key", variant, "--news_path", str(news_path),
        "--delta_v3_schema_variant", variant, "--delta_v3_refined_bank_build", "1",
        "--delta_v3_regime_ema_alpha", "1", "--time_col", "date",
    ])


def make_adapter(horizon_days=999, topic_tags=None):
    adapter = Mock()
    adapter.chat_json.return_value = json.dumps({
        "is_actionable": True,
        "topic_tags": ["other"] if topic_tags is None else topic_tags,
        "regime_vec": {key: 0.5 for key in refine.REGIME_KEYS},
        "horizon_days": horizon_days,
        "confidence": 0.8,
        "summary": "A contemporaneous external condition affects the target.",
    })
    return adapter


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TimeMMDRefineTests(unittest.TestCase):
    def test_environment_reuses_processed_files_without_running_converter(self):
        shell = r'''
source() {
    builtin source "$@"
    if [[ "$1" == */_run_forecast_common.sh ]]; then
        run_forecast_dataset_main() {
            PYTHON_BIN=/bin/false
            "$PRE_RUN_HOOK"
        }
    fi
}
source "$1"
'''
        script = REPO_ROOT / "scripts" / "Time-MMD-related" / "run_environment.sh"
        result = subprocess.run(
            ["bash", "-c", shell, "test", str(script)], cwd=REPO_ROOT,
            env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]},
            capture_output=True, text=True, check=True,
        )
        self.assertIn("processed files already exist; skipping source conversion", result.stdout)

    def test_all_shell_entrypoints_route_through_run_parser_to_domain_prompt(self):
        # Run the actual config scripts and argument builder without preparing data or training.
        shell = r'''
source() {
    builtin source "$@"
    if [[ "$1" == */_run_forecast_common.sh ]]; then
        run_forecast_dataset_main() {
            init_common_defaults
            validate_dataset_config
            build_run_args "$TASK_NAME_BASE" "${BASE_BACKBONES[0]}" "${HORIZONS[0]}" \
                "${LRS[0]}" "${SCHEDULERS[0]}" "${GRAD_ACCS[0]}" "$DEFAULT_NEWS_PATH"
            printf '%s\0' "${RUN_ARGS[@]}"
        }
    fi
}
source "$1"
'''
        script_dir = REPO_ROOT / "scripts" / "Time-MMD-related"
        scripts = sorted(script_dir.glob("run_*.sh"))
        self.assertEqual(len(scripts), 9)
        seen = set()
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-c", shell, "test", str(script)], cwd=REPO_ROOT,
                    env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]},
                    capture_output=True, check=True,
                )
                argv = result.stdout.decode().rstrip("\0").split("\0")
                args = build_parser().parse_args(argv)
                cfg = config.DeltaV3Config.from_args(args)
                self.assertEqual(cfg.schema_variant, cfg.dataset_key)
                seen.add(cfg.schema_variant)
                profile = TIME_MMD_PROFILES[cfg.schema_variant]
                cli = refine._build_cli_parser().parse_args([
                    "--news_path", args.news_path, "--schema_variant", cfg.schema_variant,
                    "--out", "unused.jsonl",
                ])
                self.assertEqual(cli.schema_variant, cfg.schema_variant)
                adapter = make_adapter()
                record = refine.refine_one_news_doc(DOC, schema_variant=cfg.schema_variant, api_adapter=adapter)
                system, user = adapter.chat_json.call_args.args
                self.assertIn(profile.target, system)
                self.assertIn(DOC["content"], user)
                self.assertNotIn("NSW", system)
                self.assertEqual(record["horizon_days"], profile.horizon_days)
                self.assertEqual(record["schema_version"], TIME_MMD_SCHEMA_VERSION)
        self.assertEqual(seen, set(TIME_MMD_VARIANTS))

    def test_mismatched_dataset_prompt_is_rejected(self):
        args = make_args("time_mmd_climate")
        for wrong in ("bitcoin", "time_mmd_environment"):
            with self.subTest(wrong=wrong):
                args.delta_v3_schema_variant = wrong
                with self.assertRaisesRegex(ValueError, "requires --delta_v3_schema_variant"):
                    config.DeltaV3Config.from_args(args)
        with self.assertRaisesRegex(ValueError, "Unknown refinement schema"):
            refine._build_prompts(DOC, "misspelled_domain")

    def test_legacy_domains_keep_original_prompts_and_limits(self):
        expected = {
            "load": "NSW electricity load", "price": "NSW electricity price",
            "gas_demand": "Netherlands gas demand", "traffic": "hourly road traffic count",
            "bitcoin": "Bitcoin hourly price",
        }
        for variant, target in expected.items():
            with self.subTest(variant=variant):
                system, _ = refine._build_prompts(DOC, variant)
                self.assertIn(target, system)
                record = refine.refine_one_news_doc(DOC, schema_variant=variant, api_adapter=make_adapter())
                self.assertEqual(record["horizon_days"], 14)
                cfg = config.DeltaV3Config.from_args(make_args(variant))
                self.assertEqual(cfg.cache_dir, config.DEFAULT_SHARED_REFINE_CACHE_DIR)

    def test_bundle_context_is_not_cut_to_a_single_article(self):
        doc = dict(DOC, content="context " * 300 + "2023-12-31: LAST_FACT")
        system, user = refine._build_prompts(doc, "time_mmd_climate")
        self.assertIn("LAST_FACT", user)
        self.assertIn("source/retrieval date is later than the as-of date", system)
        self.assertIn("NA placeholders", system)
        _, legacy_user = refine._build_prompts(doc, "load")
        self.assertNotIn("LAST_FACT", legacy_user)

    def test_non_actionable_bundle_is_zeroed(self):
        record = refine.refine_one_news_doc(
            DOC, schema_variant="time_mmd_climate", api_adapter=make_adapter(topic_tags=["retrospective"]),
        )
        self.assertFalse(record["is_actionable"])
        self.assertEqual(record["horizon_days"], 0)
        self.assertEqual(record["confidence"], 0)
        self.assertTrue(all(value == 0 for value in record["regime_vec"].values()))

    def test_cache_paths_are_isolated_by_domain_and_prompt_version(self):
        paths = set()
        for variant in TIME_MMD_VARIANTS:
            cfg = config.DeltaV3Config.from_args(make_args(variant))
            self.assertIn(variant, Path(cfg.cache_dir).parts)
            self.assertIn(TIME_MMD_SCHEMA_VERSION, Path(cfg.cache_dir).parts)
            self.assertEqual(Path(cfg.regime_bank_path).parent, Path(cfg.cache_dir))
            paths.add(cfg.regime_bank_path)
        self.assertEqual(len(paths), 9)

    def test_legacy_or_unversioned_refined_rows_are_not_reused(self):
        variant = "time_mmd_climate"
        base = refine._default_record(DOC, variant)
        invalid = [
            dict(base, schema_variant="bitcoin"),
            dict(base, schema_version="old_version"),
            {k: v for k, v in base.items() if k != "schema_version"},
            {k: v for k, v in base.items() if k != "schema_variant"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refined.jsonl"
            for row in invalid:
                with self.subTest(row=row):
                    write_jsonl(path, [row])
                    reused = refine._load_existing_refined_rows(
                        [str(path)], schema_variant=variant,
                        source_news_file="context.json", source_news_stem="context",
                    )
                    self.assertEqual(reused, {})
            write_jsonl(path, [base])
            self.assertEqual(len(refine._load_existing_refined_rows(
                [str(path)], schema_variant=variant, source_news_file="context.json", source_news_stem="context",
            )), 1)

    def test_corpus_validation_rejects_wrong_or_missing_versions(self):
        variant = "time_mmd_climate"
        with tempfile.TemporaryDirectory() as tmp:
            news = Path(tmp) / "context.json"
            news.write_text(json.dumps([DOC]), encoding="utf-8")
            refined = Path(tmp) / "refined.jsonl"
            refine.refine_dataset_news_corpus(str(news), variant, make_adapter(), str(refined))
            cfg = config.DeltaV3Config.from_args(make_args(variant, news))
            signature = refine.compute_news_corpus_signature(str(news))
            self.assertTrue(trainer._refined_corpus_matches(str(refined), cfg, news_signature=signature)[0])
            row = json.loads(refined.read_text())
            for invalid in (dict(row, schema_version="old"), dict(row, schema_variant="bitcoin")):
                write_jsonl(refined, [invalid])
                self.assertFalse(trainer._refined_corpus_matches(str(refined), cfg, news_signature=signature)[0])
            unversioned = {k: v for k, v in row.items() if k != "schema_version"}
            write_jsonl(refined, [row, unversioned])
            self.assertFalse(trainer._refined_corpus_matches(str(refined), cfg, news_signature=signature)[0])

    def test_preparation_rebuilds_legacy_caches_then_reuses_new_bank(self):
        variant = "time_mmd_climate"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            news = root / "context.json"
            news.write_text(json.dumps([DOC]), encoding="utf-8")
            cache_root = root / "cache"
            args = make_args(variant, news)
            with patch.object(config, "DEFAULT_SHARED_REFINE_CACHE_DIR", str(cache_root)):
                cfg = config.DeltaV3Config.from_args(args)
            # This was the pre-Time-MMD cache location; preserve its bytes during migration.
            old_refined = Path(config.build_refined_cache_path(str(cache_root), str(news)))
            write_jsonl(old_refined, [refine._default_record(DOC, "bitcoin")])
            old_bytes = old_refined.read_bytes()
            bundle = {
                "live_logger": Mock(),
                "train_df": pd.DataFrame({"date": ["2024-01-01"]}),
                "val_df": pd.DataFrame({"date": ["2024-02-01"]}),
                "test_df": pd.DataFrame({"date": ["2024-03-10"]}),
            }
            adapter = make_adapter()
            with patch.object(trainer, "build_news_api_adapter", return_value=adapter), patch.object(
                regime_bank, "_encode_texts", return_value=np.ones((1, 384), dtype=np.float32),
            ):
                bank, refined_path, bank_path = trainer._prepare_regime_bank(args, bundle, cfg)
            adapter.chat_json.assert_called_once()
            self.assertIn("monthly precipitation", adapter.chat_json.call_args.args[0])
            self.assertEqual(old_refined.read_bytes(), old_bytes)
            self.assertEqual(Path(refined_path).parent, Path(cfg.cache_dir))
            self.assertEqual(bank.metadata["schema_version"], TIME_MMD_SCHEMA_VERSION)
            expected = trainer._expected_regime_bank_metadata(
                args, bundle, cfg, news_signature=refine.compute_news_corpus_signature(str(news)),
            )
            self.assertTrue(trainer._regime_bank_metadata_matches(bank_path, expected)[0])
            # A monthly signal must survive the former 14-day clamp, then expire at its limit.
            self.assertEqual(int(bank.in_force_doc_count[31]), 1)
            self.assertEqual(int(bank.in_force_doc_count[62]), 1)
            self.assertEqual(int(bank.in_force_doc_count[63]), 0)
            cfg.refined_bank_build = False
            with patch.object(trainer, "build_news_api_adapter", side_effect=AssertionError("unexpected API call")), patch.object(
                regime_bank, "_encode_texts", side_effect=AssertionError("unexpected encoding"),
            ):
                reused, _, _ = trainer._prepare_regime_bank(args, bundle, cfg)
            np.testing.assert_array_equal(reused.relevance_mass, bank.relevance_mass)

    def test_bank_cache_rejects_legacy_and_missing_metadata(self):
        expected = {"schema_variant": "time_mmd_climate", "schema_version": TIME_MMD_SCHEMA_VERSION}
        for actual in ({}, {"schema_variant": "bitcoin"}, dict(expected, schema_version="old")):
            with self.subTest(actual=actual), patch.object(trainer, "read_regime_bank_metadata", return_value=actual):
                self.assertFalse(trainer._regime_bank_metadata_matches("unused.npz", expected)[0])


if __name__ == "__main__":
    unittest.main()
