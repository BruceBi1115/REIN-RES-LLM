#!/usr/bin/env python3
"""Convert Time-MMD mixed tables into REIN time-series and news inputs."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_ROOT = Path("dataset/Time-MMD")
DEFAULT_NUMERICAL_ROOT = Path("dataset/Time-MMD-numerical")
DEFAULT_TEXT_ROOT = Path("dataset/Time-MMD-text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Time-MMD tables into REIN numerical CSVs and paired-news JSON files."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--numerical-root", type=Path, default=DEFAULT_NUMERICAL_ROOT)
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Process only this domain. May be supplied more than once.",
    )
    parser.add_argument(
        "--text-column",
        default="Final_Search_6",
        help="Time-MMD text column to export (default: Final_Search_6).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    return parser.parse_args()


def read_rar_csv(path: Path) -> tuple[pd.DataFrame, str]:
    bsdtar = shutil.which("bsdtar")
    if not bsdtar:
        raise RuntimeError(
            f"Cannot read {path}: bsdtar is not installed or not available on PATH."
        )

    listing = subprocess.run(
        [bsdtar, "-tf", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    members = [name for name in listing if name.lower().endswith(".csv")]
    if len(members) != 1:
        raise ValueError(f"Expected one CSV in {path}, found {members}")

    payload = subprocess.run(
        [bsdtar, "-xOf", str(path), members[0]],
        check=True,
        capture_output=True,
    ).stdout
    return pd.read_csv(io.BytesIO(payload)), Path(members[0]).name


def read_source(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), path.name
    if path.suffix.lower() == ".rar":
        return read_rar_csv(path)
    raise ValueError(f"Unsupported Time-MMD input: {path}")


def parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce")
    return parsed


def output_stem(filename: str) -> str:
    return Path(filename).stem


def write_splits(frame: pd.DataFrame, path: Path, train_ratio: float, val_ratio: float) -> None:
    row_count = len(frame)
    train_end = int(row_count * train_ratio)
    val_end = train_end + int(row_count * val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= row_count:
        raise ValueError(
            f"Invalid chronological split for {path}: rows={row_count}, "
            f"train_ratio={train_ratio}, val_ratio={val_ratio}"
        )

    stem = path.stem
    suffix = path.suffix
    splits = {
        "trainset": frame.iloc[:train_end],
        "valset": frame.iloc[train_end:val_end],
        "testset": frame.iloc[val_end:],
    }
    for split_name, split_frame in splits.items():
        split_path = path.with_name(f"{stem}_{split_name}{suffix}")
        split_frame.to_csv(split_path, index=False)


def convert_file(
    source_path: Path,
    domain: str,
    numerical_root: Path,
    text_root: Path,
    text_column: str,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, object]:
    source, source_filename = read_source(source_path)
    if "date" not in source.columns:
        raise KeyError(f"{source_path} has no canonical 'date' column")
    if "OT" not in source.columns:
        raise KeyError(f"{source_path} has no forecasting target column 'OT'")
    if text_column not in source.columns:
        raise KeyError(f"{source_path} has no text column {text_column!r}")

    work = pd.DataFrame(
        {
            "_date": parse_dates(source["date"]),
            "OT": pd.to_numeric(source["OT"], errors="coerce"),
            "_text": source[text_column].fillna("").astype(str).str.strip(),
        }
    )
    invalid_count = int(work[["_date", "OT"]].isna().any(axis=1).sum())
    work = work.dropna(subset=["_date", "OT"])
    work = work.sort_values("_date", kind="mergesort")
    duplicate_count = int(work.duplicated(subset=["_date"], keep="last").sum())
    work = work.drop_duplicates(subset=["_date"], keep="last").reset_index(drop=True)
    if work.empty:
        raise ValueError(f"No valid date/OT rows remain in {source_path}")

    numerical_dir = numerical_root / domain
    text_dir = text_root / domain
    numerical_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    stem = output_stem(source_filename)
    numerical_path = numerical_dir / f"{stem}.csv"
    numerical = pd.DataFrame(
        {
            "date": work["_date"].dt.strftime("%Y-%m-%d"),
            "OT": work["OT"],
        }
    )
    numerical.to_csv(numerical_path, index=False)
    write_splits(numerical, numerical_path, train_ratio, val_ratio)

    news = []
    for timestamp, _, raw_text in work.itertuples(index=False, name=None):
        content = str(raw_text).strip()
        if not content or content.upper() == "NA":
            continue
        news.append(
            {
                "title": f"Time-MMD {domain} context",
                "date": timestamp.strftime("%d-%m-%Y 12:00:00 AM"),
                "content": content,
            }
        )

    text_path = text_dir / f"{stem}.json"
    with text_path.open("w", encoding="utf-8") as handle:
        json.dump(news, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return {
        "domain": domain,
        "source": str(source_path),
        "rows": len(numerical),
        "news": len(news),
        "dropped_invalid": invalid_count,
        "dropped_duplicate_dates": duplicate_count,
        "numerical": str(numerical_path),
        "text": str(text_path),
    }


def discover_sources(input_root: Path, selected_domains: set[str]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for domain_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        domain = domain_dir.name
        if selected_domains and domain not in selected_domains:
            continue
        for path in sorted(domain_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".csv", ".rar"}:
                sources.append((domain, path))
    return sources


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be less than 1")
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)

    selected_domains = set(args.domain)
    sources = discover_sources(args.input_root, selected_domains)
    found_domains = {domain for domain, _ in sources}
    missing_domains = selected_domains - found_domains
    if missing_domains:
        raise ValueError(f"No CSV or RAR source found for domains: {sorted(missing_domains)}")
    if not sources:
        raise ValueError(f"No Time-MMD CSV or RAR files found under {args.input_root}")

    for domain, path in sources:
        result = convert_file(
            path,
            domain,
            args.numerical_root,
            args.text_root,
            args.text_column,
            args.train_ratio,
            args.val_ratio,
        )
        print(
            "[Time-MMD] "
            f"domain={result['domain']} rows={result['rows']} news={result['news']} "
            f"dropped_invalid={result['dropped_invalid']} "
            f"dropped_duplicate_dates={result['dropped_duplicate_dates']}"
        )
        print(f"  numerical: {result['numerical']}")
        print(f"  text:      {result['text']}")


if __name__ == "__main__":
    main()
