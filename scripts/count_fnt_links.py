#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _iter_links(obj: Any):
    if isinstance(obj, dict):
        if "link" in obj:
            yield obj.get("link")
        for v in obj.values():
            yield from _iter_links(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_links(item)


def _load_data(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rows.append(json.loads(s))
        return rows
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_domain(link: str) -> str:
    s = str(link or "").strip()
    if not s:
        return ""
    parsed = urlparse(s)
    scheme = (parsed.scheme or "").lower()

    if scheme in {"http", "https"}:
        pass
    elif scheme == "":
        # Handle links without explicit scheme, e.g. "example.com/a" or "//example.com/a".
        if s.startswith("//"):
            parsed = urlparse("https:" + s)
        else:
            parsed = urlparse("https://" + s)
    else:
        # Ignore non-web links such as mailto:, javascript:, ftp:, etc.
        return ""
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    return host


def main():
    parser = argparse.ArgumentParser(
        description="Count unique link domain names in FNT dataset and print them."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/FNT_2019_2020_combined.json",
        help="Path to input JSON/JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output file path to save unique domains (one per line).",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = _load_data(path)
    links = []
    for v in _iter_links(data):
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        links.append(s)

    domains = []
    for link in links:
        d = _extract_domain(link)
        if d:
            domains.append(d)

    unique_domains = sorted(set(domains))

    print(f"input_file: {path}")
    print(f"link_count_total: {len(links)}")
    print(f"domain_count_total_from_links: {len(domains)}")
    print(f"domain_count_unique: {len(unique_domains)}")
    print("unique_domains:")
    for i, domain in enumerate(unique_domains, start=1):
        print(f"{i}. {domain}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for domain in unique_domains:
                f.write(domain + "\n")
        print(f"saved_unique_domains_to: {out_path}")


if __name__ == "__main__":
    main()
