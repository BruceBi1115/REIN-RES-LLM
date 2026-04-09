#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.news_datetime import normalize_news_datetime, parse_news_datetime


def normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def parse_dt(raw: str):
    dt = parse_news_datetime(raw, dayfirst=True)
    return dt


def normalize_dt(raw: str) -> str:
    return normalize_news_datetime(raw, dayfirst=True, floor="s")


def normalize_url(raw: str) -> str:
    return re.sub(r"\s+", "", str(raw or "").strip())


def identity(
    rec: dict,
    *,
    title_key: str = "title",
    date_key: str = "date",
    url_key: str = "url",
) -> tuple[str, str, str]:
    return (
        normalize_title(str(rec.get(title_key, "") or "").strip()),
        normalize_dt(str(rec.get(date_key, "") or "").strip()),
        normalize_url(str(rec.get(url_key, "") or "").strip()),
    )


def load_json_array(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError(f"{path} is not a JSON array.")
    return [x for x in obj if isinstance(x, dict)]


def save_json_array(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_identities(
    items: list[dict],
    *,
    label: str,
    title_key: str = "title",
    date_key: str = "date",
    url_key: str = "url",
) -> tuple[set[tuple[str, str, str]], int]:
    seen: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    for idx, rec in enumerate(items):
        ident = identity(rec, title_key=title_key, date_key=date_key, url_key=url_key)
        if not ident[0]:
            raise ValueError(f"{label} row {idx} is missing title.")
        if not ident[1]:
            raise ValueError(f"{label} row {idx} has invalid date: {rec.get(date_key)!r}")
        if ident in seen:
            duplicate_count += 1
            continue
        seen.add(ident)
    return seen, duplicate_count


def is_chronological(cache_items: list[dict]) -> tuple[bool, tuple[int, str, str] | None]:
    prev = None
    prev_idx = -1
    for idx, rec in enumerate(cache_items):
        cur = parse_dt(str(rec.get("date", "") or "").strip())
        if cur is None:
            continue
        if prev is not None and cur < prev:
            return False, (idx, cache_items[idx - 1].get("date", ""), rec.get("date", ""))
        prev = cur
        prev_idx = idx
    return True, None


def sort_cache_items(cache_items: list[dict]) -> list[dict]:
    indexed = list(enumerate(cache_items))
    indexed.sort(
        key=lambda item: (
            normalize_dt(str(item[1].get("date", "") or "").strip()),
            normalize_title(item[1].get("title", "")),
            normalize_url(item[1].get("url", "")),
            item[0],
        )
    )
    return [dict(rec) for _, rec in indexed]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify refined news cache identity coverage and ordering.")
    parser.add_argument("--news", required=True, help="path to the source news JSON array")
    parser.add_argument("--cache", required=True, help="path to the unified refined news cache JSON array")
    parser.add_argument("--expect-count", type=int, default=-1, help="optional expected row count for both files")
    parser.add_argument("--allow-missing", action="store_true", help="do not fail if the cache misses some source news identities")
    parser.add_argument(
        "--no-auto-sort",
        action="store_true",
        help="fail instead of rewriting the cache file when records are not sorted from early to late",
    )
    args = parser.parse_args()

    news_path = Path(args.news)
    cache_path = Path(args.cache)
    news_items = load_json_array(news_path)
    cache_items = load_json_array(cache_path)

    if args.expect_count > 0 and len(news_items) != args.expect_count:
        raise AssertionError(f"Unexpected news row count: {len(news_items)} != {args.expect_count}")

    news_identities, news_duplicate_count = collect_identities(news_items, label="source_news")
    chronological_before, bad_pair_before = is_chronological(cache_items)
    cache_auto_sorted = 0
    if (not chronological_before) and (not args.no_auto_sort):
        cache_items = sort_cache_items(cache_items)
        save_json_array(cache_path, cache_items)
        cache_auto_sorted = 1

    cache_identities, cache_duplicate_count = collect_identities(cache_items, label="cache_records")
    missing = [key for key in news_identities if key not in cache_identities]
    chronological, bad_pair = is_chronological(cache_items)
    print(f"news_rows={len(news_items)}")
    print(f"cache_rows={len(cache_items)}")
    print(f"identity_matches={len(news_identities) - len(missing)}")
    print(f"missing_identities={len(missing)}")
    print(f"source_duplicate_identities={news_duplicate_count}")
    print(f"cache_duplicate_identities={cache_duplicate_count}")
    print(f"cache_auto_sorted={cache_auto_sorted}")
    print(f"cache_chronological={int(chronological)}")
    if bad_pair_before is not None:
        idx, prev_date, cur_date = bad_pair_before
        print(f"first_descending_pair_index_before={idx} prev={prev_date!r} cur={cur_date!r}")
    if missing:
        title_key, date_key, url_key = missing[0]
        print(
            "first_missing_identity="
            f"title={title_key!r} date={date_key!r} url={url_key!r}"
        )
    if bad_pair is not None:
        idx, prev_date, cur_date = bad_pair
        print(f"first_descending_pair_index={idx} prev={prev_date!r} cur={cur_date!r}")

    if missing and (not args.allow_missing):
        title_key, date_key, url_key = missing[0]
        raise AssertionError(
            "Cache does not cover every source news identity. "
            f"First missing normalized identity: title={title_key!r} date={date_key!r} url={url_key!r}"
        )
    if not chronological:
        raise AssertionError("Cache is not sorted by date from early to late.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
