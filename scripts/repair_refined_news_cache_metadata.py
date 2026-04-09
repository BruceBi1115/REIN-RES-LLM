#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.news_datetime import normalize_news_datetime


def normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def normalize_url(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def normalize_dt(raw: str) -> str:
    return normalize_news_datetime(raw, dayfirst=True, floor="s")


def load_json_array(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError(f"{path} is not a JSON array.")
    return [x for x in obj if isinstance(x, dict)]


def build_source_index(news_items: list[dict]) -> tuple[dict[tuple[str, str], dict], dict[str, dict | None], dict[str, dict | None]]:
    by_title_url = {}
    by_text = {}
    by_title = {}
    for idx, rec in enumerate(news_items):
        title = str(rec.get("title", "") or "").strip()
        date = str(rec.get("date", "") or "").strip()
        url = str(rec.get("url", "") or "").strip()
        content = str(rec.get("content", "") or "").strip()
        if not title:
            raise ValueError(f"Source news row {idx} is missing title.")
        if not normalize_dt(date):
            raise ValueError(f"Source news row {idx} has invalid date: {date!r}")
        title_key = normalize_title(title)
        url_key = normalize_url(url)
        if url_key:
            title_url_key = (title_key, url_key)
            existing = by_title_url.get(title_url_key)
            if existing is not None and existing != rec:
                raise ValueError(f"Duplicate source title+url detected: title={title!r} url={url!r}")
            by_title_url[title_url_key] = rec
        existing_title = by_title.get(title_key)
        if existing_title is None and title_key not in by_title:
            by_title[title_key] = rec
        elif existing_title != rec:
            by_title[title_key] = None
        if content:
            existing_text = by_text.get(content)
            if existing_text is None and content not in by_text:
                by_text[content] = rec
            elif existing_text != rec:
                by_text[content] = None
    return by_title_url, by_text, by_title


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair cache metadata fields using the source news file.")
    parser.add_argument("--news", required=True, help="path to source news JSON")
    parser.add_argument("--cache", required=True, help="path to unified cache JSON")
    parser.add_argument("--check-only", action="store_true", help="do not rewrite; only report whether changes are needed")
    args = parser.parse_args()

    news_path = Path(args.news)
    cache_path = Path(args.cache)
    news_items = load_json_array(news_path)
    cache_items = load_json_array(cache_path)
    by_title_url, by_text, by_title = build_source_index(news_items)

    repaired = []
    updated = 0
    date_updates = 0
    for idx, rec in enumerate(cache_items):
        cur = dict(rec)
        title = str(cur.get("title", "") or "").strip()
        url = str(cur.get("url", "") or "").strip()
        source = by_title_url.get((normalize_title(title), normalize_url(url)))
        if source is None:
            raw_text = str(cur.get("raw_news_text", "") or "").strip()
            source = by_text.get(raw_text)
        if source is None:
            source = by_title.get(normalize_title(title))
        if source is None:
            raise KeyError(
                f"Could not map cache row {idx} back to source news: "
                f"title={title!r} url={url!r}"
            )

        new_title = str(source.get("title", "") or "").strip()
        new_date = normalize_dt(str(source.get("date", "") or "").strip())
        new_url = str(source.get("url", "") or "").strip()
        if not new_title or not new_date:
            raise ValueError(f"Mapped source news is missing title/date for cache row {idx}.")

        old_date = normalize_dt(str(cur.get("date", "") or "").strip())
        changed = (
            str(cur.get("title", "") or "").strip() != new_title
            or old_date != new_date
            or str(cur.get("url", "") or "").strip() != new_url
        )
        cur["title"] = new_title
        cur["date"] = new_date
        cur["url"] = new_url
        repaired.append(cur)
        updated += int(changed)
        date_updates += int(old_date != new_date)

    repaired.sort(
        key=lambda rec: (
            normalize_dt(str(rec.get("date", "") or "").strip()),
            normalize_title(rec.get("title", "")),
            str(rec.get("url", "") or "").strip(),
        )
    )

    print(f"cache_rows={len(cache_items)}")
    print(f"metadata_updates={updated}")
    print(f"date_updates={date_updates}")
    print(f"output_sorted=1")

    if not args.check_only:
        cache_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
