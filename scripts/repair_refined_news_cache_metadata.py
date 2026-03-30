#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).casefold()


def parse_dt(raw: str) -> datetime | None:
    s = str(raw or "").strip()
    if not s:
        return None
    iso_candidate = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_candidate)
        return dt.replace(tzinfo=None)
    except Exception:
        pass

    fmts = [
        "%d-%m-%Y %I:%M:%S %p",
        "%d/%m/%Y %I:%M:%S %p",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def normalize_dt(raw: str) -> str:
    dt = parse_dt(raw)
    return dt.isoformat() if dt is not None else ""


def load_json_array(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise TypeError(f"{path} is not a JSON array.")
    return [x for x in obj if isinstance(x, dict)]


def build_source_index(news_items: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    by_title = {}
    by_text = {}
    for idx, rec in enumerate(news_items):
        title = str(rec.get("title", "") or "").strip()
        date = str(rec.get("date", "") or "").strip()
        content = str(rec.get("content", "") or "").strip()
        if not title:
            raise ValueError(f"Source news row {idx} is missing title.")
        if not normalize_dt(date):
            raise ValueError(f"Source news row {idx} has invalid date: {date!r}")
        title_key = normalize_title(title)
        if title_key in by_title:
            raise ValueError(f"Duplicate source title detected: {title!r}")
        by_title[title_key] = rec
        if content:
            if content in by_text and by_text[content] != rec:
                raise ValueError(f"Duplicate source content maps to multiple titles: {title!r}")
            by_text[content] = rec
    return by_title, by_text


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
    by_title, by_text = build_source_index(news_items)

    repaired = []
    updated = 0
    for idx, rec in enumerate(cache_items):
        cur = dict(rec)
        title = str(cur.get("title", "") or "").strip()
        source = by_title.get(normalize_title(title))
        if source is None:
            raw_text = str(cur.get("raw_news_text", "") or "").strip()
            source = by_text.get(raw_text)
        if source is None:
            raise KeyError(f"Could not map cache row {idx} back to source news: title={title!r}")

        new_title = str(source.get("title", "") or "").strip()
        new_date = normalize_dt(str(source.get("date", "") or "").strip())
        new_url = str(source.get("url", "") or "").strip()
        if not new_title or not new_date:
            raise ValueError(f"Mapped source news is missing title/date for cache row {idx}.")

        changed = (
            str(cur.get("title", "") or "").strip() != new_title
            or normalize_dt(str(cur.get("date", "") or "").strip()) != new_date
            or str(cur.get("url", "") or "").strip() != new_url
        )
        cur["title"] = new_title
        cur["date"] = new_date
        cur["url"] = new_url
        repaired.append(cur)
        updated += int(changed)

    repaired.sort(
        key=lambda rec: (
            normalize_dt(str(rec.get("date", "") or "").strip()),
            normalize_title(rec.get("title", "")),
            str(rec.get("url", "") or "").strip(),
        )
    )

    print(f"cache_rows={len(cache_items)}")
    print(f"metadata_updates={updated}")
    print(f"output_sorted=1")

    if not args.check_only:
        cache_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
