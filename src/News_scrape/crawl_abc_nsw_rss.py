"""RSS-only news collector for ABC News (Australia), NSW-focused.

Two modes:
  live     — fetch the current RSS feeds once (typically ~30-80 most recent items
             per feed). Use this for forward-looking runs.
  archive  — backfill historical coverage by replaying Wayback Machine snapshots
             of the same RSS URLs across a date range. Needed for 2024 traffic
             backfill since live RSS only exposes recent items.

Output JSON matches the existing dataset shape used by this repo:

    [
      {"title": "...", "date": "DD-MM-YYYY HH:MM:SS AM", "content": "..."},
      ...
    ]

Design notes:
  - RSS `<description>` and `<content:encoded>` give us headline + summary; we do
    NOT follow the article link, so this stays inside ABC's allowed use of RSS.
  - Feeds are configurable via --feed (repeatable). Defaults target NSW +
    top-stories; swap/augment from https://www.abc.net.au/news/feeds/rss.
  - Dedupe is by canonicalized `<link>` URL.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from xml.etree import ElementTree as ET

import requests


DEFAULT_FEEDS: list[str] = [
    # NSW state feed
    "https://www.abc.net.au/news/feed/52498/rss.xml",
    # Top stories (nationwide — filter later if needed)
    "https://www.abc.net.au/news/feed/45910/rss.xml",
]

DEFAULT_OUTPUT = Path("dataset/abc_nsw_rss.json")
DEFAULT_TIMEOUT = 20.0
DEFAULT_DELAY = 0.7
USER_AGENT = (
    "Mozilla/5.0 (compatible; REIN-RES-LLM-Research/1.0; academic use; "
    "contact=bizhaoge@gmail.com)"
)

RSS_NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_PREFIX = "https://web.archive.org/web"


def clean_text(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def canonicalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def format_date(dt: datetime) -> str:
    # Matches the existing dataset format: "01-01-2024 03:00:00 AM"
    return dt.strftime("%d-%m-%Y %I:%M:%S %p")


def parse_pubdate(raw: str) -> datetime | None:
    text = (raw or "").strip()
    if not text:
        return None
    # RFC 822 (primary RSS format): "Mon, 01 Jan 2024 03:00:00 +1100"
    formats = (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


@dataclass
class Item:
    title: str
    date: str
    content: str
    link: str
    source_feed: str


def parse_feed_xml(xml_bytes: bytes, source_feed: str) -> list[Item]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    items: list[Item] = []
    # RSS 2.0 path: /rss/channel/item
    for node in root.iter("item"):
        title = clean_text((node.findtext("title") or ""))
        link = canonicalize_url(node.findtext("link") or "")
        pubdate_raw = node.findtext("pubDate") or node.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
        dt = parse_pubdate(pubdate_raw)
        if dt is None:
            continue

        # Prefer content:encoded (fuller), fall back to description
        content_encoded = node.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or ""
        description = node.findtext("description") or ""
        body = clean_text(content_encoded) or clean_text(description)

        if not title and not body:
            continue

        items.append(
            Item(
                title=title,
                date=format_date(dt),
                content=(title + "\n" + body).strip() if body else title,
                link=link,
                source_feed=source_feed,
            )
        )
    return items


def fetch_live_feed(session: requests.Session, url: str, timeout: float) -> list[Item]:
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return parse_feed_xml(resp.content, source_feed=url)


def list_wayback_snapshots(
    session: requests.Session,
    target_url: str,
    start: datetime,
    end: datetime,
    cadence_days: int,
    timeout: float,
) -> list[tuple[str, str]]:
    """Return list of (timestamp, original_url) tuples for one snapshot per
    `cadence_days` interval, using the CDX server."""
    params = {
        "url": target_url,
        "from": start.strftime("%Y%m%d"),
        "to": end.strftime("%Y%m%d"),
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original",
        "collapse": f"timestamp:{max(8, 8)}",  # collapse to day-level
    }
    resp = session.get(
        WAYBACK_CDX,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    try:
        rows = resp.json()
    except json.JSONDecodeError:
        return []
    if not rows or len(rows) < 2:
        return []
    # first row is header
    snaps = [(row[0], row[1]) for row in rows[1:] if len(row) >= 2]

    # Downsample to at most one snapshot per cadence_days.
    if cadence_days <= 1:
        return snaps
    kept: list[tuple[str, str]] = []
    last_ts: datetime | None = None
    for ts_str, orig in snaps:
        try:
            ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if last_ts is None or (ts - last_ts) >= timedelta(days=cadence_days):
            kept.append((ts_str, orig))
            last_ts = ts
    return kept


def fetch_wayback_feed(
    session: requests.Session,
    timestamp: str,
    original: str,
    timeout: float,
) -> list[Item]:
    # `id_` flag on Wayback returns the raw captured bytes (no rewriting).
    snapshot_url = f"{WAYBACK_PREFIX}/{timestamp}id_/{original}"
    resp = session.get(snapshot_url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return parse_feed_xml(resp.content, source_feed=original)


def collect_live(feeds: list[str], timeout: float, delay: float) -> list[Item]:
    session = requests.Session()
    collected: list[Item] = []
    for url in feeds:
        try:
            items = fetch_live_feed(session, url, timeout=timeout)
            print(f"[live] {url}  -> {len(items)} items")
            collected.extend(items)
        except Exception as exc:
            print(f"[live] FAILED {url}: {exc}")
        time.sleep(delay)
    return collected


def collect_archive(
    feeds: list[str],
    start: datetime,
    end: datetime,
    cadence_days: int,
    timeout: float,
    delay: float,
) -> list[Item]:
    session = requests.Session()
    collected: list[Item] = []
    for url in feeds:
        try:
            snaps = list_wayback_snapshots(
                session,
                target_url=url,
                start=start,
                end=end,
                cadence_days=cadence_days,
                timeout=timeout,
            )
        except Exception as exc:
            print(f"[archive] CDX failed for {url}: {exc}")
            continue
        print(f"[archive] {url}  -> {len(snaps)} snapshots selected")
        for ts_str, orig in snaps:
            try:
                items = fetch_wayback_feed(session, ts_str, orig, timeout=timeout)
            except Exception as exc:
                print(f"[archive] snapshot {ts_str} failed: {exc}")
                time.sleep(delay)
                continue
            collected.extend(items)
            time.sleep(delay)
    return collected


def dedupe_items(items: list[Item]) -> list[Item]:
    seen: set[str] = set()
    unique: list[Item] = []
    for it in items:
        key = it.link or (it.title + "|" + it.date)
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)
    # Chronological ascending (matches existing traffic_2024.json style)
    def _sort_key(x: Item) -> datetime:
        try:
            return datetime.strptime(x.date, "%d-%m-%Y %I:%M:%S %p")
        except ValueError:
            return datetime.min
    unique.sort(key=_sort_key)
    return unique


def write_output(items: list[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"title": it.title, "date": it.date, "content": it.content} for it in items]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def parse_date(value: str) -> datetime:
    dt = datetime.strptime(value, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="RSS-only ABC News collector (NSW).")
    parser.add_argument("--mode", choices=["live", "archive"], default="live")
    parser.add_argument(
        "--feed",
        action="append",
        default=None,
        help="RSS URL (repeatable). Defaults to NSW + Top Stories if omitted.",
    )
    parser.add_argument("--start", type=parse_date, default=None, help="archive mode: YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, default=None, help="archive mode: YYYY-MM-DD")
    parser.add_argument(
        "--cadence_days",
        type=int,
        default=3,
        help="archive mode: keep roughly one snapshot per N days per feed (default 3).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)

    args = parser.parse_args()
    feeds = args.feed if args.feed else DEFAULT_FEEDS

    if args.mode == "live":
        items = collect_live(feeds, timeout=args.timeout, delay=args.delay)
    else:
        if args.start is None or args.end is None:
            parser.error("--start and --end are required in archive mode")
        items = collect_archive(
            feeds,
            start=args.start,
            end=args.end,
            cadence_days=max(1, args.cadence_days),
            timeout=args.timeout,
            delay=args.delay,
        )

    unique = dedupe_items(items)
    write_output(unique, args.output)
    print(f"[done] wrote {len(unique)} unique items -> {args.output}")


if __name__ == "__main__":
    main()
