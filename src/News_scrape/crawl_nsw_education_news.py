from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


BASE_SITE_URL = "https://education.nsw.gov.au"
NEWS_FEED_URL = f"{BASE_SITE_URL}/news.newsfeed.json"
DEFAULT_TARGET_YEAR = 2024
DEFAULT_DELAY = 0.3
DEFAULT_TIMEOUT = 20.0
DEFAULT_OUTPUT = Path("dataset/education_nsw_news_2024.json")
DEFAULT_FAILED_OUTPUT = Path("dataset/education_nsw_news_2024_failed.json")
FEED_ACTIONS = ("latest", "older")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NSW-Education-News-Scraper/1.0; "
        "+https://example.com)"
    )
}
BLOCK_TAGS = {"p", "li", "blockquote", "h2", "h3", "h4"}


def clean_text(text: str) -> str:
    value = unescape(str(text or ""))
    value = value.replace("\r", "\n").replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def class_tokens(raw: str) -> set[str]:
    return {token for token in clean_text(raw).split(" ") if token}


def canonicalize_url(url: str) -> str:
    raw = clean_text(url)
    if not raw:
        return ""
    parsed = urlparse(raw)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return parsed._replace(path=path, query="", fragment="").geturl()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def first_nonempty(*values: str) -> str:
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


def extract_year(value: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", clean_text(value))
    if not match:
        return None
    return int(match.group(0))


def normalize_date(value: str) -> str:
    text = clean_text(value)
    if not text:
        return ""

    iso_candidate = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate).date().isoformat()
    except ValueError:
        pass

    candidates = [text]
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if date_match:
        candidates.insert(0, date_match.group(0))

    for candidate in dedupe_preserve_order(candidates):
        for fmt in (
            "%Y-%m-%d",
            "%d %B %Y",
            "%d %b %Y",
            "%d-%b-%Y",
            "%d-%B-%Y",
        ):
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
    return ""


def build_feed_url(action: str) -> str:
    return f"{NEWS_FEED_URL}?action={clean_text(action)}"


def build_public_article_url(path: str) -> str:
    raw = clean_text(path)
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return canonicalize_url(raw)
    public_path = raw.replace("/content/main-education/en/home", "", 1)
    if not public_path.startswith("/"):
        public_path = f"/{public_path}"
    return canonicalize_url(urljoin(BASE_SITE_URL, public_path))


def fetch_json(session: requests.Session, url: str, timeout: float) -> dict[str, Any]:
    response = session.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_html(session: requests.Session, url: str, timeout: float) -> str:
    response = session.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


@dataclass(frozen=True)
class FeedItem:
    title: str
    summary: str
    date: str
    published_at: str
    tag: str
    url: str
    action: str


class NSWEducationArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self.lead = ""
        self.date_text = ""
        self.cmp_blocks: list[str] = []
        self.media_captions: list[str] = []

        self._title_parts: list[str] | None = None
        self._lead_parts: list[str] | None = None
        self._date_parts: list[str] | None = None
        self._cmp_parts: list[str] | None = None
        self._caption_parts: list[str] | None = None

        self._lead_depth = 0
        self._date_depth = 0
        self._cmp_depth = 0
        self._caption_depth = 0

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key.lower(): (value or "") for key, value in attrs_list}
        lowered = tag.lower()
        tokens = class_tokens(attrs.get("class", ""))

        if lowered == "meta":
            key = clean_text(attrs.get("name") or attrs.get("property") or "")
            value = clean_text(attrs.get("content") or attrs.get("value") or "")
            if key and value and key not in self.meta:
                self.meta[key] = value
            return

        if self._lead_depth > 0:
            self._lead_depth += 1
        if self._date_depth > 0:
            self._date_depth += 1
        if self._cmp_depth > 0:
            self._cmp_depth += 1
        if self._caption_depth > 0:
            self._caption_depth += 1

        if lowered == "h1" and self._title_parts is None and not self.title:
            self._title_parts = []

        if "gel-lead" in tokens:
            self._lead_depth = 1
            self._lead_parts = []
        if "gel-article-date" in tokens:
            self._date_depth = 1
            self._date_parts = []
        if "cmp-text" in tokens:
            self._cmp_depth = 1
            self._cmp_parts = []
        if "gel-script__iframe--caption" in tokens:
            self._caption_depth = 1
            self._caption_parts = []

        if lowered == "br":
            self._append_to_active_collectors("\n")
            return
        if lowered in BLOCK_TAGS:
            self._append_to_active_collectors("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in BLOCK_TAGS:
            self._append_to_active_collectors("\n")

        if lowered == "h1" and self._title_parts is not None:
            self.title = clean_text("".join(self._title_parts))
            self._title_parts = None

        if self._lead_depth > 0:
            self._lead_depth -= 1
            if self._lead_depth == 0 and self._lead_parts is not None:
                self.lead = clean_text("".join(self._lead_parts))
                self._lead_parts = None

        if self._date_depth > 0:
            self._date_depth -= 1
            if self._date_depth == 0 and self._date_parts is not None:
                self.date_text = clean_text("".join(self._date_parts))
                self._date_parts = None

        if self._cmp_depth > 0:
            self._cmp_depth -= 1
            if self._cmp_depth == 0 and self._cmp_parts is not None:
                block = clean_text("".join(self._cmp_parts))
                if block:
                    self.cmp_blocks.append(block)
                self._cmp_parts = None

        if self._caption_depth > 0:
            self._caption_depth -= 1
            if self._caption_depth == 0 and self._caption_parts is not None:
                block = clean_text("".join(self._caption_parts))
                if block:
                    self.media_captions.append(block)
                self._caption_parts = None

    def handle_data(self, data: str) -> None:
        if self._title_parts is not None:
            self._title_parts.append(data)
        if self._lead_parts is not None:
            self._lead_parts.append(data)
        if self._date_parts is not None:
            self._date_parts.append(data)
        if self._cmp_parts is not None:
            self._cmp_parts.append(data)
        if self._caption_parts is not None:
            self._caption_parts.append(data)

    def _append_to_active_collectors(self, text: str) -> None:
        for parts in (self._lead_parts, self._date_parts, self._cmp_parts, self._caption_parts):
            if parts is not None:
                parts.append(text)


def parse_article_page(html: str, item: FeedItem) -> dict[str, str]:
    parser = NSWEducationArticleParser()
    parser.feed(html or "")

    title = first_nonempty(parser.meta.get("doe:article:headline", ""), parser.title, item.title)
    summary = first_nonempty(parser.lead, parser.meta.get("doe:article:description", ""), item.summary)
    date_text = first_nonempty(parser.date_text, parser.meta.get("dcterms.issued", ""), item.published_at)
    date = first_nonempty(
        normalize_date(parser.meta.get("dcterms.issued", "")),
        normalize_date(parser.meta.get("doe:article:date:published", "")),
        normalize_date(parser.date_text),
        item.date,
    )

    body_blocks = dedupe_preserve_order(parser.cmp_blocks)
    if not body_blocks:
        body_blocks = dedupe_preserve_order(parser.media_captions)

    content = clean_text("\n\n".join(body_blocks))
    if summary and content and not content.startswith(summary):
        content = clean_text(f"{summary}\n\n{content}")
    if not content:
        content = summary

    return {
        "title": title,
        "date": date,
        "date_text": date_text,
        "summary": summary,
        "content": content,
    }


def load_feed_items(session: requests.Session, action: str, timeout: float) -> list[FeedItem]:
    payload = fetch_json(session, build_feed_url(action), timeout)
    items: list[FeedItem] = []

    for row in payload.get("results", []) or []:
        title = clean_text(row.get("title", ""))
        url = build_public_article_url(row.get("path", ""))
        published_at = clean_text(row.get("last_published", ""))
        date = normalize_date(published_at)
        if not title or not url or not date:
            continue
        items.append(
            FeedItem(
                title=title,
                summary=clean_text(row.get("description", "")),
                date=date,
                published_at=published_at,
                tag=clean_text(row.get("tags", "")),
                url=url,
                action=clean_text(action),
            )
        )

    items.sort(key=lambda item: (item.published_at, item.title), reverse=True)
    return items


def discover_target_year_items(
    *,
    session: requests.Session,
    target_year: int,
    timeout: float,
) -> tuple[list[FeedItem], list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    merged: dict[str, FeedItem] = {}

    for action in FEED_ACTIONS:
        try:
            action_items = load_feed_items(session, action, timeout)
            print(f"[INFO] FEED action={action} -> {len(action_items)} items.")
            for item in action_items:
                merged.setdefault(item.url, item)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "stage": "feed",
                    "action": action,
                    "url": build_feed_url(action),
                    "error": str(exc),
                }
            )
            print(f"[WARN] FEED action={action} failed: {exc}")

    filtered = [item for item in merged.values() if extract_year(item.published_at) == int(target_year)]
    filtered.sort(key=lambda item: (item.published_at, item.title), reverse=True)
    return filtered, failures


def crawl_nsw_education_news(
    *,
    output_path: Path,
    failed_output_path: Path,
    target_year: int,
    delay: float,
    timeout: float,
    max_articles: int | None,
) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    candidates, failures = discover_target_year_items(
        session=session,
        target_year=target_year,
        timeout=timeout,
    )
    print(f"[INFO] Found {len(candidates)} candidate articles for target_year={target_year}.")

    results_by_url: dict[str, dict[str, Any]] = {}

    for item in candidates:
        if max_articles is not None and len(results_by_url) >= int(max_articles):
            print(f"[INFO] Reached max_articles={max_articles}; stop.")
            break

        try:
            html = fetch_html(session, item.url, timeout)
            article = parse_article_page(html, item)
            if not article["title"] or not article["date"]:
                raise ValueError("missing title or date")
            if not article["content"]:
                raise ValueError("missing content")

            record_year = extract_year(article["date"] or article["date_text"])
            if record_year != int(target_year):
                print(
                    f"[WARN] SKIP {item.url} -> final year {record_year} != target_year {target_year}"
                )
                continue

            record = {
                "source": "NSW Department of Education",
                "target_year": int(target_year),
                "feed_action": item.action,
                "tag": item.tag,
                "title": article["title"],
                "date": article["date"],
                "date_text": article["date_text"],
                "summary": article["summary"],
                "content": article["content"],
                "url": item.url,
            }
            results_by_url[item.url] = record
            print(f"[OK] {record['date']} {record['title']}")
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "stage": "article",
                    "action": item.action,
                    "url": item.url,
                    "title": item.title,
                    "published_at": item.published_at,
                    "error": str(exc),
                }
            )
            print(f"[WARN] ARTICLE {item.url} failed: {exc}")

        time.sleep(max(0.0, float(delay)))

    results = sorted(
        results_by_url.values(),
        key=lambda item: (str(item.get("date", "")), str(item.get("title", ""))),
        reverse=True,
    )

    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ensure_parent_dir(failed_output_path)
    failed_output_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[DONE] saved {len(results)} records to {output_path} "
        f"and {len(failures)} failures to {failed_output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape NSW Department of Education news articles for a target year. "
            "The /news page uses front-end pagination without changing the page URL, "
            "so this scraper reads the hidden news feed endpoints directly."
        )
    )
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR, help="Year to keep.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between article requests in seconds.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds.")
    parser.add_argument("--max-articles", type=int, default=None, help="Optional cap for fetched articles.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the JSON array.")
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_OUTPUT,
        help="Where to save failed feed/article fetches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        crawl_nsw_education_news(
            output_path=args.output,
            failed_output_path=args.failed_output,
            target_year=args.target_year,
            delay=args.delay,
            timeout=args.timeout,
            max_articles=args.max_articles,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
