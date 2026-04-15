from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunsplit

import requests


BASE_NEWS_URL = "https://www.gasunie.nl/en/news?page=2"
DEFAULT_TARGET_YEAR = 2025
DEFAULT_START_PAGE = 2
DEFAULT_MAX_PAGES = 24
DEFAULT_DELAY = 1.0
DEFAULT_TIMEOUT = 20.0
DEFAULT_OUTPUT = Path("dataset/gasunie_news_2025.json")
DEFAULT_FAILED_OUTPUT = Path("dataset/gasunie_news_2025_failed.json")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Gasunie-News-Scraper/1.0; "
        "+https://example.com)"
    )
}
CONTENT_TAGS = {"p", "li", "blockquote", "h2", "h3"}


def clean_text(text: str) -> str:
    value = str(text or "").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def class_tokens(raw: str) -> set[str]:
    return {token for token in clean_text(raw).split(" ") if token}


def canonicalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    clean_path = parsed.path or "/"
    if clean_path != "/" and clean_path.endswith("/"):
        clean_path = clean_path.rstrip("/")
    return parsed._replace(path=clean_path, query="", fragment="").geturl()


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


def fetch_html(session: requests.Session, url: str, timeout: float) -> str:
    response = session.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def build_listing_page_url(base_url: str, page: int) -> str:
    raw_url = str(base_url or BASE_NEWS_URL).strip() or BASE_NEWS_URL
    parts = urlsplit(raw_url)
    query_pairs = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key != "page"]
    if int(page) > 1:
        query_pairs.append(("page", str(int(page))))
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def normalize_date(date_text: str, datetime_value: str = "") -> tuple[str, str]:
    visible = clean_text(date_text)
    raw_datetime = clean_text(datetime_value)

    for candidate in [raw_datetime, visible]:
        if not candidate:
            continue
        normalized = candidate.replace("T", " ").replace("Z", "")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(normalized, fmt).date().isoformat(), visible or candidate
            except ValueError:
                continue
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(candidate, fmt).date().isoformat(), visible or candidate
            except ValueError:
                continue

    return "", visible


def extract_year(value: str) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else None


def first_nonempty(*values: str) -> str:
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ListingRecord:
    title: str
    url: str
    date: str
    date_text: str
    article_type: str
    theme: str
    reading_time: str
    page: int


@dataclass
class ArticleRecord:
    title: str
    date: str
    date_text: str
    article_type: str
    theme: str
    reading_time: str
    summary: str
    content: str


class GasunieListingParser(HTMLParser):
    def __init__(self, *, page_url: str, page_number: int) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.page_number = int(page_number)
        self.items: list[ListingRecord] = []

        self.current_item: dict[str, Any] | None = None
        self.current_meta_kind: str | None = None
        self.current_meta_div_depth = 0

        self._title_parts: list[str] | None = None
        self._article_type_parts: list[str] | None = None
        self._reading_time_parts: list[str] | None = None
        self._time_parts: list[str] | None = None
        self._time_datetime = ""

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key.lower(): (value or "") for key, value in attrs_list}
        tokens = class_tokens(attrs.get("class", ""))

        if tag == "article" and "card-blog-detail" in tokens:
            self.current_item = {
                "title": "",
                "url": "",
                "date": "",
                "date_text": "",
                "article_type": "",
                "theme": "",
                "reading_time": "",
                "page": self.page_number,
            }
            self.current_meta_kind = None
            self.current_meta_div_depth = 0
            return

        if self.current_item is None:
            return

        if tag == "div":
            if self.current_meta_kind is not None:
                self.current_meta_div_depth += 1
            elif "meta-item-date" in tokens and not self.current_item.get("date"):
                self.current_meta_kind = "date"
                self.current_meta_div_depth = 1
            elif "meta-item-clock" in tokens and not self.current_item.get("reading_time"):
                self.current_meta_kind = "reading_time"
                self.current_meta_div_depth = 1
            elif "meta-item-theme" in tokens and not self.current_item.get("article_type"):
                self.current_meta_kind = "article_type"
                self.current_meta_div_depth = 1

        if tag == "h3" and "heading" in tokens:
            self._title_parts = []
        elif tag == "a" and "card-button" in tokens:
            href = canonicalize_url(urljoin(self.page_url, attrs.get("href", "")))
            if href:
                self.current_item["url"] = href
        elif tag == "span" and "theme-icn" in tokens and attrs.get("aria-label"):
            self.current_item["theme"] = clean_text(attrs["aria-label"])
        elif tag == "span" and self.current_meta_kind == "article_type" and "text" in tokens:
            self._article_type_parts = []
        elif tag == "dd" and self.current_meta_kind == "reading_time":
            self._reading_time_parts = []
        elif tag == "time" and self.current_meta_kind == "date":
            self._time_parts = []
            self._time_datetime = attrs.get("datetime", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "h3" and self._title_parts is not None and self.current_item is not None:
            self.current_item["title"] = clean_text("".join(self._title_parts))
            self._title_parts = None
        elif tag == "span" and self._article_type_parts is not None and self.current_item is not None:
            self.current_item["article_type"] = clean_text("".join(self._article_type_parts))
            self._article_type_parts = None
        elif tag == "dd" and self._reading_time_parts is not None and self.current_item is not None:
            self.current_item["reading_time"] = clean_text("".join(self._reading_time_parts))
            self._reading_time_parts = None
        elif tag == "time" and self._time_parts is not None and self.current_item is not None:
            date, date_text = normalize_date("".join(self._time_parts), self._time_datetime)
            self.current_item["date"] = date
            self.current_item["date_text"] = date_text
            self._time_parts = None
            self._time_datetime = ""
        elif tag == "div" and self.current_meta_kind is not None:
            self.current_meta_div_depth -= 1
            if self.current_meta_div_depth <= 0:
                self.current_meta_kind = None
                self.current_meta_div_depth = 0
        elif tag == "article" and self.current_item is not None:
            record = ListingRecord(**self.current_item)
            if record.title and record.url and record.date:
                self.items.append(record)
            self.current_item = None

    def handle_data(self, data: str) -> None:
        if self._title_parts is not None:
            self._title_parts.append(data)
        if self._article_type_parts is not None:
            self._article_type_parts.append(data)
        if self._reading_time_parts is not None:
            self._reading_time_parts.append(data)
        if self._time_parts is not None:
            self._time_parts.append(data)


class GasunieArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.date = ""
        self.date_text = ""
        self.article_type = ""
        self.theme = ""
        self.reading_time = ""
        self.content_blocks: list[str] = []

        self.current_meta_kind: str | None = None
        self.current_meta_div_depth = 0
        self.paragraph_div_depth = 0
        self.reached_share = False

        self._title_parts: list[str] | None = None
        self._article_type_parts: list[str] | None = None
        self._reading_time_parts: list[str] | None = None
        self._time_parts: list[str] | None = None
        self._time_datetime = ""
        self._content_parts: list[str] | None = None
        self._content_tag: str | None = None

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key.lower(): (value or "") for key, value in attrs_list}
        tokens = class_tokens(attrs.get("class", ""))

        if tag == "div":
            if self.current_meta_kind is not None:
                self.current_meta_div_depth += 1
            elif "meta-item-date" in tokens:
                self.current_meta_kind = "date"
                self.current_meta_div_depth = 1
            elif "meta-item-clock" in tokens:
                self.current_meta_kind = "reading_time"
                self.current_meta_div_depth = 1
            elif "meta-item-theme" in tokens:
                self.current_meta_kind = "article_type"
                self.current_meta_div_depth = 1

            if self.title and "site-share" in tokens:
                self.reached_share = True

            if self.paragraph_div_depth > 0:
                self.paragraph_div_depth += 1
            elif self.title and not self.reached_share and "component-block" in tokens and "component-paragraph" in tokens:
                self.paragraph_div_depth = 1

        if tag == "h1" and "heading" in tokens and not self.title:
            self._title_parts = []
        elif tag == "span" and "theme-icn" in tokens and attrs.get("aria-label") and not self.theme:
            self.theme = clean_text(attrs["aria-label"])
        elif tag == "span" and self.current_meta_kind == "article_type" and "text" in tokens:
            self._article_type_parts = []
        elif tag == "dd" and self.current_meta_kind == "reading_time":
            self._reading_time_parts = []
        elif tag == "time" and self.current_meta_kind == "date":
            self._time_parts = []
            self._time_datetime = attrs.get("datetime", "")
        elif self.paragraph_div_depth > 0 and not self.reached_share and tag in CONTENT_TAGS:
            self._content_tag = tag
            self._content_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self._title_parts is not None:
            self.title = clean_text("".join(self._title_parts))
            self._title_parts = None
        elif tag == "span" and self._article_type_parts is not None:
            parsed_article_type = clean_text("".join(self._article_type_parts))
            if parsed_article_type and not self.article_type:
                self.article_type = parsed_article_type
            self._article_type_parts = None
        elif tag == "dd" and self._reading_time_parts is not None:
            parsed_reading_time = clean_text("".join(self._reading_time_parts))
            if parsed_reading_time and not self.reading_time:
                self.reading_time = parsed_reading_time
            self._reading_time_parts = None
        elif tag == "time" and self._time_parts is not None:
            parsed_date, parsed_date_text = normalize_date("".join(self._time_parts), self._time_datetime)
            if parsed_date and not self.date:
                self.date = parsed_date
            if parsed_date_text and not self.date_text:
                self.date_text = parsed_date_text
            self._time_parts = None
            self._time_datetime = ""
        elif tag == self._content_tag and self._content_parts is not None:
            text = clean_text("".join(self._content_parts))
            if text:
                self.content_blocks.append(text)
            self._content_tag = None
            self._content_parts = None
        elif tag == "div" and self.current_meta_kind is not None:
            self.current_meta_div_depth -= 1
            if self.current_meta_div_depth <= 0:
                self.current_meta_kind = None
                self.current_meta_div_depth = 0
        elif tag == "div" and self.paragraph_div_depth > 0:
            self.paragraph_div_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._title_parts is not None:
            self._title_parts.append(data)
        if self._article_type_parts is not None:
            self._article_type_parts.append(data)
        if self._reading_time_parts is not None:
            self._reading_time_parts.append(data)
        if self._time_parts is not None:
            self._time_parts.append(data)
        if self._content_parts is not None:
            self._content_parts.append(data)

    def to_record(self) -> ArticleRecord:
        blocks = dedupe_preserve_order(self.content_blocks)
        content = "\n\n".join(blocks)
        return ArticleRecord(
            title=self.title,
            date=self.date,
            date_text=self.date_text,
            article_type=self.article_type,
            theme=self.theme,
            reading_time=self.reading_time,
            summary=blocks[0] if blocks else "",
            content=content,
        )


def parse_listing_page(html: str, page_url: str, page_number: int) -> list[ListingRecord]:
    parser = GasunieListingParser(page_url=page_url, page_number=page_number)
    parser.feed(html)
    return parser.items


def parse_article_page(html: str) -> ArticleRecord:
    parser = GasunieArticleParser()
    parser.feed(html)
    return parser.to_record()


def crawl_gasunie_news(
    *,
    output_path: Path,
    failed_output_path: Path,
    base_url: str,
    target_year: int,
    start_page: int,
    max_pages: int,
    delay: float,
    timeout: float,
    max_articles: int | None,
) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    found_target_year = False

    for page in range(int(start_page), int(start_page) + int(max_pages)):
        page_url = build_listing_page_url(base_url, page)
        listing_html = fetch_html(session, page_url, timeout)
        listing_items = parse_listing_page(listing_html, page_url, page)

        if not listing_items:
            print(f"[INFO] LIST {page_url} -> 0 items; stop pagination.")
            break

        page_years = [extract_year(item.date or item.date_text) for item in listing_items]
        page_years = [year for year in page_years if year is not None]
        target_items = [item for item in listing_items if extract_year(item.date or item.date_text) == int(target_year)]

        print(
            f"[INFO] LIST {page_url} -> {len(listing_items)} items, "
            f"{len(target_items)} items in {target_year}."
        )

        for item in target_items:
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            found_target_year = True
            try:
                article_html = fetch_html(session, item.url, timeout)
                article = parse_article_page(article_html)
                record = {
                    "source": "Gasunie",
                    "listing_page": item.page,
                    "target_year": int(target_year),
                    "title": first_nonempty(article.title, item.title),
                    "date": first_nonempty(item.date, article.date),
                    "date_text": first_nonempty(item.date_text, article.date_text),
                    "article_type": first_nonempty(article.article_type, item.article_type),
                    "theme": first_nonempty(article.theme, item.theme),
                    "reading_time": first_nonempty(article.reading_time, item.reading_time),
                    "summary": article.summary,
                    "content": article.content,
                    "url": item.url,
                }
                record_year = extract_year(record["date"] or record["date_text"])
                if record_year != int(target_year):
                    print(
                        f"[WARN] SKIP {item.url} -> final year {record_year} != target_year {target_year}"
                    )
                    continue
                results.append(record)
                print(f"[OK] {record['date']} {record['title']}")
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "page": item.page,
                        "url": item.url,
                        "title": item.title,
                        "error": str(exc),
                    }
                )
                print(f"[WARN] ARTICLE {item.url} failed: {exc}")

            if max_articles is not None and len(results) >= int(max_articles):
                break
            time.sleep(max(0.0, float(delay)))

        if max_articles is not None and len(results) >= int(max_articles):
            print(f"[INFO] Reached max_articles={max_articles}; stop.")
            break

        oldest_year = min(page_years) if page_years else None
        newest_year = max(page_years) if page_years else None
        if found_target_year and oldest_year is not None and oldest_year < int(target_year):
            print(f"[INFO] Page {page} already crossed below {target_year}; stop.")
            break
        if newest_year is not None and newest_year < int(target_year):
            print(f"[INFO] Page {page} contains only years older than {target_year}; stop.")
            break

        time.sleep(max(0.0, float(delay)))

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
        description="Scrape Gasunie news pages and keep only articles from the target year."
    )
    parser.add_argument("--base-url", default=BASE_NEWS_URL, help="Listing URL used as pagination base.")
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR, help="Year to keep.")
    parser.add_argument("--start-page", type=int, default=DEFAULT_START_PAGE, help="First page number to crawl.")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Maximum listing pages to scan.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--max-articles", type=int, default=None, help="Optional cap for fetched target-year articles.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the scraped JSON array.")
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_OUTPUT,
        help="Where to save failed article fetches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawl_gasunie_news(
        output_path=args.output,
        failed_output_path=args.failed_output,
        base_url=args.base_url,
        target_year=args.target_year,
        start_page=args.start_page,
        max_pages=args.max_pages,
        delay=args.delay,
        timeout=args.timeout,
        max_articles=args.max_articles,
    )


if __name__ == "__main__":
    main()
