from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


BASE_ARCHIVE_URL = "https://www.coindesk.com/sitemap/archive/2024"
DEFAULT_TARGET_YEAR = 2024
DEFAULT_DELAY = 0.5
DEFAULT_TIMEOUT = 20.0
DEFAULT_START_PAGE = None
DEFAULT_MAX_ARCHIVE_PAGES = None
DEFAULT_OUTPUT = Path("dataset/news_from_sources/coindesk_archive_2024_new.json")
DEFAULT_FAILED_OUTPUT = Path("dataset/news_from_sources/coindesk_archive_2024_failed_new.json")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CoinDesk-Archive-Scraper/1.0; "
        "+https://example.com)"
    )
}
STOP_BLOCK_PREFIXES = (
    "Disclosure & Polic",
    "Edited by",
)
SKIP_BLOCKS = {
    "More For You",
    "Advertisement",
}


def clean_text(text: str) -> str:
    value = str(text or "").replace("\r", "\n")
    value = unescape(value).replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def canonicalize_url(url: str) -> str:
    raw = str(url or "").strip().strip("\\")
    raw = raw.replace("\\/", "/").replace("\\u002F", "/")
    if not raw:
        return ""
    parsed = urlparse(raw)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return parsed._replace(path=path, query="", fragment="").geturl()


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


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_html(session: requests.Session, url: str, timeout: float) -> str:
    response = session.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def extract_first_match(html: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, html or "", flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = clean_text(match.group(1))
            if value:
                return value
    return ""


def parse_output_datetime(raw_value: str) -> str:
    raw = clean_text(raw_value)
    if not raw:
        return ""

    iso_candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed.strftime("%d-%m-%Y %I:%M:%S %p")
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d-%m-%Y %I:%M:%S %p",
    ):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.strftime("%d-%m-%Y %I:%M:%S %p")
        except ValueError:
            continue
    return ""


def split_archive_root_and_start_page(archive_url: str, *, target_year: int) -> tuple[str, int]:
    parsed = urlparse(canonicalize_url(archive_url))
    segments = [segment for segment in (parsed.path or "").split("/") if segment]
    root_path = parsed.path or ""
    start_page = 1

    if (
        len(segments) >= 3
        and segments[0] == "sitemap"
        and segments[1] == "archive"
        and segments[2] == str(int(target_year))
    ):
        root_path = f"/sitemap/archive/{int(target_year)}"
        if len(segments) >= 4 and str(segments[3]).isdigit():
            start_page = max(1, int(segments[3]))

    root_url = parsed._replace(path=root_path, query="", fragment="").geturl()
    return root_url, start_page


def build_archive_page_url(archive_root_url: str, page_number: int) -> str:
    root_url = canonicalize_url(archive_root_url)
    if int(page_number) <= 1:
        return root_url
    return f"{root_url}/{int(page_number)}"


def looks_like_coindesk_article_url(url: str, *, target_year: int) -> bool:
    parsed = urlparse(canonicalize_url(url))
    host = (parsed.netloc or "").lower()
    if host not in {"coindesk.com", "www.coindesk.com"}:
        return False

    path = parsed.path or ""
    if path.startswith("/sitemap/"):
        return False

    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 4:
        return False
    if not re.fullmatch(r"20\d{2}", segments[-4] or ""):
        return False
    if int(segments[-4]) != int(target_year):
        return False
    if not re.fullmatch(r"\d{2}", segments[-3] or ""):
        return False
    if not re.fullmatch(r"\d{2}", segments[-2] or ""):
        return False
    if not segments[-1]:
        return False

    year_like_segments = [segment for segment in segments if re.fullmatch(r"20\d{2}", segment or "")]
    if len(year_like_segments) != 1:
        return False

    return True


class CoinDeskArchiveHrefParser(HTMLParser):
    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key.lower(): (value or "") for key, value in attrs_list}
        for attr_name in ("href", "content"):
            raw_value = attrs.get(attr_name, "")
            if not raw_value:
                continue
            if "coindesk.com/" not in raw_value and not raw_value.startswith("/"):
                continue
            self.urls.append(canonicalize_url(urljoin(self.page_url, raw_value)))


def extract_article_urls_from_archive(html: str, *, archive_url: str, target_year: int) -> list[str]:
    html_to_scan = str(html or "").replace("\\/", "/").replace("\\u002F", "/")
    parser = CoinDeskArchiveHrefParser(archive_url)
    parser.feed(html_to_scan)
    return dedupe_preserve_order(
        [
            url
            for url in parser.urls
            if looks_like_coindesk_article_url(url, target_year=target_year)
        ]
    )


def collect_archive_article_urls(
    session: requests.Session,
    *,
    archive_url: str,
    target_year: int,
    timeout: float,
    delay: float,
    start_page: int | None,
    max_archive_pages: int | None,
    max_articles: int | None,
) -> list[str]:
    archive_root_url, implied_start_page = split_archive_root_and_start_page(
        archive_url,
        target_year=target_year,
    )
    page_number = max(1, int(start_page or implied_start_page))
    page_count = 0
    collected_urls: list[str] = []
    seen_urls: set[str] = set()

    while True:
        if max_archive_pages is not None and page_count >= int(max_archive_pages):
            break

        page_url = build_archive_page_url(archive_root_url, page_number)
        try:
            archive_html = fetch_html(session, page_url, timeout)
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            status_code = int(response.status_code) if response is not None else None
            if status_code == 404 and page_number > max(1, int(start_page or implied_start_page)):
                print(f"[INFO] ARCHIVE page={page_number} -> 404, stop pagination.")
                break
            raise

        page_urls = extract_article_urls_from_archive(
            archive_html,
            archive_url=page_url,
            target_year=target_year,
        )
        new_urls = [url for url in page_urls if url not in seen_urls]
        print(
            f"[INFO] ARCHIVE page={page_number} total_urls={len(page_urls)} "
            f"new_urls={len(new_urls)} {page_url}"
        )

        if not page_urls:
            print(f"[INFO] ARCHIVE page={page_number} has no article urls, stop pagination.")
            break

        if not new_urls:
            print(f"[INFO] ARCHIVE page={page_number} adds no new urls, stop pagination.")
            break

        for url in new_urls:
            seen_urls.add(url)
            collected_urls.append(url)

        page_count += 1
        if max_articles is not None and len(collected_urls) >= int(max_articles):
            break

        page_number += 1
        time.sleep(max(0.0, float(delay)))

    return collected_urls


def extract_article_title(html: str) -> str:
    title = extract_first_match(
        html,
        [
            r'<meta name="parsely-title" content="([^"]+)"',
            r'<meta property="og:title" content="([^"]+)"',
            r'"headline"\s*:\s*"([^"]+)"',
            r"<title>(.*?)</title>",
        ],
    )
    title = re.sub(r"\s*\|\s*CoinDesk\s*$", "", title, flags=re.IGNORECASE)
    return clean_text(title)


def extract_article_datetime(html: str) -> str:
    raw_value = extract_first_match(
        html,
        [
            r'<meta name="parsely-pub-date" content="([^"]+)"',
            r'<meta property="article:published_time" content="([^"]+)"',
            r'"datePublished"\s*:\s*"([^"]+)"',
            r'"dateModified"\s*:\s*"([^"]+)"',
        ],
    )
    return parse_output_datetime(raw_value)


def extract_article_description(html: str) -> str:
    return extract_first_match(
        html,
        [
            r'<meta name="description" content="([^"]+)"',
            r'<meta property="og:description" content="([^"]+)"',
        ],
    )


def extract_article_fragment(html: str) -> str:
    match = re.search(
        r"<article\b[^>]*>(.*?)</article>",
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)
    match = re.search(
        r"<main\b[^>]*>(.*?)</main>",
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)
    return html or ""


class CoinDeskBlockParser(HTMLParser):
    IGNORE_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "path",
        "figure",
        "figcaption",
        "button",
        "form",
        "iframe",
        "footer",
        "nav",
        "aside",
    }
    BLOCK_TAGS = {"p", "li", "blockquote", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._ignore_depth = 0
        self._current_tag: str | None = None
        self._current_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if self._ignore_depth > 0:
            self._ignore_depth += 1
            return
        if lowered in self.IGNORE_TAGS:
            self._ignore_depth = 1
            return
        if lowered == "br" and self._current_parts is not None:
            self._current_parts.append("\n")
            return
        if lowered in self.BLOCK_TAGS:
            self._flush_current()
            self._current_tag = lowered
            self._current_parts = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._ignore_depth > 0:
            self._ignore_depth -= 1
            return
        if lowered in self.BLOCK_TAGS and self._current_tag == lowered:
            self._flush_current()

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0 or self._current_parts is None:
            return
        self._current_parts.append(data)

    def _flush_current(self) -> None:
        if self._current_parts is not None:
            text = clean_text("".join(self._current_parts))
            if text:
                self.blocks.append(text)
        self._current_tag = None
        self._current_parts = None


def extract_article_content(html: str) -> str:
    parser = CoinDeskBlockParser()
    parser.feed(extract_article_fragment(html))
    blocks: list[str] = []
    for block in dedupe_preserve_order(parser.blocks):
        if block in SKIP_BLOCKS:
            if block == "More For You":
                break
            continue
        if block.startswith(STOP_BLOCK_PREFIXES):
            break
        blocks.append(block)
    return "\n".join(blocks).strip()


def parse_sort_datetime(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%d-%m-%Y %I:%M:%S %p")
    except ValueError:
        return datetime.min


def crawl_coindesk_archive_news(
    *,
    archive_url: str,
    target_year: int,
    output_path: Path,
    failed_output_path: Path,
    delay: float,
    timeout: float,
    start_page: int | None,
    max_archive_pages: int | None,
    max_articles: int | None,
) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    article_urls = collect_archive_article_urls(
        session,
        archive_url=archive_url,
        target_year=target_year,
        timeout=timeout,
        delay=delay,
        start_page=start_page,
        max_archive_pages=max_archive_pages,
        max_articles=max_articles,
    )
    if max_articles is not None and int(max_articles) > 0:
        article_urls = article_urls[: int(max_articles)]

    results: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []

    for index, article_url in enumerate(article_urls, start=1):
        try:
            article_html = fetch_html(session, article_url, timeout)
            title = extract_article_title(article_html)
            date_value = extract_article_datetime(article_html)
            description = extract_article_description(article_html)
            content = extract_article_content(article_html)
            if not content:
                content = description

            if not title or not date_value or not content:
                raise ValueError("missing title, date, or content")

            parsed_date = parse_sort_datetime(date_value)
            if parsed_date.year != int(target_year):
                raise ValueError(
                    f"article year {parsed_date.year} != target year {target_year}"
                )

            results.append(
                {
                    "title": title,
                    "date": date_value,
                    "content": content,
                    "url": article_url,
                }
            )
            print(f"[OK] {index}/{len(article_urls)} {date_value} {title}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"url": article_url, "error": str(exc)})
            print(f"[WARN] {index}/{len(article_urls)} {article_url} failed: {exc}")
        time.sleep(max(0.0, float(delay)))

    results.sort(key=lambda item: parse_sort_datetime(item.get("date", "")), reverse=True)

    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ensure_parent_dir(failed_output_path)
    failed_output_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[DONE] saved {len(results)} records to {output_path} "
        f"and {len(failures)} failures to {failed_output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape article pages linked from CoinDesk's 2024 archive sitemap."
    )
    parser.add_argument("--archive-url", default=BASE_ARCHIVE_URL, help="CoinDesk archive page.")
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR, help="Year to crawl.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--start-page",
        type=int,
        default=DEFAULT_START_PAGE,
        help="Optional archive page number to start from. Defaults to the page encoded in --archive-url or page 1.",
    )
    parser.add_argument(
        "--max-archive-pages",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_PAGES,
        help="Optional cap for number of archive pages to scan.",
    )
    parser.add_argument("--max-articles", type=int, default=None, help="Optional cap for number of articles to fetch.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the output JSON array.")
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_OUTPUT,
        help="Where to save failed article fetches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawl_coindesk_archive_news(
        archive_url=args.archive_url,
        target_year=args.target_year,
        output_path=args.output,
        failed_output_path=args.failed_output,
        delay=args.delay,
        timeout=args.timeout,
        start_page=args.start_page,
        max_archive_pages=args.max_archive_pages,
        max_articles=args.max_articles,
    )


if __name__ == "__main__":
    main()
