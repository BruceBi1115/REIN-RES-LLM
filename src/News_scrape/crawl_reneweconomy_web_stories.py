from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests

try:
    from .crawl_pv_magazine_australia_news import (
        HEADERS,
        BeautifulSoup,
        StdlibPageParser,
        canonicalize_url,
        clean_text,
        dedupe_preserve_order,
        extract_content,
        extract_content_from_parser,
        extract_date,
        extract_date_from_parser,
        extract_json_ld_objects,
        extract_json_ld_objects_from_blocks,
        extract_title,
        extract_title_from_parser,
        fetch_html,
        get_soup,
        load_existing_json_array,
        merge_records_by_url,
    )
except ImportError:
    from crawl_pv_magazine_australia_news import (
        HEADERS,
        BeautifulSoup,
        StdlibPageParser,
        canonicalize_url,
        clean_text,
        dedupe_preserve_order,
        extract_content,
        extract_content_from_parser,
        extract_date,
        extract_date_from_parser,
        extract_json_ld_objects,
        extract_json_ld_objects_from_blocks,
        extract_title,
        extract_title_from_parser,
        fetch_html,
        get_soup,
        load_existing_json_array,
        merge_records_by_url,
    )


BASE_ARCHIVE_URL = "https://reneweconomy.com.au/all-articles/"
DEFAULT_YEAR = 2024
DEFAULT_OUTPUT = Path("dataset/reneweconomy_all_articles_2024.json")
DEFAULT_FAILED_OUTPUT = Path("dataset/reneweconomy_all_articles_2024_failed.json")
DEFAULT_DELAY = 1.0
DEFAULT_TIMEOUT = 20.0
DEFAULT_START_PAGE = 1
DEFAULT_MAX_PAGES = 512
ARCHIVE_PAGE_QUERY_KEY = "query-53-page"

STORY_URL_RE = re.compile(
    r"^https?://(?:www\.)?reneweconomy\.com\.au/"
    r"(?!all-articles/?$|web-stories(?:/.*)?$|category/|tag/|author/|page/|feed/?$)"
    r"[^/?#]+/?$"
)
TRIM_STOP_PHRASES = (
    "Get the free daily newsletter",
    "FOLLOW US ON SOCIALS",
    "Follow us on socials",
    "Subscribe to our newsletter",
    "Subscribe to the newsletter",
)


def build_archive_page_url(base_url: str, page: int) -> str:
    # RenewEconomy all-articles pagination uses `?query-53-page=N`.
    raw_url = str(base_url or BASE_ARCHIVE_URL).strip()
    if not raw_url:
        raw_url = BASE_ARCHIVE_URL
    parts = urlsplit(raw_url)
    query_pairs = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key != ARCHIVE_PAGE_QUERY_KEY]
    if page <= 1:
        query = urlencode(query_pairs, doseq=True)
    else:
        query_pairs.append((ARCHIVE_PAGE_QUERY_KEY, str(page)))
        query = urlencode(query_pairs, doseq=True)
    path = parts.path or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, query, ""))


def is_story_url(url: str) -> bool:
    return bool(STORY_URL_RE.match(canonicalize_url(url)))


def extract_listing_story_urls(soup: BeautifulSoup, page_url: str) -> list[str]:
    urls: list[str] = []
    for anchor in soup.select("h2 a[href], article a[href], main a[href]"):
        href = canonicalize_url(urljoin(page_url, anchor.get("href", "")))
        if href and is_story_url(href):
            urls.append(href)
    return dedupe_preserve_order(urls)


def extract_listing_story_urls_from_parser(parser: StdlibPageParser, page_url: str) -> list[str]:
    urls: list[str] = []
    for anchor in parser.anchors:
        href = canonicalize_url(urljoin(page_url, str(anchor.get("href", ""))))
        if href and is_story_url(href):
            urls.append(href)
    return dedupe_preserve_order(urls)


def trim_story_content(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    kept_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = clean_text(line)
        if not stripped:
            continue
        lowered = stripped.casefold()
        if any(stop.casefold() in lowered for stop in TRIM_STOP_PHRASES):
            break
        kept_lines.append(stripped)
    return "\n".join(kept_lines).strip()


def parse_story(session: requests.Session, url: str, timeout: float) -> dict[str, str]:
    if BeautifulSoup is not None:
        soup = get_soup(session, url, timeout)
        json_ld_objects = extract_json_ld_objects(soup)
        record = {
            "title": extract_title(soup, json_ld_objects),
            "date": extract_date(soup, json_ld_objects),
            "content": extract_content(soup, json_ld_objects),
            "url": canonicalize_url(url),
        }
    else:
        html = fetch_html(session, url, timeout)
        parser = StdlibPageParser(collect_content=True)
        parser.feed(html)
        json_ld_objects = extract_json_ld_objects_from_blocks(parser.json_ld_scripts)
        record = {
            "title": extract_title_from_parser(parser, json_ld_objects),
            "date": extract_date_from_parser(parser, json_ld_objects, html),
            "content": extract_content_from_parser(parser, json_ld_objects),
            "url": canonicalize_url(url),
        }

    record["content"] = trim_story_content(record.get("content", ""))
    return record


def parse_dataset_year(value: str) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    for fmt in ("%d-%m-%Y %I:%M:%S %p", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(text, fmt).year)
        except ValueError:
            continue
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else None


def crawl_web_stories(
    *,
    output_path: Path,
    failed_output_path: Path,
    base_url: str,
    target_year: int,
    delay: float,
    timeout: float,
    start_page: int,
    max_pages: int,
    max_articles: int | None,
) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    seen_story_urls: set[str] = set()
    results: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    found_target_year = False

    for page in range(int(start_page), int(start_page) + int(max_pages)):
        page_url = build_archive_page_url(base_url, page)
        try:
            if BeautifulSoup is not None:
                soup = get_soup(session, page_url, timeout)
                page_story_urls = extract_listing_story_urls(soup, page_url)
            else:
                html = fetch_html(session, page_url, timeout)
                parser = StdlibPageParser(collect_content=False)
                parser.feed(html)
                page_story_urls = extract_listing_story_urls_from_parser(parser, page_url)
        except requests.HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if page > int(start_page) and status_code in {404, 410}:
                print(f"[INFO] LIST {page_url} -> HTTP {status_code}; stop pagination.")
                break
            raise

        page_story_urls = [url for url in page_story_urls if url not in seen_story_urls]
        if not page_story_urls:
            print(f"[INFO] LIST {page_url} -> 0 stories; stop pagination.")
            break

        for url in page_story_urls:
            seen_story_urls.add(url)

        print(f"[INFO] LIST {page_url} -> {len(page_story_urls)} stories")

        page_years: list[int] = []
        for index, url in enumerate(page_story_urls, start=1):
            try:
                record = parse_story(session, url, timeout)
                if not record["title"] or not record["date"] or not record["content"]:
                    raise ValueError("missing_title_or_date_or_content")

                record_year = parse_dataset_year(record["date"])
                if record_year is not None:
                    page_years.append(record_year)

                if record_year == int(target_year):
                    results.append(record)
                    found_target_year = True
                    print(f"[OK] ({page}:{index}) {url} -> keep year={record_year}")
                else:
                    print(f"[SKIP] ({page}:{index}) {url} -> year={record_year}")
            except Exception as exc:
                failures.append({"url": url, "reason": str(exc)})
                print(f"[ERR] ({page}:{index}) {url} -> {exc}")

            if max_articles is not None and len(results) >= int(max_articles):
                break
            time.sleep(delay)

        if max_articles is not None and len(results) >= int(max_articles):
            print(f"[INFO] Reached max_articles={max_articles}; stop.")
            break

        if found_target_year and page_years and max(page_years) < int(target_year):
            print(
                f"[INFO] LIST {page_url} -> newest_story_year={max(page_years)} < target_year={target_year}; stop pagination."
            )
            break

        time.sleep(delay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = load_existing_json_array(output_path)
    existing_failures = load_existing_json_array(failed_output_path)
    merged_results = merge_records_by_url(existing_results, results)
    merged_failures = merge_records_by_url(existing_failures, failures)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_results, handle, ensure_ascii=False, indent=2)

    with failed_output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_failures, handle, ensure_ascii=False, indent=2)

    print(f"\nDone. Newly kept {len(results)} items for year={target_year}.")
    print(f"Output items after append/dedupe: {len(merged_results)} -> {output_path}")
    print(f"Failed items after append/dedupe: {len(merged_failures)} -> {failed_output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl Renew Economy all-articles archive page-by-page "
            "via ?query-53-page=N and export dataset-style JSON records."
        )
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_ARCHIVE_URL,
        help=f"Archive root URL. Default: {BASE_ARCHIVE_URL}",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Only keep stories published in this year. Default: {DEFAULT_YEAR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_OUTPUT,
        help=f"Path for failed story records. Default: {DEFAULT_FAILED_OUTPUT}",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds. Default: {DEFAULT_DELAY}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=DEFAULT_START_PAGE,
        help=f"First archive page to crawl, inclusive. Default: {DEFAULT_START_PAGE}",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=(
            "Safety cap on archive pages to scan while paging with "
            f"?{ARCHIVE_PAGE_QUERY_KEY}=N. Default: {DEFAULT_MAX_PAGES}"
        ),
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Optional limit on number of kept articles for the target year.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    crawl_web_stories(
        output_path=args.output,
        failed_output_path=args.failed_output,
        base_url=args.base_url,
        target_year=args.year,
        delay=args.delay,
        timeout=args.timeout,
        start_page=args.start_page,
        max_pages=args.max_pages,
        max_articles=args.max_articles,
    )


if __name__ == "__main__":
    main()
