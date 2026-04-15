from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


BASE_ARCHIVE_URL = "https://www.cbs.nl/en-gb/our-services/archive/archive-of-news-releases"
DEFAULT_TARGET_YEAR = 2025
DEFAULT_DELAY = 0.5
DEFAULT_TIMEOUT = 20.0
DEFAULT_OUTPUT = Path("dataset/cbs_archive_news_2025.json")
DEFAULT_FAILED_OUTPUT = Path("dataset/cbs_archive_news_2025_failed.json")
THEME_URL_RE = re.compile(
    r'href="(?P<href>/en-gb/our-services/archive/archive-of-news-releases/themes/[^"#?]+)"',
    flags=re.IGNORECASE,
)
ARTICLE_URL_RE = re.compile(
    r"^https?://(?:www\.)?cbs\.nl/en-gb/(?P<kind>news|background)/(?P<year>20\d{2})/[^?#]+$",
    flags=re.IGNORECASE,
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CBS-Archive-News-Scraper/1.0; "
        "+https://example.com)"
    )
}


def clean_text(text: str) -> str:
    value = str(text or "").replace("\r", "\n")
    value = unescape(value)
    value = value.replace("\xa0", " ")
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


def extract_first_match(html: str, pattern: str, *, flags: int = re.IGNORECASE | re.DOTALL) -> str:
    match = re.search(pattern, html or "", flags)
    if not match:
        return ""
    return clean_text(match.group(1))


def parse_cbs_datetime(datetime_value: str) -> str:
    raw = clean_text(datetime_value)
    if not raw:
        return ""
    iso_candidate = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate).date().isoformat()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%d-%m-%YT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def parse_article_kind(url: str) -> str:
    match = ARTICLE_URL_RE.match(canonicalize_url(url))
    if not match:
        return ""
    kind = str(match.group("kind")).strip().lower()
    if kind == "background":
        return "Background"
    if kind == "news":
        return "News release"
    return kind.title()


def year_tab_url(theme_url: str, target_year: int) -> str:
    base = canonicalize_url(theme_url)
    return f"{base}?tab={int(target_year)}"


class CBSFragmentTextParser(HTMLParser):
    IGNORE_TAGS = {
        "script",
        "style",
        "noscript",
        "figure",
        "figcaption",
        "svg",
        "path",
        "button",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
        "div",
        "aside",
    }
    BLOCK_TAGS = {"p", "h2", "h3", "li", "section", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if self.ignore_depth > 0:
            self.ignore_depth += 1
            return
        if lowered in self.IGNORE_TAGS:
            self.ignore_depth = 1
            return
        if lowered == "br":
            self.parts.append("\n")
            return
        if lowered in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self.ignore_depth > 0:
            self.ignore_depth -= 1
            return
        if lowered in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.ignore_depth > 0:
            return
        self.parts.append(data)

    def get_text(self) -> str:
        return clean_text("".join(self.parts))


def html_fragment_to_text(fragment: str) -> str:
    parser = CBSFragmentTextParser()
    parser.feed(fragment or "")
    return parser.get_text()


@dataclass
class ThemeListingEntry:
    title: str
    url: str
    theme: str
    theme_url: str
    article_type: str


def extract_theme_links(archive_html: str, archive_url: str) -> list[str]:
    urls = [
        canonicalize_url(urljoin(archive_url, match.group("href")))
        for match in THEME_URL_RE.finditer(archive_html or "")
    ]
    return dedupe_preserve_order(urls)


def parse_theme_page(html: str, page_url: str, target_year: int) -> tuple[str, list[ThemeListingEntry]]:
    theme_name = extract_first_match(html, r"<h1>(.*?)</h1>")
    list_match = re.search(
        r'<ol class="filtered-list-items[^"]*">(.*?)</ol>',
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not list_match:
        return theme_name, []

    list_html = list_match.group(1)
    entries: list[ThemeListingEntry] = []
    for href, title_html in re.findall(
        r'<a href="([^"]+)">\s*(.*?)\s*</a>',
        list_html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        url = canonicalize_url(urljoin(page_url, href))
        match = ARTICLE_URL_RE.match(url)
        if not match or int(match.group("year")) != int(target_year):
            continue
        title = clean_text(re.sub(r"<[^>]+>", " ", title_html))
        if not title or not url:
            continue
        entries.append(
            ThemeListingEntry(
                title=title,
                url=url,
                theme=theme_name,
                theme_url=canonicalize_url(page_url),
                article_type=parse_article_kind(url),
            )
        )
    return theme_name, entries


def extract_article_title(html: str) -> str:
    for pattern in [
        r'<h1 class="main-title">(.*?)</h1>',
        r'<meta name="DCTERMS\.title" content="([^"]+)"',
        r'<meta property="og:title" content="([^"]+)"',
        r"<title>(.*?)</title>",
    ]:
        value = extract_first_match(html, pattern)
        if value:
            return value
    return ""


def extract_article_date_values(html: str) -> tuple[str, str]:
    time_match = re.search(
        r'<time[^>]+datetime="([^"]+)"[^>]*>(.*?)</time>',
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if time_match:
        date_iso = parse_cbs_datetime(time_match.group(1))
        date_text = clean_text(re.sub(r"<[^>]+>", " ", time_match.group(2)))
        return date_iso, date_text

    meta_datetime = extract_first_match(html, r'<meta name="DCTERMS\.modified"[^>]+content="([^"]+)"')
    if meta_datetime:
        return parse_cbs_datetime(meta_datetime), meta_datetime
    return "", ""


def extract_article_summary(html: str) -> str:
    lead_match = re.search(
        r'<section class="leadtext[^"]*">(.*?)</section>',
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if lead_match:
        summary = html_fragment_to_text(lead_match.group(1))
        if summary:
            return summary

    for pattern in [
        r'<meta name="description" content="([^"]+)"',
        r'"description"\s*:\s*"([^"]+)"',
    ]:
        value = extract_first_match(html, pattern)
        if value:
            return value
    return ""


def extract_article_content(html: str) -> str:
    article_match = re.search(
        r"<article>(.*?)</article>",
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not article_match:
        return ""

    article_html = article_match.group(1)
    body_html = re.sub(
        r"<header>(.*?)</header>",
        "",
        article_html,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return html_fragment_to_text(body_html)


def crawl_cbs_archive_news(
    *,
    output_path: Path,
    failed_output_path: Path,
    archive_url: str,
    target_year: int,
    delay: float,
    timeout: float,
    max_themes: int | None,
    max_articles: int | None,
) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    archive_html = fetch_html(session, archive_url, timeout)
    theme_urls = extract_theme_links(archive_html, archive_url)
    if max_themes is not None and int(max_themes) > 0:
        theme_urls = theme_urls[: int(max_themes)]

    results_by_url: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for index, theme_url in enumerate(theme_urls, start=1):
        year_url = year_tab_url(theme_url, target_year)
        try:
            theme_html = fetch_html(session, year_url, timeout)
            theme_name, entries = parse_theme_page(theme_html, year_url, target_year)
            print(
                f"[INFO] THEME {index}/{len(theme_urls)} {theme_name or theme_url} -> "
                f"{len(entries)} entries in {target_year}."
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"theme_url": theme_url, "error": str(exc)})
            print(f"[WARN] THEME {theme_url} failed: {exc}")
            continue

        for entry in entries:
            record = results_by_url.get(entry.url)
            if record is None:
                try:
                    article_html = fetch_html(session, entry.url, timeout)
                    title = extract_article_title(article_html) or entry.title
                    date, date_text = extract_article_date_values(article_html)
                    summary = extract_article_summary(article_html)
                    content = extract_article_content(article_html)
                    if summary and content and not content.startswith(summary):
                        content = f"{summary}\n\n{content}"
                    if not content:
                        content = summary
                    if not title or not date:
                        raise ValueError("missing title or date")
                    if int(date[:4]) != int(target_year):
                        raise ValueError(f"article year {date[:4]} != target year {target_year}")

                    record = {
                        "source": "CBS",
                        "target_year": int(target_year),
                        "title": title,
                        "date": date,
                        "date_text": date_text,
                        "article_type": entry.article_type,
                        "theme": entry.theme,
                        "themes": [entry.theme] if entry.theme else [],
                        "summary": summary,
                        "content": content,
                        "url": entry.url,
                    }
                    results_by_url[entry.url] = record
                    print(f"[OK] {date} {title}")
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "theme": entry.theme,
                            "theme_url": entry.theme_url,
                            "url": entry.url,
                            "title": entry.title,
                            "error": str(exc),
                        }
                    )
                    print(f"[WARN] ARTICLE {entry.url} failed: {exc}")
                    continue
                time.sleep(max(0.0, float(delay)))
            else:
                themes = list(record.get("themes", []))
                if entry.theme and entry.theme not in themes:
                    themes.append(entry.theme)
                    record["themes"] = themes

            if max_articles is not None and len(results_by_url) >= int(max_articles):
                break

        if max_articles is not None and len(results_by_url) >= int(max_articles):
            print(f"[INFO] Reached max_articles={max_articles}; stop.")
            break

        time.sleep(max(0.0, float(delay)))

    results = sorted(
        results_by_url.values(),
        key=lambda item: (
            str(item.get("date", "")),
            str(item.get("title", "")),
        ),
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
        description="Scrape 2025 archived CBS news releases for every theme."
    )
    parser.add_argument("--archive-url", default=BASE_ARCHIVE_URL, help="CBS archive landing page.")
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR, help="Year to crawl.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between requests in seconds.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--max-themes", type=int, default=None, help="Optional cap for number of themes to crawl.")
    parser.add_argument("--max-articles", type=int, default=None, help="Optional cap for number of unique articles to fetch.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the output JSON array.")
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_OUTPUT,
        help="Where to save failed theme/article fetches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawl_cbs_archive_news(
        output_path=args.output,
        failed_output_path=args.failed_output,
        archive_url=args.archive_url,
        target_year=args.target_year,
        delay=args.delay,
        timeout=args.timeout,
        max_themes=args.max_themes,
        max_articles=args.max_articles,
    )


if __name__ == "__main__":
    main()
