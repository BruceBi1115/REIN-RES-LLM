from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = None


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BTC-News-JSON-Builder/1.0; +https://example.com)"
}
FIXED_START_DT = datetime(2021, 1, 1, 0, 0, 0)
FIXED_END_DT = datetime(2025, 12, 31, 23, 59, 59)

CONTENT_SELECTORS = [
    "article",
    "main",
    "[role='main']",
    ".article-content",
    ".post-content",
    ".entry-content",
    ".article__content",
    ".article-body",
    ".post-body",
    ".story-content",
    ".story-body",
    ".content",
    "#content",
    ".single-content",
]

CRYPTO_PANIC_DESCRIPTION_SELECTORS = [
    "#detail_pane .description .description-body",
    ".news-detail .description .description-body",
    ".description .description-body",
    ".description-body",
]

CRYPTO_PANIC_TITLE_SELECTORS = [
    "#detail_pane .post-title .text",
    ".news-detail .post-title .text",
    ".post-title .text",
    "h1.post-title",
]

NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "header",
    "footer",
    "nav",
    "aside",
    ".newsletter",
    ".related",
    ".recommended",
    ".advertisement",
    ".ad",
    ".ads",
    ".promo",
    ".social-share",
    ".share",
]


def is_null_like(value: str | None) -> bool:
    return not value or str(value).strip().upper() in {"", "NULL", "NAN", "NONE"}


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_datetime(raw_value: str) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return raw
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y, %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%d-%m-%Y %I:%M:%S %p")
        except ValueError:
            continue
    return raw


def parse_source_datetime(raw_value: str) -> datetime | None:
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y, %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def canonicalize_url(url: str) -> str:
    return str(url or "").strip()


def is_cryptopanic_url(url: str) -> bool:
    return "cryptopanic.com" in urlparse(canonicalize_url(url)).netloc.lower()


def fetch_soup(session: requests.Session, url: str, timeout: float) -> tuple[BeautifulSoup, str]:
    if BeautifulSoup is None:
        raise ModuleNotFoundError("bs4 is required. Install it with: pip install beautifulsoup4")
    response = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return BeautifulSoup(response.text, "html.parser"), response.url


def pick_external_link(page_url: str, soup: BeautifulSoup, preferred_domain: str) -> str | None:
    page_host = urlparse(page_url).netloc.lower()
    preferred_domain = str(preferred_domain or "").strip().lower()

    link_candidates: list[str] = []

    for selector in [
        "a[href]",
        "link[rel='canonical']",
        "meta[property='og:url']",
        "meta[name='twitter:url']",
    ]:
        for node in soup.select(selector):
            href = node.get("href") or node.get("content")
            if not href:
                continue
            link = canonicalize_url(href)
            host = urlparse(link).netloc.lower()
            if not host:
                continue
            if "cryptopanic.com" in host:
                continue
            link_candidates.append(link)

    for candidate in link_candidates:
        host = urlparse(candidate).netloc.lower()
        if preferred_domain and preferred_domain in host:
            return candidate

    for candidate in link_candidates:
        host = urlparse(candidate).netloc.lower()
        if host and host != page_host:
            return candidate
    return None


def extract_text_from_node(node: BeautifulSoup) -> str:
    paragraphs: list[str] = []
    for tag in node.find_all(["p", "li"]):
        text = clean_text(tag.get_text(" ", strip=True))
        if len(text) >= 40:
            paragraphs.append(text)
    return "\n".join(paragraphs).strip()


def extract_article_text(soup: BeautifulSoup) -> str:
    if BeautifulSoup is None:
        raise ModuleNotFoundError("bs4 is required. Install it with: pip install beautifulsoup4")
    work_soup = BeautifulSoup(str(soup), "html.parser")
    for selector in NOISE_SELECTORS:
        for node in work_soup.select(selector):
            node.decompose()

    best_text = ""
    for selector in CONTENT_SELECTORS:
        for node in work_soup.select(selector):
            text = extract_text_from_node(node)
            if len(text) > len(best_text):
                best_text = text

    if best_text:
        return best_text

    body = work_soup.body or work_soup
    best_text = extract_text_from_node(body)
    if best_text:
        return best_text

    meta_desc = (
        work_soup.find("meta", attrs={"property": "og:description"})
        or work_soup.find("meta", attrs={"name": "description"})
    )
    if meta_desc and meta_desc.get("content"):
        return clean_text(meta_desc["content"])
    return ""


def extract_cryptopanic_detail(soup: BeautifulSoup) -> dict[str, str]:
    content = ""
    description_node = soup.find(class_="description-body")
    if description_node is not None:
        content = clean_text(description_node.get_text(" ", strip=True))

    if not content:
        for selector in CRYPTO_PANIC_DESCRIPTION_SELECTORS:
            node = soup.select_one(selector)
            if node is None:
                continue
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                content = text
                break

    title = ""
    for selector in CRYPTO_PANIC_TITLE_SELECTORS:
        node = soup.select_one(selector)
        if node is None:
            continue
        text = clean_text(node.get_text(" ", strip=True))
        if text:
            title = text
            break

    date = ""
    time_node = soup.select_one("#detail_pane .post-source time, .news-detail .post-source time, .post-source time")
    if time_node is not None:
        date = normalize_datetime(
            time_node.get("title")
            or time_node.get("datetime")
            or time_node.get_text(" ", strip=True)
        )

    return {
        "title": title,
        "date": date,
        "content": content,
    }


def build_record_from_cryptopanic(
    *,
    title: str,
    date: str,
    soup: BeautifulSoup,
    final_url: str,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    detail = extract_cryptopanic_detail(soup)
    content = clean_text(detail.get("content", ""))
    if not content:
        return None, {"url": final_url, "reason": "cryptopanic_description_not_found"}

    record = {
        "title": title or clean_text(detail.get("title", "")),
        "date": date or normalize_datetime(detail.get("date", "")),
        "content": content,
        "url": final_url,
    }
    return record, None


def build_record_from_cryptopanic_url(
    *,
    title: str,
    date: str,
    session: requests.Session,
    candidate_urls: Iterable[str],
    timeout: float,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    seen: set[str] = set()
    last_failure: dict[str, str] | None = None

    for candidate in candidate_urls:
        url = canonicalize_url(candidate)
        if not url or url in seen or not is_cryptopanic_url(url):
            continue
        seen.add(url)
        try:
            soup, final_url = fetch_soup(session, url, timeout)
        except Exception as exc:
            last_failure = {"url": url, "reason": f"cryptopanic_fetch_failed:{exc}"}
            continue

        if not is_cryptopanic_url(final_url):
            continue

        record, failure = build_record_from_cryptopanic(
            title=title,
            date=date,
            soup=soup,
            final_url=final_url,
        )
        if record is not None:
            return record, None
        last_failure = failure

    return None, last_failure


def resolve_article_url(
    session: requests.Session,
    csv_url: str,
    source_domain: str,
    timeout: float,
) -> tuple[str | None, str | None]:
    landing_url = canonicalize_url(csv_url)
    if not landing_url:
        return None, "missing_url"

    if not is_cryptopanic_url(landing_url):
        return landing_url, None

    try:
        soup, final_url = fetch_soup(session, landing_url, timeout)
    except Exception as exc:
        return None, f"resolve_fetch_failed:{exc}"

    if not is_cryptopanic_url(final_url):
        return final_url, None

    external = pick_external_link(final_url, soup, preferred_domain=source_domain)
    if external:
        return external, None
    return None, "source_link_not_found"


def build_record(
    *,
    row: dict[str, str],
    session: requests.Session,
    timeout: float,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    title = clean_text(row.get("title", ""))
    date = normalize_datetime(row.get("newsDatetime", ""))
    csv_url = row.get("url", "")
    source_domain = row.get("sourceDomain", "")
    cryptopanic_candidates = [csv_url]

    resolved_url, resolve_error = resolve_article_url(
        session=session,
        csv_url=csv_url,
        source_domain=source_domain,
        timeout=timeout,
    )
    if resolved_url is None:
        fallback_record, fallback_failure = build_record_from_cryptopanic_url(
            title=title,
            date=date,
            session=session,
            candidate_urls=cryptopanic_candidates,
            timeout=timeout,
        )
        if fallback_record is not None:
            return fallback_record, None
        return None, fallback_failure or {"url": csv_url, "reason": resolve_error or "resolve_failed"}

    try:
        soup, final_url = fetch_soup(session, resolved_url, timeout)
    except Exception as exc:
        fallback_record, fallback_failure = build_record_from_cryptopanic_url(
            title=title,
            date=date,
            session=session,
            candidate_urls=[resolved_url, *cryptopanic_candidates],
            timeout=timeout,
        )
        if fallback_record is not None:
            return fallback_record, None
        return None, fallback_failure or {"url": resolved_url, "reason": f"fetch_failed:{exc}"}

    if is_cryptopanic_url(final_url):
        cryptopanic_record, cryptopanic_failure = build_record_from_cryptopanic(
            title=title,
            date=date,
            soup=soup,
            final_url=final_url,
        )
        if cryptopanic_record is not None:
            return cryptopanic_record, None
        return None, cryptopanic_failure or {"url": final_url, "reason": "cryptopanic_page_not_source"}

    content = extract_article_text(soup)
    if not content:
        fallback_record, fallback_failure = build_record_from_cryptopanic_url(
            title=title,
            date=date,
            session=session,
            candidate_urls=[csv_url],
            timeout=timeout,
        )
        if fallback_record is not None:
            return fallback_record, None
        return None, fallback_failure or {"url": final_url, "reason": "empty_content"}

    if not title:
        page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        title = clean_text(page_title)

    record = {
        "title": title,
        "date": date,
        "content": content,
        "url": final_url,
    }
    return record, None


def iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = parse_source_datetime(row.get("newsDatetime", ""))
            if dt is None:
                continue
            if FIXED_START_DT <= dt <= FIXED_END_DT:
                yield row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl BTC news article text from CSV URLs and export JSON with title/date/content/url keys."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/News_scrape/news_currencies_source_joinedResult_BTC.csv"),
        help="Input BTC CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/News_scrape/news_currencies_source_joinedResult_BTC_2021_2025.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=Path("src/News_scrape/news_currencies_source_joinedResult_BTC_2021_2025_failed.json"),
        help="Failed rows log path.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between successful requests in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    rows = list(iter_rows(args.input))

    results: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    total = len(rows)
    print(
        "Using fixed datetime window: "
        f"{FIXED_START_DT.strftime('%Y-%m-%d %H:%M:%S')} -> {FIXED_END_DT.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    for idx, row in enumerate(rows, start=1):
        try:
            record, failure = build_record(
                row=row,
                session=session,
                timeout=float(args.timeout),
            )
            if record is not None:
                results.append(record)
                print(f"[OK] {idx}/{total} {record['url']}")
            else:
                failures.append(
                    {
                        "title": row.get("title", ""),
                        "newsDatetime": row.get("newsDatetime", ""),
                        "url": row.get("url", ""),
                        "sourceDomain": row.get("sourceDomain", ""),
                        "sourceUrl": row.get("sourceUrl", ""),
                        "reason": (failure or {}).get("reason", "unknown"),
                    }
                )
                print(f"[SKIP] {idx}/{total} {(failure or {}).get('reason', 'unknown')} :: {row.get('url', '')}")
        except Exception as exc:
            failures.append(
                {
                    "title": row.get("title", ""),
                    "newsDatetime": row.get("newsDatetime", ""),
                    "url": row.get("url", ""),
                    "sourceDomain": row.get("sourceDomain", ""),
                    "sourceUrl": row.get("sourceUrl", ""),
                    "reason": f"unexpected_error:{exc}",
                }
            )
            print(f"[ERR] {idx}/{total} unexpected_error:{exc} :: {row.get('url', '')}")
        time.sleep(max(0.0, float(args.delay)))

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with args.failed_output.open("w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(results)} records to {args.output}")
    print(f"Done. Saved {len(failures)} failed records to {args.failed_output}")


if __name__ == "__main__":
    main()
