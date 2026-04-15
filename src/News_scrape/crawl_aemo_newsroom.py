from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - depends on local environment
    PlaywrightTimeoutError = RuntimeError
    sync_playwright = None


DEFAULT_TARGET_YEAR = 2024
DEFAULT_DELAY = 0.5
DEFAULT_TIMEOUT = 30.0
DEFAULT_SETTLE_MS = 1500
DEFAULT_MAX_SCROLL_ROUNDS = 80
DEFAULT_CHALLENGE_TIMEOUT = 45.0
DEFAULT_OUTPUT = Path("dataset/aemo_newsroom_2024.json")
DEFAULT_FAILED_OUTPUT = Path("dataset/aemo_newsroom_2024_failed.json")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
CHALLENGE_TEXT_MARKERS = (
    "Just a moment",
    "Enable JavaScript and cookies to continue",
    "Checking your browser",
)
STOP_TEXT_RE = re.compile(
    r"^(related information|share on facebook|share on linkedin|share on twitter|about aemo|download aemo energy live|cookies help us improve your website experience\.)$",
    flags=re.IGNORECASE,
)
READING_TIME_RE = re.compile(r"^\d+\s+min(?:ute)?s?$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class SectionConfig:
    name: str
    label: str
    listing_url_template: str
    article_path_prefix: str


SECTION_CONFIGS = [
    SectionConfig(
        name="news-updates",
        label="News Updates",
        listing_url_template="https://www.aemo.com.au/newsroom/news-updates#year={year}",
        article_path_prefix="/newsroom/news-updates/",
    ),
    SectionConfig(
        name="media-release",
        label="Media Releases",
        listing_url_template="https://www.aemo.com.au/newsroom/media-release#year={year}",
        article_path_prefix="/newsroom/media-release/",
    ),
]


def clean_text(text: str) -> str:
    value = str(text or "").replace("\r", "\n").replace("\xa0", " ")
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


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_date(value: str) -> str:
    text = clean_text(value)
    if not text:
        return ""

    candidates: list[str] = []
    iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if iso_match:
        candidates.append(iso_match.group(0))
    slash_match = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
    if slash_match:
        candidates.append(slash_match.group(0))
    month_match = re.search(r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", text)
    if month_match:
        candidates.append(month_match.group(0))
    if text not in candidates:
        candidates.append(text)

    for candidate in candidates:
        normalized = candidate.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).date().isoformat()
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
    return ""


def extract_year(date_value: str) -> int | None:
    normalized = normalize_date(date_value)
    if not normalized:
        return None
    return int(normalized[:4])


def looks_like_date(value: str) -> bool:
    return bool(normalize_date(value))


def looks_like_reading_time(value: str) -> bool:
    return bool(READING_TIME_RE.match(clean_text(value)))


def require_playwright() -> None:
    if sync_playwright is not None:
        return
    raise RuntimeError(
        "This scraper needs Playwright because AEMO newsroom pages are client-rendered.\n"
        "Install it with:\n"
        "  pip install playwright\n"
        "  playwright install chromium\n"
        "If AEMO still shows a challenge page, rerun with --headful."
    )


def page_text(page: Any) -> str:
    try:
        return clean_text(page.locator("body").inner_text())
    except Exception:  # noqa: BLE001
        return ""


def is_challenge_page(page: Any) -> bool:
    title = clean_text(page.title())
    body = page_text(page)
    return any(marker.lower() in f"{title}\n{body}".lower() for marker in CHALLENGE_TEXT_MARKERS)


def dismiss_cookie_banner(page: Any) -> None:
    for label in ("Confirm", "Accept", "I agree"):
        try:
            locator = page.get_by_role("button", name=label)
            if locator.count() > 0 and locator.first.is_visible():
                locator.first.click(timeout=1000)
                page.wait_for_timeout(500)
                return
        except Exception:  # noqa: BLE001
            continue


def navigate(page: Any, url: str, timeout_ms: int, challenge_timeout_ms: int) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    dismiss_cookie_banner(page)
    deadline = time.time() + max(1.0, challenge_timeout_ms / 1000.0)
    while True:
        dismiss_cookie_banner(page)
        if not is_challenge_page(page):
            break
        if time.time() >= deadline:
            raise RuntimeError(
                f"AEMO challenge page did not clear for {url}. "
                "Try rerunning with --headful so Chromium can complete the challenge."
            )
        try:
            page.wait_for_load_state("networkidle", timeout=2000)
        except PlaywrightTimeoutError:
            pass
        page.wait_for_timeout(1000)


def apply_year_filter(page: Any, target_year: int, settle_ms: int) -> None:
    year_text = str(int(target_year))
    for locator_factory in (
        lambda: page.get_by_role("button", name=year_text),
        lambda: page.get_by_role("link", name=year_text),
        lambda: page.get_by_text(year_text, exact=True),
    ):
        try:
            locator = locator_factory()
            if locator.count() > 0 and locator.first.is_visible():
                locator.first.click(timeout=1000)
                page.wait_for_timeout(settle_ms)
                dismiss_cookie_banner(page)
                return
        except Exception:  # noqa: BLE001
            continue


def extract_section_urls(page: Any, section: SectionConfig, listing_url: str) -> list[str]:
    listing_path = urlparse(listing_url).path.rstrip("/")
    prefix = section.article_path_prefix.rstrip("/") + "/"
    raw_urls = page.evaluate(
        """
        ({ prefix, listingPath }) => {
          const values = [];
          for (const anchor of document.querySelectorAll('a[href]')) {
            const raw = anchor.getAttribute('href');
            if (!raw) continue;
            let resolved;
            try {
              resolved = new URL(raw, location.href);
            } catch (_error) {
              continue;
            }
            const path = resolved.pathname.replace(/\\/+$/, '');
            if (!path.startsWith(prefix.replace(/\\/+$/, ''))) continue;
            if (path === listingPath) continue;
            values.push(resolved.toString());
          }
          return values;
        }
        """,
        {"prefix": prefix, "listingPath": listing_path},
    )
    seen: set[str] = set()
    results: list[str] = []
    for raw_url in raw_urls or []:
        url = canonicalize_url(str(raw_url))
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(url)
    return results


def click_load_more(page: Any) -> bool:
    clicked_text = page.evaluate(
        """
        () => {
          const patterns = [
            /^load more$/i,
            /^show more$/i,
            /^view more$/i,
            /^more results$/i,
            /^see more$/i,
          ];
          const isVisible = (el) => {
            if (!(el instanceof HTMLElement)) return false;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') return false;
            return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
          };
          for (const el of document.querySelectorAll('button, a, [role="button"]')) {
            const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
            if (!text || !isVisible(el)) continue;
            if (!patterns.some((pattern) => pattern.test(text))) continue;
            el.click();
            return text;
          }
          return '';
        }
        """
    )
    return bool(clean_text(clicked_text))


def discover_listing_urls(
    *,
    page: Any,
    section: SectionConfig,
    target_year: int,
    timeout_ms: int,
    challenge_timeout_ms: int,
    settle_ms: int,
    max_scroll_rounds: int,
) -> list[str]:
    listing_url = section.listing_url_template.format(year=int(target_year))
    navigate(page, listing_url, timeout_ms, challenge_timeout_ms)
    dismiss_cookie_banner(page)
    page.wait_for_timeout(settle_ms)

    urls = extract_section_urls(page, section, listing_url)
    if not urls:
        apply_year_filter(page, target_year, settle_ms)
        urls = extract_section_urls(page, section, listing_url)

    stagnant_rounds = 0
    for _ in range(max(1, int(max_scroll_rounds))):
        before = len(urls)

        if click_load_more(page):
            page.wait_for_timeout(settle_ms)
        else:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(settle_ms)

        dismiss_cookie_banner(page)
        try:
            page.wait_for_load_state("networkidle", timeout=max(1000, settle_ms))
        except PlaywrightTimeoutError:
            pass

        urls = extract_section_urls(page, section, listing_url)
        if len(urls) == before:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0

        if stagnant_rounds >= 3:
            break

    return urls


def extract_time_candidates(page: Any) -> list[dict[str, str]]:
    values = page.evaluate(
        """
        () => Array.from(document.querySelectorAll('time')).map((node) => ({
          text: (node.innerText || node.textContent || '').trim(),
          datetime: (node.getAttribute('datetime') || '').trim(),
        }))
        """
    )
    results: list[dict[str, str]] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "text": clean_text(str(item.get("text", ""))),
                "datetime": clean_text(str(item.get("datetime", ""))),
            }
        )
    return results


def choose_article_root_text(page: Any, title: str) -> str:
    raw_text = page.evaluate(
        """
        (title) => {
          const selectors = ['main article', 'article', '[role="main"] article', 'main', '[role="main"]', 'body'];
          for (const selector of selectors) {
            const node = document.querySelector(selector);
            if (!node) continue;
            const text = (node.innerText || node.textContent || '').replace(/\\u00a0/g, ' ').trim();
            if (!text) continue;
            if (!title || text.includes(title)) return text;
          }
          return (document.body?.innerText || '').replace(/\\u00a0/g, ' ').trim();
        }
        """,
        title,
    )
    return clean_text(str(raw_text or ""))


def choose_title(page: Any) -> str:
    try:
        locator = page.locator("h1")
        if locator.count() > 0:
            title = clean_text(locator.first.inner_text())
            if title:
                return title
    except Exception:  # noqa: BLE001
        pass

    page_title = clean_text(page.title())
    if "|" in page_title:
        _, suffix = page_title.split("|", 1)
        candidate = clean_text(suffix)
        if candidate:
            return candidate
    return page_title


def choose_date_from_page(page: Any, fallback_lines: list[str]) -> tuple[str, str]:
    for candidate in extract_time_candidates(page):
        date_iso = normalize_date(candidate["datetime"]) or normalize_date(candidate["text"])
        if date_iso:
            return date_iso, candidate["text"] or date_iso

    for line in fallback_lines[:6]:
        date_iso = normalize_date(line)
        if date_iso:
            return date_iso, line
    return "", ""


def choose_start_index(lines: list[str], title: str) -> int:
    title_indexes = [index for index, line in enumerate(lines) if clean_text(line) == clean_text(title)]
    if not title_indexes:
        return 0

    best_index = title_indexes[0]
    best_score = -1
    for index in title_indexes:
        lookahead = lines[index + 1 : index + 5]
        score = sum(1 for line in lookahead if looks_like_date(line) or looks_like_reading_time(line))
        if score > best_score or (score == best_score and index > best_index):
            best_index = index
            best_score = score
    return best_index


def parse_article_content(raw_text: str, title: str, date_text: str, reading_time: str) -> str:
    lines = [clean_text(line) for line in str(raw_text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    start_index = choose_start_index(lines, title)
    lines = lines[start_index:]

    if lines and clean_text(lines[0]) == clean_text(title):
        lines = lines[1:]

    if lines and date_text and clean_text(lines[0]) == clean_text(date_text):
        lines = lines[1:]
    elif lines and looks_like_date(lines[0]):
        lines = lines[1:]

    if lines and reading_time and clean_text(lines[0]) == clean_text(reading_time):
        lines = lines[1:]
    elif lines and looks_like_reading_time(lines[0]):
        lines = lines[1:]

    content_lines: list[str] = []
    for line in lines:
        if STOP_TEXT_RE.match(line):
            break
        if line.lower().startswith("share on "):
            break
        if line == title or (date_text and line == date_text) or (reading_time and line == reading_time):
            continue
        content_lines.append(line)

    return clean_text("\n\n".join(content_lines))


def scrape_article(
    *,
    page: Any,
    url: str,
    target_year: int,
    timeout_ms: int,
    challenge_timeout_ms: int,
) -> dict[str, str]:
    navigate(page, url, timeout_ms, challenge_timeout_ms)
    title = choose_title(page)
    if not title:
        raise ValueError("missing title")

    raw_text = choose_article_root_text(page, title)
    lines = [clean_text(line) for line in raw_text.splitlines() if clean_text(line)]
    date_iso, date_text = choose_date_from_page(page, lines)
    if not date_iso:
        raise ValueError("missing date")
    if int(date_iso[:4]) != int(target_year):
        raise ValueError(f"article year {date_iso[:4]} != target_year {target_year}")

    reading_time = ""
    for line in lines[:8]:
        if looks_like_reading_time(line):
            reading_time = line
            break

    content = parse_article_content(raw_text, title, date_text, reading_time)
    if not content:
        raise ValueError("missing content")

    return {
        "title": title,
        "date": date_iso,
        "content": content,
        "url": canonicalize_url(url),
    }


def crawl_aemo_newsroom(
    *,
    output_path: Path,
    failed_output_path: Path,
    target_year: int,
    delay: float,
    timeout: float,
    challenge_timeout: float,
    settle_ms: int,
    max_scroll_rounds: int,
    max_articles: int | None,
    headful: bool,
) -> None:
    require_playwright()
    timeout_ms = int(max(1.0, float(timeout)) * 1000)
    challenge_timeout_ms = int(max(1.0, float(challenge_timeout)) * 1000)

    results_by_url: dict[str, dict[str, str]] = {}
    failures: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not headful,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=USER_AGENT,
            locale="en-AU",
            timezone_id="Australia/Sydney",
            viewport={"width": 1440, "height": 1600},
        )

        listing_page = context.new_page()
        article_page = context.new_page()

        try:
            for section in SECTION_CONFIGS:
                try:
                    urls = discover_listing_urls(
                        page=listing_page,
                        section=section,
                        target_year=target_year,
                        timeout_ms=timeout_ms,
                        challenge_timeout_ms=challenge_timeout_ms,
                        settle_ms=int(settle_ms),
                        max_scroll_rounds=int(max_scroll_rounds),
                    )
                    print(f"[INFO] {section.label}: discovered {len(urls)} candidate URLs.")
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "section": section.name,
                            "listing_url": section.listing_url_template.format(year=int(target_year)),
                            "error": str(exc),
                        }
                    )
                    print(f"[WARN] LIST {section.label} failed: {exc}")
                    continue

                for url in urls:
                    if url in results_by_url:
                        continue
                    try:
                        record = scrape_article(
                            page=article_page,
                            url=url,
                            target_year=target_year,
                            timeout_ms=timeout_ms,
                            challenge_timeout_ms=challenge_timeout_ms,
                        )
                        results_by_url[url] = record
                        print(f"[OK] {record['date']} {record['title']}")
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            {
                                "section": section.name,
                                "url": url,
                                "error": str(exc),
                            }
                        )
                        print(f"[WARN] ARTICLE {url} failed: {exc}")

                    if max_articles is not None and len(results_by_url) >= int(max_articles):
                        break
                    time.sleep(max(0.0, float(delay)))

                if max_articles is not None and len(results_by_url) >= int(max_articles):
                    print(f"[INFO] Reached max_articles={max_articles}; stop.")
                    break
        finally:
            context.close()
            browser.close()

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
        description="Scrape AEMO newsroom News Updates and Media Releases for a target year."
    )
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR, help="Year to keep.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between article requests in seconds.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-page timeout in seconds.")
    parser.add_argument(
        "--challenge-timeout",
        type=float,
        default=DEFAULT_CHALLENGE_TIMEOUT,
        help="How long to wait for AEMO's challenge page to clear, in seconds.",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=DEFAULT_SETTLE_MS,
        help="Wait after scrolls or clicks on the listing page, in milliseconds.",
    )
    parser.add_argument(
        "--max-scroll-rounds",
        type=int,
        default=DEFAULT_MAX_SCROLL_ROUNDS,
        help="Maximum load-more / scroll attempts per listing page.",
    )
    parser.add_argument("--max-articles", type=int, default=None, help="Optional cap for fetched articles.")
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Chromium with a visible window. Useful when AEMO shows a challenge page.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save the combined JSON array.")
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_OUTPUT,
        help="Where to save failed listings/article fetches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        crawl_aemo_newsroom(
            output_path=args.output,
            failed_output_path=args.failed_output,
            target_year=args.target_year,
            delay=args.delay,
            timeout=args.timeout,
            challenge_timeout=args.challenge_timeout,
            settle_ms=args.settle_ms,
            max_scroll_rounds=args.max_scroll_rounds,
            max_articles=args.max_articles,
            headful=bool(args.headful),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
