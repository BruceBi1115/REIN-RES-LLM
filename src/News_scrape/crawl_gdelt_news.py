"""GDELT 2.0 DOC API collector for historical news backfill.

GDELT catalogs news articles globally with daily coverage back to 2017. No API
key required; all data is public/academic. This collector slices a date range
into weekly windows (to stay within the 250-records-per-query cap) and supports
domain and keyword filters.

Why this over RSS archive / direct scraping:
  - Complete historical coverage (every day of 2024 for major outlets).
  - Legitimate academic API, no ToS issues.
  - Structured: title + URL + date + source country + language.

Output JSON matches the repo convention:
    [{"title", "date": "DD-MM-YYYY HH:MM:SS AM", "content"}, ...]

`content` defaults to the title (GDELT does not publish article bodies). This is
usually enough for the refine stage since titles carry the headline event. If
you need bodies, run a second pass with a legit source (the outlet's own API or
Wayback Machine by URL) over the `link` field captured in the optional raw
output.

Usage:

  # Australian traffic-adjacent news, 2024 full year
  python src/News_scrape/crawl_gdelt_news.py \\
      --start 2024-01-01 --end 2024-12-31 \\
      --domain abc.net.au --domain smh.com.au --domain 9news.com.au \\
      --keywords "traffic,crash,accident,storm,flood,bushfire,closure,weather" \\
      --output dataset/traffic_au_2024_gdelt.json

  # Netherlands gas-adjacent news, 2024
  python src/News_scrape/crawl_gdelt_news.py \\
      --start 2024-01-01 --end 2024-12-31 \\
      --domain reuters.com --domain gasunie.nl --domain nos.nl --domain rtlnieuws.nl \\
      --keywords "gas,LNG,pipeline,heating,energy,cold,storage,ENTSOG" \\
      --output dataset/gas_nl_2024_gdelt.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_INTERVAL_SEC = 6.5   # use a small safety margin above GDELT's ≥5s guidance
GDELT_MAX_RECORDS = 250        # hard cap per query
DEFAULT_TIMEOUT = 30.0
USER_AGENT = (
    "Mozilla/5.0 (compatible; REIN-RES-LLM-Research/1.0; academic use; "
    "contact=bizhaoge@gmail.com)"
)


@dataclass
class Article:
    title: str
    date: str           # "DD-MM-YYYY HH:MM:SS AM"
    content: str        # title by default
    link: str
    domain: str
    language: str
    seendate_utc: str   # "YYYY-MM-DDTHH:MM:SSZ" — for dedup / debugging


@dataclass
class FetchResult:
    articles: list[dict]
    ok: bool
    detail: str
    request_url: str


def iso_to_display(seendate: str) -> str:
    # GDELT returns "20240603T011500Z"
    try:
        dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return ""
    return dt.strftime("%d-%m-%Y %I:%M:%S %p")


def to_gdelt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def normalize_domain_filter(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    host = host.strip().lower()
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def normalize_keyword_filter(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith('"') and text.endswith('"'):
        return text
    # Preserve explicit GDELT operators such as theme:TERROR or near20:"a b".
    if any(marker in text for marker in (":", "(", ")", "<", ">")):
        return text
    if " " in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def build_query(domains: list[str], keywords: list[str], domain_operator: str = "domain") -> str:
    parts: list[str] = []
    if domains:
        normalized_domains = [d for d in (normalize_domain_filter(x) for x in domains) if d]
        if not normalized_domains:
            raise ValueError("No valid --domain values after normalization")
        if len(normalized_domains) == 1:
            parts.append(f"{domain_operator}:{normalized_domains[0]}")
        else:
            parts.append("(" + " OR ".join(f"{domain_operator}:{d}" for d in normalized_domains) + ")")
    if keywords:
        normalized_keywords = [k for k in (normalize_keyword_filter(x) for x in keywords) if k]
        if not normalized_keywords:
            raise ValueError("No valid --keywords values after normalization")
        if len(normalized_keywords) == 1:
            parts.append(normalized_keywords[0])
        else:
            parts.append("(" + " OR ".join(normalized_keywords) + ")")
    if not parts:
        raise ValueError("Need at least one of --domain / --keywords")
    return " ".join(parts)


class RateLimiter:
    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval = float(min_interval_sec)
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


def fetch_window(
    session: requests.Session,
    limiter: RateLimiter,
    query: str,
    start: datetime,
    end: datetime,
    timeout: float,
    max_attempts: int = 6,
) -> FetchResult:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(GDELT_MAX_RECORDS),
        "startdatetime": to_gdelt_datetime(start),
        "enddatetime": to_gdelt_datetime(end),
        "sort": "dateasc",
    }
    request_url = requests.Request("GET", GDELT_DOC_URL, params=params).prepare().url or GDELT_DOC_URL
    backoff = 8.0
    backoff_ceiling = 90.0
    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        try:
            resp = session.get(
                GDELT_DOC_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            print(f"  [warn] attempt {attempt}: {exc}; sleeping {backoff:.1f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 1.7, backoff_ceiling)
            continue
        if resp.status_code == 429:
            print(f"  [warn] attempt {attempt}: 429 rate-limited, sleeping {backoff:.1f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 1.7, backoff_ceiling)
            continue
        if 500 <= resp.status_code < 600:
            print(f"  [warn] attempt {attempt}: HTTP {resp.status_code}, sleeping {backoff:.1f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 1.7, backoff_ceiling)
            continue
        if resp.status_code != 200:
            body_preview = resp.text.strip().replace("\n", " ")[:160]
            return FetchResult([], False, f"HTTP {resp.status_code}: {body_preview!r}", request_url)
        text = resp.text.strip()
        if not text or text.startswith("Invalid") or text.startswith("Please"):
            # "Please" typically = "Please try your request again" = soft rate limit
            if "try your request" in text.lower() or text.startswith("Please"):
                print(
                    f"  [warn] attempt {attempt}: soft rate-limit body={text[:80]!r}; "
                    f"sleeping {backoff:.1f}s",
                    flush=True,
                )
                time.sleep(backoff)
                backoff = min(backoff * 1.7, backoff_ceiling)
                continue
            return FetchResult([], False, f"non-JSON response: {text[:160]!r}", request_url)
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            return FetchResult([], False, f"JSON decode failed: {text[:160]!r}", request_url)
        articles = payload.get("articles", [])
        if not isinstance(articles, list):
            return FetchResult([], False, "JSON response missing 'articles' list", request_url)
        if articles:
            return FetchResult(articles, True, "ok", request_url)
        return FetchResult([], True, "no matches", request_url)
    detail = f"exhausted {max_attempts} attempts for window {start.date()} -> {end.date()}"
    print(f"  [error] {detail}", flush=True)
    return FetchResult([], False, detail, request_url)


def iter_windows(
    start: datetime, end: datetime, window_days: int
) -> Iterable[tuple[datetime, datetime]]:
    cursor = start
    step = timedelta(days=max(1, int(window_days)))
    while cursor < end:
        window_end = min(cursor + step, end)
        yield cursor, window_end
        cursor = window_end


def parse_article(raw: dict) -> Article | None:
    title = str(raw.get("title") or "").strip()
    link = str(raw.get("url") or "").strip()
    seendate = str(raw.get("seendate") or "").strip()
    if not title or not seendate:
        return None
    display = iso_to_display(seendate)
    if not display:
        return None
    return Article(
        title=title,
        date=display,
        content=title,
        link=link,
        domain=str(raw.get("domain") or "").strip(),
        language=str(raw.get("language") or "").strip(),
        seendate_utc=seendate,
    )


def collect(
    query: str,
    start: datetime,
    end: datetime,
    window_days: int,
    timeout: float,
    min_interval_sec: float,
    show_request_url: bool,
) -> list[Article]:
    session = requests.Session()
    limiter = RateLimiter(min_interval_sec)
    seen_urls: set[str] = set()
    articles: list[Article] = []
    failed_windows = 0

    windows = list(iter_windows(start, end, window_days))
    print(f"[gdelt] query: {query}", flush=True)
    print(
        f"[gdelt] {len(windows)} windows of {window_days}d from {start.date()} to {end.date()} "
        f"(min interval {min_interval_sec:.1f}s)",
        flush=True,
    )

    for idx, (w_start, w_end) in enumerate(windows, start=1):
        fetch = fetch_window(session, limiter, query, w_start, w_end, timeout=timeout)
        raws = fetch.articles
        kept = 0
        for raw in raws:
            art = parse_article(raw)
            if art is None:
                continue
            key = art.link or (art.title + "|" + art.seendate_utc)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            articles.append(art)
            kept += 1
        saturated = " [SATURATED]" if len(raws) >= GDELT_MAX_RECORDS else ""
        detail = ""
        if not fetch.ok:
            failed_windows += 1
            detail = f" [REQUEST_FAILED: {fetch.detail}]"
        elif not raws:
            detail = " [NO_MATCHES]"
        print(
            f"  [{idx:3d}/{len(windows)}] "
            f"{w_start.date()} -> {w_end.date()}  "
            f"raw={len(raws):3d} kept={kept:3d} total={len(articles)}{saturated}{detail}",
            flush=True,
        )
        if show_request_url or (not fetch.ok and idx == 1):
            print(f"        url: {fetch.request_url}", flush=True)
    if failed_windows:
        print(
            f"[gdelt] {failed_windows}/{len(windows)} windows failed before returning JSON. "
            f"Those windows are not true zero-result windows.",
            flush=True,
        )
    if not articles:
        if failed_windows == len(windows) and windows:
            print(
                "[gdelt] no articles were collected because every window failed. "
                "This usually means rate limiting, network issues, or an invalid query.",
                flush=True,
            )
        else:
            print(
                "[gdelt] no unique articles matched across the successful windows. "
                "If this is unexpected, try domains-only or keywords-only once to isolate the filter.",
                flush=True,
            )
    return articles


def sort_chronological(items: list[Article]) -> list[Article]:
    def _key(x: Article) -> datetime:
        try:
            return datetime.strptime(x.date, "%d-%m-%Y %I:%M:%S %p")
        except ValueError:
            return datetime.min
    return sorted(items, key=_key)


def write_output(items: list[Article], path: Path, raw_path: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"title": it.title, "date": it.date, "content": it.content} for it in items]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = [
            {
                "title": it.title, "date": it.date, "content": it.content,
                "link": it.link, "domain": it.domain,
                "language": it.language, "seendate_utc": it.seendate_utc,
            }
            for it in items
        ]
        with raw_path.open("w", encoding="utf-8") as fh:
            json.dump(raw_payload, fh, ensure_ascii=False, indent=2)


def parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_csv(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for chunk in values:
        for tok in str(chunk).split(","):
            tok = tok.strip()
            if tok:
                out.append(tok)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="GDELT 2.0 DOC API historical news collector.")
    parser.add_argument("--start", type=parse_date, required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", type=parse_date, required=True, help="YYYY-MM-DD (exclusive)")
    parser.add_argument(
        "--domain", action="append", default=None,
        help="Outlet domain or URL, e.g. abc.net.au or https://www.abc.net.au/news "
             "(repeatable; comma-lists also accepted).",
    )
    parser.add_argument(
        "--domain_operator", choices=("domain", "domainis"), default="domain",
        help="GDELT domain operator to use (default: domain). Use domainis for exact host matches.",
    )
    parser.add_argument(
        "--keywords", action="append", default=None,
        help="Keyword(s) to require in addition to any domain filter, e.g. "
             "'traffic,crash,storm' (repeatable; comma-lists accepted). Multi-word "
             "phrases are quoted automatically.",
    )
    parser.add_argument(
        "--window_days", type=int, default=7,
        help="Slice the date range into windows of this many days (default 7). "
             "If you see [SATURATED] in logs, lower this.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw_output", type=Path, default=None,
        help="Optional secondary JSON with URL/domain/language metadata for later enrichment.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--min_interval_sec", type=float, default=GDELT_MIN_INTERVAL_SEC,
        help="Minimum pause between GDELT requests (default: %(default).1f). Increase this if you hit 429s.",
    )
    parser.add_argument(
        "--show_request_url", action="store_true",
        help="Print the fully encoded GDELT request URL for each window for debugging.",
    )
    args = parser.parse_args()

    domains = parse_csv(args.domain)
    keywords = parse_csv(args.keywords)
    query = build_query(domains, keywords, domain_operator=args.domain_operator)

    articles = collect(
        query=query,
        start=args.start,
        end=args.end,
        window_days=args.window_days,
        timeout=args.timeout,
        min_interval_sec=args.min_interval_sec,
        show_request_url=args.show_request_url,
    )
    articles = sort_chronological(articles)
    write_output(articles, args.output, args.raw_output)
    print(f"[done] wrote {len(articles)} unique items -> {args.output}", flush=True)
    if args.raw_output is not None:
        print(f"[done] wrote raw metadata -> {args.raw_output}", flush=True)


if __name__ == "__main__":
    main()
