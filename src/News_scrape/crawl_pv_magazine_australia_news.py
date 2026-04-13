from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = None


BASE_NEWS_URL = "https://www.pv-magazine-australia.com/news/"
DEFAULT_OUTPUT = Path("dataset/pv_magazine_australia_news.json")
DEFAULT_FAILED_OUTPUT = Path("dataset/pv_magazine_australia_news_failed.json")
DEFAULT_DELAY = 1.0
DEFAULT_TIMEOUT = 20.0
DEFAULT_START_PAGE = 172
DEFAULT_END_PAGE = 313
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PV-Magazine-Australia-News-Scraper/1.0; "
        "+https://example.com)"
    )
}

ARTICLE_URL_RE = re.compile(
    r"^https?://(?:www\.)?pv-magazine-australia\.com/\d{4}/\d{2}/\d{2}/[^/?#]+/?$"
)
LISTING_URL_RE = re.compile(
    r"^https?://(?:www\.)?pv-magazine-australia\.com/news(?:/page/\d+)?$"
)
VISIBLE_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) "
    r"\d{1,2}, \d{4}\b"
)

CONTENT_SELECTORS = [
    "article .entry-content",
    "article .post-content",
    "article .article-content",
    "article .td-post-content",
    "article .single-post-content",
    "article [itemprop='articleBody']",
    ".entry-content",
    ".post-content",
    ".article-content",
    ".td-post-content",
    "article",
    "main article",
    "main",
]

NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "iframe",
    "header",
    "footer",
    "nav",
    "aside",
    "figure",
    "figcaption",
    ".share",
    ".sharedaddy",
    ".social-share",
    ".author",
    ".author-box",
    ".related",
    ".related-posts",
    ".jp-relatedposts",
    ".popular",
    ".newsletter",
    ".comments",
    ".comment-respond",
    ".post-tags",
    ".tags",
    ".advertisement",
    ".ad",
    ".ads",
    ".promo",
]

STOP_PHRASES = [
    "This content is protected by copyright and may not be reused.",
    "Popular content",
    "Related content",
    "Leave a Reply",
    "Most popular",
    "Keep up to date",
    "Subscribe to our global magazine",
    "Newsletter",
]

CONTENT_TAGS = {"h2", "h3", "p", "li", "blockquote"}
NOISE_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "iframe",
    "header",
    "footer",
    "nav",
    "aside",
    "figure",
}
NOISE_CLASS_KEYWORDS = {
    "share",
    "social",
    "author",
    "related",
    "popular",
    "newsletter",
    "comment",
    "tag",
    "advertisement",
    "ad",
    "ads",
    "promo",
}


class StdlibPageParser(HTMLParser):
    def __init__(self, *, collect_content: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.collect_content = collect_content
        self.tag_stack: list[str] = []
        self.meta_tags: list[dict[str, str]] = []
        self.anchors: list[dict[str, Any]] = []
        self.h1_texts: list[str] = []
        self.time_values: list[str] = []
        self.json_ld_scripts: list[str] = []
        self.content_blocks: list[str] = []
        self.title_text = ""

        self._current_anchor: dict[str, Any] | None = None
        self._title_parts: list[str] = []
        self._h1_parts: list[str] | None = None
        self._time_parts: list[str] | None = None
        self._current_time_datetime = ""
        self._script_parts: list[str] | None = None
        self._content_parts: list[str] | None = None
        self._content_tag: str | None = None
        self._article_depth = 0
        self._main_depth = 0
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key.lower(): (value or "") for key, value in attrs_list}
        lowered_tag = tag.lower()

        if lowered_tag == "meta":
            self.meta_tags.append(attrs)

        in_h2 = "h2" in self.tag_stack
        if lowered_tag == "a":
            self._current_anchor = {
                "href": attrs.get("href", ""),
                "text_parts": [],
                "in_h2": in_h2,
            }

        if lowered_tag == "title":
            self._title_parts = []
        elif lowered_tag == "h1":
            self._h1_parts = []
        elif lowered_tag == "time":
            self._time_parts = []
            self._current_time_datetime = attrs.get("datetime", "")
        elif lowered_tag == "script" and attrs.get("type", "").casefold() == "application/ld+json":
            self._script_parts = []

        self.tag_stack.append(lowered_tag)

        if lowered_tag == "article":
            self._article_depth += 1
        elif lowered_tag == "main":
            self._main_depth += 1

        if not self.collect_content:
            return

        if self._ignore_depth > 0:
            self._ignore_depth += 1
            return

        if self._inside_content_scope() and self._should_ignore_tag(lowered_tag, attrs):
            self._ignore_depth = 1
            return

        if self._inside_content_scope() and lowered_tag in CONTENT_TAGS:
            self._content_tag = lowered_tag
            self._content_parts = []

    def handle_endtag(self, tag: str) -> None:
        lowered_tag = tag.lower()

        if lowered_tag == "a" and self._current_anchor is not None:
            text = clean_text("".join(self._current_anchor["text_parts"]))
            self.anchors.append(
                {
                    "href": self._current_anchor["href"],
                    "text": text,
                    "in_h2": bool(self._current_anchor["in_h2"]),
                }
            )
            self._current_anchor = None

        if lowered_tag == "title":
            self.title_text = clean_text("".join(self._title_parts))
            self._title_parts = []
        elif lowered_tag == "h1" and self._h1_parts is not None:
            text = clean_text("".join(self._h1_parts))
            if text:
                self.h1_texts.append(text)
            self._h1_parts = None
        elif lowered_tag == "time" and self._time_parts is not None:
            text = clean_text(self._current_time_datetime or "".join(self._time_parts))
            if text:
                self.time_values.append(text)
            self._time_parts = None
            self._current_time_datetime = ""
        elif lowered_tag == "script" and self._script_parts is not None:
            text = "".join(self._script_parts).strip()
            if text:
                self.json_ld_scripts.append(text)
            self._script_parts = None

        if self.collect_content and self._content_tag == lowered_tag and self._content_parts is not None:
            text = clean_text("".join(self._content_parts))
            if text:
                self.content_blocks.append(text)
            self._content_tag = None
            self._content_parts = None

        if self.collect_content and self._ignore_depth > 0:
            self._ignore_depth -= 1
        elif lowered_tag == "article" and self._article_depth > 0:
            self._article_depth -= 1
        elif lowered_tag == "main" and self._main_depth > 0:
            self._main_depth -= 1

        if self.tag_stack and self.tag_stack[-1] == lowered_tag:
            self.tag_stack.pop()
        elif lowered_tag in self.tag_stack:
            for index in range(len(self.tag_stack) - 1, -1, -1):
                if self.tag_stack[index] == lowered_tag:
                    self.tag_stack.pop(index)
                    break

    def handle_data(self, data: str) -> None:
        if self._current_anchor is not None:
            self._current_anchor["text_parts"].append(data)
        if self._title_parts is not None and "title" in self.tag_stack:
            self._title_parts.append(data)
        if self._h1_parts is not None and "h1" in self.tag_stack:
            self._h1_parts.append(data)
        if self._time_parts is not None and "time" in self.tag_stack:
            self._time_parts.append(data)
        if self._script_parts is not None and "script" in self.tag_stack:
            self._script_parts.append(data)
        if (
            self.collect_content
            and self._content_parts is not None
            and self._inside_content_scope()
            and self._ignore_depth == 0
        ):
            self._content_parts.append(data)

    def _inside_content_scope(self) -> bool:
        return self._article_depth > 0 or self._main_depth > 0

    def _should_ignore_tag(self, tag: str, attrs: dict[str, str]) -> bool:
        if tag in NOISE_TAGS:
            return True
        tokens = self._tokenize_attrs(attrs)
        return any(keyword in tokens for keyword in NOISE_CLASS_KEYWORDS)

    @staticmethod
    def _tokenize_attrs(attrs: dict[str, str]) -> set[str]:
        tokens: set[str] = set()
        for key in ("class", "id"):
            raw = attrs.get(key, "")
            for piece in re.split(r"[\s_-]+", raw.casefold()):
                if piece:
                    tokens.add(piece)
        return tokens


def require_bs4() -> None:
    if BeautifulSoup is None:
        raise ModuleNotFoundError("bs4 is required. Install it with: pip install beautifulsoup4")


def clean_text(text: str) -> str:
    text = str(text or "").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html_tags(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    text = unescape(text)
    return clean_text(text)


def canonicalize_url(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    clean_path = parsed.path or "/"
    if clean_path != "/" and clean_path.endswith("/"):
        clean_path = clean_path.rstrip("/")
    return parsed._replace(query="", fragment="", path=clean_path).geturl()


def get_soup(session: requests.Session, url: str, timeout: float) -> BeautifulSoup:
    require_bs4()
    html = fetch_html(session, url, timeout)
    return BeautifulSoup(html, "html.parser")


def fetch_html(session: requests.Session, url: str, timeout: float) -> str:
    response = session.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def iter_json_objects(value: Any) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    if isinstance(value, dict):
        objects.append(value)
        for nested in value.values():
            objects.extend(iter_json_objects(nested))
    elif isinstance(value, list):
        for item in value:
            objects.extend(iter_json_objects(item))
    return objects


def extract_json_ld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for node in soup.select("script[type='application/ld+json']"):
        raw = node.string or node.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        objects.extend(iter_json_objects(payload))
    return objects


def extract_json_ld_objects_from_blocks(raw_blocks: list[str]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for raw in raw_blocks:
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        objects.extend(iter_json_objects(payload))
    return objects


def get_first_nonempty(values: list[str]) -> str:
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


def meta_candidates_from_parser(
    parser: StdlibPageParser,
    *,
    property_name: str | None = None,
    name: str | None = None,
    itemprop: str | None = None,
) -> list[str]:
    values: list[str] = []
    for attrs in parser.meta_tags:
        matches = True
        if property_name is not None:
            matches = matches and attrs.get("property", "").casefold() == property_name.casefold()
        if name is not None:
            matches = matches and attrs.get("name", "").casefold() == name.casefold()
        if itemprop is not None:
            matches = matches and attrs.get("itemprop", "").casefold() == itemprop.casefold()
        if matches and attrs.get("content"):
            values.append(attrs["content"])
    return values


def extract_title(soup: BeautifulSoup, json_ld_objects: list[dict[str, Any]]) -> str:
    title_candidates: list[str] = []

    h1 = soup.select_one("h1")
    if h1 is not None:
        title_candidates.append(h1.get_text(" ", strip=True))

    for selector in [
        "meta[property='og:title']",
        "meta[name='twitter:title']",
        "meta[name='title']",
    ]:
        node = soup.select_one(selector)
        if node is not None and node.get("content"):
            title_candidates.append(node["content"])

    for obj in json_ld_objects:
        if obj.get("@type") in {"NewsArticle", "Article", "ReportageNewsArticle", "WebPage"}:
            for key in ["headline", "name"]:
                value = obj.get(key)
                if isinstance(value, str):
                    title_candidates.append(value)

    if soup.title is not None:
        title_candidates.append(soup.title.get_text(" ", strip=True))

    title = get_first_nonempty(title_candidates)
    title = re.sub(r"\s+(?:–|-)\s+pv magazine Australia$", "", title, flags=re.IGNORECASE)
    return title


def extract_title_from_parser(parser: StdlibPageParser, json_ld_objects: list[dict[str, Any]]) -> str:
    title_candidates: list[str] = []

    title_candidates.extend(parser.h1_texts)
    title_candidates.extend(meta_candidates_from_parser(parser, property_name="og:title"))
    title_candidates.extend(meta_candidates_from_parser(parser, name="twitter:title"))
    title_candidates.extend(meta_candidates_from_parser(parser, name="title"))

    for obj in json_ld_objects:
        if obj.get("@type") in {"NewsArticle", "Article", "ReportageNewsArticle", "WebPage"}:
            for key in ["headline", "name"]:
                value = obj.get(key)
                if isinstance(value, str):
                    title_candidates.append(value)

    if parser.title_text:
        title_candidates.append(parser.title_text)

    title = get_first_nonempty(title_candidates)
    title = re.sub(r"\s+(?:–|-)\s+pv magazine Australia$", "", title, flags=re.IGNORECASE)
    return title


def parse_datetime_to_dataset_format(raw_value: str) -> str:
    raw = clean_text(raw_value)
    if not raw:
        return ""

    iso_candidate = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_candidate)
        return dt.strftime("%d-%m-%Y %I:%M:%S %p")
    except ValueError:
        pass

    if raw.endswith(" UTC"):
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S UTC")
            return dt.strftime("%d-%m-%Y %I:%M:%S %p")
        except ValueError:
            pass

    for fmt in (
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%d-%m-%Y %I:%M:%S %p",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%d-%m-%Y %I:%M:%S %p")
        except ValueError:
            continue

    return raw


def extract_visible_date_text(soup: BeautifulSoup) -> str:
    scope = soup.select_one("article") or soup.select_one("main") or soup
    text = scope.get_text("\n", strip=True)
    match = VISIBLE_DATE_RE.search(text)
    return match.group(0) if match else ""


def extract_visible_date_text_from_html(html: str) -> str:
    match = VISIBLE_DATE_RE.search(strip_html_tags(html))
    return match.group(0) if match else ""


def extract_date(soup: BeautifulSoup, json_ld_objects: list[dict[str, Any]]) -> str:
    date_candidates: list[str] = []

    for selector in [
        "meta[property='article:published_time']",
        "meta[property='article:modified_time']",
        "meta[name='date']",
        "meta[itemprop='datePublished']",
        "time[datetime]",
    ]:
        for node in soup.select(selector):
            value = node.get("content") or node.get("datetime") or node.get_text(" ", strip=True)
            if value:
                date_candidates.append(value)

    for obj in json_ld_objects:
        if obj.get("@type") in {"NewsArticle", "Article", "ReportageNewsArticle", "WebPage"}:
            for key in ["datePublished", "dateModified", "uploadDate"]:
                value = obj.get(key)
                if isinstance(value, str):
                    date_candidates.append(value)

    visible_date = extract_visible_date_text(soup)
    if visible_date:
        date_candidates.append(visible_date)

    for raw in date_candidates:
        formatted = parse_datetime_to_dataset_format(raw)
        if formatted:
            return formatted
    return ""


def extract_date_from_parser(
    parser: StdlibPageParser,
    json_ld_objects: list[dict[str, Any]],
    html: str,
) -> str:
    date_candidates: list[str] = []
    date_candidates.extend(meta_candidates_from_parser(parser, property_name="article:published_time"))
    date_candidates.extend(meta_candidates_from_parser(parser, property_name="article:modified_time"))
    date_candidates.extend(meta_candidates_from_parser(parser, name="date"))
    date_candidates.extend(meta_candidates_from_parser(parser, itemprop="datePublished"))
    date_candidates.extend(parser.time_values)

    for obj in json_ld_objects:
        if obj.get("@type") in {"NewsArticle", "Article", "ReportageNewsArticle", "WebPage"}:
            for key in ["datePublished", "dateModified", "uploadDate"]:
                value = obj.get(key)
                if isinstance(value, str):
                    date_candidates.append(value)

    visible_date = extract_visible_date_text_from_html(html)
    if visible_date:
        date_candidates.append(visible_date)

    for raw in date_candidates:
        formatted = parse_datetime_to_dataset_format(raw)
        if formatted:
            return formatted
    return ""


def extract_json_ld_article_body(json_ld_objects: list[dict[str, Any]]) -> str:
    for obj in json_ld_objects:
        if obj.get("@type") not in {"NewsArticle", "Article", "ReportageNewsArticle"}:
            continue
        body = obj.get("articleBody")
        if isinstance(body, str):
            text = clean_text(body)
            if len(text) >= 80:
                return text
    return ""


def should_stop_on_text(text: str) -> bool:
    lowered = text.casefold()
    return any(stop.casefold() in lowered for stop in STOP_PHRASES)


def trim_content_tail(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    cut_index = len(cleaned)
    lowered = cleaned.casefold()
    for stop in STOP_PHRASES:
        idx = lowered.find(stop.casefold())
        if idx != -1:
            cut_index = min(cut_index, idx)
    return cleaned[:cut_index].strip()


def extract_text_from_node(node: BeautifulSoup) -> str:
    paragraphs: list[str] = []
    for tag in node.find_all(["h2", "h3", "p", "li", "blockquote"]):
        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            continue
        if should_stop_on_text(text):
            break
        paragraphs.append(text)
    return "\n".join(paragraphs).strip()


def extract_content(soup: BeautifulSoup, json_ld_objects: list[dict[str, Any]]) -> str:
    content = extract_json_ld_article_body(json_ld_objects)
    if content:
        return trim_content_tail(content)

    require_bs4()
    best_text = ""
    for selector in CONTENT_SELECTORS:
        for node in soup.select(selector):
            work_node = BeautifulSoup(str(node), "html.parser")
            for noise_selector in NOISE_SELECTORS:
                for noise in work_node.select(noise_selector):
                    noise.decompose()
            text = extract_text_from_node(work_node)
            if len(text) > len(best_text):
                best_text = text

    if best_text:
        return trim_content_tail(best_text)

    meta_desc = (
        soup.find("meta", attrs={"property": "og:description"})
        or soup.find("meta", attrs={"name": "description"})
    )
    if meta_desc is not None and meta_desc.get("content"):
        return trim_content_tail(meta_desc["content"])
    return ""


def extract_content_from_parser(
    parser: StdlibPageParser,
    json_ld_objects: list[dict[str, Any]],
) -> str:
    content = extract_json_ld_article_body(json_ld_objects)
    if content:
        return trim_content_tail(content)

    blocks: list[str] = []
    for block in parser.content_blocks:
        if should_stop_on_text(block):
            break
        blocks.append(block)

    if blocks:
        return trim_content_tail("\n".join(blocks))

    meta_desc_candidates = meta_candidates_from_parser(parser, property_name="og:description")
    meta_desc_candidates.extend(meta_candidates_from_parser(parser, name="description"))
    meta_desc = get_first_nonempty(meta_desc_candidates)
    if meta_desc:
        return trim_content_tail(meta_desc)
    return ""


def is_article_url(url: str) -> bool:
    return bool(ARTICLE_URL_RE.match(canonicalize_url(url)))


def is_listing_url(url: str) -> bool:
    return bool(LISTING_URL_RE.match(canonicalize_url(url)))


def page_number_from_url(url: str) -> int:
    normalized = canonicalize_url(url)
    match = re.search(r"/news/page/(\d+)$", normalized)
    if match:
        return int(match.group(1))
    return 1


def build_listing_page_url(base_url: str, page: int) -> str:
    root = canonicalize_url(base_url)
    if page <= 1:
        return root
    return f"{root}/page/{page}"


def build_listing_page_urls(base_url: str, start_page: int, end_page: int) -> list[str]:
    if start_page <= 0 or end_page <= 0:
        raise ValueError("start_page and end_page must be positive integers")
    if start_page > end_page:
        raise ValueError("start_page must be less than or equal to end_page")
    return [build_listing_page_url(base_url, page) for page in range(start_page, end_page + 1)]


def extract_listing_article_urls(soup: BeautifulSoup, page_url: str) -> list[str]:
    urls: list[str] = []

    for anchor in soup.select("h2 a[href]"):
        href = canonicalize_url(urljoin(page_url, anchor.get("href", "")))
        if href and is_article_url(href):
            urls.append(href)

    if urls:
        return dedupe_preserve_order(urls)

    for anchor in soup.select("article a[href], main a[href]"):
        href = canonicalize_url(urljoin(page_url, anchor.get("href", "")))
        if href and is_article_url(href):
            urls.append(href)
    return dedupe_preserve_order(urls)


def extract_listing_article_urls_from_parser(parser: StdlibPageParser, page_url: str) -> list[str]:
    urls: list[str] = []
    for anchor in parser.anchors:
        href = canonicalize_url(urljoin(page_url, str(anchor.get("href", ""))))
        if not href or not is_article_url(href):
            continue
        if anchor.get("in_h2"):
            urls.append(href)

    if urls:
        return dedupe_preserve_order(urls)

    for anchor in parser.anchors:
        href = canonicalize_url(urljoin(page_url, str(anchor.get("href", ""))))
        if href and is_article_url(href):
            urls.append(href)
    return dedupe_preserve_order(urls)


def find_next_listing_url(soup: BeautifulSoup, page_url: str) -> str | None:
    current_page = page_number_from_url(page_url)
    candidates: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = canonicalize_url(urljoin(page_url, anchor["href"]))
        if not is_listing_url(href):
            continue
        text = clean_text(anchor.get_text(" ", strip=True))
        if "next" in text.casefold():
            return href
        candidates.append(href)

    next_page = current_page + 1
    for candidate in candidates:
        if page_number_from_url(candidate) == next_page:
            return candidate
    return None


def find_next_listing_url_from_parser(parser: StdlibPageParser, page_url: str) -> str | None:
    current_page = page_number_from_url(page_url)
    candidates: list[str] = []

    for anchor in parser.anchors:
        href = canonicalize_url(urljoin(page_url, str(anchor.get("href", ""))))
        if not is_listing_url(href):
            continue
        text = clean_text(str(anchor.get("text", "")))
        if "next" in text.casefold():
            return href
        candidates.append(href)

    next_page = current_page + 1
    for candidate in candidates:
        if page_number_from_url(candidate) == next_page:
            return candidate
    return None


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def crawl_listing_urls(
    *,
    session: requests.Session,
    base_url: str,
    timeout: float,
    delay: float,
    start_page: int,
    end_page: int,
    max_articles: int | None,
) -> list[str]:
    article_urls: list[str] = []
    seen_articles: set[str] = set()
    page_urls = build_listing_page_urls(base_url, start_page, end_page)

    for page_url in page_urls:
        if BeautifulSoup is not None:
            soup = get_soup(session, page_url, timeout)
            page_article_urls = extract_listing_article_urls(soup, page_url)
        else:
            html = fetch_html(session, page_url, timeout)
            parser = StdlibPageParser(collect_content=False)
            parser.feed(html)
            page_article_urls = extract_listing_article_urls_from_parser(parser, page_url)

        for url in page_article_urls:
            if url in seen_articles:
                continue
            seen_articles.add(url)
            article_urls.append(url)
            if max_articles is not None and len(article_urls) >= max_articles:
                return article_urls

        print(f"[INFO] LIST {page_url} -> {len(page_article_urls)} articles")
        time.sleep(delay)

    return article_urls


def parse_article(session: requests.Session, url: str, timeout: float) -> dict[str, str]:
    if BeautifulSoup is not None:
        soup = get_soup(session, url, timeout)
        json_ld_objects = extract_json_ld_objects(soup)
        return {
            "title": extract_title(soup, json_ld_objects),
            "date": extract_date(soup, json_ld_objects),
            "content": extract_content(soup, json_ld_objects),
            "url": canonicalize_url(url),
        }

    html = fetch_html(session, url, timeout)
    parser = StdlibPageParser(collect_content=True)
    parser.feed(html)
    json_ld_objects = extract_json_ld_objects_from_blocks(parser.json_ld_scripts)
    return {
        "title": extract_title_from_parser(parser, json_ld_objects),
        "date": extract_date_from_parser(parser, json_ld_objects, html),
        "content": extract_content_from_parser(parser, json_ld_objects),
        "url": canonicalize_url(url),
    }


def load_existing_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    data = json.loads(content)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError(f"Existing file is not a JSON array: {path}")


def merge_records_by_url(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index_by_url: dict[str, int] = {}

    for record in existing:
        url = canonicalize_url(str(record.get("url", "")))
        merged.append(record)
        if url:
            index_by_url[url] = len(merged) - 1

    for record in incoming:
        url = canonicalize_url(str(record.get("url", "")))
        if url and url in index_by_url:
            merged[index_by_url[url]] = record
        else:
            merged.append(record)
            if url:
                index_by_url[url] = len(merged) - 1

    return merged


def crawl_news(
    *,
    output_path: Path,
    failed_output_path: Path,
    base_url: str,
    delay: float,
    timeout: float,
    start_page: int,
    end_page: int,
    max_articles: int | None,
) -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    article_urls = crawl_listing_urls(
        session=session,
        base_url=base_url,
        timeout=timeout,
        delay=delay,
        start_page=start_page,
        end_page=end_page,
        max_articles=max_articles,
    )

    print(f"[INFO] Total article URLs collected: {len(article_urls)}")

    results: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    total = len(article_urls)

    for index, url in enumerate(article_urls, start=1):
        try:
            record = parse_article(session, url, timeout)
            if not record["title"] or not record["content"]:
                raise ValueError("missing_title_or_content")
            results.append(record)
            print(f"[OK] ({index}/{total}) {url}")
        except Exception as exc:
            failures.append({"url": url, "reason": str(exc)})
            print(f"[ERR] ({index}/{total}) {url} -> {exc}")
        time.sleep(delay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = load_existing_json_array(output_path)
    existing_failures = load_existing_json_array(failed_output_path)
    merged_results = merge_records_by_url(existing_results, results)
    merged_failures = merge_records_by_url(existing_failures, failures)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    with failed_output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_failures, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Newly crawled {len(results)} items.")
    print(f"Output items after append/dedupe: {len(merged_results)} -> {output_path}")
    print(f"Failed items after append/dedupe: {len(merged_failures)} -> {failed_output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl pv magazine Australia news pages and export dataset-style JSON."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_NEWS_URL,
        help=f"News listing root URL. Default: {BASE_NEWS_URL}",
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
        help=f"Path for failed article records. Default: {DEFAULT_FAILED_OUTPUT}",
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
        help=f"First listing page to crawl, inclusive. Default: {DEFAULT_START_PAGE}",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=DEFAULT_END_PAGE,
        help=f"Last listing page to crawl, inclusive. Default: {DEFAULT_END_PAGE}",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Optional limit on number of articles to export.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    crawl_news(
        output_path=args.output,
        failed_output_path=args.failed_output,
        base_url=args.base_url,
        delay=args.delay,
        timeout=args.timeout,
        start_page=args.start_page,
        end_page=args.end_page,
        max_articles=args.max_articles,
    )


if __name__ == "__main__":
    main()
