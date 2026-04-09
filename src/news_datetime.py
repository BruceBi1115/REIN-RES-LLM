from __future__ import annotations

import re
from datetime import datetime
from typing import Any


_COMMON_NEWS_DATETIME_FORMATS = (
    "%Y-%m-%d %I:%M:%S %p",
    "%Y-%m-%d %I:%M %p",
    "%Y/%m/%d %I:%M:%S %p",
    "%Y/%m/%d %I:%M %p",
    "%d-%m-%Y %I:%M:%S %p",
    "%d-%m-%Y %I:%M %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M %p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%b %d, %Y %I:%M:%S %p",
    "%b %d, %Y %I:%M %p",
    "%B %d, %Y %I:%M:%S %p",
    "%B %d, %Y %I:%M %p",
    "%b %d, %Y %H:%M:%S",
    "%b %d, %Y %H:%M",
    "%B %d, %Y %H:%M:%S",
    "%B %d, %Y %H:%M",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
)


def _normalize_datetime_value(dt: Any) -> datetime | None:
    if dt is None:
        return None
    if hasattr(dt, "to_pydatetime"):
        try:
            dt = dt.to_pydatetime()
        except Exception:
            pass
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def parse_news_datetime(raw: Any, *, dayfirst: bool = False) -> datetime | None:
    existing = _normalize_datetime_value(raw)
    if existing is not None:
        return existing

    text = re.sub(r"\s+", " ", str(raw or "").strip())
    if not text:
        return None

    iso_candidate = text.replace("Z", "+00:00")
    try:
        return _normalize_datetime_value(datetime.fromisoformat(iso_candidate))
    except Exception:
        pass

    for fmt in _COMMON_NEWS_DATETIME_FORMATS:
        candidate = iso_candidate if "%z" in fmt else text
        try:
            return _normalize_datetime_value(datetime.strptime(candidate, fmt))
        except Exception:
            continue

    return None


def normalize_news_datetime(raw: Any, *, dayfirst: bool = False, floor: str = "s") -> str:
    dt = parse_news_datetime(raw, dayfirst=dayfirst)
    if dt is None:
        return ""
    if floor == "s":
        dt = dt.replace(microsecond=0)
    return dt.isoformat()
