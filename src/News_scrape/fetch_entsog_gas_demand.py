from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests


BASE_API_URL = "https://transparency.entsog.eu/api/v1"
FINAL_CONSUMER_POINT_TYPE = "Aggregated Point - Final Consumers"
DEFAULT_COUNTRY = "DE"
DEFAULT_YEAR = 2025
DEFAULT_PERIOD_TYPE = "day"
DEFAULT_INDICATOR = "Physical Flow"
DEFAULT_TIMEZONE = "WET"
DEFAULT_TIMEOUT = 60.0
DEFAULT_DELAY = 0.2
DEFAULT_CHUNK_SIZE = 25
DEFAULT_RETRIES = 3
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ENTSOG-Gas-Demand-Downloader/1.0; "
        "+https://example.com)"
    )
}

COUNTRY_ALIASES = {
    "DE": "DE",
    "DEU": "DE",
    "GERMANY": "DE",
    "NL": "NL",
    "NLD": "NL",
    "NETHERLANDS": "NL",
    "THE NETHERLANDS": "NL",
    "HOLLAND": "NL",
    "UK": "UK",
    "GB": "UK",
    "GBR": "UK",
    "UNITED KINGDOM": "UK",
    "GREAT BRITAIN": "UK",
}

COUNTRY_NAMES = {
    "DE": "Germany",
    "NL": "Netherlands",
    "UK": "United Kingdom",
}


@dataclass(frozen=True)
class MonthWindow:
    index: int
    start: date
    end: date


def normalize_country_code(value: str) -> str:
    key = str(value or "").strip().upper()
    if not key:
        raise ValueError("country cannot be empty")
    return COUNTRY_ALIASES.get(key, key)


def default_output_paths(country_code: str, year: int, period_type: str) -> tuple[Path, Path, Path]:
    stem = f"entsog_gas_demand_{country_code.lower()}_{year}_{period_type}"
    return (
        Path("dataset") / f"{stem}.csv",
        Path("dataset") / f"{stem}_raw.csv",
        Path("dataset") / f"{stem}_points.json",
    )


def iter_month_windows(year: int, max_months: int | None = None) -> list[MonthWindow]:
    windows: list[MonthWindow] = []
    cursor = date(int(year), 1, 1)
    limit = int(max_months) if max_months is not None else None
    index = 0
    while cursor.year == int(year):
        month_start = cursor
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        month_end = next_month - timedelta(days=1)
        windows.append(MonthWindow(index=index + 1, start=month_start, end=month_end))
        index += 1
        cursor = next_month
        if limit is not None and index >= limit:
            break
    return windows


def chunked(values: list[str], chunk_size: int) -> Iterable[list[str]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(values), size):
        yield values[start : start + size]


def request_json(
    session: requests.Session,
    endpoint: str,
    *,
    params: dict[str, Any],
    timeout: float,
    retries: int,
    delay: float,
) -> dict[str, Any]:
    url = f"{BASE_API_URL}/{endpoint}"
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"unexpected payload type for {endpoint}: {type(payload)!r}")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max(1, int(retries)):
                break
            sleep_s = delay * attempt
            print(f"[WARN] {endpoint} attempt={attempt} failed: {exc}; retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    assert last_error is not None
    raise last_error


def unwrap_items(payload: dict[str, Any], preferred_key: str) -> list[dict[str, Any]]:
    candidate = payload.get(preferred_key)
    if isinstance(candidate, list):
        return [item for item in candidate if isinstance(item, dict)]
    for key, value in payload.items():
        if key == "meta":
            continue
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "none", "null"}


def build_point_direction(item: dict[str, Any]) -> str:
    operator_key = str(item.get("operatorKey") or "").strip()
    point_key = str(item.get("pointKey") or "").strip()
    direction_key = str(item.get("directionKey") or "").strip()
    if not operator_key or not point_key or not direction_key:
        return ""
    return f"{operator_key}{point_key}{direction_key}"


def fetch_final_consumer_points(
    session: requests.Session,
    *,
    country_code: str,
    timeout: float,
    retries: int,
    delay: float,
) -> list[dict[str, Any]]:
    payload = request_json(
        session,
        "operatorpointdirections",
        params={
            "pointType": FINAL_CONSUMER_POINT_TYPE,
            "tSOCountry": country_code,
            "hasData": 1,
            "limit": -1,
        },
        timeout=timeout,
        retries=retries,
        delay=delay,
    )
    items = unwrap_items(payload, "operatorpointdirections")
    deduped: dict[str, dict[str, Any]] = {}
    for item in items:
        if str(item.get("pointType") or "").strip() != FINAL_CONSUMER_POINT_TYPE:
            continue
        if str(item.get("directionKey") or "").strip().lower() != "exit":
            continue
        if str(item.get("tSOCountry") or "").strip().upper() != country_code:
            continue
        if not is_truthy(item.get("hasData")):
            continue
        if is_truthy(item.get("isInvalid")):
            continue
        point_direction = build_point_direction(item)
        if not point_direction:
            continue
        enriched = dict(item)
        enriched["pointDirection"] = point_direction
        deduped[point_direction] = enriched
    points = sorted(
        deduped.values(),
        key=lambda item: (
            str(item.get("operatorKey") or ""),
            str(item.get("pointKey") or ""),
            str(item.get("directionKey") or ""),
        ),
    )
    return points


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_iso_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def fetch_operational_rows(
    session: requests.Session,
    *,
    point_directions: list[str],
    indicator: str,
    period_type: str,
    window: MonthWindow,
    time_zone: str,
    timeout: float,
    retries: int,
    delay: float,
    chunk_size: int,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunked(point_directions, chunk_size), start=1):
        params = {
            "pointDirection": ",".join(chunk),
            "indicator": indicator,
            "periodType": period_type,
            "from": window.start.isoformat(),
            "to": window.end.isoformat(),
            "timeZone": time_zone,
            "includeExemptions": 0,
            "sorting": "periodFrom",
            "limit": -1,
        }
        payload = request_json(
            session,
            "operationaldatas",
            params=params,
            timeout=timeout,
            retries=retries,
            delay=delay,
        )
        rows = unwrap_items(payload, "operationaldatas")
        print(
            f"[INFO] {window.start.isoformat()}..{window.end.isoformat()} "
            f"chunk={chunk_index} points={len(chunk)} rows={len(rows)}"
        )
        all_rows.extend(rows)
        time.sleep(delay)
    return all_rows


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    country_code: str,
    country_name: str,
    indicator: str,
    period_type: str,
    year: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []

    for row in rows:
        value = parse_float(row.get("value"))
        if value is None:
            continue
        period_from = str(row.get("periodFrom") or "").strip()
        period_to = str(row.get("periodTo") or "").strip()
        if not period_from or not period_to:
            continue
        period_from_dt = parse_iso_dt(period_from)
        if period_from_dt.year != int(year):
            continue

        unit = str(row.get("unit") or "").strip()
        operator_key = str(row.get("operatorKey") or "").strip()
        point_key = str(row.get("pointKey") or "").strip()
        direction_key = str(row.get("directionKey") or "").strip()
        point_direction = f"{operator_key}{point_key}{direction_key}"

        raw_rows.append(
            {
                "country_code": country_code,
                "country_name": country_name,
                "year": year,
                "indicator": indicator,
                "period_type": period_type,
                "period_from": period_from,
                "period_to": period_to,
                "operator_key": operator_key,
                "operator_label": str(row.get("operatorLabel") or "").strip(),
                "point_key": point_key,
                "point_label": str(row.get("pointLabel") or "").strip(),
                "direction_key": direction_key,
                "point_direction": point_direction,
                "unit": unit,
                "value": value,
                "flow_status": str(row.get("flowStatus") or "").strip(),
            }
        )

        key = (period_from, period_to, unit)
        slot = aggregates.setdefault(
            key,
            {
                "country_code": country_code,
                "country_name": country_name,
                "year": year,
                "indicator": indicator,
                "period_type": period_type,
                "period_from": period_from,
                "period_to": period_to,
                "unit": unit,
                "total_value": 0.0,
                "point_directions": set(),
                "record_count": 0,
            },
        )
        slot["total_value"] += value
        slot["record_count"] += 1
        slot["point_directions"].add(point_direction)

    aggregate_rows_out: list[dict[str, Any]] = []
    for slot in sorted(aggregates.values(), key=lambda item: item["period_from"]):
        point_directions = sorted(str(item) for item in slot.pop("point_directions"))
        slot["total_value"] = round(float(slot["total_value"]), 6)
        slot["point_count"] = len(point_directions)
        slot["point_directions"] = "|".join(point_directions)
        aggregate_rows_out.append(slot)

    raw_rows.sort(key=lambda item: (item["period_from"], item["operator_key"], item["point_key"]))
    return aggregate_rows_out, raw_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download ENTSOG Transparency Platform final-consumer gas-demand proxy "
            "series for a single country and year."
        )
    )
    parser.add_argument(
        "--country",
        type=str,
        default=DEFAULT_COUNTRY,
        help="Country code or alias. Examples: DE, Germany, NL, UK. Default: DE",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Calendar year to download. Default: {DEFAULT_YEAR}",
    )
    parser.add_argument(
        "--period-type",
        type=str,
        default=DEFAULT_PERIOD_TYPE,
        choices=("day", "hour"),
        help=f"ENTSOG periodType. Default: {DEFAULT_PERIOD_TYPE}",
    )
    parser.add_argument(
        "--indicator",
        type=str,
        default=DEFAULT_INDICATOR,
        help=(
            "ENTSOG indicator to use. Exact case matters. "
            f"Default: {DEFAULT_INDICATOR}"
        ),
    )
    parser.add_argument(
        "--time-zone",
        type=str,
        default=DEFAULT_TIMEZONE,
        help=f"ENTSOG timeZone parameter. Default: {DEFAULT_TIMEZONE}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Number of pointDirections per API call. Default: {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between API calls in seconds. Default: {DEFAULT_DELAY}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retry count per request. Default: {DEFAULT_RETRIES}",
    )
    parser.add_argument(
        "--max-months",
        type=int,
        default=None,
        help="Optional debug cap on how many months to download.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Aggregated output CSV path. Default: dataset/entsog_gas_demand_<country>_<year>_<period>.csv",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Point-level raw output CSV path.",
    )
    parser.add_argument(
        "--points-output",
        type=Path,
        default=None,
        help="Point metadata JSON path.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    country_code = normalize_country_code(args.country)
    country_name = COUNTRY_NAMES.get(country_code, country_code)
    output_path, raw_output_path, points_output_path = default_output_paths(
        country_code,
        int(args.year),
        str(args.period_type),
    )
    if args.output is not None:
        output_path = args.output
    if args.raw_output is not None:
        raw_output_path = args.raw_output
    if args.points_output is not None:
        points_output_path = args.points_output

    session = requests.Session()
    session.headers.update(HEADERS)

    print(
        f"[INFO] country={country_code} ({country_name}) year={args.year} "
        f"period_type={args.period_type} indicator={args.indicator}"
    )
    points = fetch_final_consumer_points(
        session,
        country_code=country_code,
        timeout=float(args.timeout),
        retries=int(args.retries),
        delay=float(args.delay),
    )
    if not points:
        raise SystemExit(f"No final-consumer pointDirections found for country={country_code}")

    point_directions = [str(item["pointDirection"]) for item in points]
    print(f"[INFO] discovered {len(point_directions)} exit pointDirections")

    all_rows: list[dict[str, Any]] = []
    windows = iter_month_windows(int(args.year), args.max_months)
    for window in windows:
        rows = fetch_operational_rows(
            session,
            point_directions=point_directions,
            indicator=str(args.indicator),
            period_type=str(args.period_type),
            window=window,
            time_zone=str(args.time_zone),
            timeout=float(args.timeout),
            retries=int(args.retries),
            delay=float(args.delay),
            chunk_size=int(args.chunk_size),
        )
        all_rows.extend(rows)

    aggregate_rows_out, raw_rows = aggregate_rows(
        all_rows,
        country_code=country_code,
        country_name=country_name,
        indicator=str(args.indicator),
        period_type=str(args.period_type),
        year=int(args.year),
    )

    points_payload = {
        "source": BASE_API_URL,
        "country_code": country_code,
        "country_name": country_name,
        "year": int(args.year),
        "period_type": str(args.period_type),
        "indicator": str(args.indicator),
        "point_type": FINAL_CONSUMER_POINT_TYPE,
        "point_count": len(points),
        "points": points,
    }

    write_csv(output_path, aggregate_rows_out)
    write_csv(raw_output_path, raw_rows)
    write_json(points_output_path, points_payload)

    print(f"[DONE] aggregate rows={len(aggregate_rows_out)} -> {output_path}")
    print(f"[DONE] raw rows={len(raw_rows)} -> {raw_output_path}")
    print(f"[DONE] point metadata={len(points)} -> {points_output_path}")


if __name__ == "__main__":
    main()
