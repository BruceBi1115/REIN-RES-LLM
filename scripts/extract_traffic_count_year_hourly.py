#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


HOUR_COLUMNS = [f"hour_{hour:02d}" for hour in range(24)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one year from the daily traffic-count table and expand it into an hourly CSV. "
            "By default, all matched rows in the same hour are aggregated with sum."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/traffic_count/road_traffic_counts_hourly_permanent3.csv",
        help="Source CSV path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/traffic_count/road_traffic_counts_hourly_permanent3_2024_hourly.csv",
        help="Destination CSV path.",
    )
    parser.add_argument("--year", type=int, default=2024, help="Target year to extract.")
    parser.add_argument(
        "--aggregate",
        type=str,
        default="sum",
        choices=["sum", "mean", "first", "none"],
        help="How to combine multiple matched records within the same hour.",
    )
    parser.add_argument("--station-key", type=str, default="", help="Optional station_key filter.")
    parser.add_argument(
        "--traffic-direction-seq",
        type=str,
        default="",
        help="Optional traffic_direction_seq filter.",
    )
    parser.add_argument(
        "--cardinal-direction-seq",
        type=str,
        default="",
        help="Optional cardinal_direction_seq filter.",
    )
    parser.add_argument(
        "--classification-seq",
        type=str,
        default="",
        help="Optional classification_seq filter.",
    )
    return parser


def _clean_text(value: str | None) -> str:
    return str(value or "").strip()


def _parse_day(value: str | None) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    date_part = text[:10]
    try:
        return datetime.strptime(date_part, "%Y-%m-%d")
    except ValueError:
        return None


def _parse_count(value: str | None) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _row_matches(row: dict[str, str], args) -> bool:
    if _clean_text(row.get("year")) != str(int(args.year)):
        return False

    filters = {
        "station_key": _clean_text(args.station_key),
        "traffic_direction_seq": _clean_text(args.traffic_direction_seq),
        "cardinal_direction_seq": _clean_text(args.cardinal_direction_seq),
        "classification_seq": _clean_text(args.classification_seq),
    }
    for key, expected in filters.items():
        if expected and _clean_text(row.get(key)) != expected:
            return False
    return True


def _format_count(value: float | None) -> str:
    if value is None:
        return ""
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _iter_expanded_rows(input_path: Path, args) -> Iterable[dict[str, object]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not _row_matches(row, args):
                continue
            day = _parse_day(row.get("date"))
            if day is None:
                continue
            base_payload = {
                "record_id": _clean_text(row.get("record_id")),
                "station_key": _clean_text(row.get("station_key")),
                "traffic_direction_seq": _clean_text(row.get("traffic_direction_seq")),
                "cardinal_direction_seq": _clean_text(row.get("cardinal_direction_seq")),
                "classification_seq": _clean_text(row.get("classification_seq")),
            }
            for hour, hour_col in enumerate(HOUR_COLUMNS):
                traffic_count = _parse_count(row.get(hour_col))
                yield {
                    **base_payload,
                    "time": day + timedelta(hours=hour),
                    "traffic_count": traffic_count,
                }


def _write_aggregated_csv(output_path: Path, args, expanded_rows: list[dict[str, object]]) -> None:
    start = datetime(int(args.year), 1, 1, 0, 0, 0)
    end = datetime(int(args.year) + 1, 1, 1, 0, 0, 0)

    if args.aggregate == "sum":
        sums: dict[datetime, float] = defaultdict(float)
        has_value: dict[datetime, bool] = defaultdict(bool)
        for row in expanded_rows:
            value = row["traffic_count"]
            if value is None:
                continue
            ts = row["time"]
            sums[ts] += float(value)
            has_value[ts] = True

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "traffic_count"])
            ts = start
            while ts < end:
                writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), _format_count(sums[ts] if has_value[ts] else None)])
                ts += timedelta(hours=1)
        return

    if args.aggregate == "mean":
        sums: dict[datetime, float] = defaultdict(float)
        counts: dict[datetime, int] = defaultdict(int)
        for row in expanded_rows:
            value = row["traffic_count"]
            if value is None:
                continue
            ts = row["time"]
            sums[ts] += float(value)
            counts[ts] += 1

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "traffic_count"])
            ts = start
            while ts < end:
                value = (sums[ts] / counts[ts]) if counts[ts] > 0 else None
                writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), _format_count(value)])
                ts += timedelta(hours=1)
        return

    if args.aggregate == "first":
        first_value: dict[datetime, float] = {}
        for row in expanded_rows:
            value = row["traffic_count"]
            ts = row["time"]
            if value is None or ts in first_value:
                continue
            first_value[ts] = float(value)

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "traffic_count"])
            ts = start
            while ts < end:
                writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), _format_count(first_value.get(ts))])
                ts += timedelta(hours=1)
        return

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "traffic_count",
                "record_id",
                "station_key",
                "traffic_direction_seq",
                "cardinal_direction_seq",
                "classification_seq",
            ]
        )
        for row in sorted(expanded_rows, key=lambda item: (item["time"], item["station_key"], item["record_id"])):
            writer.writerow(
                [
                    row["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    _format_count(row["traffic_count"]),
                    row["record_id"],
                    row["station_key"],
                    row["traffic_direction_seq"],
                    row["cardinal_direction_seq"],
                    row["classification_seq"],
                ]
            )


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    expanded_rows = list(_iter_expanded_rows(input_path, args))
    if not expanded_rows:
        raise SystemExit(
            "No matching rows were found. "
            f"year={args.year} station_key={_clean_text(args.station_key) or '*'} "
            f"traffic_direction_seq={_clean_text(args.traffic_direction_seq) or '*'} "
            f"cardinal_direction_seq={_clean_text(args.cardinal_direction_seq) or '*'} "
            f"classification_seq={_clean_text(args.classification_seq) or '*'}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_aggregated_csv(output_path, args, expanded_rows)

    print(
        "Done. "
        f"matched_daily_rows={len(expanded_rows) // 24} "
        f"expanded_hourly_rows={len(expanded_rows)} "
        f"aggregate={args.aggregate} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
