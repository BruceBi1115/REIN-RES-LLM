#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path


MISSING_TOKENS = {"", "na", "nan", "none", "null"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-fill missing values in a regular hourly CSV. By default, "
            "only complete missing runs of at most three hours are filled."
        )
    )
    parser.add_argument(
        "--input",
        default=(
            "dataset/EPA_criteria_gas_2025/"
            "sacramento_ozone_site0012_poc1_hourly_2025.csv"
        ),
        help="Input hourly CSV.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV. Defaults to INPUT_ffill_maxN.csv.",
    )
    parser.add_argument("--time-column", default="date")
    parser.add_argument("--value-column", default="ozone_ppm")
    parser.add_argument(
        "--max-gap-hours",
        type=int,
        default=3,
        help=(
            "Fill a missing run only when its full length is at most this value. "
            "Use 0 for unlimited forward fill."
        ),
    )
    parser.add_argument(
        "--mask-column",
        default="is_imputed",
        help="Column marking values created by forward fill.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not add the imputation-mask column.",
    )
    return parser


def is_missing(value: str | None) -> bool:
    return str(value or "").strip().lower() in MISSING_TOKENS


def parse_timestamp(value: str, column: str, row_number: int) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid timestamp in column {column!r} at CSV row {row_number}: {value!r}"
        ) from exc


def validate_numeric(value: str, column: str, row_number: int) -> None:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value in column {column!r} at CSV row {row_number}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite value in column {column!r} at CSV row {row_number}: {value!r}"
        )


def default_output_path(input_path: Path, max_gap_hours: int) -> Path:
    suffix = "all" if max_gap_hours == 0 else str(max_gap_hours)
    return input_path.with_name(f"{input_path.stem}_ffill_max{suffix}{input_path.suffix}")


def read_and_validate(
    input_path: Path,
    time_column: str,
    value_column: str,
) -> tuple[list[str], list[dict[str, str]], list[datetime]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fieldnames = list(reader.fieldnames or [])
        missing_columns = {time_column, value_column} - set(fieldnames)
        if missing_columns:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing_columns)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV has no data rows: {input_path}")

    timestamps: list[datetime] = []
    previous: datetime | None = None
    for row_number, row in enumerate(rows, start=2):
        timestamp = parse_timestamp(row[time_column], time_column, row_number)
        if previous is not None and timestamp - previous != timedelta(hours=1):
            raise ValueError(
                "Input timestamps are not strictly hourly and contiguous: "
                f"{previous.isoformat()} -> {timestamp.isoformat()} at CSV row {row_number}."
            )
        previous = timestamp
        timestamps.append(timestamp)

        value = str(row[value_column] or "").strip()
        if not is_missing(value):
            validate_numeric(value, value_column, row_number)
            row[value_column] = value

    return fieldnames, rows, timestamps


def forward_fill_runs(
    rows: list[dict[str, str]],
    value_column: str,
    mask_column: str,
    max_gap_hours: int,
) -> dict[str, int]:
    original_missing = 0
    filled_values = 0
    gap_runs = 0
    filled_gap_runs = 0
    skipped_gap_runs = 0
    max_gap = 0
    last_observed_value: str | None = None
    index = 0

    for row in rows:
        row[mask_column] = "0"

    while index < len(rows):
        value = rows[index][value_column]
        if not is_missing(value):
            last_observed_value = value
            index += 1
            continue

        gap_start = index
        while index < len(rows) and is_missing(rows[index][value_column]):
            index += 1
        gap_length = index - gap_start
        original_missing += gap_length
        gap_runs += 1
        max_gap = max(max_gap, gap_length)

        within_limit = max_gap_hours == 0 or gap_length <= max_gap_hours
        if last_observed_value is None or not within_limit:
            skipped_gap_runs += 1
            continue

        filled_gap_runs += 1
        for gap_index in range(gap_start, index):
            rows[gap_index][value_column] = last_observed_value
            rows[gap_index][mask_column] = "1"
            filled_values += 1

    return {
        "total_rows": len(rows),
        "original_missing_values": original_missing,
        "filled_values": filled_values,
        "remaining_missing_values": original_missing - filled_values,
        "gap_runs": gap_runs,
        "filled_gap_runs": filled_gap_runs,
        "skipped_gap_runs": skipped_gap_runs,
        "maximum_gap_hours": max_gap,
    }


def write_output(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    mask_column: str,
    include_mask: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = list(fieldnames)
    if include_mask and mask_column not in output_fields:
        output_fields.append(mask_column)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.max_gap_hours < 0:
        raise ValueError("--max-gap-hours must be zero or a positive integer.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(input_path, args.max_gap_hours)
    )
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output must differ from input; the source file is never overwritten.")

    fieldnames, rows, timestamps = read_and_validate(
        input_path,
        args.time_column,
        args.value_column,
    )
    stats = forward_fill_runs(
        rows,
        args.value_column,
        args.mask_column,
        args.max_gap_hours,
    )
    write_output(
        output_path,
        fieldnames,
        rows,
        args.mask_column,
        include_mask=not args.no_mask,
    )

    metadata = {
        "source": str(input_path),
        "output": str(output_path),
        "time_column": args.time_column,
        "value_column": args.value_column,
        "mask_column": None if args.no_mask else args.mask_column,
        "method": "forward_fill",
        "max_gap_hours": None if args.max_gap_hours == 0 else args.max_gap_hours,
        "unlimited_fill": args.max_gap_hours == 0,
        "time_start": timestamps[0].isoformat(sep=" "),
        "time_end": timestamps[-1].isoformat(sep=" "),
        **stats,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    print(
        "Done. "
        f"rows={stats['total_rows']} "
        f"original_missing={stats['original_missing_values']} "
        f"filled={stats['filled_values']} "
        f"remaining_missing={stats['remaining_missing_values']} "
        f"max_gap={stats['maximum_gap_hours']}h "
        f"output={output_path} metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
