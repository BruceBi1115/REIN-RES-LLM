#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


OZONE_PARAMETER_CODE = "44201"
OZONE_UNIT = "Parts per million"
HOURS_PER_NON_LEAP_YEAR = 365 * 24


@dataclass(frozen=True)
class Sample:
    gmt_time: datetime
    local_time: datetime
    value: float
    qualifier: str
    method_code: str


@dataclass
class MonitorGroup:
    site_num: str
    poc: str
    latitude: str = ""
    longitude: str = ""
    parameter_name: str = ""
    rows: int = 0
    invalid_measurements: int = 0
    units: set[str] = field(default_factory=set)
    method_types: set[str] = field(default_factory=set)
    method_codes: set[str] = field(default_factory=set)
    qualified_rows: int = 0
    samples: list[Sample] = field(default_factory=list)

    @property
    def unique_gmt_hours(self) -> int:
        return len({sample.gmt_time for sample in self.samples})

    @property
    def duplicate_hours(self) -> int:
        return len(self.samples) - self.unique_gmt_hours


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one Sacramento County EPA AQS hourly Ozone monitor/POC "
            "into a model-ready hourly CSV. If site/POC are omitted, the monitor "
            "with the most unique valid hours is selected."
        )
    )
    parser.add_argument(
        "--input",
        default="dataset/EPA_criteria_gas_2025/hourly_44201_2025.csv",
        help="EPA AirData hourly Ozone CSV.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output CSV. By default, a filename containing the selected site and "
            "POC is created beside the input file."
        ),
    )
    parser.add_argument(
        "--coverage-output",
        default="",
        help="Coverage report CSV. Defaults to the input directory.",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--state-code", default="06", help="AQS state code; 06 is California.")
    parser.add_argument("--county-code", default="067", help="AQS county code; 067 is Sacramento.")
    parser.add_argument("--parameter-code", default=OZONE_PARAMETER_CODE)
    parser.add_argument(
        "--site-num",
        default="",
        help="Optional AQS site number. Omit to select by coverage.",
    )
    parser.add_argument(
        "--poc",
        default="",
        help="Optional Parameter Occurrence Code. Omit to select by coverage.",
    )
    parser.add_argument(
        "--time-basis",
        choices=("gmt", "local"),
        default="gmt",
        help="Timestamp basis in the output. GMT is recommended for news alignment.",
    )
    return parser


def normalize_code(value: str, width: int) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(width)


def parse_time(date_value: str, time_value: str) -> datetime | None:
    text = f"{str(date_value or '').strip()} {str(time_value or '').strip()}"
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def parse_measurement(value: str) -> float | None:
    try:
        parsed = float(str(value or "").strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def source_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def ensure_complete_line(path: Path) -> None:
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {path}")
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        if handle.read(1) not in (b"\n", b"\r"):
            raise ValueError(
                "Input CSV does not end with a complete line. "
                "The download may still be running or may have been truncated."
            )


def required_columns() -> set[str]:
    return {
        "State Code",
        "County Code",
        "Site Num",
        "Parameter Code",
        "POC",
        "Latitude",
        "Longitude",
        "Parameter Name",
        "Date Local",
        "Time Local",
        "Date GMT",
        "Time GMT",
        "Sample Measurement",
        "Units of Measure",
        "Qualifier",
        "Method Type",
        "Method Code",
    }


def collect_groups(path: Path, args: argparse.Namespace) -> dict[tuple[str, str], MonitorGroup]:
    groups: dict[tuple[str, str], MonitorGroup] = {}
    state_code = normalize_code(args.state_code, 2)
    county_code = normalize_code(args.county_code, 3)
    parameter_code = normalize_code(args.parameter_code, 5)
    requested_site = normalize_code(args.site_num, 4) if args.site_num else ""
    requested_poc = str(args.poc).strip()

    initial_signature = source_signature(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            missing = required_columns() - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

            for row in reader:
                if normalize_code(row["State Code"], 2) != state_code:
                    continue
                if normalize_code(row["County Code"], 3) != county_code:
                    continue
                if normalize_code(row["Parameter Code"], 5) != parameter_code:
                    continue
                if not str(row["Date Local"] or "").startswith(f"{args.year:04d}-"):
                    continue

                site_num = normalize_code(row["Site Num"], 4)
                poc = str(row["POC"] or "").strip()
                if requested_site and site_num != requested_site:
                    continue
                if requested_poc and poc != requested_poc:
                    continue

                key = (site_num, poc)
                group = groups.setdefault(key, MonitorGroup(site_num=site_num, poc=poc))
                group.rows += 1
                group.latitude = group.latitude or str(row["Latitude"] or "").strip()
                group.longitude = group.longitude or str(row["Longitude"] or "").strip()
                group.parameter_name = group.parameter_name or str(row["Parameter Name"] or "").strip()

                unit = str(row["Units of Measure"] or "").strip()
                method_type = str(row["Method Type"] or "").strip()
                method_code = str(row["Method Code"] or "").strip()
                qualifier = str(row["Qualifier"] or "").strip()
                if unit:
                    group.units.add(unit)
                if method_type:
                    group.method_types.add(method_type)
                if method_code:
                    group.method_codes.add(method_code)
                if qualifier:
                    group.qualified_rows += 1

                local_time = parse_time(row["Date Local"], row["Time Local"])
                gmt_time = parse_time(row["Date GMT"], row["Time GMT"])
                value = parse_measurement(row["Sample Measurement"])
                if local_time is None or gmt_time is None or value is None:
                    group.invalid_measurements += 1
                    continue
                group.samples.append(
                    Sample(
                        gmt_time=gmt_time,
                        local_time=local_time,
                        value=value,
                        qualifier=qualifier,
                        method_code=method_code,
                    )
                )
    except csv.Error as exc:
        raise ValueError(
            "Malformed or incomplete CSV encountered. "
            "The source download may not have finished."
        ) from exc

    if source_signature(path) != initial_signature:
        raise RuntimeError(
            "Input CSV changed while it was being read. "
            "Wait for the download to finish, then run the script again."
        )
    return groups


def ranked_groups(groups: dict[tuple[str, str], MonitorGroup]) -> list[MonitorGroup]:
    return sorted(
        groups.values(),
        key=lambda group: (
            group.unique_gmt_hours,
            len(group.samples),
            -group.invalid_measurements,
            group.site_num,
            group.poc,
        ),
        reverse=True,
    )


def write_coverage(path: Path, groups: list[MonitorGroup], expected_hours: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "site_num",
                "poc",
                "valid_unique_hours",
                "expected_hours",
                "coverage_ratio",
                "source_rows",
                "invalid_measurements",
                "duplicate_hours",
                "qualified_rows",
                "start_gmt",
                "end_gmt",
                "units",
                "method_types",
                "method_codes",
                "latitude",
                "longitude",
            ]
        )
        for rank, group in enumerate(groups, start=1):
            gmt_times = [sample.gmt_time for sample in group.samples]
            writer.writerow(
                [
                    rank,
                    group.site_num,
                    group.poc,
                    group.unique_gmt_hours,
                    expected_hours,
                    f"{group.unique_gmt_hours / expected_hours:.6f}",
                    group.rows,
                    group.invalid_measurements,
                    group.duplicate_hours,
                    group.qualified_rows,
                    min(gmt_times).strftime("%Y-%m-%d %H:%M:%S") if gmt_times else "",
                    max(gmt_times).strftime("%Y-%m-%d %H:%M:%S") if gmt_times else "",
                    "|".join(sorted(group.units)),
                    "|".join(sorted(group.method_types)),
                    "|".join(sorted(group.method_codes)),
                    group.latitude,
                    group.longitude,
                ]
            )


def modal_gmt_offset(samples: list[Sample]) -> timedelta:
    offsets = Counter(sample.gmt_time - sample.local_time for sample in samples)
    if not offsets:
        raise ValueError("The selected monitor has no valid timestamps.")
    return offsets.most_common(1)[0][0]


def selected_samples_by_time(
    group: MonitorGroup,
    time_basis: str,
) -> tuple[dict[datetime, Sample], int]:
    buckets: dict[datetime, list[Sample]] = {}
    for sample in group.samples:
        timestamp = sample.gmt_time if time_basis == "gmt" else sample.local_time
        buckets.setdefault(timestamp, []).append(sample)

    selected: dict[datetime, Sample] = {}
    conflicting_duplicates = 0
    for timestamp, candidates in buckets.items():
        values = {sample.value for sample in candidates}
        if len(values) > 1:
            conflicting_duplicates += 1
        candidates.sort(key=lambda sample: (bool(sample.qualifier), sample.method_code))
        selected[timestamp] = candidates[0]
    return selected, conflicting_duplicates


def expected_grid(year: int, time_basis: str, gmt_offset: timedelta) -> list[datetime]:
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    if time_basis == "gmt":
        start += gmt_offset
        end += gmt_offset

    timestamps: list[datetime] = []
    current = start
    while current < end:
        timestamps.append(current)
        current += timedelta(hours=1)
    return timestamps


def format_measurement(value: float) -> str:
    return f"{value:.12g}"


def write_hourly_output(
    path: Path,
    group: MonitorGroup,
    year: int,
    time_basis: str,
) -> dict[str, int | float | str]:
    gmt_offset = modal_gmt_offset(group.samples)
    samples_by_time, conflicting_duplicates = selected_samples_by_time(group, time_basis)
    grid = expected_grid(year, time_basis, gmt_offset)

    observed_hours = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "ozone_ppm"])
        for timestamp in grid:
            sample = samples_by_time.get(timestamp)
            if sample is None:
                writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), ""])
                continue
            observed_hours += 1
            writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), format_measurement(sample.value)])

    return {
        "expected_hours": len(grid),
        "observed_hours": observed_hours,
        "missing_hours": len(grid) - observed_hours,
        "coverage_ratio": observed_hours / len(grid),
        "conflicting_duplicate_hours": conflicting_duplicates,
        "gmt_minus_local_hours": gmt_offset.total_seconds() / 3600,
        "grid_start": grid[0].strftime("%Y-%m-%d %H:%M:%S"),
        "grid_end": grid[-1].strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    ensure_complete_line(input_path)
    groups = ranked_groups(collect_groups(input_path, args))
    if not groups:
        raise SystemExit(
            "No matching Ozone rows found for "
            f"state={normalize_code(args.state_code, 2)} "
            f"county={normalize_code(args.county_code, 3)} "
            f"site={args.site_num or '*'} poc={args.poc or '*'} year={args.year}."
        )

    expected_hours = (
        HOURS_PER_NON_LEAP_YEAR + 24
        if args.year % 4 == 0 and (args.year % 100 != 0 or args.year % 400 == 0)
        else HOURS_PER_NON_LEAP_YEAR
    )
    coverage_path = (
        Path(args.coverage_output)
        if args.coverage_output
        else input_path.parent / f"sacramento_ozone_poc_coverage_{args.year}.csv"
    )
    write_coverage(coverage_path, groups, expected_hours)

    selected = groups[0]
    if selected.parameter_name.lower() != "ozone":
        raise ValueError(f"Selected parameter is not Ozone: {selected.parameter_name!r}")
    if selected.units != {OZONE_UNIT}:
        raise ValueError(f"Expected Ozone unit {OZONE_UNIT!r}, found {sorted(selected.units)!r}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent
        / (
            f"sacramento_ozone_site{selected.site_num}_poc{selected.poc}"
            f"_hourly_{args.year}.csv"
        )
    )
    output_stats = write_hourly_output(
        output_path,
        selected,
        args.year,
        args.time_basis,
    )

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "source": str(input_path),
        "source_size_bytes": input_path.stat().st_size,
        "output": str(output_path),
        "coverage_report": str(coverage_path),
        "target_column": "ozone_ppm",
        "time_basis": "UTC" if args.time_basis == "gmt" else "AQS local standard time",
        "filters": {
            "year_local": args.year,
            "state_code": normalize_code(args.state_code, 2),
            "county_code": normalize_code(args.county_code, 3),
            "parameter_code": normalize_code(args.parameter_code, 5),
            "site_num": selected.site_num,
            "poc": selected.poc,
        },
        "monitor": {
            "parameter_name": selected.parameter_name,
            "units": sorted(selected.units),
            "latitude": selected.latitude,
            "longitude": selected.longitude,
            "method_types": sorted(selected.method_types),
            "method_codes": sorted(selected.method_codes),
            "qualified_rows_retained": selected.qualified_rows,
        },
        "selection": (
            "Explicit site/POC filter" if args.site_num or args.poc else "Maximum unique valid hourly coverage"
        ),
        **output_stats,
        "missing_value_policy": (
            "Missing monitor hours are blank. They are not converted to zero or interpolated."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    print(
        "Done. "
        f"selected_site={selected.site_num} selected_poc={selected.poc} "
        f"observed_hours={output_stats['observed_hours']}/{output_stats['expected_hours']} "
        f"coverage={output_stats['coverage_ratio']:.2%} "
        f"output={output_path} metadata={metadata_path} coverage_report={coverage_path}"
    )


if __name__ == "__main__":
    main()
