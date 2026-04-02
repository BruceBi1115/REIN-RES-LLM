from __future__ import annotations

import argparse
import csv
from pathlib import Path


def has_currency_token(value: str, target: str) -> bool:
    tokens = [item.strip().upper() for item in str(value or "").split(",")]
    return target.upper() in {token for token in tokens if token and token != "NULL"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter rows from news_currencies_source_joinedResult.csv by currencies column."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/News_scrape/news_currencies_source_joinedResult.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/News_scrape/news_currencies_source_joinedResult_BTC.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--currency",
        type=str,
        default="BTC",
        help="Currency token to filter by.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Only keep rows where currencies exactly equals the target value.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    matched = 0
    total = 0
    target = args.currency.strip().upper()

    with args.input.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No header found in input CSV: {args.input}")
        if "currencies" not in fieldnames:
            raise ValueError("Input CSV does not contain a 'currencies' column.")

        with args.output.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total += 1
                value = str(row.get("currencies") or "").strip()
                keep = value.upper() == target if args.exact else has_currency_token(value, target)
                if keep:
                    writer.writerow(row)
                    matched += 1

    mode = "exact" if args.exact else "token"
    print(
        f"Done. Wrote {matched} / {total} rows to {args.output} "
        f"(currency={target}, mode={mode})."
    )


if __name__ == "__main__":
    main()
