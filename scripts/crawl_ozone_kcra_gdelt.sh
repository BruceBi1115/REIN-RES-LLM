#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
START_DATE="${START_DATE:-2025-01-01}"
END_DATE="${END_DATE:-2026-01-01}"
WINDOW_DAYS="${WINDOW_DAYS:-7}"
MIN_INTERVAL_SEC="${MIN_INTERVAL_SEC:-10}"
TIMEOUT_SEC="${TIMEOUT_SEC:-45}"

# Keep the requested spelling in the output filenames.
OUTPUT_PATH="${OUTPUT_PATH:-dataset/[ozone]_GDLET_KCA.json}"
RAW_OUTPUT_PATH="${RAW_OUTPUT_PATH:-dataset/[ozone]_GDLET_KCA_raw.json}"

# KCRA is already a Sacramento-area source. The second term block prevents
# unrelated national KCRA stories from entering the Ozone news collection.
QUERY='(ozone OR smog OR "air quality" OR "air pollution" OR smoke OR wildfire OR heat OR wind OR emissions) (Sacramento OR Folsom OR "Sacramento Valley" OR "Northern California")'

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import requests" >/dev/null 2>&1; then
  echo "[ERROR] '$PYTHON_BIN' does not provide the requests package." >&2
  echo "        Set PYTHON_BIN to the project environment before running." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
tmp_output="$tmp_dir/ozone_kcra.json"
tmp_raw_output="$tmp_dir/ozone_kcra_raw.json"
collector_log="$tmp_dir/collector.log"

echo "[config] source=KCRA domain=kcra.com"
echo "[config] range=$START_DATE to $END_DATE (end exclusive)"
echo "[config] output=$OUTPUT_PATH"
echo "[config] query=$QUERY"

set +e
"$PYTHON_BIN" src/News_scrape/crawl_gdelt_news.py \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --domain kcra.com \
  --keywords "$QUERY" \
  --window_days "$WINDOW_DAYS" \
  --min_interval_sec "$MIN_INTERVAL_SEC" \
  --timeout "$TIMEOUT_SEC" \
  --output "$tmp_output" \
  --raw_output "$tmp_raw_output" 2>&1 | tee "$collector_log"
collector_status="${PIPESTATUS[0]}"
set -e

if [[ "$collector_status" -ne 0 ]]; then
  echo "[ERROR] GDELT collector exited with status $collector_status." >&2
  exit "$collector_status"
fi

if grep -Fq "[REQUEST_FAILED:" "$collector_log"; then
  echo "[ERROR] At least one GDELT date window failed after retries." >&2
  echo "        No final output was replaced. Run the script again." >&2
  exit 1
fi

if grep -Fq "[SATURATED]" "$collector_log"; then
  echo "[ERROR] A GDELT date window reached the 250-article limit." >&2
  echo "        Rerun with a smaller window, for example WINDOW_DAYS=3." >&2
  exit 1
fi

"$PYTHON_BIN" - "$tmp_output" "$tmp_raw_output" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path


main_path = Path(sys.argv[1])
raw_path = Path(sys.argv[2])

with main_path.open("r", encoding="utf-8") as handle:
    rows = json.load(handle)
with raw_path.open("r", encoding="utf-8") as handle:
    raw_rows = json.load(handle)

if not isinstance(rows, list) or not rows:
    raise SystemExit("[ERROR] GDELT returned no matching KCRA articles.")
if len(rows) != len(raw_rows):
    raise SystemExit(
        f"[ERROR] Main/raw output length mismatch: {len(rows)} != {len(raw_rows)}"
    )

required_main = {"title", "date", "content"}
for index, row in enumerate(rows):
    missing = required_main.difference(row)
    if missing:
        raise SystemExit(f"[ERROR] Row {index} is missing fields: {sorted(missing)}")
    datetime.strptime(row["date"], "%d-%m-%Y %I:%M:%S %p")

unexpected_domains = sorted(
    {
        str(row.get("domain", "")).strip().lower()
        for row in raw_rows
        if not str(row.get("domain", "")).strip().lower().endswith("kcra.com")
    }
)
if unexpected_domains:
    raise SystemExit(
        f"[ERROR] Non-KCRA domains found in raw output: {unexpected_domains}"
    )

dates = [
    datetime.strptime(row["date"], "%d-%m-%Y %I:%M:%S %p")
    for row in rows
]
print(
    "[validate] "
    f"articles={len(rows)} "
    f"first={min(dates).isoformat(sep=' ')} "
    f"last={max(dates).isoformat(sep=' ')}"
)
PY

mkdir -p "$(dirname -- "$OUTPUT_PATH")" "$(dirname -- "$RAW_OUTPUT_PATH")"
mv "$tmp_output" "$OUTPUT_PATH"
mv "$tmp_raw_output" "$RAW_OUTPUT_PATH"

echo "[done] REIN news file: $OUTPUT_PATH"
echo "[done] Audit metadata:  $RAW_OUTPUT_PATH"
