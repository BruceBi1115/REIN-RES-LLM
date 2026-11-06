#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT_NAME="scripts/run_ozone.sh"
RUN_SCRIPT_PATH="$SCRIPT_DIR/run_ozone.sh"

cd "$REPO_ROOT"

RAW_FILE="dataset/EPA_criteria_gas_2025/sacramento_ozone_site0012_poc1_hourly_2025_ffill_max3.csv"
TRAIN_FILE="dataset/EPA_criteria_gas_2025/ozone_hourly_2025_trainset.csv"
VAL_FILE="dataset/EPA_criteria_gas_2025/ozone_hourly_2025_valset.csv"
TEST_FILE="dataset/EPA_criteria_gas_2025/ozone_hourly_2025_testset.csv"

DEFAULT_NEWS_PATH="dataset/ozone_freenews.json"
# DEFAULT_REGIME_BANK_PATH="_shared_refine_cache/v4/regime_bank_load_freeNews_reneweconomy_pvmag_watt.npz"
# DEFAULT_REFINED_JSONL_PATH="_shared_refine_cache/v4/refined_load_freeNews_reneweconomy_pvmag_watt.jsonl"

DATASET_KEY="Ozone"
TIME_COL="date"
VALUE_COL="ozone_ppm"
ID_COL="segment_id"
UNIT="ppm"
DESCRIPTION="This dataset records hourly ozone concentrations at EPA AQS site 06-067-0012, POC 1, in Sacramento County during 2025."
REGION="Sacramento County, California, USA"
FREQ_MIN="60"
DAY_FIRST="0"

# Base-only for now: no news file, LLM call, or delta-stage training is used.
STAGE="all"
# DEFAULT_NEWS_PATH=""
NEWS_API_ENABLE="1"

BASE_EPOCHS="40"
BATCH_SIZE="16"
GRAD_ACCS=("4")
LRS=("1e-4")
HORIZONS=("48" "96" "192" "336" "720")
SCHEDULERS=("1")
BASE_BACKBONES=("dlinear")
NORMALIZATION_MODE="zscore"
EARLY_STOP_PATIENCE="5"

TASK_NAME_BASE="ozone"
PRE_RUN_HOOK="prepare_ozone_splits"

# Required by the shared runner even though these delta settings are inactive
# while STAGE=base.
DELTA_V3_REFINED_BANK_BUILD="1"
DELTA_V3_ACTIVE_MASS_THRESHOLDS=("0.7")
DELTA_V3_SCHEMA_VARIANT="ozone"

prepare_ozone_splits() {
  "$PYTHON_BIN" - \
    "$RAW_FILE" \
    "$TRAIN_FILE" \
    "$VAL_FILE" \
    "$TEST_FILE" \
    "$HISTORY_LEN" \
    "${HORIZONS[@]}" <<'PY'
import math
import sys
from pathlib import Path

import pandas as pd


raw_path = Path(sys.argv[1])
train_path = Path(sys.argv[2])
val_path = Path(sys.argv[3])
test_path = Path(sys.argv[4])
history_len = int(sys.argv[5])
horizons = [int(value) for value in sys.argv[6:]]

if not raw_path.is_file():
    raise FileNotFoundError(raw_path)

source = pd.read_csv(raw_path)
required_columns = {"date", "ozone_ppm"}
missing_columns = sorted(required_columns.difference(source.columns))
if missing_columns:
    raise KeyError(f"Missing required Ozone columns: {missing_columns}")

# Select the model inputs explicitly. In particular, is_imputed is diagnostic
# metadata and is never written to the model-facing split files.
work = source.loc[:, ["date", "ozone_ppm"]].copy()
work["date"] = pd.to_datetime(work["date"], errors="coerce")
work["ozone_ppm"] = pd.to_numeric(work["ozone_ppm"], errors="coerce")

if work["date"].isna().any():
    examples = source.loc[work["date"].isna(), "date"].head(5).tolist()
    raise ValueError(f"Unparseable Ozone timestamps: {examples}")

work = work.sort_values("date", kind="mergesort").reset_index(drop=True)
if work["date"].duplicated().any():
    examples = work.loc[work["date"].duplicated(keep=False), "date"].head(5).tolist()
    raise ValueError(f"Duplicate Ozone timestamps: {examples}")

time_steps = work["date"].diff().dropna()
unexpected_steps = time_steps.ne(pd.Timedelta(hours=1))
if unexpected_steps.any():
    examples = time_steps.loc[unexpected_steps].head(5).astype(str).tolist()
    raise ValueError(f"Ozone source is not on a complete hourly grid: {examples}")

row_count = len(work)
if row_count == 0:
    raise ValueError("Ozone source file is empty.")

train_end = math.floor(row_count * 0.70)
val_end = train_end + math.floor(row_count * 0.10)
work["_position"] = range(row_count)

valid = work["ozone_ppm"].notna()
segment_starts = valid & ~valid.shift(1, fill_value=False)
work["segment_id"] = segment_starts.cumsum()

clean = work.loc[valid].copy()
clean["segment_id"] = clean["segment_id"].map(lambda value: f"ozone_{int(value):04d}")

splits = {
    "train": clean.loc[clean["_position"] < train_end].copy(),
    "val": clean.loc[
        (clean["_position"] >= train_end) & (clean["_position"] < val_end)
    ].copy(),
    "test": clean.loc[clean["_position"] >= val_end].copy(),
}
output_paths = {"train": train_path, "val": val_path, "test": test_path}

for split_name, frame in splits.items():
    if frame.empty:
        raise ValueError(f"Ozone {split_name} split is empty.")
    output = frame.loc[:, ["date", "ozone_ppm", "segment_id"]].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output_paths[split_name].parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_paths[split_name], index=False)


def window_count(current, previous, horizon):
    previous_sizes = (
        previous.groupby("segment_id", sort=False).size().to_dict()
        if previous is not None
        else {}
    )
    total = 0
    for segment_id, frame in current.groupby("segment_id", sort=False):
        prefix = min(history_len, int(previous_sizes.get(segment_id, 0)))
        total += max(0, prefix + len(frame) - history_len - horizon + 1)
    return total


source_imputed = pd.to_numeric(
    source.get("is_imputed", pd.Series(0, index=source.index)),
    errors="coerce",
).fillna(0).eq(1)
source_imputed_retained = int((source_imputed & valid).sum())

print(
    "[DATASET] prepared Ozone chronological splits "
    f"train={len(splits['train'])} val={len(splits['val'])} "
    f"test={len(splits['test'])}"
)
print(
    "[DATASET] "
    f"source_rows={row_count} retained_rows={len(clean)} "
    f"remaining_missing_removed={int((~valid).sum())} "
    f"imputed_values_retained={source_imputed_retained} "
    "ignored_column=is_imputed"
)

for horizon in horizons:
    counts = {
        "train": window_count(splits["train"], None, horizon),
        "val": window_count(splits["val"], splits["train"], horizon),
        "test": window_count(splits["test"], splits["val"], horizon),
    }
    if min(counts.values()) == 0:
        raise ValueError(
            f"No complete Ozone window for horizon={horizon}: {counts}"
        )
    print(f"[DATASET] horizon={horizon} complete_windows={counts}")
PY
}

source "$SCRIPT_DIR/_run_forecast_common.sh"
run_forecast_dataset_main "$@"
