#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# All-horizon load runner for single-field regime-bank ablations, excluding regime_vec.
# Override HORIZONS_CSV or ABLATIONS_CSV from the shell if needed.
export HORIZONS_CSV="${HORIZONS_CSV:-48,96,192,336,720}"
export ABLATIONS_CSV="${ABLATIONS_CSV:-no_relevance_mass,no_in_force_doc_count}"

exec bash "$SCRIPT_DIR/run_nsw_load_ablations.sh" "$@"
