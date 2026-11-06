#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Price h48 smoke/diagnostic runner for single-field regime-bank ablations.
# Override any variable below from the shell if needed.
export HORIZONS_CSV="${HORIZONS_CSV:-48}"
export ABLATIONS_CSV="${ABLATIONS_CSV:-full,no_regime_vec,no_topic_mass,no_text_emb,no_relevance_mass,no_in_force_doc_count}"
export STAGE="${STAGE:-all}"

exec bash "$SCRIPT_DIR/run_nsw_price_ablations.sh" "$@"
