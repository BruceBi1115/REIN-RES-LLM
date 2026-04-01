#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Legacy entrypoint kept for compatibility.
exec bash "$SCRIPT_DIR/run_tinynews_experiment.sh" --dataset nsw_load "$@"
