#!/usr/bin/env bash
set -euo pipefail

TIME_MMD_DOMAIN="Economy"
TIME_MMD_STEM="US_TradeBalance_Month"
TIME_MMD_SLUG="economy"
TIME_MMD_DESCRIPTION="Monthly United States trade balance from Time-MMD."
TIME_MMD_REGION="United States"
TIME_MMD_UNIT="million USD"
TIME_MMD_SCHEMA_VARIANT="time_mmd_economy"
TIME_MMD_HORIZONS="3 6 12 24"
TIME_MMD_HISTORY_LEN="36"
TIME_MMD_PATCH_LEN="4"
TIME_MMD_PATCH_STRIDE="2"
TIME_MMD_NEWS_WINDOW_DAYS="31"
TIME_MMD_REGIME_TAU_DAYS="14.0"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_run_time_mmd_domain.sh"
run_time_mmd_domain "$@"
