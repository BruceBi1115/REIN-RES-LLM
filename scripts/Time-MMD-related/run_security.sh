#!/usr/bin/env bash
set -euo pipefail

TIME_MMD_DOMAIN="Security"
TIME_MMD_STEM="US_FEMAGrant_Month"
TIME_MMD_SLUG="security"
TIME_MMD_DESCRIPTION="Monthly United States FEMA grant amount from Time-MMD."
TIME_MMD_REGION="United States"
TIME_MMD_UNIT="USD"
TIME_MMD_SCHEMA_VARIANT="time_mmd_security"
TIME_MMD_HORIZONS="3 6 12 24"
TIME_MMD_HISTORY_LEN="36"
TIME_MMD_PATCH_LEN="4"
TIME_MMD_PATCH_STRIDE="2"
TIME_MMD_NEWS_WINDOW_DAYS="31"
TIME_MMD_REGIME_TAU_DAYS="14.0"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_run_time_mmd_domain.sh"
run_time_mmd_domain "$@"
