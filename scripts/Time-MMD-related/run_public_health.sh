#!/usr/bin/env bash
set -euo pipefail

TIME_MMD_DOMAIN="Public_Health"
TIME_MMD_STEM="US_FLURATIO_Week"
TIME_MMD_SLUG="public_health"
TIME_MMD_DESCRIPTION="Weekly unweighted influenza-like illness visit percentage for the United States from Time-MMD."
TIME_MMD_REGION="United States"
TIME_MMD_UNIT="percent"
TIME_MMD_SCHEMA_VARIANT="time_mmd_public_health"
TIME_MMD_HORIZONS="4 8 12 24"
TIME_MMD_HISTORY_LEN="52"
TIME_MMD_PATCH_LEN="4"
TIME_MMD_PATCH_STRIDE="2"
TIME_MMD_NEWS_WINDOW_DAYS="7"
TIME_MMD_REGIME_TAU_DAYS="7.0"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_run_time_mmd_domain.sh"
run_time_mmd_domain "$@"
