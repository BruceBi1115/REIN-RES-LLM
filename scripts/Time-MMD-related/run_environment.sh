#!/usr/bin/env bash
set -euo pipefail

TIME_MMD_DOMAIN="Environment"
TIME_MMD_STEM="NewYork_AQI_Day"
TIME_MMD_SLUG="environment"
TIME_MMD_DESCRIPTION="Daily maximum Air Quality Index for the New York metropolitan area from Time-MMD."
TIME_MMD_REGION="New York-Newark-Jersey City, US"
TIME_MMD_UNIT="AQI"
TIME_MMD_SCHEMA_VARIANT="time_mmd_environment"
TIME_MMD_HORIZONS="48 96 192 336 720"
TIME_MMD_HISTORY_LEN="48"
TIME_MMD_PATCH_LEN="7"
TIME_MMD_PATCH_STRIDE="3"
TIME_MMD_NEWS_WINDOW_DAYS="1"
TIME_MMD_REGIME_TAU_DAYS="5.0"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_run_time_mmd_domain.sh"
run_time_mmd_domain "$@"
