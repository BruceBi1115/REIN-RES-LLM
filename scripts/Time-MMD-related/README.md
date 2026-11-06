# Time-MMD REIN Runs

From the repository root, run any domain script, for example:

```bash
bash scripts/Time-MMD-related/run_algriculture.sh
bash scripts/Time-MMD-related/run_climate.sh
```

Each script checks its numerical and text inputs, then runs the existing base and
delta stages. Existing non-empty processed files are reused. Missing files are
regenerated from `dataset/Time-MMD/<domain>`; set `TIME_MMD_FORCE_PREPARE=1` to
force regeneration. Domain prompts live in `src/time_mmd_prompts.py` and are routed
by `--delta_v3_schema_variant`. The script and Python configuration reject a
Time-MMD dataset key paired with a different schema.

| Script | Schema | Target | Maximum text effect duration |
| --- | --- | --- | --- |
| `run_algriculture.sh` | `time_mmd_algriculture` | Monthly US retail broiler price | 62 days |
| `run_climate.sh` | `time_mmd_climate` | Monthly contiguous-US precipitation | 62 days |
| `run_economy.sh` | `time_mmd_economy` | Monthly US trade balance, exports minus imports | 62 days |
| `run_energy.sh` | `time_mmd_energy` | Weekly US retail gasoline price | 28 days |
| `run_environment.sh` | `time_mmd_environment` | Daily New York metro AQI | 14 days |
| `run_public_health.sh` | `time_mmd_public_health` | Weekly US unweighted ILI visit percentage | 28 days |
| `run_security.sh` | `time_mmd_security` | Monthly FEMA grant amount | 62 days |
| `run_socialgood.sh` | `time_mmd_socialgood` | Monthly US unadjusted unemployment rate | 62 days |
| `run_traffic.sh` | `time_mmd_traffic` | Monthly US vehicle miles traveled | 62 days |

Effect durations are conservative configurable-in-code caps for the new prompts,
not validated optimal hyperparameters and not the forecast horizon in samples.
The LLM estimates a supported duration within the cap; both output normalization
and regime-bank construction use that cap. Existing non-Time-MMD schemas retain
their 14-day cap and prompts. The five regime fields and topic vocabulary are
unchanged; each domain prompt defines their interpretation and signs.

New caches are isolated under:

```text
_shared_refine_cache/v4/<time_mmd_domain>/v4_regime_v2_time_mmd_v1/
```

Old top-level caches generated with load/price/ozone/traffic prompts are preserved.
The next domain run refines into the new directory and records the domain and
prompt version. Wrong-version, wrong-domain, and unversioned caches are rejected.
Increment `TIME_MMD_SCHEMA_VERSION` when changing prompt semantics so subsequent
runs cannot silently reuse stale refinement. The log records the selected prompt:

```text
[DELTA_V3][REFINE_CONFIG] schema_variant=time_mmd_climate schema_version=v4_regime_v2_time_mmd_v1 max_horizon_days=62
```

Restart an already running process to load the new prompts. Previous evaluation
results are not updated by changing code; rerun the corresponding experiment.
First refinement requires the configured news API credentials and may incur API
costs. Subsequent refinement can reuse compatible per-document records.

Time-MMD text bundles are exported with the numerical row's date. This is an as-of
reference, not independently verified publication metadata. Some supplied Climate
rows contain later-dated facts. Prompts explicitly instruct the LLM to exclude
those facts and NA placeholders; this instruction alone does not prove the source
data is free of temporal leakage. No source dates are rewritten by this change.

Offline regression checks (use the project's Python environment):

```bash
python -B -m unittest discover -s tests -p 'test_time_mmd_refine.py' -v
```

These checks exercise all nine shell configurations through `run.py`, fake-LLM
refinement, version validation, cache migration/reuse, and monthly effect expiry.
They do not call an external API or download a text encoder.
