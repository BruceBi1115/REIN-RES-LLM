# Delta V3 Acceptance

Status: pending end-to-end validation runs.

| Metric | LOAD target | PRICE target | Measured | Pass? |
|---|---:|---:|---:|---|
| `delta_mae / base_mae` | `<= 0.92` | `<= 0.85` | pending | pending |
| `top10pct_residual_mae` improvement | `>= 15%` | `>= 25%` | pending | pending |
| `delta_helped_rate_top10pct_residual` | `>= 0.70` | `>= 0.70` | pending | pending |
| forced news dropout stays within `base * 1.02` | yes | yes | pending | pending |

Notes:
- Fill this table after running the full NSW LOAD and NSW PRICE delta_v3 evaluations.
- Keep the measured values aligned with the final metrics emitted by `results/test_results.csv` and the delta_v3 evaluation logs.
