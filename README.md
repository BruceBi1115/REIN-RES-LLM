# REIN-RES-LLM

REIN-RES-LLM is a time-series forecasting framework that combines a numeric base forecaster with a news-aware residual correction module. The main idea is to first learn a strong pure time-series baseline, then use structured news-derived regime signals to adjust the residual forecast when external events are relevant.

## Framework Overview

The framework has two stages:

1. **Base forecasting**
   - Trains a standard time-series backbone on historical values.
   - Supported backbones include MLP, DLinear, NLinear, and PatchTST.
   - Produces the base prediction and optional hidden features.

2. **Delta V3 residual forecasting**
   - Predicts a residual correction on top of the base forecast.
   - Encodes recent time-series history with a PatchTST-style encoder.
   - Encodes news-derived regime features with a regime projector.
   - Uses modulation heads to control the residual strength, shape gain, and spike bias.

The final prediction is:

```text
final_prediction = base_prediction + lambda_delta * residual_prediction
```

where `lambda_delta` is learned from both time-series context and news regime relevance.

## News Regime Pipeline

News articles are converted into structured regime signals before training. Each relevant article is refined into:

- `regime_vec`: numeric regime dimensions such as tightness, demand outlook, volatility tone, and policy effect.
- `topic_tag_mass`: topic tags such as heatwave, outage, renewable surge, policy active, and routine.
- `text_emb`: text embedding of the article summary.
- `relevance_mass`: total active news relevance for a target timestamp.

These signals are stored in a regime bank and looked up during training or evaluation.

## Key Files

- `run.py`: main training entry point.
- `src/base_backbone.py`: base forecasting backbones.
- `src/base_delta_decoouple_trainer.py`: base and delta training orchestration.
- `src/delta_v3/model.py`: Delta V3 model definition.
- `src/delta_v3/encoder_ts.py`: PatchTST-style time-series encoder.
- `src/delta_v3/modulation_heads.py`: regime projector and scalar modulation MLPs.
- `src/delta_v3/regime_bank.py`: regime bank construction and lookup.
- `src/delta_v3/schema_refine_v2.py`: news-to-regime schema refinement.
- `scripts/run_nsw_load.sh`: NSW electricity load experiment.
- `scripts/run_nsw_price.sh`: NSW electricity price experiment.
- `scripts/run_gas.sh`: gas demand experiment.
- `scripts/run_traffic.sh`: traffic count experiment.
- `scripts/run_bitcoin_hourly.sh`: Bitcoin hourly experiment.

## Quick Start

Run one of the prepared experiment scripts:

```bash
bash scripts/run_traffic.sh
```

or call the main entry point directly:

```bash
python run.py \
  --stage all \
  --train_file path/to/train.csv \
  --val_file path/to/val.csv \
  --test_file path/to/test.csv \
  --time_col date \
  --value_col value \
  --base_backbone dlinear \
  --news_path path/to/news.json \
  --news_api_enable 1 \
  --delta_v3_schema_variant traffic
```

Use `--stage base` to train only the base model, `--stage delta` to train only the residual module from an existing base checkpoint, or `--stage all` to run both stages.

## Outputs

Training checkpoints are written under `checkpoints/` by default. Evaluation summaries and diagnostic plots are written under `results/`.

Typical evaluation outputs include:

- base and delta forecast metrics.
- residual and spike-related diagnostics.
- counterfactual tests with blanked or permuted regime signals.
- temporal component attribution plots.

## Notes

The framework is designed for event-sensitive forecasting tasks where external information can explain residual movement beyond the historical signal. It is especially useful for domains such as electricity load, electricity price, gas demand, traffic, and crypto markets.
