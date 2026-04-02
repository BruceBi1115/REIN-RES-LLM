# SignNet DELTA Text Evolution Plan

## Goal

This file records the framework upgrade paths for the current `Base + DELTA + SignNet + temporal text` system so we can iterate safely, compare designs, and switch paths later if one direction underperforms.

The target is to improve:

- residual direction prediction
- residual magnitude prediction
- text utilization ceiling
- robustness under sparse-news and domain-shift settings

This document is a planning reference.

Current status:

- the original summary-based path is still available as `summary_gated`
- the executable experimental branch is now `plan_c_mvp`
- the older intermediate multimodal branch has been removed from code because it was not a good fit for the current framework goals

## Current Baseline

Current framework behavior:

- `Base` predicts the main forecast anchor
- `DELTA` predicts residual magnitude and residual context
- in `delta_sign_mode=signnet_binary`, external `SignNet` predicts direction/state and overrides DELTA's internal sign path
- final prediction is produced by fusing `base_pred` with the composed residual

Current text path:

- a shared `TemporalTextTower` encodes time-aligned text into:
  - `patch_context`
  - `text_summary`
  - `text_strength`
- `DELTA` uses:
  - patch-level gated fusion in `summary_gated`
  - sample-level `text_summary`
  - optional regime-aware expert routing in `plan_c_mvp`
- `SignNet` uses:
  - `text_summary + text_strength`
  - optional regime-aware expert routing in `plan_c_mvp`

Current bottlenecks:

- direction and magnitude are still learned in two separate models and combined late
- text is still compressed before entering external `SignNet`
- sparse-news and dense-news samples can need different residual behavior
- validation can look strong while test degrades under news coverage shift

## Shared Design Principles

All upgrade plans should follow these rules:

- keep `Base` as the stable anchor model
- keep a clear separation between `base forecast` and `residual reasoning`
- make text use stronger, but also make text influence suppressible
- support sparse-news fallback behavior
- preserve ablation ability through flags rather than hard replacement

Recommended future top-level switch:

```text
--residual_arch {current,plan_a,plan_c}
```

Recommended supporting switches:

```text
--direction_source {external_signnet,shared_head}
--text_fusion_mode {summary_only,gated_add,regime_routed}
--residual_confidence_enable {0,1}
--structured_token_mode {vector,tokens}
```

## Plan A

### Summary

Plan A upgrades the current system into a shared residual reasoner while staying close to the existing codebase.

Core idea:

- stop treating `SignNet` and `DELTA` as two mostly independent models
- build one shared multimodal residual trunk
- predict direction, magnitude, and confidence from that shared trunk

### Architecture

Inputs:

- TS patches
- temporal text features from `TemporalTextTower`
- structured news features
- base prediction embedding

Shared trunk outputs:

- residual context

Heads:

- `direction_head`
- `magnitude_head`
- `confidence_head`

Final residual:

```text
delta = confidence * direction_score * magnitude
```

Final prediction:

```text
pred = fuse(base_pred, delta)
```

### Text Handling

Text should no longer feed only one generic summary.

Recommended split:

- `dir_text_summary`
- `mag_text_summary`

Reason:

- direction cues and magnitude cues are often different
- polarity and event intent help direction
- intensity and persistence help magnitude

### Why Plan A

Pros:

- highest value-to-effort ratio
- easiest migration from current code
- removes late-stage mismatch between `SignNet` and `DELTA`
- enables explicit confidence suppression

Risks:

- still summary-heavy compared with a true token interaction model
- may leave some text ceiling untapped

Best fit:

- first major upgrade from the current framework
- moderate data regimes
- when we want better generalization before making the model much heavier

### Implementation Scope

Likely code areas:

- replace external `SignNet` override path with shared residual trunk
- keep `TemporalTextTower`
- add confidence head and consistency loss
- restructure DELTA forward path around shared residual reasoning

### Success Signals

- better test stability than current `signnet_binary`
- fewer cases where val improves but test drops
- stronger sparse-news robustness

### Failure Signals

- still large val-test gap under sparse-news test splits
- confidence head collapses near `1.0` everywhere
- direction and magnitude remain weakly coupled

## Plan C

### Summary

Plan C is the regime-aware residual mixture path and is now the active experimental branch in code.

Core idea:

- different market and dataset regimes need different residual reasoning behavior
- the system should decide whether to apply no correction, weak correction, or expert-specific correction

### Architecture

Modules:

- `Base backbone`
- `Residual reasoner experts`
- `Regime router`

Current executable MVP:

- router outputs routes `none / trend / event / reversal / sparse`
- `none` behaves like an abstention path and reduces correction confidence
- the remaining routes mix expert-specific residual transformations
- both DELTA and external SignNet can use this regime-aware mixture

Suggested expert roles:

- `trend`
- `event`
- `reversal`
- `sparse_news`

Router inputs:

- history volatility
- base prediction dispersion
- text strength
- structured-news strength
- combined news strength

Final behavior:

- route each sample to one or more residual experts
- shrink residual confidence when the router prefers the abstention route
- let dense-news and sparse-news samples follow different correction behavior

### Text Handling

Text is routed, not only fused.

Examples:

- dense reliable news -> stronger event route probability
- no news or weak confidence -> sparse or abstention route
- reversal-like signal -> reversal route

The current MVP still uses `TemporalTextTower` summaries rather than full token-level routing, but it adds regime-specific text usage on top of those summaries.

### Why Plan C

Pros:

- better match for heterogeneous datasets
- natural way to handle sparse-news vs dense-news mismatch
- gives both DELTA and SignNet a route-aware correction policy
- includes an explicit abstention mechanism

Risks:

- larger engineering surface than the original summary path
- harder training dynamics
- risk of expert collapse or router instability

Best fit:

- when one architecture cannot serve all datasets well
- when sparse-news behavior is a main failure mode
- when cross-dataset robustness matters more than a single dense-news ceiling

### Implementation Scope

Likely code areas:

- introduce router and expert modules
- add routing diagnostics and expert usage logging
- let SignNet and DELTA share the same regime-aware design language
- preserve `summary_gated` as the stable fallback

### Success Signals

- better cross-dataset robustness
- less degradation under news coverage shift
- cleaner specialization by dataset regime
- more conservative behavior on no-news slices

### Failure Signals

- expert imbalance
- router collapse to one expert
- unstable or non-reproducible training

## Recommended Development Order

Recommended order now:

1. keep `summary_gated` as the stable baseline
2. evaluate `plan_c_mvp`
3. if Plan C is too unstable, fall back to Plan A style simplification

Reason:

- `summary_gated` is still the simplest trustworthy fallback
- Plan C directly targets the sparse-news and regime-mismatch failures we actually observed
- the removed intermediate multimodal branch increased complexity without fitting the current priorities well enough

## When To Switch Plans

Use this rule of thumb:

- stay on `summary_gated` if it remains the most stable option
- move to Plan A if the main problem is still late coupling between direction and magnitude
- move to Plan C if dataset heterogeneity, sparse-news behavior, or regime differences are the main failure mode

Fallback guidance:

- if Plan C becomes too hard to control, return to `summary_gated`
- if Plan C improves robustness but still leaves direction-magnitude mismatch, revisit Plan A style shared-head refactoring

## Suggested Evaluation Policy

For every plan transition, compare against:

- current baseline
- no signnet
- no temporal text
- no structured features
- base only

Track at minimum:

- val/test MAE
- val/test MSE
- direction/state accuracy
- residual magnitude calibration
- sparse-news vs dense-news slice performance
- no-news slice performance
- route usage distribution
- abstention-rate distribution

## Immediate Next Step

If implementation continues, the practical next move is:

- keep `summary_gated` as the default production-like path
- use `delta_multimodal_arch=plan_c_mvp` as the experimental branch
- monitor route collapse, abstention rate, and expert balance before making larger architectural jumps
