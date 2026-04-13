# Text Fusion Redesign — Macro News as Regime Modulator (Not Step-Level Signal)

> Target system: `delta_v3` residual head on top of a frozen TS backbone (MLP / PatchTST).
> Driving problem: the news corpora we have (e.g. `watt_free_2024_price.json`, wattclarity-style NEM commentary) are **macro, slow-moving, retrospective** — *not* tick-aligned event news. The current design treats them as step-level additive priors, and that is the wrong shape of signal.
> This document is a **design plan only**. No code changes.

---

## 1. What's wrong with the current design

### 1.1 Symptoms observed
- `delta_v3_nsw_price` log: after pretrain `val_slow_r2 = -0.30` (negative → worse than predicting zero), then at delta epoch 1 `val_mae = 4263.75` vs base `33.62` — a **127×** blow-up.
- `gate = 0.74` despite `news_gate_init_bias = -1.0` — gate opens aggressively within one epoch and destroys the base.
- `spike_tgt = 0.0000` — the spike head never fires on target, but the model still commits large z-space outputs.
- Global zscore for price: `σ = 635` (spike-polluted). Any small z-space drift from a news-conditioned head becomes a large raw-space error on inverse-normalize.

### 1.2 Root causes (not patches, root causes)
1. **Temporal resolution mismatch.**
   Series is 30-min, horizon is 24 h. News is daily at best and mostly *retrospective* (quarterly reviews, market-design reports, AER/AEMC releases). There is no lawful way a headline like "AER released Wholesale Market Performance Report 2024" causes the price at 14:30 tomorrow to move by `$X`. Yet the current design wires it to exactly that prediction target.

2. **Fielded schema forces fabrication.**
   `schema_refine.py` asks the LLM for `{direction, magnitude_bucket, persistence_hours, strength}` per article. For honest general news the answer is *none/none/0/0*, but LLMs rarely output all-zero records and push a weak `magnitude=small`. Those fabricated fields then feed `num_feat` and `_record_weight`, injecting structured noise into the residual target.

3. **Additive news head on a residual predictor.**
   `model.py:90-94`:
   ```
   slow_hat  = slow_head(fused)  + gate * slow_news_head(news_summary)
   shape_hat = shape_head(fused) + news_shape_scale * shape_news_hat
   residual_hat = slow_hat + shape_hat + spike_gate * spike_hat
   ```
   The news branch writes **directly into the residual in z-space**. Combined with `σ_price = 635`, a news-induced z-error of 1 becomes a raw-space MAE of ~635. There is no mechanism that prevents this.

4. **The "cross-attention fusion" isn't cross-attention.**
   Read `cross_attn.py`: `news_kv_tokens` is ignored (`_ = news_kv_tokens`). The module just emits a scalar utility gate from `[ts_summary, news_summary, vol, vol]` and applies it to the news heads. There is no token-level interaction and no mechanism to *select which news matters for which patch*.

5. **Day-level softmax attention is the wrong question.**
   `NewsDayEncoder` softmaxes over a 3-day window to decide "which of the last few days' news matters". For macro news the right question is the opposite: *which slow regime are we in?* — an EMA/window aggregate, not a winner-take-one.

6. **Mean-pooled per-day text embedding destroys topicality.**
   A day might carry 12 disparate articles (market design + battery policy + heatwave outlook + retrospective). Mean-pooling e5 embeddings smears them into a centroid that answers no question well.

**Bottom line:** the current pipeline treats macro news as a *step-level additive residual generator*. Macro news is physically incapable of that. We need a design whose every component matches what macro news can actually tell us.

---

## 2. Design philosophy

> **Macro news should modulate the *distributional properties* of the residual (scale, spike probability, shape gain, trust-on-base) — never add a residual offset directly.**

Three non-negotiable principles:

| Principle | Concrete consequence |
|---|---|
| **P1. Resolution match** | News feeds a *slow* regime variable on a 5–14 day timescale, never a per-step value. |
| **P2. Multiplicative-only coupling** | News enters the model only through `×` (gains, gates, priors), never through `+` into the residual. Worst case: news is uninformative → modulation ≈ identity → model degrades gracefully to TS-only delta. |
| **P3. Labels from the series, not from the LLM** | The news encoder is trained against *self-supervised targets derived from the time series itself* (next-day volatility, spike indicator, shape deviation), not against hand-crafted fields. This is the only way to eliminate fabricated-label rot. |

A supporting principle:

> **P4. Base must remain viable.** At any moment we must be able to zero out news and recover a usable forecast. The design treats news as a *conditional boost*, not a pillar.

---

## 3. What macro/general NEM news can actually tell us

Looking at the corpus content honestly, the information content of a typical wattclarity / AEMC / AER piece is one of:

1. **Regime tightness** — reserve margin, supply headroom, interconnector availability over the next days–weeks
2. **Demand outlook** — heatwave / cold-snap / holiday / industrial-maintenance forecasts
3. **Renewable-share outlook** — wind/solar over- or under-supply expectations
4. **Policy-in-effect flag** — new market rule, price cap change, rebidding guideline active *now*
5. **Background volatility tone** — "stressed market", "orderly trading", "calm week"
6. **Event horizon** — how many days this described condition is claimed to last

None of these are step-level multipliers. All of them are **slow scalars** that modify the *distribution* of the next 24–72 hours of residuals:
- They can widen or tighten the expected shape amplitude.
- They can raise or lower the prior probability of a spike.
- They cannot tell you that the 14:30 price will be `$X`.

The redesign uses exactly these six signals and nothing else.

---

## 4. Proposed architecture

Five modules, top-down. Items marked **[REMOVE]** should be deleted from the current code path; **[NEW]** is new; **[REUSE]** stays.

### 4.1 Module A — Regime-oriented news refinement (offline, once per dataset)

Replace `schema_refine.py`'s fielded JSON with a regime descriptor.

**[REMOVE]** `event_type`, `direction`, `magnitude_bucket`, `persistence_hours`, `strength` — these force fabrication.

**[NEW]** `schema_refine_v2` output per article:

```json
{
  "doc_key": "...",
  "published_at": "...",
  "is_actionable": true,          // strict: does this describe a *currently in-force* market condition in the next 14 days?
  "topic_tags": ["supply_tight", "heatwave", "policy_active"],   // multi-label, closed vocab
  "regime_vec": {
    "tightness":       -0.6,      // [-1, +1]  relaxed ↔ tight
    "demand_outlook":  +0.4,      // [-1, +1]  weak ↔ heavy
    "renewable_surplus":-0.2,     // [-1, +1]  below-avg ↔ surplus
    "volatility_tone":  0.7,      // [ 0, +1]  calm ↔ expect big swings
    "policy_in_effect": 1.0       // { 0, 1 }  rule change active now
  },
  "horizon_days": 7,              // integer, hard-capped at 14
  "confidence":   0.8,            // LLM self-rated
  "summary": "60 words max, factual"
}
```

Key contract: **if `is_actionable == false`, every downstream weight for this doc is 0**. This is the explicit escape hatch that lets the LLM say "this is retrospective/irrelevant" without being forced into a `small/none` fabrication. Prompt must emphasize that `false` is the correct answer for retrospectives, market-structure analyses, and anything referring to events > 14 days in the past.

Topic tags come from a fixed closed vocabulary (≤ 15 items, e.g. `supply_tight`, `supply_surplus`, `heatwave`, `cold_snap`, `holiday`, `outage`, `fuel_shock`, `renewable_surge`, `renewable_drought`, `interconnector_limit`, `policy_active`, `market_intervention`, `retrospective`, `routine`, `other`). `retrospective`/`routine` tags force `is_actionable=false`.

Store per doc: `(published_at, is_actionable, topic_tags, regime_vec, horizon_days, text_embed_384)`.

### 4.2 Module B — Regime bank (daily rolling aggregate, not per-day snapshot)

Replace `build_daily_news_bank` (selects top-K per calendar day, mean-pools, builds a single day vector).

**[NEW]** `build_regime_bank`: for each calendar day `d` compute a smoothed regime embedding `r_d ∈ R^H` by aggregating **all articles currently in-force**.

Algorithm (pseudocode):
```
for each calendar day d in [train_start, test_end]:
    in_force = { doc : published_at(doc) ≤ d  AND  published_at(doc) + horizon_days(doc) ≥ d
                       AND  is_actionable(doc) == True }
    if in_force is empty:
        r_d = 0;  relevance_mass[d] = 0
        continue
    for doc in in_force:
        age = (d - published_at(doc)).days
        w_doc = confidence(doc) * exp(-age / tau)    # tau = 5 days
    w_doc /= sum(w_doc)
    regime_concat = concat( weighted_mean(regime_vec),
                            weighted_mean(topic_tag_onehot),
                            weighted_mean(text_embed_384) )
    r_d_raw = RegimeProjMLP(regime_concat)            # (H,)
    relevance_mass[d] = sum_before_norm(w_doc)         # scalar
# apply temporal EMA to prevent day-to-day jitter
r_d = EMA(r_d_raw, alpha=0.5, window=5)
```

Store at bank: `{ dates: [N], r: [N, H], relevance_mass: [N], topic_tag_mass: [N, num_tags] }`. The topic-tag mass is kept separately so downstream modules can apply topic-specific modulations if desired.

**Why this shape:**
- *In-force filter* matches the claim that news describes a condition persisting for `horizon_days` — the first time we actually use the LLM's horizon estimate for something real.
- *Exponential decay with τ=5* is the natural smoothness for news whose nominal shelf-life is a week.
- *EMA over 5 days* kills day-to-day noise from single-article pulses.
- *relevance_mass* is the "how much grounded signal is currently in-force" scalar — the hard switch used later.

### 4.3 Module C — Multiplicative modulation head (the only way news touches the residual)

**[REMOVE]** `slow_news_head`, `shape_news_head`, `CrossAttentionFusion`'s gate_mlp additive output.

**[NEW]** Three small MLPs read `r_d` and emit **multiplicative knobs** on the existing TS-only residual prediction:

```
# --- TS-only residual (unchanged) ---
slow_ts, shape_ts, spike_ts, spike_gate_logits = ts_residual_heads(ts_summary, ts_tokens)
# residual_ts_z = slow_ts.unsqueeze(-1) + shape_ts + sigmoid(spike_gate_logits) * spike_ts
# (we DO NOT compute residual_ts_z directly anymore; we compose below)

# --- News-only modulation knobs (new) ---
trust_logit    = MLP_trust(r_d)         # scalar per sample
shape_gain_raw = MLP_shape_gain(r_d)    # scalar per sample
spike_bias     = MLP_spike_prior(r_d)   # scalar per sample

# Hard gates: if no in-force news, force knobs to identity
active = (relevance_mass[d] > 0).float()
trust_logit    = trust_logit    * active
shape_gain_raw = shape_gain_raw * active
spike_bias     = spike_bias     * active

lambda_base  = 0.8 * torch.sigmoid(trust_logit - 2.0)    # range [0, 0.8], bias-init away from 1
shape_gain   = 1.0 + 0.3 * torch.tanh(shape_gain_raw)    # range [0.7, 1.3]
spike_logits = spike_gate_logits + spike_bias            # news can nudge spike probability only

# --- Composition (the new residual assembly) ---
slow_z    = slow_ts
shape_z   = shape_ts * shape_gain.unsqueeze(-1)
spike_gate= sigmoid(spike_logits)
residual_z= slow_z.unsqueeze(-1) + shape_z + spike_gate * spike_ts
y_hat_z   = base_pred_z + lambda_base.unsqueeze(-1) * residual_z
```

What each knob does:

| Knob | Meaning | Safety cap |
|---|---|---|
| `lambda_base ∈ [0, 0.8]` | How much of the residual correction we trust today. Base is always at least 20%. | Hard: 0.8 max, init near 0.1 |
| `shape_gain ∈ [0.7, 1.3]` | News can stretch/shrink the diurnal shape amplitude by ≤ 30%. | Hard: ±0.3 |
| `spike_bias ∈ R` | News can raise/lower the spike-gate logit (probability), never the magnitude. | Soft L2 reg |

**Crucial invariants**:
1. **There is no additive news residual term anywhere.** `r_d` enters only through `×` or via the sigmoid of spike_gate.
2. **When `relevance_mass=0`, all three knobs are forced to identity** → `lambda_base = 0.8 * σ(-2) ≈ 0.1`, `shape_gain = 1`, `spike_bias = 0`. The model reduces to base + (small-trust) TS residual. Note that `lambda_base` is never exactly zero; by design we keep residual turned down but never off when news is absent. If the dataset benefits from a full mute, change to `lambda_base = 0.8 * σ(trust_logit - 2) * active + λ₀ * (1 - active)` with a learned baseline `λ₀`.
3. **News cannot originate a spike in magnitude.** It can only raise `P(spike)`; the $ amount of the spike comes from `spike_ts` which only sees the TS encoder. This is the key barrier against a news-triggered price blow-up.
4. **Scale-space safety.** Worst-case news-induced z-error is `Δshape * 0.3 + small_logit_wiggle` — bounded. With price `σ=635`, the bounded multiplicative factor can only distort shape by ≤ 30%, not inject free dollars.

### 4.4 Module D — Training regime (blanking, dropout, pretraining)

**[NEW]** Self-supervised regime pretraining (replaces `pretrain.py`'s slow-R² warmup, which is the thing that output R²=-0.30):

Before the residual stage, train `NewsDayEncoder + RegimeProjMLP + (three modulation MLPs)` against *three targets computed from the time series itself*, with no residual supervision:

| Pretrain target | Label source | Loss |
|---|---|---|
| `vol_ratio_{d+1}` = log(std(y_{d+1}) / std(y_{d-5:d})) | from series | MSE |
| `spike_flag_{d+1}` = 1 if any step in day d+1 has `|y - y_rolling_mean| > 3σ_rolling` | from series | BCE |
| `shape_dev_{d+1}` = L2(normalized shape of day d+1, 30-day avg shape) | from series | MSE |

Heads for these three targets read `r_d` through their own small MLPs; they share the RegimeProjMLP. After 10–15 epochs:
- If the news corpus contains no signal for these targets → the encoder converges to outputting ≈ 0, which is the correct behavior. No fabrication penalty.
- If it contains signal → the encoder learns exactly the representation that *predicts series behavior*, which is what we want before we expose it to the residual task.

Then freeze `NewsDayEncoder + RegimeProjMLP`, initialize the three modulation MLPs from a shallow transfer (copy the vol/spike/shape heads' first layer as init for MLP_shape_gain/MLP_spike_prior/MLP_trust respectively), and train only the modulation MLPs + the TS residual heads on the main residual objective.

**[NEW]** Training-time news blanking:
```
p_blank = 0.3            # force model to work without news
per-sample mask: with prob p_blank set relevance_mass[d] = 0 for this sample
```
Blanking is applied in addition to the natural `active=0` rows. This prevents gate collapse and forces the base path to stay strong.

**[NEW]** News-off counterfactual consistency loss:
```
loss_consistency = 0.05 * || y_hat_z(news_on) - y_hat_z(news_off) ||^2 * (1 - active)
```
Only penalizes rows where there is no in-force news; says "if there is no news, the two paths must match". This catches any latent leakage of the RegimeProjMLP into the output when it shouldn't be contributing.

**[NEW]** Residual loss in *raw space with robust winsorization* for price targets:
```
y_raw = denormalize(y_hat_z)
y_raw_w = clip(y_raw, q_0.005, q_0.995)           # winsorize at 0.5%/99.5% of train distribution
y_true_w = clip(y_true, q_0.005, q_0.995)
loss_point = SmoothL1(y_raw_w, y_true_w) + 0.1 * Quantile(y_raw, y_true, q=[0.1, 0.5, 0.9])
```
This removes the `σ=635` amplification pathway: gradients don't see raw spikes at all, so they can't drive the model toward large z-space outputs. For load (small σ, no extreme tails) you can stay in z-space SmoothL1 as before.

### 4.5 Module E — Diagnostics & evaluation (how we prove it's working)

Three mandatory eval artifacts, all cheap to compute:

1. **Blank-news counterfactual test.**
   At eval time, run two passes: (a) normal, (b) with `r_d = 0` on all rows. Report:
   - `mae_normal vs mae_blank` on the **news-active subset** (`relevance_mass > threshold_0.7`)
   - `mae_normal vs mae_blank` on the **news-inactive subset**
   - Accept only if news-active subset improves and news-inactive subset is **within ±1%** (i.e. news does nothing when it shouldn't).

2. **Date-shuffle permutation test.**
   Randomly misalign news dates to time series dates; retrain briefly or just re-evaluate with a scrambled bank lookup. Eval MAE on news-active subset must degrade clearly vs the aligned version, otherwise the model isn't using news at all.

3. **Knob distribution logging.**
   Every epoch, dump histograms of `(lambda_base, shape_gain, spike_bias)` stratified by `active`. Expected pattern:
   - `active=0`: all three collapse to identity (mode on 0.1 / 1.0 / 0.0) — if not, the consistency loss is too weak.
   - `active=1, high relevance_mass`: `lambda_base` spread to the right, `shape_gain` with real variance, `spike_bias` non-trivial.

   Any violation → stop and debug before believing the validation metric.

### 4.6 Data-normalization side note (outside the fusion redesign proper, but blocks it)

The price dataset's global `σ=635` poisons every fusion design. Before (or alongside) this redesign, do **one** of the following to the price pipeline:
- (a) **Robust stats**: compute `μ, σ` from the winsorized (0.5%/99.5%) training distribution.
- (b) **Per-series log1p**: forecast `log1p(y)` instead of `y`, denormalize at the end.
- (c) **Per-window local zscore**: normalize each 48-step history by its own local stats.

(b) is the simplest and usually the most stable for NEM prices. This is not optional — with `σ=635`, even a *correct* fusion design would look broken in the MAE metric.

---

## 5. Why this design will work — theoretical analysis

This section explains *why*, not *that*. Each argument is tied to the failure modes from §1.

### 5.1 It matches the information content of macro news
Macro news describes slow, distributional, window-valid conditions. The regime bank encodes exactly that: a smoothed, in-force-weighted vector with an explicit validity window. We are no longer asking the encoder to answer "what did the LLM say about direction × magnitude at this timestamp" (a question macro news cannot answer); we are asking "what is the market-wide context for the next few days" (a question it *can* answer).

### 5.2 Multiplicative-only coupling is fail-safe
The fundamental failure mode of §1 was that a bad news path added raw-space dollars to the prediction. In the new design, news can only scale or gate terms that are themselves bounded (shape with amplitude bounded by the TS encoder, spike gate ∈ [0,1]). Even a maximally wrong news encoder can shift the answer by ≤ 30% of the TS-predicted shape amplitude. The blow-up pathway is physically closed.

### 5.3 Self-supervised targets eliminate label rot
The current pipeline trusts the LLM to emit `direction/magnitude`, then trains on that. If the LLM is wrong (which on retrospective/regulatory articles it always is), the model learns wrong labels. In the new design, the encoder is trained *against targets derived from the series itself* (vol, spike, shape deviation). If a piece of news has no predictive value for those three quantities, the encoder is rewarded for ignoring it. There is no way to get fooled by LLM overconfidence — we never promote LLM outputs to training targets.

### 5.4 Hard sparsity prevents gate-collapse
The current model went from `gate_init_bias = -1` to `gate = 0.74` in one epoch because the additive news head had easy gradient wins on high-variance outlier days. In the new design, news-active rows are a strict minority (let's say 25% of training), and news-blanking (p=0.3) plus the consistency loss further suppress reliance. The model learns that on the 75%+ of rows with no relevant news, knobs must be identity — and once it learns that, the TS residual path stays in shape.

### 5.5 Spike probability ≠ spike magnitude is the physically correct factorization
The only defensible claim from a heatwave article is "the probability of a price spike tomorrow is higher". It is *not* a claim about the dollar amount. The new design enforces this factorization at the architecture level: `MLP_spike_prior` can only modify the logit of the spike gate; `spike_ts` (the magnitude head) sees only the time series. Cannot hallucinate a $1000 spike from a weather tweet.

### 5.6 Raw-space winsorized loss closes the zscore amplification loop
Even if the news path were perfectly calibrated, the current global zscore on price guarantees that any z-output of 1 reverse-maps to 635 raw MAE. The redesign fixes this at two independent points: (a) multiplicative coupling means the news cannot produce a free z-offset in the first place, and (b) the raw-space winsorized loss ensures gradients don't reward overshooting during training. Two belts, one pair of trousers.

### 5.7 Graceful degradation is a property, not a hope
The explicit promise: when `relevance_mass=0`, everything collapses to a *slightly-muted TS-only delta*. This is enforced by the `active` gate and tested by the consistency loss. The current design has no such property — it can actively harm the base when news is uninformative, which is exactly what the log shows.

### 5.8 This is aligned with how analysts actually use this news
Read the wattclarity dataset from a human trader's perspective. No one trades the open on "AER released quarterly report". What they do use it for is: am I in a stressed-market week (wider stops), is tomorrow a heatwave day (expect higher intraday peaks), is a policy in effect that changes rebidding (expect volatility). These are *exactly* `lambda_base`, `shape_gain`, and `spike_bias` — respectively. The design encodes the traders' own mental model.

---

## 6. Scope of change, in practice

**Stays unchanged**:
- `base_backbone` + frozen base forecast
- `PatchTSTTSEncoder` + `ts_tokens/ts_summary`
- Slow / shape / spike TS heads (just rewired downstream)
- Hard residual sampler conceptually (may tune hit_rate later)

**Removed**:
- `slow_news_head`, `shape_news_head` and all additive news branches in `model.py:90-94`
- `CrossAttentionFusion.gate_mlp` path as currently wired (the gate scalar is repurposed but the additive coupling is gone)
- `schema_refine.py` fielded event/magnitude/direction outputs
- `pretrain.py` slow-R² warmup (replaced by self-supervised vol/spike/shape pretrain)
- `NewsDayEncoder`'s 3-day softmax "which day" attention

**New**:
- `schema_refine_v2` (regime descriptors, `is_actionable` escape hatch, closed topic vocab)
- `build_regime_bank` (in-force filter, exponential decay, EMA smoothing, topic-tag mass)
- `MLP_trust`, `MLP_shape_gain`, `MLP_spike_prior` modulation heads
- Self-supervised pretraining stage with vol/spike/shape heads
- Training-time news blanking + consistency loss + optional news-off counterfactual eval
- Raw-space winsorized loss for price targets
- Knob-histogram logging per epoch

Approximate code surface: ~1 file touched to remove fused additive heads (`model.py`), ~2 files replaced (`schema_refine_v2`, `regime_bank`), ~2 new files (`pretrain_v2`, `modulation_heads`), ~1 file edited for training/eval loop (`trainer.py`). No change to dataloader or base backbone.

---

## 7. Acceptance criteria (how we know it worked)

A run is considered successful **only** if all four of these hold on `nsw_price`:

1. **No regression without news.** `val_mae` on the news-inactive subset within ±1% of base `val_mae = 33.6`. (Proof that the system degrades gracefully.)
2. **Clear gain with news.** `val_mae` on the news-active subset **strictly below** base `val_mae` by a margin ≥ 2 × std of 5-seed noise.
3. **Permutation collapse.** Under date-shuffled news, news-active subset's MAE rises back to base level (proof that the gain is actually from news alignment, not from architectural noise).
4. **Knob sanity.** In the final-epoch histograms, `active=0` rows show `lambda_base<0.15`, `shape_gain∈[0.99,1.01]`, `spike_bias≈0`; `active=1` rows show spread.

If any of the four fails, we do **not** ship the design — we iterate on Module D (training recipe), not on Modules B/C (architecture).

---

## 8. Open questions for follow-up

- **Topic-conditional modulation.** Should `MLP_spike_prior` be three separate heads routed by `topic_tag_mass`, one each for `heatwave / outage / policy`? Plausibly yes, but start with the single-head version and only add topic routing if knob histograms show it's warranted.
- **Shared regime bank across datasets.** Australian NEM load and price are the same physical market; the regime embedding should be shareable between `nsw_load` and `nsw_price`. Consider building one bank per market region, not per dataset.
- **Cross-dataset pretraining.** Self-supervised vol/spike/shape targets exist for both load and price; pretraining the RegimeProjMLP on *both* series' targets simultaneously should be strictly better than per-dataset pretraining.
- **Short-horizon news decay τ.** τ=5 is a guess. A grid {3, 5, 7, 10} should be swept once on a single seed.
