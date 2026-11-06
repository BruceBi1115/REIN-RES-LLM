## Q1 Could the authors provide a component-wise ablation study of the three residual channels?

Yes. This is the most direct way to show whether the slow/shape/spike decomposition is useful beyond adding model capacity. We will add component-wise ablations where each residual channel is removed while keeping the rest of the training and regime-bank pipeline unchanged. Concretely, we will evaluate `w/o slow`, `w/o shape`, and `w/o spike`, and compare them with the full REIN model and with a single-head residual variant.

These ablations will also help interpret the role of each channel across datasets. For example, we expect the slow channel to matter more for persistent demand or gas-demand shifts, the shape channel to matter more for traffic and holiday-related intra-day changes, and the spike channel to matter most for heavy-tailed price or incident-driven corrections. If the ablation results do not support this pattern, we will revise the interpretation accordingly rather than overclaiming channel semantics.

[Need additional experiments: w/o slow, w/o shape, w/o spike, and preferably single residual head. Report both MSE and MAE across datasets/horizons or at least representative horizons with full results in appendix.]

## Q2 REIN fails to outperform numerical baselines on the Traffic dataset.

We will clarify this limitation. Traffic is the weakest setting for REIN in the current evaluation, and we should not present the results as uniformly dominant across all domains. The likely reason is that the traffic target is highly local and sensor-dependent, while the available GDELT-derived news stream is broader and does not always capture the specific road incidents, local closures, commuter shifts, or station-level disruptions that drive hourly traffic counts. In that situation, the bounded design helps prevent large degradation, but it also limits the possible gain from weakly matched news.

We will revise the discussion to make this point explicit. The Traffic result is useful because it shows that REIN does not automatically improve a forecast simply by adding text; it improves most when the news corpus contains timely, actionable information aligned with the target. We will also strengthen the active/blank/permuted analysis for Traffic so that the reader can see where aligned news helps and where the corpus coverage is insufficient.

[Need manuscript revision: explicitly discuss Traffic as a limitation. Optional additional analysis: Traffic active subset by news relevance mass, event type, or high-error windows.]

## Q3 Relying on an LLM to process raw news adds computational overhead and architectural complexity to the pipeline.

This concern is fair, and we will make the cost profile clearer. In REIN, the LLM is used offline to refine raw articles into structured regime records; it is not called during model training or test-time forecasting. Once the regime bank is built, the online forecaster only retrieves compact numerical vectors, topic masses, embeddings, and relevance mass from cache. This keeps the forecasting-time cost close to an ordinary residual model rather than an LLM-in-the-loop system.

We will add a clearer runtime and complexity discussion. The current implementation caches refined records and reuses them across all horizons and ablations, so the LLM cost is a one-time preprocessing cost for each corpus. We will also report the number of processed articles, refinement time, and the delta-stage parameter count so the added complexity can be evaluated against the forecasting gains and the bounded-safety mechanism.

[Need manuscript revision: add or strengthen computational-cost paragraph. If not already prominent, report offline refinement time, number of articles, cache reuse, and no LLM calls at forecast time.]
