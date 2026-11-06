## Q1 I suggest dropping the underscore z to indicate z-space normalization since the reader already expects the data to be normalized and the notation may mislead the reader into thinking that z is another variable.

This is a helpful suggestion. We used the subscript mainly to remind readers that the bounded residual guarantee is stated in normalized space, but we agree that repeated `_z` notation can make the equations look heavier than necessary and may suggest an additional variable. We will simplify the notation in the main text and state once, before the forecasting equations, that all variables in that section are expressed in normalized space unless otherwise specified. When we return to original-scale metrics, we will make that conversion explicit in the experiment section.

[Need manuscript revision: simplify z-space notation in Section 3 and add one clear normalization sentence.]

## Q2 I think Figure 1 would benefit from being larger and including variable names, or another solution could be to add a table in the Appendices to gather the notation used and their relation to Figure 1.

We agree that the connection between Figure 1 and the equations can be made easier to follow. The current figure emphasizes the pipeline, but it does not expose enough of the mathematical notation used later in Section 3. We will enlarge the figure where space allows and add key variable names to the delta-stage nodes, such as the slow residual, shape residual, spike residual, and the news-conditioned modulation knobs. To avoid overcrowding the figure, we will also add an appendix notation table that maps each visual block in Figure 1 to the corresponding symbol and equation.

[Need manuscript revision: update Figure 1 or add variable labels; add appendix notation table linking figure blocks to equations.]

## Q3 In the JSON produced by the LLM, the term renewable_surplus may be too specific to the energy sector and may act more generally as the opposite of tightness in non-energy domains.

You are right that the name `renewable_surplus` is energy-specific and becomes awkward in the Traffic setting. In the implementation, this slot is interpreted through domain-specific prompt instructions, so for Traffic it functions as a "mobility relief / alternate capacity" axis rather than literal renewable supply. However, the naming in the paper can make the schema look less domain-general than the actual usage.

We will revise the description to distinguish between the canonical schema name used in our implementation and the domain-specific interpretation of that axis. A clearer name such as `relief_or_substitution` would better express the intended role across domains, while the energy-specific interpretation can remain as one instantiation. We will also clarify in the prompt appendix that the five axes are semantic slots whose meanings are adapted per domain under fixed range constraints.

[Need manuscript revision: clarify the domain-specific meaning of the `renewable_surplus` axis; optionally rename the axis in the paper to a more general term.]

## Q4 In Eq. (6), I am not sure what the overline in the variables refers to.

Thank you for pointing this out. The overline is meant to denote aggregated regime-bank quantities after confidence weighting, time decay, and causal EMA smoothing. This is explained in the regime-bank construction, but the notation is not sufficiently local to Eq. (6). We will revise the equation text so that the meaning of the overline is stated immediately when the variables first appear.

[Need manuscript revision: define the overline near Eq. (6) as confidence/time-decay aggregated and causally smoothed regime-bank features.]

## Q5 In Appendix K, adding plots of the base model prediction against the target would help understand what the correction is trying to correct.

This is a good point. The current Appendix K focuses on decomposing the final residual correction, but it does not show the base forecast and target in the same panel. Adding them would make it much clearer whether the correction is compensating for a level bias, a missing daily pattern, or a localized spike. We will update the appendix figures to include the base prediction and target, or add an adjacent panel for them if the original decomposition plot becomes too crowded.

For the Electricity Load example you mention, this would also help determine whether the visible daily pattern in the correction reflects residual seasonality left by the frozen base model or a news-conditioned change in intra-day shape.

[Need manuscript revision: update Appendix K figures to include base forecast and target, or add a paired base-vs-target panel.]

## Q6 Figure in Appendix K would be cleaner if it did not include the name of the .csv file below each plot.

Agreed. Those file names are internal debugging labels and should not appear in the final paper figures. We will remove them from the Appendix K plots.

[Need manuscript revision: regenerate Appendix K figures without CSV file names.]

## Q7 Although this is not necessary for the relevance of the work, I am wondering how the approach would perform if tested on the COVID period without seeing it during training.

This is an interesting stress test because COVID-like lockdowns are exactly the kind of exogenous regime shift where historical numerical patterns alone can be insufficient. We did not include a COVID-period experiment in the current version, mainly because our present datasets and news corpora are from later periods and were chosen to avoid leakage-controlled alignment issues across very different data sources.

We will mention this as a valuable future evaluation setting. If data access and consistent news coverage permit, we can also add a preliminary out-of-distribution experiment in which the model is trained before a major disruption period and tested on the disruption period. We would expect the result to depend strongly on whether the news bank contains timely actionable lockdown or mobility-restriction signals, which is consistent with REIN's design: the method can help only when the relevant external event is represented in the news stream.

[Optional additional experiment: COVID or other major disruption out-of-distribution test. Otherwise add as future work/limitation.]
