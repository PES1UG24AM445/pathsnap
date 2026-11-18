# PathSnap held-out evaluator results

All 10 trips from the reserved evaluator split returned valid API responses, covering 19,674 GPS observations. The evaluation used frozen LightGBM and Viterbi settings from commit `48ec0db`.

| Metric | LightGBM only | LightGBM + Viterbi |
|---|---:|---:|
| Pooled Top-1 | 80.97% | 90.78% |
| Mean per-trip Top-1 | 88.63% | 93.95% |
| Mean route continuity | 98.75% | 99.94% |

Pooled Top-1 counts each observation equally. Mean per-trip Top-1 gives every trip equal weight. Route continuity measures connected transitions under the reconstruction's bounded directed-path rule.

Viterbi improved Top-1 on 10 of 10 trips. Median server time was 1.00 seconds on the local API run.

Model and decoder selection were completed on the development splits before this run. The API loaded the reference routes after prediction and used them to compute aggregate scores.

[Machine-readable aggregate results](../results/evaluator_summary.json)
