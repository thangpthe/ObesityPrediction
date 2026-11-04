# Phase 2 — Permutation Feature Importance

- Dataset SHA-256: `28b6991062855d3d15642e38b1aa9d1b32fb7abbb5abf5b096f1868cd706154b`
- Holdout samples: 418
- Baseline holdout Macro-F1: 0.8379
- Method: permutation importance
- Scoring: macro-F1
- Repeats: 30
- Random state: 42

Importance is the decrease in Macro-F1 after randomly permuting one input feature. Larger positive values indicate greater reliance by this fitted model.

| Rank | Feature | Mean importance | Std |
|---:|---|---:|---:|
| 1 | Gender | 0.26261 | 0.01590 |
| 2 | Age | 0.17435 | 0.01446 |
| 3 | FCVC | 0.16828 | 0.01554 |
| 4 | family_history_with_overweight | 0.16056 | 0.01437 |
| 5 | CALC | 0.11080 | 0.01352 |
| 6 | NCP | 0.07641 | 0.00868 |
| 7 | CAEC | 0.06957 | 0.00955 |
| 8 | TUE | 0.05088 | 0.00972 |
| 9 | CH2O | 0.04486 | 0.00909 |
| 10 | FAF | 0.04035 | 0.01082 |
| 11 | MTRANS | 0.03195 | 0.00856 |
| 12 | FAVC | 0.02719 | 0.00516 |
| 13 | SCC | 0.00661 | 0.00185 |
| 14 | SMOKE | 0.00027 | 0.00081 |

## Interpretation cautions

- Importance measures model reliance, not causality.
- Correlated features can share or mask importance.
- Near-zero or negative values do not prove a feature is useless.
- Results are specific to this dataset and holdout split.
- The dataset contains synthetic observations; external validation is still required.
