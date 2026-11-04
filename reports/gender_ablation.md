# Phase 2 — Gender ablation study

- Training samples: 1669
- Evaluation: 5-fold Stratified Cross-Validation
- Estimator: Random Forest
- Same random state and hyperparameters in both experiments
- Holdout test was not used for model selection

## Results

| Experiment | Features | Macro-F1 mean | Macro-F1 std | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|---:|
| With Gender | 14 | 0.8384 | 0.0132 | 0.8388 | 0.8424 |
| Without Gender | 13 | 0.8060 | 0.0191 | 0.8066 | 0.8101 |

## Difference

- Macro-F1 difference (with minus without Gender): +0.0324
- Balanced accuracy difference: +0.0322

## Interpretation

- A positive difference means including Gender improved the CV score.
- A negative difference means the model performed better without Gender.
- A small difference suggests Gender adds limited overall predictive value under this setup.
- This experiment measures predictive contribution, not causality or fairness.
- Subgroup performance and class distributions should also be considered.
- Results are specific to this dataset and CV design.
