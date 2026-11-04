# Phase 2 — UCI obesity lifestyle model evaluation

> Training timestamp (UTC): 2026-11-04T09:26:34.237385+00:00
> Educational classification only; not a medical diagnostic tool.

## Dataset and leakage controls

- Source file: `data\raw\ObesityDataSet_raw_and_data_sinthetic.csv`
- SHA-256: `28b6991062855d3d15642e38b1aa9d1b32fb7abbb5abf5b096f1868cd706154b`
- Rows before exact deduplication: 2111
- Exact duplicate rows removed before splitting: 24
- Rows after deduplication: 2087
- Target: `NObeyesdad` (seven original classes)
- Features used (14): `Age`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`, `Gender`, `family_history_with_overweight`, `FAVC`, `CAEC`, `SMOKE`, `SCC`, `CALC`, `MTRANS`
- Excluded from primary model: `Height`, `Weight` (anthropometric shortcut/leakage risk because target categories are BMI-derived).
- All imputers, scaling and one-hot encoding are fitted inside each training fold via scikit-learn Pipeline.
- Exact duplicate rows are removed before the stratified holdout split. The source also contains synthetic records; related synthetic neighbours may still cross splits.

## Cross-validation comparison

Five-fold StratifiedKFold on the training partition; selection metric: macro-F1.

| Model | Macro-F1 mean | Macro-F1 std | Balanced accuracy mean |
|---|---:|---:|---:|
| Random Forest | 0.8384 | 0.0132 | 0.8388 |
| Decision Tree | 0.7221 | 0.0265 | 0.7253 |
| Logistic Regression | 0.5730 | 0.0312 | 0.5921 |
| Dummy (most_frequent) | 0.0412 | 0.0002 | 0.1429 |

## Final holdout test — Random Forest

The holdout test is evaluated once after model selection using cross-validation.

| Metric | Value |
|---|---:|
| accuracy | 0.8421 |
| balanced_accuracy | 0.8394 |
| macro_precision | 0.8382 |
| macro_recall | 0.8394 |
| macro_f1 | 0.8379 |
| weighted_f1 | 0.8415 |

### Per-class classification report

```text
                     precision    recall  f1-score   support

Insufficient_Weight       0.86      0.91      0.88        53
      Normal_Weight       0.71      0.77      0.74        57
     Obesity_Type_I       0.88      0.83      0.85        70
    Obesity_Type_II       0.88      0.95      0.91        60
   Obesity_Type_III       1.00      0.98      0.99        65
 Overweight_Level_I       0.80      0.75      0.77        55
Overweight_Level_II       0.74      0.69      0.71        58

           accuracy                           0.84       418
          macro avg       0.84      0.84      0.84       418
       weighted avg       0.84      0.84      0.84       418
```

### Confusion matrix

```text
[
  [
    48,
    4,
    0,
    0,
    0,
    0,
    1
  ],
  [
    3,
    44,
    1,
    0,
    0,
    4,
    5
  ],
  [
    0,
    3,
    58,
    4,
    0,
    1,
    4
  ],
  [
    0,
    1,
    0,
    57,
    0,
    1,
    1
  ],
  [
    0,
    1,
    0,
    0,
    64,
    0,
    0
  ],
  [
    3,
    4,
    3,
    1,
    0,
    41,
    3
  ],
  [
    2,
    5,
    4,
    3,
    0,
    4,
    40
  ]
]
```

## Limitations

- UCI reports that 77% of the 2,111 records are synthetic (Weka/SMOTE); the CSV has no row-level provenance flag.
- Exact deduplication does not prevent related synthetic observations from appearing in both train and test partitions.
- The target is BMI-derived. This experiment predicts the dataset's BMI-defined category from survey/lifestyle variables; it does not establish clinical prediction or population prevalence.
- Results are dataset-specific and require independent external validation before any real-world use.

## Reproducibility

- Random state: 42
- Split: stratified 80% train / 20% holdout test
- CV: 5-fold stratified, training partition only
- Python: 3.14.6
- scikit-learn: 1.9.1
