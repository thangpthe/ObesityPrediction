# Phase 2 — Random Forest hyperparameter tuning

- Training samples: 1669
- Holdout samples: 418
- Search method: RandomizedSearchCV
- Number of configurations: 30
- Cross-validation: 5-fold StratifiedKFold
- Selection metric: Macro-F1
- Features: 14, including Gender

## Baseline comparison

- Baseline CV Macro-F1: 0.8384
- Baseline CV standard deviation: 0.0132
- Best tuned CV Macro-F1: 0.8384
- Difference: +0.0000

## Best hyperparameters

```json
{
  "classifier__n_estimators": 400,
  "classifier__min_samples_split": 2,
  "classifier__min_samples_leaf": 2,
  "classifier__max_features": "sqrt",
  "classifier__max_depth": 30,
  "classifier__class_weight": "balanced"
}
```

## Top 10 configurations

| Rank | Mean CV Macro-F1 | Std |
|---:|---:|---:|
| 1 | 0.8384 | 0.0132 |
| 2 | 0.8357 | 0.0257 |
| 3 | 0.8331 | 0.0254 |
| 4 | 0.8323 | 0.0266 |
| 5 | 0.8321 | 0.0166 |
| 6 | 0.8300 | 0.0264 |
| 7 | 0.8294 | 0.0159 |
| 8 | 0.8264 | 0.0153 |
| 9 | 0.8220 | 0.0273 |
| 10 | 0.8197 | 0.0288 |

## Exploratory holdout results

The holdout has already been examined in earlier Phase 2 analyses. These values are reported for transparency only and must not be used for further model selection.

| Metric | Value |
|---|---:|
| accuracy | 0.8421 |
| balanced_accuracy | 0.8394 |
| macro_f1 | 0.8379 |
| weighted_f1 | 0.8415 |

### Classification report

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

## Limitations

- Randomized search CV score is subject to model-selection optimism.
- A separate external dataset is needed for independent validation.
- Holdout metrics are exploratory because the holdout was examined previously.
- This dataset is not a clinical validation cohort.
