# Phase 2 — Gender subgroup evaluation

- Holdout samples: 418
- Overall holdout Macro-F1: 0.8379

## Metrics by gender

| Gender | Samples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Female | 198 | 0.8990 | 0.7235 |

### Classification report — Female

```text
                     precision    recall  f1-score   support

Insufficient_Weight       0.85      0.97      0.90        34
      Normal_Weight       0.86      0.78      0.82        23
     Obesity_Type_I       0.91      0.89      0.90        36
    Obesity_Type_II       0.00      0.00      0.00         0
   Obesity_Type_III       1.00      1.00      1.00        64
 Overweight_Level_I       0.83      0.89      0.86        27
Overweight_Level_II       0.70      0.50      0.58        14

           accuracy                           0.90       198
          macro avg       0.74      0.72      0.72       198
       weighted avg       0.90      0.90      0.90       198
```
| Male | 220 | 0.7909 | 0.6670 |

### Classification report — Male

```text
                     precision    recall  f1-score   support

Insufficient_Weight       0.88      0.79      0.83        19
      Normal_Weight       0.63      0.76      0.69        34
     Obesity_Type_I       0.84      0.76      0.80        34
    Obesity_Type_II       0.88      0.95      0.91        60
   Obesity_Type_III       0.00      0.00      0.00         1
 Overweight_Level_I       0.77      0.61      0.68        28
Overweight_Level_II       0.75      0.75      0.75        44

           accuracy                           0.79       220
          macro avg       0.68      0.66      0.67       220
       weighted avg       0.79      0.79      0.79       220
```

## Actual target distribution by gender

NObeyesdad  Insufficient_Weight  Normal_Weight  Obesity_Type_I  Obesity_Type_II  Obesity_Type_III  Overweight_Level_I  Overweight_Level_II
Gender                                                                                                                                    
Female                   0.1717         0.1162          0.1818           0.0000            0.3232              0.1364               0.0707
Male                     0.0864         0.1545          0.1545           0.2727            0.0045              0.1273               0.2000

## Interpretation cautions

- Subgroup metrics depend on subgroup sample size.
- A difference in metrics does not by itself establish bias.
- Gender importance measures model reliance, not causality.
- The dataset contains synthetic observations and requires independent external validation.
