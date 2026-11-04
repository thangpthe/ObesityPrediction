# ObesityPrediction

A machine learning project for multi-class obesity-level classification using the UCI Estimation of Obesity Levels Based on Eating Habits and Physical Condition dataset.

## Project Overview

The project explores machine learning methods for classifying seven obesity-related categories based on eating habits and physical condition.

The repository contains the original project implementation and a reproducible machine learning pipeline developed in Phase 2.

**Important:** This is an educational research project, not a clinical diagnostic system.

## Dataset

- Original records: 2,111
- Exact duplicate records removed: 24
- Records after deduplication: 2,087
- Target classes: 7
- Primary model features: 14

`Height` and `Weight` are excluded from the primary model because the target is BMI-derived.

The UCI dataset contains synthetic records. Results should therefore be interpreted cautiously and should not be considered evidence of clinical validity or external generalization.

## Phase 2 – Model Development, Evaluation and Analysis

Phase 2 establishes a reproducible machine learning pipeline and evaluates multiple classification algorithms.

### Models Compared

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest

Random Forest achieved the highest baseline cross-validation Macro-F1.

| Metric                    |   Random Forest |
| ------------------------- | --------------: |
| CV Macro-F1               | 0.8384 ± 0.0132 |
| Holdout Accuracy          |          0.8421 |
| Holdout Balanced Accuracy |          0.8394 |
| Holdout Macro-F1          |          0.8379 |
| Holdout Weighted-F1       |          0.8415 |

The holdout set has been examined in multiple analyses. Its metrics are reported for transparency and should not be interpreted as an untouched final evaluation.

### Additional Analyses

- Permutation feature importance.
- Gender subgroup performance evaluation.
- Gender feature ablation study.
- Random Forest hyperparameter tuning using RandomizedSearchCV.

Hyperparameter tuning did not improve the baseline CV Macro-F1. The original Random Forest configuration is retained as the primary model.

## Repository Structure

```text
ObesityPrediction/
├── data/
├── models/
├── reports/
│   ├── data_audit.md
│   ├── model_comparison.csv
│   ├── model_evaluation.md
│   ├── feature_importance.md
│   ├── gender_performance.md
│   ├── gender_ablation.md
│   ├── random_forest_tuning.md
│   └── phase2_final_report.md
├── src/
│   └── training/
│       ├── train_uci.py
│       ├── analyze_feature_importance.py
│       ├── analyze_gender_performance.py
│       ├── analyze_gender_ablation.py
│       └── tune_random_forest.py
├── requirements-phase2.txt
└── README.md
```

## Setup

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-phase2.txt
```

Python 3.10 or later is recommended.

## Run the ML Pipeline

Run commands from the repository root.

Train and evaluate baseline models:

```powershell
python -m src.training.train_uci
```

Analyze feature importance:

```powershell
python -m src.training.analyze_feature_importance
```

Evaluate performance by gender:

```powershell
python -m src.training.analyze_gender_performance
```

Run gender ablation:

```powershell
python -m src.training.analyze_gender_ablation
```

Tune Random Forest hyperparameters:

```powershell
python -m src.training.tune_random_forest
```

See [src/training/README.md](src/training/README.md) for detailed instructions.

## Reports

The main Phase 2 report is available at:

[reports/phase2_final_report.md](reports/phase2_final_report.md)

Additional reports and CSV results are stored in the `reports/` directory.

## Limitations

- The dataset contains synthetic observations.
- Related observations may occur across training and holdout partitions.
- The holdout set has been examined during exploratory analyses.
- Subgroup target distributions are imbalanced.
- Independent external validation is required before making stronger generalization claims.
- The experiment does not establish clinical usefulness or diagnostic reliability.

## Project Status

**Phase 2: Completed – baseline model development, evaluation, and exploratory analysis.**
