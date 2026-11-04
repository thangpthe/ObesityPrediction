# Phase 2 — ML pipeline

This package trains and evaluates the primary seven-class UCI obesity-level
experiment. It intentionally excludes `Height` and `Weight` from predictors
because the target is BMI-derived.

## Setup (Windows PowerShell)

Run from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-phase2.txt
```

If Python 3.11 is not installed, use an installed Python 3.10+ version.

## Train and evaluate

```powershell
python -m src.training.train_uci
```

Optional custom paths:

```powershell
python -m src.training.train_uci --data data/raw/ObesityDataSet_raw_and_data_sinthetic.csv
```

## Outputs

- `models/uci_obesity_lifestyle_v1.joblib`: selected preprocessing + estimator pipeline.
- `models/uci_obesity_lifestyle_v1.json`: model and dataset metadata.
- `reports/model_comparison.csv`: five-fold CV comparison.
- `reports/model_evaluation.md`: final holdout metrics, class report and confusion matrix.

Generated model artifacts are ignored by Git. Commit the evaluation report and
comparison CSV only after reviewing the actual run results.

## Method

- Remove exact duplicate rows before splitting.
- Stratified 80/20 holdout split; compare models with five-fold stratified CV on
  the training partition only.
- Compare DummyClassifier, Logistic Regression, Decision Tree and Random Forest.
- Select by CV macro-F1, then evaluate the selected model once on holdout test.
- Fit imputation, scaling and one-hot encoding inside each Pipeline fold.
- Do not interpret scores as clinical performance: UCI reports that 77% of
  records are synthetic, and related synthetic observations may cross splits.
