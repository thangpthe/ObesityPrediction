# Phase 2 — ML Pipeline

This package trains, evaluates, and analyzes the primary seven-class UCI obesity-level classification experiment.

The primary model intentionally excludes `Height` and `Weight` because the target is BMI-derived.

## Setup (Windows PowerShell)

Run from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-phase2.txt
```

Python 3.10 or later can also be used.

## 1. Train and Evaluate Baseline Models

```powershell
python -m src.training.train_uci
```

This script:

- Removes exact duplicate rows.
- Creates a stratified 80/20 train-holdout split.
- Compares DummyClassifier, Logistic Regression, Decision Tree, and Random Forest.
- Uses five-fold StratifiedKFold cross-validation on the training partition.
- Selects the model using CV Macro-F1.
- Evaluates the selected model on the holdout partition.

## 2. Feature Importance

```powershell
python -m src.training.analyze_feature_importance
```

Performs permutation importance analysis and generates:

- `reports/permutation_importance.csv`
- `reports/feature_importance.md`

## 3. Gender Subgroup Evaluation

```powershell
python -m src.training.analyze_gender_performance
```

Evaluates model performance separately for Female and Male holdout subgroups.

Generates:

- `reports/gender_performance.md`
- `reports/gender_subgroup_metrics.csv`
- `reports/gender_target_distribution.csv`

## 4. Gender Ablation Study

```powershell
python -m src.training.analyze_gender_ablation
```

Compares cross-validation performance with and without the Gender feature.

Generates:

- `reports/gender_ablation.md`
- `reports/gender_ablation.csv`

## 5. Random Forest Hyperparameter Tuning

```powershell
python -m src.training.tune_random_forest
```

Uses RandomizedSearchCV with 30 configurations and five-fold stratified cross-validation.

Generates:

- `reports/random_forest_tuning.md`
- `reports/random_forest_tuning_results.csv`

The tuned model is saved separately. The baseline model is not overwritten.

## Outputs

Baseline model artifacts:

- `models/uci_obesity_lifestyle_v1.joblib`
- `models/uci_obesity_lifestyle_v1.json`

Tuned model artifacts:

- `models/uci_obesity_lifestyle_tuned_v1.joblib`
- `models/uci_obesity_lifestyle_tuned_v1.json`

Generated reports are stored in `reports/`.

Generated model artifacts are ignored by Git. Do not overwrite legacy model files.

## Methodological Notes

- Remove exact duplicate rows before splitting.
- Use stratified train-holdout splitting and stratified cross-validation.
- Fit imputation, scaling, and one-hot encoding inside the pipeline.
- Use Macro-F1 as the primary model selection metric.
- Treat previously examined holdout results as exploratory.
- Interpret permutation importance as model reliance, not causality.
- Do not interpret subgroup differences alone as proof of bias or fairness.
- Do not interpret model scores as clinical performance.

The UCI dataset contains a substantial proportion of synthetic records. Related observations may cross data partitions, and independent external validation is needed to assess generalization.

See `reports/phase2_final_report.md` for the consolidated Phase 2 results.
