
"""Randomized hyperparameter tuning for Phase 2 Random Forest.

Run from repository root:
    python -m src.training.tune_random_forest

The original baseline model is never overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"

BASELINE_PATH = ROOT / "models/uci_obesity_lifestyle_v1.joblib"
BASELINE_METADATA_PATH = ROOT / "models/uci_obesity_lifestyle_v1.json"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "NObeyesdad"

NUMERIC_FEATURES = [
    "Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
N_ITER = 30


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", classifier),
    ])


def main() -> None:
    for path in [DATA_PATH, BASELINE_PATH, BASELINE_METADATA_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    X = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }

    # Baseline CV comparison using the original configuration.
    baseline = build_pipeline()
    baseline.set_params(
        classifier__n_estimators=400,
        classifier__min_samples_leaf=2,
        classifier__class_weight="balanced",
    )

    print("Evaluating baseline with 5-fold CV...")

    baseline_scores = cross_validate(
        baseline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    baseline_macro_f1 = baseline_scores["test_macro_f1"].mean()
    baseline_std = baseline_scores["test_macro_f1"].std()

    print(
        f"Baseline Macro-F1: {baseline_macro_f1:.4f} "
        f"+/- {baseline_std:.4f}"
    )

    # Search space.
    param_distributions = {
        "classifier__n_estimators": [300, 400, 500, 700],
        "classifier__max_depth": [None, 10, 15, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 3, 4],
        "classifier__max_features": ["sqrt", 0.7, 1.0],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample",
        ],
    }

    search = RandomizedSearchCV(
        estimator=build_pipeline(),
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1,
        return_train_score=False,
    )

    print("\nStarting RandomizedSearchCV...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_score = search.best_score_

    print("\nBest parameters:")
    print(search.best_params_)

    print(
        f"\nBest CV Macro-F1: {best_score:.4f}"
    )

    print(
        f"Difference from baseline CV: "
        f"{best_score - baseline_macro_f1:+.4f}"
    )

    # Holdout is reported for transparency only.
    # It must not be used to select further tuning iterations.
    predictions = best_model.predict(X_test)

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted")
        ),
    }

    labels = sorted(y.unique())
    class_report = classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
    )

    # Save as a separate model; do not overwrite baseline.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "uci_obesity_lifestyle_tuned_v1.joblib"
    metadata_path = MODEL_DIR / "uci_obesity_lifestyle_tuned_v1.json"

    joblib.dump(best_model, model_path)

    metadata = {
        "model_name": "Random Forest tuned",
        "model_file": model_path.name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "n_iter": N_ITER,
        "cv_folds": 5,
        "selection_metric": "f1_macro",
        "features": FEATURES,
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(best_score),
        "baseline_cv_macro_f1": float(baseline_macro_f1),
        "baseline_cv_macro_f1_std": float(baseline_std),
        "holdout_metrics_exploratory": test_metrics,
        "scikit_learn_version": sklearn.__version__,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Save all search results.
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.sort_values(
        "rank_test_score"
    )

    cv_results.to_csv(
        REPORT_DIR / "random_forest_tuning_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_results = cv_results.head(10)

    lines = [
        "# Phase 2 — Random Forest hyperparameter tuning",
        "",
        f"- Training samples: {len(X_train)}",
        f"- Holdout samples: {len(X_test)}",
        f"- Search method: RandomizedSearchCV",
        f"- Number of configurations: {N_ITER}",
        "- Cross-validation: 5-fold StratifiedKFold",
        "- Selection metric: Macro-F1",
        "- Features: 14, including Gender",
        "",
        "## Baseline comparison",
        "",
        f"- Baseline CV Macro-F1: {baseline_macro_f1:.4f}",
        f"- Baseline CV standard deviation: {baseline_std:.4f}",
        f"- Best tuned CV Macro-F1: {best_score:.4f}",
        f"- Difference: {best_score - baseline_macro_f1:+.4f}",
        "",
        "## Best hyperparameters",
        "",
        "```json",
        json.dumps(search.best_params_, indent=2),
        "```",
        "",
        "## Top 10 configurations",
        "",
        "| Rank | Mean CV Macro-F1 | Std |",
        "|---:|---:|---:|",
    ]

    for _, row in top_results.iterrows():
        lines.append(
            f"| {int(row['rank_test_score'])} "
            f"| {row['mean_test_score']:.4f} "
            f"| {row['std_test_score']:.4f} |"
        )

    lines += [
        "",
        "## Exploratory holdout results",
        "",
        "The holdout has already been examined in earlier Phase 2 analyses. "
        "These values are reported for transparency only and must not be "
        "used for further model selection.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for key, value in test_metrics.items():
        lines.append(f"| {key} | {value:.4f} |")

    lines += [
        "",
        "### Classification report",
        "",
        "```text",
        class_report.rstrip(),
        "```",
        "",
        "## Limitations",
        "",
        "- Randomized search CV score is subject to model-selection optimism.",
        "- A separate external dataset is needed for independent validation.",
        "- Holdout metrics are exploratory because the holdout was examined previously.",
        "- This dataset is not a clinical validation cohort.",
        "",
    ]

    report_path = REPORT_DIR / "random_forest_tuning.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nExploratory holdout metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nSaved files:")
    print(report_path)
    print(REPORT_DIR / "random_forest_tuning_results.csv")
    print(model_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
