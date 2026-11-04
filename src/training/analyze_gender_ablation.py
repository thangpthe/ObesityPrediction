
"""Ablation study: measure the contribution of Gender.

Run from repository root:
    python -m src.training.analyze_gender_ablation

This experiment uses CV on the training partition only.
It does not overwrite the saved baseline model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "NObeyesdad"
NUMERIC_FEATURES = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
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
RANDOM_STATE = 42


def build_pipeline(include_gender: bool) -> Pipeline:
    categorical = CATEGORICAL_FEATURES.copy()

    if not include_gender:
        categorical.remove("Gender")

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
        ("categorical", categorical_pipeline, categorical),
    ])

    classifier = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", classifier),
    ])


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = data[all_features].copy()
    y = data[TARGET].astype(str).copy()

    X_train, _, y_train, _ = train_test_split(
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

    experiments = [
        ("With Gender", True),
        ("Without Gender", False),
    ]

    results = []

    print(f"Training samples: {len(X_train)}")
    print("Evaluation: 5-fold Stratified CV")
    print("Holdout test is not used in this experiment.")

    for name, include_gender in experiments:
        features = all_features if include_gender else [
            col for col in all_features if col != "Gender"
        ]

        print(f"\n[CV] {name}")
        pipeline = build_pipeline(include_gender)

        scores = cross_validate(
            pipeline,
            X_train[features],
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        result = {
            "experiment": name,
            "n_features": len(features),
            "macro_f1_mean": scores["test_macro_f1"].mean(),
            "macro_f1_std": scores["test_macro_f1"].std(),
            "balanced_accuracy_mean": scores[
                "test_balanced_accuracy"
            ].mean(),
            "accuracy_mean": scores["test_accuracy"].mean(),
        }

        results.append(result)

        print(
            f"Macro-F1: {result['macro_f1_mean']:.4f} "
            f"+/- {result['macro_f1_std']:.4f}"
        )
        print(
            "Balanced accuracy: "
            f"{result['balanced_accuracy_mean']:.4f}"
        )
        print(f"Accuracy: {result['accuracy_mean']:.4f}")

    result_df = pd.DataFrame(results)

    with_gender = result_df.iloc[0]
    without_gender = result_df.iloc[1]

    delta_f1 = (
        with_gender["macro_f1_mean"]
        - without_gender["macro_f1_mean"]
    )
    delta_balanced = (
        with_gender["balanced_accuracy_mean"]
        - without_gender["balanced_accuracy_mean"]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(
        REPORT_DIR / "gender_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# Phase 2 — Gender ablation study",
        "",
        f"- Training samples: {len(X_train)}",
        "- Evaluation: 5-fold Stratified Cross-Validation",
        "- Estimator: Random Forest",
        "- Same random state and hyperparameters in both experiments",
        "- Holdout test was not used for model selection",
        "",
        "## Results",
        "",
        "| Experiment | Features | Macro-F1 mean | Macro-F1 std | Balanced accuracy | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in result_df.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['n_features']} "
            f"| {row['macro_f1_mean']:.4f} "
            f"| {row['macro_f1_std']:.4f} "
            f"| {row['balanced_accuracy_mean']:.4f} "
            f"| {row['accuracy_mean']:.4f} |"
        )

    lines += [
        "",
        "## Difference",
        "",
        f"- Macro-F1 difference (with minus without Gender): {delta_f1:+.4f}",
        f"- Balanced accuracy difference: {delta_balanced:+.4f}",
        "",
        "## Interpretation",
        "",
        "- A positive difference means including Gender improved the CV score.",
        "- A negative difference means the model performed better without Gender.",
        "- A small difference suggests Gender adds limited overall predictive value under this setup.",
        "- This experiment measures predictive contribution, not causality or fairness.",
        "- Subgroup performance and class distributions should also be considered.",
        "- Results are specific to this dataset and CV design.",
        "",
    ]

    (REPORT_DIR / "gender_ablation.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nAblation results:")
    print(result_df.to_string(index=False))
    print(f"\nMacro-F1 difference: {delta_f1:+.4f}")
    print("\nSaved:")
    print(REPORT_DIR / "gender_ablation.md")
    print(REPORT_DIR / "gender_ablation.csv")


if __name__ == "__main__":
    main()
