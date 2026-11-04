"""Reproducible Phase 2 training pipeline for the UCI obesity-level dataset.

Primary experiment: predict the original seven NObeyesdad categories from
demographic/lifestyle survey variables. Height and Weight are deliberately
excluded because the source target is BMI-derived.

Run from repository root:
    python -m src.training.train_uci
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
DEFAULT_MODEL_DIR = ROOT / "models"
DEFAULT_REPORT_DIR = ROOT / "reports"

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
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES + ["Height", "Weight", TARGET]
RANDOM_STATE = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")
    raw = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Dataset thiếu các cột bắt buộc: {missing}")

    original_rows = len(raw)
    exact_duplicates = int(raw.duplicated().sum())
    data = raw.drop_duplicates().reset_index(drop=True)
    if data[TARGET].isna().any():
        raise ValueError(f"Cột target {TARGET} có giá trị rỗng.")
    if data[FEATURES].isna().any().any():
        raise ValueError("Feature có giá trị rỗng; cần audit trước khi train.")
    classes = sorted(data[TARGET].astype(str).unique().tolist())
    if len(classes) != 7:
        raise ValueError(
            f"Kỳ vọng target UCI có 7 lớp, thực tế tìm thấy {len(classes)}: {classes}"
        )
    audit = {
        "source_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256_file(path),
        "rows_before_deduplication": original_rows,
        "exact_duplicate_rows_removed": exact_duplicates,
        "rows_after_deduplication": len(data),
        "n_features": len(FEATURES),
        "target": TARGET,
        "classes": classes,
        "excluded_features": ["Height", "Weight"],
    }
    return data, audit


def build_pipeline(estimator) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    prep = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", prep), ("classifier", estimator)])


def model_candidates() -> dict[str, Pipeline]:
    return {
        "Dummy (most_frequent)": build_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "Logistic Regression": build_pipeline(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Decision Tree": build_pipeline(
            DecisionTreeClassifier(
                class_weight="balanced",
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    }


def write_report(
    path: Path,
    audit: dict,
    cv_results: list[dict],
    selected_name: str,
    test_metrics: dict,
    report_text: str,
    matrix: list[list[int]],
) -> None:
    lines = [
        "# Phase 2 — UCI obesity lifestyle model evaluation",
        "",
        f"> Training timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        "> Educational classification only; not a medical diagnostic tool.",
        "",
        "## Dataset and leakage controls",
        "",
        f"- Source file: `{audit['source_file']}`",
        f"- SHA-256: `{audit['sha256']}`",
        f"- Rows before exact deduplication: {audit['rows_before_deduplication']}",
        f"- Exact duplicate rows removed before splitting: {audit['exact_duplicate_rows_removed']}",
        f"- Rows after deduplication: {audit['rows_after_deduplication']}",
        f"- Target: `{TARGET}` (seven original classes)",
        f"- Features used ({len(FEATURES)}): " + ", ".join(f"`{c}`" for c in FEATURES),
        "- Excluded from primary model: `Height`, `Weight` (anthropometric shortcut/leakage risk because target categories are BMI-derived).",
        "- All imputers, scaling and one-hot encoding are fitted inside each training fold via scikit-learn Pipeline.",
        "- Exact duplicate rows are removed before the stratified holdout split. The source also contains synthetic records; related synthetic neighbours may still cross splits.",
        "",
        "## Cross-validation comparison",
        "",
        "Five-fold StratifiedKFold on the training partition; selection metric: macro-F1.",
        "",
        "| Model | Macro-F1 mean | Macro-F1 std | Balanced accuracy mean |",
        "|---|---:|---:|---:|",
    ]
    for item in cv_results:
        lines.append(
            f"| {item['model']} | {item['macro_f1_mean']:.4f} | "
            f"{item['macro_f1_std']:.4f} | {item['balanced_accuracy_mean']:.4f} |"
        )
    lines += [
        "",
        f"## Final holdout test — {selected_name}",
        "",
        "The holdout test is evaluated once after model selection using cross-validation.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in test_metrics.items():
        lines.append(f"| {key} | {value:.4f} |")
    lines += [
        "",
        "### Per-class classification report",
        "",
        "```text",
        report_text.rstrip(),
        "```",
        "",
        "### Confusion matrix",
        "",
        "```text",
        json.dumps(matrix, indent=2),
        "```",
        "",
        "## Limitations",
        "",
        "- UCI reports that 77% of the 2,111 records are synthetic (Weka/SMOTE); the CSV has no row-level provenance flag.",
        "- Exact deduplication does not prevent related synthetic observations from appearing in both train and test partitions.",
        "- The target is BMI-derived. This experiment predicts the dataset's BMI-defined category from survey/lifestyle variables; it does not establish clinical prediction or population prevalence.",
        "- Results are dataset-specific and require independent external validation before any real-world use.",
        "",
        "## Reproducibility",
        "",
        f"- Random state: {RANDOM_STATE}",
        "- Split: stratified 80% train / 20% holdout test",
        "- CV: 5-fold stratified, training partition only",
        f"- Python: {platform.python_version()}",
        f"- scikit-learn: {sklearn.__version__}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(data_path: Path, model_dir: Path, report_dir: Path) -> None:
    data, audit = load_dataset(data_path)
    X = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    candidates = model_candidates()
    results = []
    fitted = {}

    print(f"Dataset: {len(data)} rows after exact deduplication")
    print(f"Train: {len(X_train)} | Holdout test: {len(X_test)}")
    print(f"Target classes: {sorted(y.unique())}")
    print("Primary features exclude Height and Weight.")

    for name, pipeline in candidates.items():
        print(f"\n[CV] {name}")
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        item = {
            "model": name,
            "macro_f1_mean": float(scores["test_macro_f1"].mean()),
            "macro_f1_std": float(scores["test_macro_f1"].std()),
            "balanced_accuracy_mean": float(
                scores["test_balanced_accuracy"].mean()
            ),
        }
        results.append(item)
        print(
            f"  macro-F1={item['macro_f1_mean']:.4f} "
            f"+/- {item['macro_f1_std']:.4f}; "
            f"balanced_accuracy={item['balanced_accuracy_mean']:.4f}"
        )

    results.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    selected_name = results[0]["model"]
    selected = candidates[selected_name]
    print(f"\n[SELECTED] {selected_name} (highest CV macro-F1)")

    selected.fit(X_train, y_train)
    predictions = selected.predict(X_test)
    labels = sorted(y.unique())
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_precision": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
    }
    report_text = classification_report(
        y_test, predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "uci_obesity_lifestyle_v1.joblib"
    metadata_path = model_dir / "uci_obesity_lifestyle_v1.json"
    joblib.dump(selected, model_path)
    metadata = {
        **audit,
        "model_name": selected_name,
        "model_file": model_path.name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_metrics": test_metrics,
        "scikit_learn_version": sklearn.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(
        report_dir / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    write_report(
        report_dir / "model_evaluation.md",
        audit,
        results,
        selected_name,
        test_metrics,
        report_text,
        matrix,
    )

    print("\n[FINAL TEST]")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved report: {report_dir / 'model_evaluation.md'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.model_dir, args.report_dir)
