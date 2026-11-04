
"""Evaluate the saved Phase 2 model across gender subgroups.

Run from repository root:
    python -m src.training.analyze_gender_performance
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_PATH = ROOT / "models/uci_obesity_lifestyle_v1.joblib"
METADATA_PATH = ROOT / "models/uci_obesity_lifestyle_v1.json"
REPORT_DIR = ROOT / "reports"

TARGET = "NObeyesdad"
FEATURES = [
    "Age", "FCVC", "NCP", "CH2O", "FAF", "TUE",
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]
RANDOM_STATE = 42


def main() -> None:
    for path in [DATA_PATH, MODEL_PATH, METADATA_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()

    # Reproduce the original Phase 2 holdout split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    expected_f1 = metadata["test_metrics"]["macro_f1"]
    actual_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    if abs(actual_f1 - expected_f1) > 1e-6:
        raise ValueError(
            f"Baseline Macro-F1 mismatch: {actual_f1:.6f} "
            f"versus metadata {expected_f1:.6f}"
        )

    labels = sorted(y.unique())
    results = []
    report_sections = [
        "# Phase 2 — Gender subgroup evaluation",
        "",
        f"- Holdout samples: {len(X_test)}",
        f"- Overall holdout Macro-F1: {actual_f1:.4f}",
        "",
        "## Metrics by gender",
        "",
        "| Gender | Samples | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]

    for gender in sorted(X_test["Gender"].astype(str).unique()):
        mask = X_test["Gender"].astype(str) == gender
        y_group = y_test.loc[mask]
        pred_group = pd.Series(predictions, index=y_test.index).loc[mask]

        accuracy = accuracy_score(y_group, pred_group)
        macro_f1 = f1_score(
            y_group, pred_group, labels=labels,
            average="macro", zero_division=0
        )

        results.append({
            "gender": gender,
            "samples": int(mask.sum()),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        report_sections.append(
            f"| {gender} | {int(mask.sum())} | "
            f"{accuracy:.4f} | {macro_f1:.4f} |"
        )

        report_sections += [
            "",
            f"### Classification report — {gender}",
            "",
            "```text",
            classification_report(
                y_group, pred_group,
                labels=labels, zero_division=0
            ).rstrip(),
            "```",
        ]

    # Show actual target distribution within each gender group.
    distribution = pd.crosstab(
        X_test["Gender"].astype(str),
        y_test,
        normalize="index"
    ).round(4)

    report_sections += [
        "",
        "## Actual target distribution by gender",
        "",
        distribution.to_string(),
        "",
        "## Interpretation cautions",
        "",
        "- Subgroup metrics depend on subgroup sample size.",
        "- A difference in metrics does not by itself establish bias.",
        "- Gender importance measures model reliance, not causality.",
        "- The dataset contains synthetic observations and requires "
        "independent external validation.",
        "",
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "gender_subgroup_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    distribution.to_csv(
        REPORT_DIR / "gender_target_distribution.csv",
        encoding="utf-8-sig",
    )

    (REPORT_DIR / "gender_performance.md").write_text(
        "\n".join(report_sections), encoding="utf-8"
    )

    print("Gender subgroup metrics:")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nTarget distribution by gender:")
    print(distribution)
    print("\nSaved reports:")
    print(REPORT_DIR / "gender_performance.md")
    print(REPORT_DIR / "gender_subgroup_metrics.csv")
    print(REPORT_DIR / "gender_target_distribution.csv")


if __name__ == "__main__":
    main()
