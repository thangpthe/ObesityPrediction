
"""Analyze permutation importance for the saved Phase 2 UCI model.

Run from repository root:
    python -m src.training.analyze_feature_importance
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path in [DATA_PATH, MODEL_PATH, METADATA_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    actual_hash = sha256_file(DATA_PATH)

    if actual_hash != metadata["sha256"]:
        raise ValueError(
            "Dataset SHA-256 does not match the saved model metadata. "
            "Stop to avoid analyzing a different dataset."
        )

    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

    X = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()

    # Reproduce exactly the original Phase 2 holdout split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    # Verify that the saved model reproduces the reported holdout score.
    predictions = model.predict(X_test)
    baseline_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    expected_f1 = metadata["test_metrics"]["macro_f1"]
    if abs(baseline_f1 - expected_f1) > 1e-6:
        raise ValueError(
            f"Holdout Macro-F1 mismatch: reproduced={baseline_f1:.6f}, "
            f"metadata={expected_f1:.6f}"
        )

    print(f"Holdout rows: {len(X_test)}")
    print(f"Baseline Macro-F1: {baseline_f1:.4f}")
    print("Calculating permutation importance (30 repeats)...")

    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring="f1_macro",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)

    importance.insert(0, "rank", range(1, len(importance) + 1))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / "permutation_importance.csv"
    md_path = REPORT_DIR / "feature_importance.md"

    importance.to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    lines = [
        "# Phase 2 — Permutation Feature Importance",
        "",
        f"- Dataset SHA-256: `{actual_hash}`",
        f"- Holdout samples: {len(X_test)}",
        f"- Baseline holdout Macro-F1: {baseline_f1:.4f}",
        "- Method: permutation importance",
        "- Scoring: macro-F1",
        "- Repeats: 30",
        "- Random state: 42",
        "",
        "Importance is the decrease in Macro-F1 after randomly "
        "permuting one input feature. Larger positive values indicate "
        "greater reliance by this fitted model.",
        "",
        "| Rank | Feature | Mean importance | Std |",
        "|---:|---|---:|---:|",
    ]

    for row in importance.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.feature} | "
            f"{row.importance_mean:.5f} | {row.importance_std:.5f} |"
        )

    lines += [
        "",
        "## Interpretation cautions",
        "",
        "- Importance measures model reliance, not causality.",
        "- Correlated features can share or mask importance.",
        "- Near-zero or negative values do not prove a feature is useless.",
        "- Results are specific to this dataset and holdout split.",
        "- The dataset contains synthetic observations; external validation "
        "is still required.",
        "",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nTop features:")
    print(importance.head(10).to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved report: {md_path}")


if __name__ == "__main__":
    main()
