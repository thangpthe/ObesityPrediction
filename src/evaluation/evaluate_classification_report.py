
"""
Phase 3 - Classification Report Evaluation

Evaluate the existing deployed model on the original holdout test set.
Does not retrain or overwrite the model.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.training.train_uci import (
    FEATURES,
    TARGET,
    RANDOM_STATE,
    DEFAULT_DATA,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "uci_obesity_lifestyle_v1.joblib"
METADATA_PATH = ROOT / "models" / "uci_obesity_lifestyle_v1.json"

OUTPUT_DIR = ROOT / "reports" / "phase3"

LABELS_VI = {
    "Insufficient_Weight": "Thiếu cân",
    "Normal_Weight": "Cân nặng bình thường",
    "Obesity_Type_I": "Béo phì loại I",
    "Obesity_Type_II": "Béo phì loại II",
    "Obesity_Type_III": "Béo phì loại III",
    "Overweight_Level_I": "Thừa cân mức I",
    "Overweight_Level_II": "Thừa cân mức II",
}


def main():
    print("=" * 65)
    print("PHASE 3 - CLASSIFICATION REPORT")
    print("=" * 65)

    # 1. Check required files
    for path in [DEFAULT_DATA, MODEL_PATH, METADATA_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    # 2. Read model metadata
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    # 3. Load and verify dataset
    data, audit = load_dataset(DEFAULT_DATA)

    if audit["sha256"] != metadata.get("sha256"):
        raise ValueError(
            "SHA-256 dataset không khớp metadata của mô hình. "
            "Dừng đánh giá để tránh sai lệch."
        )

    if metadata.get("features") != FEATURES:
        raise ValueError("Danh sách features không khớp metadata.")

    if metadata.get("rows_after_deduplication") != len(data):
        raise ValueError("Số lượng bản ghi không khớp metadata.")

    # 4. Recreate original train/test split
    X = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 5. Load existing model - no retraining
    model = joblib.load(MODEL_PATH)

    # 6. Predict
    y_pred = model.predict(X_test)

    labels = sorted(y.unique())

    # 7. Calculate classification report
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[LABELS_VI[label] for label in labels],
        zero_division=0,
    )

    # 8. Calculate overall metrics
    overall_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    # 9. Build per-class table
    rows = []

    for label in labels:
        item = report_dict[label]

        rows.append({
            "class": label,
            "label_vi": LABELS_VI[label],
            "precision": item["precision"],
            "recall": item["recall"],
            "f1_score": item["f1-score"],
            "support": int(item["support"]),
        })

    class_df = pd.DataFrame(rows)

    # 10. Export files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "classification_report.csv"
    class_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    json_result = {
        "model_file": MODEL_PATH.name,
        "dataset_sha256": audit["sha256"],
        "total_rows": len(data),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": RANDOM_STATE,
        "overall_metrics": overall_metrics,
        "per_class_metrics": rows,
        "averages": {
            "macro_avg": report_dict["macro avg"],
            "weighted_avg": report_dict["weighted avg"],
        },
    }

    json_path = OUTPUT_DIR / "classification_report.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            json_result,
            file,
            ensure_ascii=False,
            indent=4,
        )

    txt_path = OUTPUT_DIR / "classification_report.txt"

    with txt_path.open("w", encoding="utf-8") as file:
        file.write(
            "PHASE 3 - CLASSIFICATION REPORT\n"
            "UCI Obesity Lifestyle Dataset\n\n"
        )
        file.write(report_text)
        file.write("\n\nOVERALL METRICS\n")

        for name, value in overall_metrics.items():
            file.write(f"{name}: {value:.6f}\n")

        file.write(
            "\nEducational classification only; "
            "not a medical diagnostic tool.\n"
        )

    # 11. Print results
    print("\nDATASET INFORMATION")
    print(f"Total rows: {len(data)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nCLASSIFICATION REPORT")
    print(report_text)

    print("\nOVERALL METRICS")
    for name, value in overall_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nEXPORTED FILES")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"TXT: {txt_path}")

    print("\nClassification Report evaluation completed!")


if __name__ == "__main__":
    main()
