
"""
Phase 3 - Confusion Matrix Evaluation

Load the existing trained model and reproduce the original
stratified holdout split. Does not retrain or overwrite the model.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.training.train_uci import (
    FEATURES,
    TARGET,
    RANDOM_STATE,
    DEFAULT_DATA,
    load_dataset,
    sha256_file,
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
    print("=" * 60)
    print("PHASE 3 - CONFUSION MATRIX")
    print("=" * 60)

    # 1. Check files
    for path in [DEFAULT_DATA, MODEL_PATH, METADATA_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    # 2. Read metadata
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    # 3. Load and audit original dataset
    data, audit = load_dataset(DEFAULT_DATA)

    # Verify dataset is exactly the same as training dataset
    if audit["sha256"] != metadata.get("sha256"):
        raise ValueError(
            "SHA-256 dataset không khớp với metadata của mô hình. "
            "Dừng đánh giá để tránh sử dụng sai tập dữ liệu."
        )

    if metadata.get("features") != FEATURES:
        raise ValueError("Danh sách features không khớp metadata.")

    if metadata.get("rows_after_deduplication") != len(data):
        raise ValueError("Số dòng dataset không khớp metadata.")

    # 4. Recreate exactly the original holdout split
    X = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 5. Load existing model - DO NOT retrain
    model = joblib.load(MODEL_PATH)

    # 6. Predict on original holdout test set
    y_pred = model.predict(X_test)

    labels = sorted(y.unique())
    labels_vi = [LABELS_VI[label] for label in labels]

    # 7. Calculate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 8. Export confusion matrix CSV
    matrix_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    matrix_df.index.name = "Actual"
    matrix_df.columns.name = "Predicted"

    csv_path = OUTPUT_DIR / "confusion_matrix.csv"
    matrix_df.to_csv(csv_path, encoding="utf-8-sig")

    # 9. Draw confusion matrix
    plt.figure(figsize=(13, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels_vi,
    )

    display.plot(
        cmap="Blues",
        values_format="d",
        xticks_rotation=35,
        colorbar=True,
    )

    plt.title(
        "Confusion Matrix - Random Forest\n"
        "UCI Obesity Lifestyle Dataset",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    plt.xlabel("Nhãn dự đoán", fontsize=11)
    plt.ylabel("Nhãn thực tế", fontsize=11)

    plt.tight_layout()

    image_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # 10. Save evaluation information
    summary = {
        "model_file": MODEL_PATH.name,
        "dataset_sha256": audit["sha256"],
        "total_rows": len(data),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "classes": labels,
        "confusion_matrix": cm.tolist(),
    }

    summary_path = OUTPUT_DIR / "confusion_matrix_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    # 11. Print results
    print("\nDATASET INFORMATION")
    print(f"Total rows: {len(data)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {len(labels)}")

    print("\nCONFUSION MATRIX")
    print(matrix_df)

    print("\nEXPORTED FILES")
    print(f"PNG: {image_path}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")

    print("\nConfusion Matrix evaluation completed!")


if __name__ == "__main__":
    main()
