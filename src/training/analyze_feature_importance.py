
"""
Phase 3 - Permutation Feature Importance

Analyze the saved Phase 2 model using the original holdout test set.
Does not retrain or overwrite the model.

Run from repository root:
    python -m src.training.analyze_feature_importance
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/raw/ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_PATH = ROOT / "models/uci_obesity_lifestyle_v1.joblib"
METADATA_PATH = ROOT / "models/uci_obesity_lifestyle_v1.json"

REPORT_DIR = ROOT / "reports/phase3"

TARGET = "NObeyesdad"

FEATURES = [
    "Age", "FCVC", "NCP", "CH2O", "FAF", "TUE",
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
N_REPEATS = 30

FEATURE_LABELS = {
    "Age": "Tuổi",
    "FCVC": "Tần suất ăn rau",
    "NCP": "Số bữa ăn chính",
    "CH2O": "Lượng nước uống",
    "FAF": "Hoạt động thể chất",
    "TUE": "Thời gian sử dụng thiết bị điện tử",
    "Gender": "Giới tính",
    "family_history_with_overweight": "Tiền sử gia đình thừa cân",
    "FAVC": "Thường xuyên ăn thực phẩm giàu calo",
    "CAEC": "Ăn vặt giữa các bữa",
    "SMOKE": "Hút thuốc",
    "SCC": "Theo dõi lượng calo",
    "CALC": "Sử dụng đồ uống có cồn",
    "MTRANS": "Phương tiện di chuyển",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:

    print("=" * 65)
    print("PHASE 3 - FEATURE IMPORTANCE")
    print("=" * 65)

    # 1. Check required files
    for path in [DATA_PATH, MODEL_PATH, METADATA_PATH]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path}"
            )

    # 2. Read model metadata
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    # 3. Verify dataset
    actual_hash = sha256_file(DATA_PATH)

    if actual_hash != metadata.get("sha256"):
        raise ValueError(
            "Dataset SHA-256 không khớp metadata. "
            "Dừng phân tích để tránh sử dụng sai dữ liệu."
        )

    if metadata.get("features") != FEATURES:
        raise ValueError(
            "Danh sách features không khớp metadata."
        )

    # 4. Load and clean dataset
    data = (
        pd.read_csv(DATA_PATH)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if TARGET not in data.columns:
        raise ValueError(
            f"Dataset không có cột target: {TARGET}"
        )

    missing_features = [
        feature for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Dataset thiếu features: {missing_features}"
        )

    if len(data) != metadata.get("rows_after_deduplication"):
        raise ValueError(
            "Số dòng dataset không khớp metadata."
        )

    X = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()

    # 5. Reproduce original Phase 2 holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 6. Load existing model - no retraining
    model = joblib.load(MODEL_PATH)

    # 7. Verify original holdout Macro-F1
    predictions = model.predict(X_test)

    baseline_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    expected_f1 = metadata["test_metrics"]["macro_f1"]

    if abs(baseline_f1 - expected_f1) > 1e-6:
        raise ValueError(
            f"Holdout Macro-F1 không khớp: "
            f"reproduced={baseline_f1:.6f}, "
            f"metadata={expected_f1:.6f}"
        )

    print("\nDATASET INFORMATION")
    print(f"Total rows: {len(data)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Baseline Macro-F1: {baseline_f1:.4f}")

    # 8. Calculate permutation importance
    print(
        f"\nCalculating permutation importance "
        f"({N_REPEATS} repeats)..."
    )

    result = permutation_importance(
        estimator=model,
        X=X_test,
        y=y_test,
        scoring="f1_macro",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 9. Build importance table
    importance = pd.DataFrame({
        "feature": FEATURES,
        "feature_label_vi": [
            FEATURE_LABELS[feature]
            for feature in FEATURES
        ],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance = importance.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance.insert(
        0,
        "rank",
        range(1, len(importance) + 1),
    )

    # 10. Create output directory
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 11. Export CSV
    csv_path = REPORT_DIR / "feature_importance.csv"

    importance.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 12. Export JSON
    json_path = REPORT_DIR / "feature_importance.json"

    json_result = {
        "model_file": MODEL_PATH.name,
        "dataset_sha256": actual_hash,
        "total_rows": len(data),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(FEATURES),
        "baseline_macro_f1": float(baseline_f1),
        "method": "Permutation Importance",
        "scoring": "f1_macro",
        "n_repeats": N_REPEATS,
        "random_state": RANDOM_STATE,
        "features": importance.to_dict(
            orient="records"
        ),
    }

    json_path.write_text(
        json.dumps(
            json_result,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    # 13. Create horizontal bar chart
    plot_data = importance.sort_values(
        "importance_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.barh(
        plot_data["feature_label_vi"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std"],
        capsize=3,
    )

    ax.axvline(
        x=0,
        linewidth=1,
        linestyle="--",
    )

    ax.set_title(
        "Permutation Feature Importance\n"
        "UCI Obesity Lifestyle Dataset",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Mức giảm Macro-F1 sau khi hoán đổi đặc trưng",
        fontsize=10,
    )

    ax.set_ylabel("Đặc trưng đầu vào", fontsize=10)

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.35,
    )

    fig.tight_layout()

    png_path = REPORT_DIR / "feature_importance.png"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # 14. Export Markdown report
    md_path = REPORT_DIR / "feature_importance.md"

    lines = [
        "# Phase 3 - Permutation Feature Importance",
        "",
        f"- Dataset SHA-256: `{actual_hash}`",
        f"- Total rows: {len(data)}",
        f"- Holdout samples: {len(X_test)}",
        f"- Baseline Macro-F1: {baseline_f1:.4f}",
        "- Method: Permutation Importance",
        "- Scoring: Macro-F1",
        f"- Repeats: {N_REPEATS}",
        f"- Random state: {RANDOM_STATE}",
        "",
        "## Feature ranking",
        "",
        "| Rank | Feature | Mean importance | Std |",
        "|---:|---|---:|---:|",
    ]

    for row in importance.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.feature_label_vi} "
            f"(`{row.feature}`) | "
            f"{row.importance_mean:.6f} | "
            f"{row.importance_std:.6f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Permutation importance measures the decrease in "
        "Macro-F1 after randomly permuting one input feature.",
        "",
        "A larger positive value indicates greater reliance "
        "by this fitted model on that feature.",
        "",
        "## Limitations",
        "",
        "- Importance measures model reliance, not causality.",
        "- Correlated features can share or mask importance.",
        "- Near-zero or negative values do not prove a feature is useless.",
        "- Results are specific to this dataset and holdout split.",
        "- The dataset contains synthetic observations.",
        "- Independent external validation is still required.",
        "",
        "Educational classification only; "
        "not a medical diagnostic tool.",
        "",
    ]

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # 15. Print results
    print("\nFEATURE IMPORTANCE RANKING")

    print(
        importance[
            [
                "rank",
                "feature_label_vi",
                "importance_mean",
                "importance_std",
            ]
        ].to_string(index=False)
    )

    print("\nEXPORTED FILES")
    print(f"PNG: {png_path}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")

    print("\nFeature Importance evaluation completed!")


if __name__ == "__main__":
    main()
