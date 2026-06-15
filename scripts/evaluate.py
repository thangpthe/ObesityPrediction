"""
scripts/evaluate.py
────────────────────
Đánh giá mô hình đã huấn luyện trên tập test độc lập.
Tính: Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report.
"""

import os
import sys
import pandas as pd
import joblib

from config import TEST_CSV as TEST_PATH, MODEL_RF as MODEL_PATH, FEATURE_COLS, TARGET_COL
from metrics import print_metrics

FEATURES = FEATURE_COLS   # backward-compat alias
TARGET   = TARGET_COL
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Đọc dữ liệu test
    if not os.path.exists(TEST_PATH):
        print(f"[!] Không tìm thấy tập test: {TEST_PATH}")
        print("    Hãy chạy scripts/train.py trước.")
        sys.exit(1)

    print(f"[+] Đọc tập test: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    X_test  = test_df[FEATURES]
    y_test  = test_df[TARGET]

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Không tìm thấy model: {MODEL_PATH}")
        sys.exit(1)

    print(f"[+] Load model: {MODEL_PATH}")
    model   = joblib.load(MODEL_PATH)
    y_pred  = model.predict(X_test)

    # Kết quả
    print("\n" + "═" * 55)
    print("  KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (tập test độc lập)")
    print("═" * 55)
    print_metrics(
        "Random Forest (test)",
        y_test, y_pred,
        extended=True,
        class_names=["Không béo phì", "Béo phì"],
    )


if __name__ == "__main__":
    main()
