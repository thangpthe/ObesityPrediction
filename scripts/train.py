"""
scripts/train.py
────────────────
Huấn luyện các mô hình học máy có giám sát:
  - Naive Bayes  (CategoricalNB)
  - Decision Tree
  - Random Forest  ← mô hình chính

Kết quả (file .pkl) được lưu vào thư mục models/
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import (
    FINAL_CSV as DATA_PATH,
    MODELS_DIR,
    TEST_CSV as TEST_OUT,
    FEATURE_COLS,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_RATIO,
)
from metrics import print_metrics

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TEST_OUT), exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    print(f"[+] Đọc dữ liệu: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLS]
    y = data[TARGET_COL]
    print(f"    Số mẫu: {len(data)} | Tỷ lệ lớp: {y.value_counts().to_dict()}")
    return X, y


def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=y_train_val
    )
    # Lưu tập test để đánh giá độc lập
    test_df = pd.concat([X_test.reset_index(drop=True),
                          y_test.reset_index(drop=True)], axis=1)
    test_df.to_csv(TEST_OUT, index=False)
    print(f"[+] Đã lưu tập test → {TEST_OUT}")
    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, y_train, y_val


def evaluate(name: str, model, X_val, y_val):
    y_pred = model.predict(X_val)
    print_metrics(name, y_val, y_pred)


def train_and_save(name: str, model, X_train, y_train, X_val, y_val, filename: str):
    print(f"\n[+] Huấn luyện {name}...")
    model.fit(X_train, y_train)
    evaluate(name, model, X_val, y_val)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"[+] Đã lưu mô hình → {path}")
    return model


def main():
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)

    # ── Naive Bayes ──────────────────────────────────────────────────────────
    # Lưu ý: CategoricalNB yêu cầu dữ liệu nguyên dương → cần discretize trước
    # train_and_save("Naive Bayes", CategoricalNB(),
    #                X_train, y_train, X_val, y_val, "modelNB.pkl")

    # ── Decision Tree ────────────────────────────────────────────────────────
    # train_and_save("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE),
    #                X_train, y_train, X_val, y_val, "modelDT.pkl")

    # ── Random Forest (mô hình chính) ────────────────────────────────────────
    train_and_save(
        "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        X_train, y_train, X_val, y_val,
        "modelRF.pkl",
    )

    print("\n[✓] Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
