"""
scripts/pre_processing.py
──────────────────────────
Tiền xử lý dữ liệu thô:
  - Gộp dữ liệu từ nhiều nguồn
  - Tính chỉ số BMI
  - Cân bằng lớp (RandomUnderSampler)
  - Xuất file dataset_final.csv
"""

import os
import pandas as pd
from sklearn.utils import resample

from config import PROC_DIR, RAW_CSV as INPUT_FILE, FINAL_CSV as OUTPUT_FILE

# ── Cấu hình ──────────────────────────────────────────────────────────────────
os.makedirs(PROC_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[+] Đọc dữ liệu: {path}  →  {len(df)} dòng")
    # Bỏ giá trị null
    df.dropna(inplace=True)
    return df


def compute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Tính BMI nếu cột chưa có."""
    if "BMI" not in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
        df["BMI"] = df["BMI"].round(2)
    return df


def balance_classes(df: pd.DataFrame, target_col: str = "Obesity") -> pd.DataFrame:
    """Cân bằng lớp bằng RandomUnderSampler (sklearn)."""
    majority = df[df[target_col] == df[target_col].value_counts().idxmax()]
    minority = df[df[target_col] != df[target_col].value_counts().idxmax()]
    majority_down = resample(majority, replace=False,
                             n_samples=len(minority), random_state=42)
    balanced = pd.concat([majority_down, minority]).sample(
        frac=1, random_state=42).reset_index(drop=True)
    print(f"[+] Sau cân bằng: {balanced[target_col].value_counts().to_dict()}")
    return balanced


def main():
    df = load_and_clean(INPUT_FILE)
    df = compute_bmi(df)

    # Chỉ giữ các cột cần thiết
    from config import FEATURE_COLS, TARGET_COL
    COLS = FEATURE_COLS + [TARGET_COL]
    df = df[[c for c in COLS if c in df.columns]]

    df = balance_classes(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[✓] Đã lưu dataset đã xử lý → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
