"""
scripts/config.py
─────────────────
Shared constants for all scripts in this project.
Import from here instead of re-deriving paths / column lists in every file.
"""

import os

# ── Project root (two levels up from this file) ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Directory paths ───────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── File paths ────────────────────────────────────────────────────────────────
RAW_CSV    = os.path.join(RAW_DIR,  "obesity_prediction.csv")
FINAL_CSV  = os.path.join(RAW_DIR,  "obesity_prediction_dataset_final.csv")
TEST_CSV   = os.path.join(PROC_DIR, "test_data.csv")
MODEL_RF   = os.path.join(MODELS_DIR, "modelRF.pkl")
MODEL_DT   = os.path.join(MODELS_DIR, "modelDT.pkl")
MODEL_NB   = os.path.join(MODELS_DIR, "modelNB.pkl")

# ── Dataset schema ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "BMI",
    "family_history_with_overweight",
    "CH2O",
    "FAF",
]
TARGET_COL = "Obesity"

# ── Training hyper-parameters ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80 / 20 train-test split
VAL_RATIO    = 0.25   # 0.25 of train → 60 / 20 / 20 final split
