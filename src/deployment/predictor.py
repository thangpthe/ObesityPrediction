"""Prediction service for the Phase 3 FastAPI application."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "uci_obesity_lifestyle_v1.joblib"
METADATA_PATH = ROOT_DIR / "models" / "uci_obesity_lifestyle_v1.json"

CLASS_LABELS_VI = {
    "Insufficient_Weight": "Thiếu cân",
    "Normal_Weight": "Cân nặng bình thường",
    "Obesity_Type_I": "Béo phì loại I",
    "Obesity_Type_II": "Béo phì loại II",
    "Obesity_Type_III": "Béo phì loại III",
    "Overweight_Level_I": "Thừa cân mức I",
    "Overweight_Level_II": "Thừa cân mức II",
}


class ObesityPredictor:
    """Load the trained pipeline and perform obesity-level predictions."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy metadata: {METADATA_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.features = self.metadata["features"]
        self.numeric_features = self.metadata["numeric_features"]
        self.categorical_features = self.metadata[
            "categorical_features"
        ]

        if len(self.features) != 14:
            raise ValueError("Model metadata không có đủ 14 đặc trưng.")

        self.classes = [
            str(label) for label in self.model.classes_
        ]

    def get_category_options(self):
        """
        Read valid categorical values directly from the fitted
        OneHotEncoder to avoid inventing or mismatching input values.
        """
        preprocessor = self.model.named_steps["preprocess"]
        categorical_pipeline = (
            preprocessor.named_transformers_["categorical"]
        )
        encoder = categorical_pipeline.named_steps["onehot"]

        return {
            feature: [
                str(value) for value in categories
            ]
            for feature, categories in zip(
                self.categorical_features,
                encoder.categories_,
            )
        }

    def validate_input(self, input_data):
        """Validate and arrange input data in the model's feature order."""

        if not isinstance(input_data, dict):
            raise ValueError("Dữ liệu đầu vào phải ở dạng dictionary.")

        missing = [
            feature
            for feature in self.features
            if feature not in input_data
        ]

        if missing:
            raise ValueError(
                "Thiếu các trường dữ liệu: " + ", ".join(missing)
            )

        row = {}

        for feature in self.numeric_features:
            try:
                value = float(input_data[feature])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Giá trị của {feature} phải là số."
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"Giá trị của {feature} không hợp lệ."
                )

            if feature == "Age" and value <= 0:
                raise ValueError("Tuổi phải lớn hơn 0.")

            row[feature] = value

        category_options = self.get_category_options()

        for feature in self.categorical_features:
            value = str(input_data[feature])

            if value not in category_options[feature]:
                raise ValueError(
                    f"Giá trị '{value}' không hợp lệ cho {feature}. "
                    f"Các giá trị hợp lệ: {category_options[feature]}"
                )

            row[feature] = value

        return pd.DataFrame(
            [[row[feature] for feature in self.features]],
            columns=self.features,
        )

    def predict(self, input_data):
        """Return predicted class and probability for all seven classes."""

        dataframe = self.validate_input(input_data)

        predicted_class = str(self.model.predict(dataframe)[0])
        probabilities = self.model.predict_proba(dataframe)[0]

        probability_results = []

        for label, probability in zip(self.classes, probabilities):
            probability_results.append({
                "class": label,
                "label_vi": CLASS_LABELS_VI.get(label, label),
                "probability": float(probability),
                "percentage": round(float(probability) * 100, 2),
            })

        probability_results.sort(
            key=lambda item: item["probability"],
            reverse=True,
        )

        return {
            "predicted_class": predicted_class,
            "predicted_label_vi": CLASS_LABELS_VI.get(
                predicted_class,
                predicted_class,
            ),
            "probabilities": probability_results,
        }


# Initialize one shared predictor instance.
predictor = ObesityPredictor()