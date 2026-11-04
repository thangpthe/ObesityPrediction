# ObesityPrediction

**A Machine Learning Project for Multi-Class Obesity Level Classification**

An educational machine learning project that explores the classification of obesity-related levels based on eating habits, physical condition, and lifestyle characteristics using the UCI Estimation of Obesity Levels Based on Eating Habits and Physical Condition dataset.

The project includes a reproducible machine learning pipeline, model evaluation and analysis, and a web application developed with FastAPI.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and must not be used to make clinical decisions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Features and Target Classes](#features-and-target-classes)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Development – Phase 2](#model-development--phase-2)
- [Model Evaluation – Phase 3](#model-evaluation--phase-3)
- [Feature Importance Analysis](#feature-importance-analysis)
- [FastAPI Web Application](#fastapi-web-application)
- [API Documentation](#api-documentation)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Running the ML Pipeline](#running-the-ml-pipeline)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Project Status](#project-status)

---

## Project Overview

The project investigates machine learning approaches for classifying seven obesity-related categories using lifestyle and eating-habit characteristics.

The repository includes:

- Data auditing and preprocessing.
- Baseline model comparison.
- Random Forest model development.
- Cross-validation and holdout evaluation.
- Hyperparameter tuning.
- Feature importance analysis.
- Gender subgroup performance and feature ablation studies.
- Confusion matrix and classification report generation.
- FastAPI-based prediction application.
- Web interface for interactive predictions and model evaluation.

## Project Objectives

1. Explore the relationship between lifestyle characteristics and obesity-level categories in the dataset.
2. Develop and compare multiple machine learning classification algorithms.
3. Build a reproducible preprocessing and model training pipeline.
4. Evaluate classification performance using appropriate metrics.
5. Analyze feature importance and model behavior.
6. Deploy the selected model through a web application and REST API.

## Dataset

**Dataset:** UCI Estimation of Obesity Levels Based on Eating Habits and Physical Condition.

The dataset contains information about individuals' eating habits, physical condition, and lifestyle characteristics.

### Dataset Summary

| Description                     | Value |
| ------------------------------- | ----: |
| Original records                | 2,111 |
| Exact duplicate records removed |    24 |
| Records after deduplication     | 2,087 |
| Input features                  |    14 |
| Target classes                  |     7 |

The target variable is `NObeyesdad`.

`Height` and `Weight` are excluded from the primary model because the obesity-level target is BMI-derived. Including these variables could introduce target-related information leakage.

The dataset contains a substantial proportion of synthetic observations. Therefore, model performance should be interpreted cautiously.

## Features and Target Classes

### Input Features

The primary model uses 14 features.

| Feature                        | Description                               |
| ------------------------------ | ----------------------------------------- |
| Age                            | Age                                       |
| FCVC                           | Frequency of vegetable consumption        |
| NCP                            | Number of main meals                      |
| CH2O                           | Daily water consumption                   |
| FAF                            | Physical activity frequency               |
| TUE                            | Time spent using technological devices    |
| Gender                         | Gender                                    |
| family_history_with_overweight | Family history of overweight              |
| FAVC                           | Frequent consumption of high-calorie food |
| CAEC                           | Food consumption between meals            |
| SMOKE                          | Smoking habit                             |
| SCC                            | Calorie consumption monitoring            |
| CALC                           | Alcohol consumption                       |
| MTRANS                         | Transportation method                     |

### Target Classes

The model predicts one of seven categories:

1. `Insufficient_Weight`
2. `Normal_Weight`
3. `Obesity_Type_I`
4. `Obesity_Type_II`
5. `Obesity_Type_III`
6. `Overweight_Level_I`
7. `Overweight_Level_II`

The web application also displays Vietnamese labels for these categories.

## Machine Learning Workflow

The project follows a structured machine learning workflow:

1. Load and audit the original dataset.
2. Remove exact duplicate records.
3. Separate input features and target labels.
4. Split data using stratified train-test splitting.
5. Build preprocessing pipelines for numerical and categorical features.
6. Compare classification algorithms using stratified cross-validation.
7. Select the primary model based on CV Macro-F1.
8. Evaluate the selected model on the holdout partition.
9. Perform additional model analyses.
10. Export trained model artifacts and evaluation reports.
11. Integrate the model into the FastAPI application.

The preprocessing pipeline performs imputation, numerical scaling, and categorical one-hot encoding.

Preprocessing is fitted within the machine learning pipeline to reduce data leakage during model evaluation.

## Model Development – Phase 2

Phase 2 established the reproducible machine learning pipeline and compared multiple classification algorithms.

### Models Compared

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest

Random Forest achieved the highest baseline cross-validation Macro-F1 and was retained as the primary model.

### Baseline Model Performance

| Metric                    |   Random Forest |
| ------------------------- | --------------: |
| CV Macro-F1               | 0.8384 ± 0.0132 |
| Holdout Accuracy          |          0.8421 |
| Holdout Balanced Accuracy |          0.8394 |
| Holdout Macro-F1          |          0.8379 |
| Holdout Weighted-F1       |          0.8415 |

The holdout partition contains 418 observations.

### Additional Phase 2 Analyses

- Permutation feature importance.
- Gender subgroup performance evaluation.
- Gender feature ablation study.
- Random Forest hyperparameter tuning using RandomizedSearchCV.

Hyperparameter tuning did not improve the baseline cross-validation Macro-F1 sufficiently to replace the original Random Forest configuration.

The baseline model is therefore retained as the deployed model.

## Model Evaluation – Phase 3

Phase 3 focuses on detailed model evaluation and integration of evaluation results into the web application.

The evaluation scripts reproduce the original stratified train-test split and load the existing trained model. They do not retrain the model during evaluation.

### Evaluation Metrics

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 84.21% |
| Balanced Accuracy | 83.94% |
| Macro Precision   | 83.82% |
| Macro Recall      | 83.94% |
| Macro F1-score    | 83.79% |
| Weighted F1-score | 84.15% |

### Confusion Matrix

The confusion matrix provides an overview of correct and incorrect predictions across the seven target classes.

File: `reports/phase3/confusion_matrix.png`

### Classification Report

The classification report provides precision, recall, F1-score, and support for each target class.

File: `reports/phase3/classification_report.csv`

Additional formats:

- `reports/phase3/classification_report.json`
- `reports/phase3/classification_report.txt`

### Evaluation Scripts

```powershell
python -m src.evaluation.evaluate_confusion_matrix
```

```powershell
python -m src.evaluation.evaluate_classification_report
```

These scripts use the existing model and reproduce the original holdout partition.

## Feature Importance Analysis

Permutation feature importance is calculated on the holdout partition using Macro-F1 as the scoring metric.

The analysis uses 30 repeats and a fixed random state for reproducibility.

### Top Five Features

| Rank | Feature                        | Importance Mean |
| ---: | ------------------------------ | --------------: |
|    1 | Gender                         |        0.262609 |
|    2 | Age                            |        0.174352 |
|    3 | FCVC                           |        0.168280 |
|    4 | Family history with overweight |        0.160557 |
|    5 | CALC                           |        0.110797 |

The complete ranking is available in:

- `reports/phase3/feature_importance.csv`
- `reports/phase3/feature_importance.json`
- `reports/phase3/feature_importance.md`
- `reports/phase3/feature_importance.png`

Permutation importance measures how much the model's predictive performance depends on a feature under the evaluation procedure. It does not establish causation or clinical importance.

## FastAPI Web Application

The project includes a web application developed using:

- FastAPI
- Uvicorn
- Jinja2
- HTML and CSS
- Scikit-learn
- Joblib

### Application Features

- Home page introducing the project.
- Prediction form for 14 input features.
- Prediction result with Vietnamese and English class labels.
- Probability distribution across all seven classes.
- About page displaying model evaluation metrics.
- Confusion matrix visualization.
- Classification report table.
- Feature importance visualization and ranking.
- Health-check endpoint.

### Model Artifacts

The deployed application loads the baseline model:

```text
models/
├── uci_obesity_lifestyle_v1.joblib
└── uci_obesity_lifestyle_v1.json
```

The tuned model is retained as a research artifact:

```text
models/
├── uci_obesity_lifestyle_tuned_v1.joblib
└── uci_obesity_lifestyle_tuned_v1.json
```

The tuned model is not the model currently used by the prediction application.

## API Documentation

The application provides the following endpoints.

| Method | Endpoint       | Description                           |
| ------ | -------------- | ------------------------------------- |
| GET    | `/`            | Home page                             |
| GET    | `/predict`     | Prediction form                       |
| POST   | `/predict`     | Submit prediction form                |
| GET    | `/about`       | Model information and evaluation      |
| POST   | `/api/predict` | Prediction REST API                   |
| GET    | `/health`      | Application health check              |
| GET    | `/docs`        | Interactive Swagger API documentation |

### Prediction API

Endpoint:

```text
POST /api/predict
```

The API accepts a JSON object containing all 14 model features.

Example request structure:

```json
{
  "Age": 25,
  "FCVC": 2,
  "NCP": 3,
  "CH2O": 2,
  "FAF": 1,
  "TUE": 1,
  "Gender": "<valid category>",
  "family_history_with_overweight": "<valid category>",
  "FAVC": "<valid category>",
  "CAEC": "<valid category>",
  "SMOKE": "<valid category>",
  "SCC": "<valid category>",
  "CALC": "<valid category>",
  "MTRANS": "<valid category>"
}
```

Replace categorical placeholders with values accepted by the trained model. The valid values can be inspected through the prediction form or model preprocessing configuration.

### Response Structure

A successful prediction returns:

- `predicted_class`: Predicted class identifier.
- `predicted_label_vi`: Vietnamese class label.
- `probabilities`: Probability and percentage for each of the seven classes.

Example response structure:

```json
{
  "predicted_class": "Normal_Weight",
  "predicted_label_vi": "Cân nặng bình thường",
  "probabilities": [
    {
      "class": "Normal_Weight",
      "label_vi": "Cân nặng bình thường",
      "probability": 0.85,
      "percentage": 85.0
    }
  ]
}
```

The probability list contains all seven classes and is sorted by descending probability. The values above illustrate the response structure, not an actual prediction result.

### Health Check

Endpoint:

```text
GET /health
```

Returns application status, model loading status, model name, number of features, and number of target classes.

## Repository Structure

```text
ObesityPrediction/
│
├── data/
│   └── raw/
│
├── dataset/
│
├── models/
│   ├── uci_obesity_lifestyle_v1.joblib
│   ├── uci_obesity_lifestyle_v1.json
│   ├── uci_obesity_lifestyle_tuned_v1.joblib
│   └── uci_obesity_lifestyle_tuned_v1.json
│
├── reports/
│   ├── feature_importance.md
│   ├── gender_ablation.csv
│   ├── gender_ablation.md
│   ├── gender_performance.md
│   ├── gender_subgroup_metrics.csv
│   ├── gender_target_distribution.csv
│   ├── model_comparison.csv
│   ├── model_evaluation.md
│   ├── permutation_importance.csv
│   ├── phase2_final_report.md
│   ├── random_forest_tuning.md
│   ├── random_forest_tuning_results.csv
│   ├── uci_dataset_assessment.md
│   └── phase3/
│       ├── classification_report.csv
│       ├── classification_report.json
│       ├── classification_report.txt
│       ├── confusion_matrix.csv
│       ├── confusion_matrix.png
│       ├── confusion_matrix_summary.json
│       ├── feature_importance.csv
│       ├── feature_importance.json
│       ├── feature_importance.md
│       └── feature_importance.png
├── src/
│   ├── deployment/
│   │   └── __init__.py
│   │   └── predictor.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_classification_report.py
│   │   └── evaluate_confusion_matrix.py
│   │
│   └── training/
│       ├── train_uci.py
│       ├── analyze_feature_importance.py
│       ├── analyze_gender_performance.py
│       ├── analyze_gender_ablation.py
│       ├── tune_random_forest.py
│       └── README.md
│
├── scripts/
├── static/
│   └── style.css
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── about.html
│
├── app.py
├── requirements.txt
├── requirements-phase2.txt
├── .gitattributes
├── .gitignore
└── README.md
```

Some auxiliary files and directories are omitted from this overview.

## Installation and Setup

### Requirements

- Python 3.10 or later, with a compatible package environment.
- Git.
- Git LFS for downloading model artifacts.

### 1. Clone the Repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ObesityPrediction
```

Replace the placeholder with the actual repository URL.

### 2. Download Git LFS Artifacts

```powershell
git lfs install
git lfs pull
```

This downloads the trained `.joblib` model files tracked by Git LFS.

### 3. Create a Virtual Environment

Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

Run commands from the repository root.

Start the FastAPI application:

```powershell
python -m uvicorn app:app --reload
```

Open the following URLs in a browser:

| Page              | URL                           |
| ----------------- | ----------------------------- |
| Home              | http://127.0.0.1:8000/        |
| Prediction        | http://127.0.0.1:8000/predict |
| About             | http://127.0.0.1:8000/about   |
| API Documentation | http://127.0.0.1:8000/docs    |
| Health Check      | http://127.0.0.1:8000/health  |

Stop the server with `Ctrl + C`.

## Running the ML Pipeline

The following commands run the research and training scripts from the repository root.

### Train and Compare Models

```powershell
python -m src.training.train_uci
```

### Analyze Feature Importance

```powershell
python -m src.training.analyze_feature_importance
```

### Evaluate Gender Subgroups

```powershell
python -m src.training.analyze_gender_performance
```

### Run Gender Feature Ablation

```powershell
python -m src.training.analyze_gender_ablation
```

### Tune Random Forest

```powershell
python -m src.training.tune_random_forest
```

### Generate Phase 3 Evaluation Reports

```powershell
python -m src.evaluation.evaluate_confusion_matrix
```

```powershell
python -m src.evaluation.evaluate_classification_report
```

See `src/training/README.md` for additional details about the training workflow.

**Note:** Running the training and evaluation scripts may regenerate model artifacts or reports. The deployed application uses the existing baseline model unless the deployment configuration is deliberately changed.

## Limitations

- The dataset contains a substantial proportion of synthetic observations.
- Related observations may occur across training and holdout partitions.
- The holdout set has been examined in multiple analyses and should not be considered an untouched final evaluation.
- Target class distributions are not perfectly balanced.
- Feature importance indicates model reliance, not causal relationships.
- Subgroup performance differences alone do not establish bias or fairness.
- The project does not establish clinical usefulness, diagnostic reliability, or external generalization.

Independent external validation on representative real-world data is required before stronger generalization claims can be made.

## Future Improvements

Potential future development directions include:

- External validation using independent real-world datasets.
- More extensive model calibration and uncertainty analysis.
- Improved API input validation and automated testing.
- Model monitoring and version management.
- Additional interpretability and subgroup evaluation.
- Deployment using a production-ready server configuration.
- Improved user experience and accessibility.

## Project Status

| Phase   | Description                            | Status      |
| ------- | -------------------------------------- | ----------- |
| Phase 1 | Dataset exploration and preprocessing  | Completed   |
| Phase 2 | Model development and analysis         | Completed   |
| Phase 3 | Evaluation and FastAPI integration     | Completed   |
| Phase 4 | Documentation and project presentation | Completed   |

---

**Project:** ObesityPrediction

**Framework:** FastAPI

**Machine Learning:** Scikit-learn

**Purpose:** Educational research and machine learning experimentation
