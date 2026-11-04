"""FastAPI application for the obesity prediction project."""

import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.deployment.predictor import predictor


ROOT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
REPORTS_DIR = ROOT_DIR / "reports" / "phase3"


app = FastAPI(
    title="Obesity Prediction API",
    description=(
        "Machine learning system for educational "
        "and research support in obesity-level prediction."
    ),
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Các chỉ số được tính trên tập kiểm tra gồm 418 mẫu.
MODEL_METRICS = [
    {"name": "Accuracy", "value": 84.21},
    {"name": "Balanced Accuracy", "value": 83.94},
    {"name": "Macro Precision", "value": 83.82},
    {"name": "Macro Recall", "value": 83.94},
    {"name": "Macro F1-score", "value": 83.79},
    {"name": "Weighted F1-score", "value": 84.15},
]


def read_csv_report(filename: str):
    """Read a CSV report from reports/phase3 safely."""

    file_path = REPORTS_DIR / filename

    if not file_path.exists():
        return [], []

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []
        rows = list(reader)

    return headers, rows


def get_evaluation_data():
    """Load evaluation tables and check whether report images exist."""

    classification_headers, classification_rows = read_csv_report(
        "classification_report.csv"
    )

    feature_headers, feature_rows = read_csv_report(
        "feature_importance.csv"
    )

    confusion_matrix_exists = (
        REPORTS_DIR / "confusion_matrix.png"
    ).exists()

    feature_importance_exists = (
        REPORTS_DIR / "feature_importance.png"
    ).exists()

    return {
        "metrics": MODEL_METRICS,
        "classification_headers": classification_headers,
        "classification_rows": classification_rows,
        "feature_headers": feature_headers,
        "feature_rows": feature_rows,
        "confusion_matrix_exists": confusion_matrix_exists,
        "feature_importance_exists": feature_importance_exists,
    }


class PredictionRequest(BaseModel):
    """Input schema containing the 14 model features."""

    Age: float = Field(gt=0)
    FCVC: float
    NCP: float
    CH2O: float
    FAF: float
    TUE: float

    Gender: str
    family_history_with_overweight: str
    FAVC: str
    CAEC: str
    SMOKE: str
    SCC: str
    CALC: str
    MTRANS: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "category_options": predictor.get_category_options(),
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    form = await request.form()
    input_data = dict(form)

    try:
        result = predictor.predict(input_data)

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "result": result,
                "input_data": input_data,
            },
        )

    except (ValueError, TypeError) as exc:
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "category_options": predictor.get_category_options(),
                "error": str(exc),
                "form_data": input_data,
            },
            status_code=400,
        )


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    context = {
        "metadata": predictor.metadata,
        **get_evaluation_data(),
    }

    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context=context,
    )


@app.get("/reports/phase3/{filename}")
async def evaluation_report_image(filename: str):
    """Serve only the two approved evaluation images."""

    allowed_files = {
        "confusion_matrix.png",
        "feature_importance.png",
    }

    if filename not in allowed_files:
        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found.",
        )

    file_path = REPORTS_DIR / filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evaluation image has not been generated.",
        )

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=filename,
    )


@app.post("/api/predict")
async def predict_api(data: PredictionRequest):
    try:
        result = predictor.predict(data.model_dump())
        return result

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_name": predictor.metadata.get("model_name"),
        "n_features": len(predictor.features),
        "n_classes": len(predictor.classes),
    }