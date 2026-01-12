# app/api/routes.py
# /predict, /health, /dashboard, /monitoring/run

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from app.api.schemas import PredictionRequest, PredictionResponse
from app.inference.predictor import Predictor
from app.core.logging import log_prediction
from app.monitoring.data_loader import load_production_data
from app.monitoring.drift import run_drift_check
from app.monitoring.governance import run_governance_checks
import pandas as pd
import os
from app.core.templates import templates
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")

router = APIRouter()
predictor = Predictor()


# Endpoint for CSV upload & prediction with drift
@router.post("/predict")
async def predict_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    df = pd.read_csv(file.file)
    predictions, probability = predictor.predict(df)

    reference_df = pd.read_csv("models/v1/reference_data.csv")
    background_tasks.add_task(
        run_drift_check, df, reference_df, "v1"
    )

    return JSONResponse({
        "predictions": predictions.tolist() if hasattr(predictions, "tolist") else predictions,
        "drift": "scheduled"
    })




@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/run-drift")
def run_drift():
    current_df = load_production_data()
    report_path = run_drift_check(current_df)
    return {
        "status": "drift_check_completed",
        "report_path": report_path
    }


@router.get("/monitoring/run")
def monitoring_run(background_tasks: BackgroundTasks, model_version: str = "v1"):
    """
    Run production monitoring including drift + governance checks in background.
    """
    # Load current and reference data
    current_data = pd.read_csv("data/processed/current_data.csv")
    reference_data = pd.read_csv("data/processed/credit_default_clean.csv")  # reference

    # Schedule background tasks
    background_tasks.add_task(run_drift_check, current_data, reference_data, model_version=model_version)
    background_tasks.add_task(run_governance_checks, current_data, model_version=model_version)

    return {"status": "monitoring triggered", "model_version": model_version}


# Dashboard endpoint
@router.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )
