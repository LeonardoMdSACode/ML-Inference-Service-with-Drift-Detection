# app/api/routes.py
# /predict, /health, /dashboard, /monitoring/run

from fastapi import APIRouter, BackgroundTasks
from app.api.schemas import PredictionRequest, PredictionResponse
from app.inference.predictor import Predictor
from app.core.logging import log_prediction
from app.monitoring.data_loader import load_production_data
from app.monitoring.drift import run_drift_check
from app.monitoring.governance import run_governance_checks
import pandas as pd

router = APIRouter()
predictor = Predictor()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    payload = request.dict()
    prediction, probability = predictor.predict(payload)

    log_prediction(payload, prediction, probability)

    return {
        "prediction": prediction,
        "probability": probability
    }


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
    Step 6: Run production monitoring including drift + governance checks in background.
    """
    # Load current and reference data
    current_data = pd.read_csv("data/processed/current_data.csv")
    reference_data = pd.read_csv("data/processed/credit_default_clean.csv")  # reference

    # Schedule background tasks
    background_tasks.add_task(run_drift_check, current_data, reference_data, model_version=model_version)
    background_tasks.add_task(run_governance_checks, current_data, model_version=model_version)

    return {"status": "monitoring triggered", "model_version": model_version}
