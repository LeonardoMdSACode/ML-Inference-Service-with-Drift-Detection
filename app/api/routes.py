# /predict, /health, /dashboard

from fastapi import APIRouter
from app.api.schemas import PredictionRequest, PredictionResponse
from app.inference.predictor import Predictor
from app.core.logging import log_prediction
from app.monitoring.data_loader import load_production_data
from app.monitoring.drift import run_drift_check


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
