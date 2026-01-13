# app/api/routes.py
# /predict, /health, /dashboard, /monitoring/run

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.inference.predictor import Predictor
from app.monitoring.data_loader import load_production_data
from app.monitoring.drift import run_drift_check
from app.monitoring.governance import run_governance_checks

import pandas as pd

templates = Jinja2Templates(directory="app/templates")

router = APIRouter()
predictor = Predictor()


# CSV upload & prediction
@router.post("/predict")
async def predict_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    df = pd.read_csv(file.file)

    # ---- STRICT MODE: schema enforcement ----
    missing = set(predictor.features) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid schema. Missing required columns: {sorted(missing)}"
        )

    # ---- Model inference ----
    preds, probas = predictor.predict(df)

    results = []
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        results.append({
            "row": i,
            "probability": round(float(proba), 4),
            "prediction": "Default" if pred == 1 else "No Default",
            "risk_level": (
                "High" if proba >= 0.75 else
                "Medium" if proba >= 0.5 else
                "Low"
            )
        })

    # ---- Drift scheduled in background ----
    reference_df = pd.read_csv("models/v1/reference_data.csv")
    background_tasks.add_task(
        run_drift_check, df[predictor.features], reference_df[predictor.features], "v1"
    )

    return JSONResponse({
        "n_rows": len(results),
        "results": results,
        "drift": "scheduled"
    })


# Health
@router.get("/health")
def health():
    return {"status": "ok"}


# Manual drift run
@router.get("/run-drift")
def run_drift():
    current_df = load_production_data()
    report_path = run_drift_check(current_df)
    return {
        "status": "drift_check_completed",
        "report_path": report_path
    }


# Monitoring pipeline
@router.get("/monitoring/run")
def monitoring_run(background_tasks: BackgroundTasks, model_version: str = "v1"):
    current_data = pd.read_csv("data/processed/current_data.csv")
    reference_data = pd.read_csv("data/processed/credit_default_clean.csv")

    background_tasks.add_task(
        run_drift_check,
        current_data[predictor.features],
        reference_data[predictor.features],
        model_version
    )
    background_tasks.add_task(
        run_governance_checks,
        current_data,
        model_version=model_version
    )

    return {
        "status": "monitoring triggered",
        "model_version": model_version
    }


# Dashboard
@router.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )
