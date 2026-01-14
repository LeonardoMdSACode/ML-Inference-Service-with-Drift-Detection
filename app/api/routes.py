# app/api/routes.py
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.inference.predictor import Predictor
from app.monitoring.data_loader import load_production_data
from app.monitoring.drift import run_drift_check
from app.monitoring.governance import run_governance_checks

import pandas as pd
import numpy as np
import json
import os

templates = Jinja2Templates(directory="app/templates")
router = APIRouter()
predictor = Predictor()

# Production log file
PROD_LOG = "data/production/predictions_log.csv"

# ------------------------------------------------------------------
# ENSURE production log exists at server startup (CRITICAL FIX)
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(PROD_LOG), exist_ok=True)

if not os.path.exists(PROD_LOG):
    # Create empty production log with correct schema
    base_cols = list(predictor.features)
    extra_cols = [
        "prediction",
        "probability",
        "risk_level",
        "model_version",
        "timestamp",
    ]
    empty_df = pd.DataFrame(columns=base_cols + extra_cols)
    empty_df.to_csv(PROD_LOG, index=False)
# ------------------------------------------------------------------


@router.post("/predict")
async def predict_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # ---- STRICT MODE: schema enforcement ----
    missing = set(predictor.features) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid schema. Missing required columns: {sorted(missing)}",
        )

    # ---- Model inference ----
    preds, probas = predictor.predict(df)
    results = []
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        results.append({
            "row": i,
            "probability": round(float(proba), 4),
            "prediction": "Default" if pred == 1 else "No Default",
            "risk_level": "High" if proba >= 0.75 else "Medium" if proba >= 0.5 else "Low"
        })

    # ---- Drift: immediate for frontend ----
    reference_df = pd.read_csv("models/v1/reference_data.csv")
    _, drift_dict = run_drift_check(
        df[predictor.features],
        reference_df[predictor.features],
        "v1",
    )

    drift_for_chart = []
    for col, score in drift_dict.items():
        try:
            score_value = float(score)
            if not np.isfinite(score_value):
                score_value = 0.0
        except Exception:
            score_value = 0.0
        drift_for_chart.append({"column": col, "score": score_value})

    # ---- Append predictions to production log ----
    df_log = df.copy()

    # ---- FIX: Remove existing prediction/risk/probability/etc columns to avoid extra column issue ----
    for col in ["prediction", "probability", "risk_level", "model_version", "timestamp"]:
        if col in df_log.columns:
            df_log = df_log.drop(columns=[col])

    df_log["prediction"] = preds
    df_log["probability"] = probas
    df_log["risk_level"] = [
        "High" if p >= 0.75 else "Medium" if p >= 0.5 else "Low"
        for p in probas
    ]
    df_log["model_version"] = predictor.model_version
    df_log["timestamp"] = pd.Timestamp.utcnow()

    df_log.to_csv(PROD_LOG, mode="a", header=False, index=False)

    # ---- Dashboard JSON ----
    DASHBOARD_JSON = "reports/evidently/drift_report.json"

    dashboard_payload = {
        "n_rows": len(results),
        "results": results,
        "drift": drift_for_chart,
    }

    os.makedirs(os.path.dirname(DASHBOARD_JSON), exist_ok=True)
    tmp_path = DASHBOARD_JSON + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(dashboard_payload, f, indent=2)
    os.replace(tmp_path, DASHBOARD_JSON)

    return JSONResponse({
        "n_rows": len(results),
        "results": results,
        "drift": drift_for_chart,
    })


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/run-drift")
def run_drift():
    current_df = load_production_data()
    report_path = run_drift_check(current_df)
    return {"status": "drift_check_completed", "report_path": report_path}


@router.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})
