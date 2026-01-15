# app/api/routes.py
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.inference.predictor import Predictor
from app.monitoring.data_loader import load_production_data
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
# ENSURE production log exists at server startup
# ------------------------------------------------------------------
os.makedirs(os.path.dirname(PROD_LOG), exist_ok=True)

if not os.path.exists(PROD_LOG):
    base_cols = list(predictor.features)
    extra_cols = [
        "target",            # true label
        "model_prediction",  # model output
        "model_probability",
        "model_risk_level",
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

    # ---- Append predictions to production log (minimal, fast) ----
    df_log = df.copy()

    # Keep true target if present
    if "target" in df.columns:
        df_log["target"] = df["target"]
    else:
        df_log["target"] = np.nan

    # Remove any old model prediction columns to prevent duplicates
    for col in ["model_prediction", "model_probability", "model_risk_level", "model_version", "timestamp"]:
        if col in df_log.columns:
            df_log = df_log.drop(columns=[col])

    df_log["model_prediction"] = preds
    df_log["model_probability"] = probas
    df_log["model_risk_level"] = [
        "High" if p >= 0.75 else "Medium" if p >= 0.5 else "Low"
        for p in probas
    ]
    df_log["model_version"] = predictor.model_version
    df_log["timestamp"] = pd.Timestamp.utcnow()

    df_log.to_csv(PROD_LOG, mode="a", header=False, index=False)

    return JSONResponse({
        "n_rows": len(results),
        "results": results,
    })


@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})
