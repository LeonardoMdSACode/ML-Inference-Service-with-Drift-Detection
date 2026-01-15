# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import asyncio
import os
import pandas as pd
import random
import json
from datetime import datetime

from app.api.routes import router
from app.api.dashboard_data import router as dashboard_data_router
from app.inference.predictor import Predictor
from app.monitoring.drift import run_drift_check
from app.core.logging import init_db

# ---- Constants ----
PROD_LOG_PATH = "data/production/predictions_log.csv"
REFERENCE_PATH = "models/v1/reference_data.csv"
DASHBOARD_JSON = "reports/evidently/drift_report.json"
SOURCE_DATA = "data/processed/current_data.csv"

# ---- Config ----
STARTUP_DELAY = 5
MIN_SLEEP = 2
MAX_SLEEP = 8
MIN_BATCH = 1
MAX_BATCH = 5
MAX_DRIFT_ROWS = 9000
MAX_DISPLAY = 101  # last N predictions for dashboard

predictor = Predictor()
os.makedirs(os.path.dirname(DASHBOARD_JSON), exist_ok=True)

# ---- Traffic daemon in-process (no HTTP call) ----
async def traffic_loop():
    await asyncio.sleep(STARTUP_DELAY)
    if not os.path.exists(SOURCE_DATA):
        print("Traffic daemon: source data not found, disabled.")
        return

    df_source = pd.read_csv(SOURCE_DATA)
    print("Traffic daemon started (in-process).")

    while True:
        try:
            batch_size = random.randint(MIN_BATCH, MAX_BATCH)
            sample = df_source.sample(batch_size)
            # In-process prediction instead of requests.post
            preds, probas = predictor.predict(sample)
            df_log = sample.copy()
            df_log["model_prediction"] = preds
            df_log["model_probability"] = probas
            df_log["model_risk_level"] = [
                "High" if p >= 0.75 else "Medium" if p >= 0.5 else "Low"
                for p in probas
            ]
            df_log["model_version"] = predictor.model_version
            df_log["timestamp"] = pd.Timestamp.utcnow()
            df_log.to_csv(PROD_LOG_PATH, mode="a", header=not os.path.exists(PROD_LOG_PATH), index=False)

        except Exception as e:
            print("Traffic daemon error:", e)

        await asyncio.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


# ---- Drift loop ----
async def drift_loop(interval_seconds: int = 10):
    while True:
        try:
            if not os.path.exists(PROD_LOG_PATH):
                await asyncio.sleep(interval_seconds)
                continue

            prod_df = pd.read_csv(PROD_LOG_PATH)
            if len(prod_df) > MAX_DRIFT_ROWS:
                prod_df = prod_df.tail(MAX_DRIFT_ROWS)
                prod_df.to_csv(PROD_LOG_PATH, index=False)

            missing_features = set(predictor.features) - set(prod_df.columns)
            if missing_features:
                await asyncio.sleep(interval_seconds)
                continue

            prod_df = prod_df.dropna(subset=predictor.features)
            if prod_df.empty:
                await asyncio.sleep(interval_seconds)
                continue

            reference_df = pd.read_csv(REFERENCE_PATH)
            _, drift_dict = run_drift_check(
                prod_df[predictor.features],
                reference_df[predictor.features],
                model_version="v1"
            )

            # Prepare last N predictions for dashboard
            results = []
            log_cols = ["model_prediction", "model_probability", "model_risk_level"]
            if all(c in prod_df.columns for c in log_cols):
                for i, row in prod_df.tail(MAX_DISPLAY).iterrows():
                    results.append({
                        "row": i,
                        "prediction": "Default" if row["model_prediction"] == 1 else "No Default",
                        "probability": round(float(row["model_probability"]), 4),
                        "risk_level": row.get("model_risk_level", "Unknown")
                    })

            dashboard_payload = {
                "n_rows": len(prod_df),
                "results": results,
                "drift": [
                    {"column": col, "score": float(score)}
                    for col, score in drift_dict.items()
                ],
            }

            tmp_path = DASHBOARD_JSON + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(dashboard_payload, f, indent=2)
            os.replace(tmp_path, DASHBOARD_JSON)

        except Exception as e:
            print("Drift loop error:", e)

        await asyncio.sleep(interval_seconds)


# ---- HF-compatible lifespan ----
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    tasks = [
        asyncio.create_task(traffic_loop()),
        asyncio.create_task(drift_loop(10))
    ]
    yield
    for t in tasks:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass


# ---- FastAPI app ----
app = FastAPI(title="ML Inference Service", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.include_router(router)
app.include_router(dashboard_data_router)
