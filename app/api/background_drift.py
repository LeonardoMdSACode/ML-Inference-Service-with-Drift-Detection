# app/api/background_drift.py
import asyncio
import pandas as pd
import os
import json

from app.monitoring.drift import run_drift_check
from app.inference.predictor import Predictor

predictor = Predictor()

REFERENCE_PATH = "models/v1/reference_data.csv"
PROD_LOG_PATH = "data/production/predictions_log.csv"
DASHBOARD_JSON = "reports/evidently/drift_report.json"

MAX_ROWS = 5000  # rolling window
os.makedirs(os.path.dirname(DASHBOARD_JSON), exist_ok=True)

async def drift_loop(interval_seconds: int = 10):
    while True:
        try:
            if not os.path.exists(PROD_LOG_PATH):
                await asyncio.sleep(interval_seconds)
                continue

            prod_df = pd.read_csv(PROD_LOG_PATH)

            # Retention window
            if len(prod_df) > MAX_ROWS:
                prod_df = prod_df.tail(MAX_ROWS)
                prod_df.to_csv(PROD_LOG_PATH, index=False)

            # Keep only rows with all required features
            missing_features = set(predictor.features) - set(prod_df.columns)
            if missing_features:
                print(f"Skipping drift check, missing features: {missing_features}")
                await asyncio.sleep(interval_seconds)
                continue

            prod_df = prod_df.dropna(subset=predictor.features)
            if prod_df.empty:
                await asyncio.sleep(interval_seconds)
                continue

            reference_df = pd.read_csv(REFERENCE_PATH)

            # ---- Run drift on features only ----
            _, drift_dict = run_drift_check(
                prod_df[predictor.features],
                reference_df[predictor.features],
                model_version="v1"
            )

            # ---- Populate predictions for dashboard ----
            results = []
            if "model_prediction" in prod_df.columns and "model_probability" in prod_df.columns:
                for i, row in prod_df.tail(50).iterrows():  # last 50 rows
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
