# app/api/background_drift.py
import asyncio
import pandas as pd
import os
import json
import numpy as np

from app.monitoring.drift import run_drift_check
from app.inference.predictor import Predictor

predictor = Predictor()
REFERENCE_PATH = "models/v1/reference_data.csv"
CURRENT_DATA_PATH = "data/production/predictions_log.csv"
DASHBOARD_JSON = "reports/evidently/drift_report.json"

# Ensure folder exists and JSON file exists at startup
os.makedirs(os.path.dirname(DASHBOARD_JSON), exist_ok=True)
if not os.path.exists(DASHBOARD_JSON):
    with open(DASHBOARD_JSON, "w") as f:
        json.dump({"n_rows": 0, "results": [], "drift": [{"column": feat, "score": 0.0} for feat in predictor.features]}, f, indent=2)

async def drift_loop(interval_seconds: int = 30):
    """
    Continuously run drift checks and update dashboard JSON.
    """
    while True:
        try:
            current_df = pd.read_csv(CURRENT_DATA_PATH)
            reference_df = pd.read_csv(REFERENCE_PATH)

            _, drift_dict = run_drift_check(
                current_df[predictor.features],
                reference_df[predictor.features],
                "v1"
            )

            # Ensure numeric safe drift values
            drift_for_chart = []
            for col, score in drift_dict.items():
                try:
                    val = float(score)
                    if not np.isfinite(val):
                        val = 0.0
                except Exception:
                    val = 0.0
                drift_for_chart.append({"column": col, "score": val})

            dashboard_payload = {
                "n_rows": len(current_df),
                "results": [],  # predictions not included in background loop
                "drift": drift_for_chart
            }

            # Atomic write to avoid read/write collision
            tmp_path = DASHBOARD_JSON + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(dashboard_payload, f, indent=2)
            os.replace(tmp_path, DASHBOARD_JSON)

        except Exception as e:
            print("Drift loop error:", e)

        await asyncio.sleep(interval_seconds)
