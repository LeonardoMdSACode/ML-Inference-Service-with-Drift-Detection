# app/api/dashboard_data.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import json
import os

router = APIRouter()

DATA_FILE = "reports/evidently/drift_report.json"  # we will write drift info here

@router.get("/dashboard/data")
def get_dashboard_data():
    """
    Return the latest drift and prediction summary for the frontend dashboard.
    """
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            return JSONResponse({"status": "ok", "data": data})
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    else:
        return JSONResponse({"status": "error", "message": "No data available"}, status_code=404)
