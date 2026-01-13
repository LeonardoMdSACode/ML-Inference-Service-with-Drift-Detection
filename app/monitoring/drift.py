# app/monitoring/drift.py
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from app.monitoring.governance import Governance

REFERENCE_DATA_PATH = "models/v1/reference_data.csv"
REPORT_DIR = "reports/evidently"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")

# Thresholds configuration
thresholds = {
    "psi": 0.2,
    "accuracy_drop": 0.05,
    "f1": 0.7
}

governance = Governance(thresholds=thresholds)


def run_drift_check(current_data: pd.DataFrame, reference_data: pd.DataFrame, model_version="v1"):
    """
    Run Evidently DataDriftPreset on current vs reference data,
    save HTML report, and run governance checks.
    Returns a tuple: (alerts, drift_scores)
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(current_data=current_data, reference_data=reference_data)
    report.save_html(REPORT_PATH)

    # Extract numeric drift scores per column
    report_dict = report.as_dict() if hasattr(report, "as_dict") else {}
    drift_scores = {}

    metrics_list = report_dict.get("metrics", [])

    for metric in metrics_list:
        result = metric.get("result", {})
        # Check column-level drift
        drift_by_columns = result.get("drift_by_columns", {})
        if drift_by_columns:
            for col, info in drift_by_columns.items():
                score = info.get("drift_score", 0.0)
                if score is None or not pd.notna(score):
                    score = 0.0
                drift_scores[col] = float(score)
        # fallback: Dataset-level drift metric (PSI share)
        elif metric.get("metric") == "DatasetDriftMetric":
            drift_scores["dataset"] = float(result.get("share_of_drifted_columns", 0.0))

    # Run governance checks (keeps existing alerts)
    alerts = governance.check_metrics(report_dict, model_version=model_version)

    return alerts, drift_scores
