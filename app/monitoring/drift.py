# Evidently logic
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
    Returns a tuple: (alerts, report_metrics)
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(current_data=current_data, reference_data=reference_data)
    report.save_html(REPORT_PATH)

    # report.as_dict() returns a dict; in newer Evidently versions it can be a list
    report_metrics = report.as_dict() if hasattr(report, "as_dict") else list(report)

    # Run governance checks
    alerts = governance.check_metrics(report_metrics, model_version=model_version)

    return alerts, report_metrics
