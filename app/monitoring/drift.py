# Evidently logic

import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

REFERENCE_DATA_PATH = "models/v1/reference_data.csv"
REPORT_DIR = "reports/evidently"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")


def run_drift_check(current_df: pd.DataFrame):
    reference_df = pd.read_csv(REFERENCE_DATA_PATH)

    os.makedirs(REPORT_DIR, exist_ok=True)

    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(
        reference_data=reference_df.drop(columns=["target"]),
        current_data=current_df
    )

    report.save_html(REPORT_PATH)

    return REPORT_PATH
