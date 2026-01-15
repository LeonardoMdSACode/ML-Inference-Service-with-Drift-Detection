# app/monitoring/governance.py
import json
import logging
from datetime import datetime
import os
from app.utils.alerts import send_email_alert, send_slack_alert
from app.core.config import LOGS_PATH  # configurable logs folder

# ensure logs folder exists
os.makedirs(LOGS_PATH, exist_ok=True)

# setup logger
logger = logging.getLogger("governance")
logger.setLevel(logging.INFO)

# Remove all existing handlers
if logger.hasHandlers():
    logger.handlers.clear()
    
handler = logging.FileHandler(os.path.join(LOGS_PATH, "governance_alerts.log"))
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Governance:
    def __init__(self, thresholds: dict):
        """
        thresholds example:
        {
            "psi": 0.2,
            "accuracy_drop": 0.05,
            "f1": 0.7
        }
        """
        self.thresholds = thresholds

    def check_metrics(self, report_dict: dict, model_version: str):
        alerts = []

        # Normalize report_dict to a metrics dict
        metrics = {}
        if isinstance(report_dict, dict):
            raw_metrics = report_dict.get("metrics")
            if isinstance(raw_metrics, list):
                for item in raw_metrics:
                    metric_name = item.get("metric")
                    result = item.get("result", {})
                    if metric_name:
                        metrics[metric_name] = result
            else:
                metrics = raw_metrics or {}
        elif isinstance(report_dict, list):
            for item in report_dict:
                metric_name = item.get("metric")
                result = item.get("result", {})
                if metric_name:
                    metrics[metric_name] = result

        # Data drift (project-level)
        psi_metric = metrics.get("DatasetDriftMetric", {})
        psi = psi_metric.get("share_of_drifted_columns", 0)
        if psi > self.thresholds.get("psi", 0.2):
            alerts.append(f"Data drift detected (PSI={psi})")

        # Column-level drift alerts
        data_drift_table = metrics.get("DataDriftTable", {}).get("drift_by_columns", {})
        if data_drift_table:
            for col, info in data_drift_table.items():
                if isinstance(info, dict) and info.get("drift_detected", False):
                    alert_msg = f"Drift detected in column {col} (score={info.get('drift_score')})"
                    alerts.append(alert_msg)

        # Classification performance
        f1 = metrics.get("ClassificationPreset", {}).get("f1_score", 1.0)
        if f1 < self.thresholds.get("f1", 0.7):
            alerts.append(f"F1 drop detected (F1={f1})")

        # Regression accuracy
        accuracy_drop = metrics.get("RegressionPreset", {}).get("accuracy_drop", 0)
        if accuracy_drop > self.thresholds.get("accuracy_drop", 0.05):
            alerts.append(f"Accuracy drop detected ({accuracy_drop})")

        # Log and send alerts
        for alert in alerts:
            self.log_alert(alert, model_version)
        try:
            send_email_alert(alert)
            send_slack_alert(alert)
        except Exception:
            pass

        return alerts

    @staticmethod
    def log_alert(message: str, model_version: str):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "alert": message
        }
        logger.info(json.dumps(log_entry))


def run_governance_checks(report_dict: dict, model_version: str = "v1", thresholds: dict = None):
    """
    Convenience wrapper to run governance checks using default thresholds.
    """
    thresholds = thresholds or {"psi": 0.2, "accuracy_drop": 0.05, "f1": 0.7}
    governance = Governance(thresholds)
    return governance.check_metrics(report_dict, model_version)
