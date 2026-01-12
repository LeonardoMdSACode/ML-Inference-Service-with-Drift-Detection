# This file implements threshold checking, governance signals logging, and notifications.

import json
import logging
from datetime import datetime
from app.utils.alerts import send_email_alert, send_slack_alert
import os

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("governance")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/governance_alerts.log")
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

        # Example: data drift
        psi = report_dict.get("metrics", {}).get("DataDriftPreset", {}).get("result", {}).get("dataset_drift", 0)
        if psi > self.thresholds.get("psi", 0.2):
            alerts.append(f"Data drift detected (PSI={psi})")

        # Example: classification performance
        f1 = report_dict.get("metrics", {}).get("ClassificationPreset", {}).get("result", {}).get("f1_score", 1.0)
        if f1 < self.thresholds.get("f1", 0.7):
            alerts.append(f"F1 drop detected (F1={f1})")

        # Example: regression accuracy
        accuracy_drop = report_dict.get("metrics", {}).get("RegressionPreset", {}).get("result", {}).get("accuracy_drop", 0)
        if accuracy_drop > self.thresholds.get("accuracy_drop", 0.05):
            alerts.append(f"Accuracy drop detected ({accuracy_drop})")

        # Log alerts
        for alert in alerts:
            self.log_alert(alert, model_version)

        # Optional notifications
        for alert in alerts:
            send_email_alert(alert)
            send_slack_alert(alert)

        return alerts

    @staticmethod
    def log_alert(message: str, model_version: str):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "alert": message
        }
        logger.info(json.dumps(log_entry))
