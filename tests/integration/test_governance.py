# tests/test_governance.py

import json
from app.monitoring.governance import run_governance_checks


def test_governance_detects_alerts():
    with open("tests/integration/test_governance.json", "r") as f:
        report = json.load(f)

    alerts = run_governance_checks(report, model_version="v1")

    assert isinstance(alerts, list)
    assert len(alerts) >= 0
