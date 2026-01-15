# tests/integration/test_api.py

import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_valid_csv():
    df = pd.DataFrame([
        {
            "credit_limit": 50000.0,
            "age": 35,
            "pay_delay_sep": 0,
            "pay_delay_aug": -1,
            "bill_amt_sep": 12000.0,
            "bill_amt_aug": 11000.0,
            "pay_amt_sep": 3000.0,
            "pay_amt_aug": 2500.0
        },
        {
            "credit_limit": 200000.0,
            "age": 42,
            "pay_delay_sep": 2,
            "pay_delay_aug": 0,
            "bill_amt_sep": 60000.0,
            "bill_amt_aug": 58000.0,
            "pay_amt_sep": 10000.0,
            "pay_amt_aug": 9000.0
        }
    ])

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    file = io.BytesIO(csv_bytes)

    response = client.post(
        "/predict",
        files={"file": ("test.csv", file, "text/csv")}
    )

    assert response.status_code == 200

    body = response.json()
    # Only check n_rows and results; do not expect drift here
    assert "results" in body
    assert body["n_rows"] == 2
    # Optional: basic validation of result structure
    for r in body["results"]:
        assert "prediction" in r
        assert "probability" in r
        assert "risk_level" in r
        assert "row" in r


def test_predict_endpoint_missing_columns():
    df = pd.DataFrame([
        {
            "credit_limit": 50000.0,
            "age": 35
        }
    ])

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    file = io.BytesIO(csv_bytes)

    response = client.post(
        "/predict",
        files={"file": ("bad.csv", file, "text/csv")}
    )

    assert response.status_code == 400
    assert "Invalid schema" in response.json()["detail"]
