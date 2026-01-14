# tests/test_schemas.py

import pytest
from pydantic import ValidationError
from app.api.schemas import PredictionRequest, PredictionResponse


def test_prediction_request_valid():
    payload = {
        "credit_limit": 50000.0,
        "age": 35,
        "pay_delay_sep": 0,
        "pay_delay_aug": -1,
        "bill_amt_sep": 12000.0,
        "bill_amt_aug": 11000.0,
        "pay_amt_sep": 3000.0,
        "pay_amt_aug": 2500.0
    }

    req = PredictionRequest(**payload)
    assert req.credit_limit == 50000.0
    assert req.age == 35


def test_prediction_request_missing_field():
    payload = {
        "credit_limit": 50000.0,
        "age": 35
    }

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_prediction_request_invalid_type():
    payload = {
        "credit_limit": "not-a-number",
        "age": "thirty",
        "pay_delay_sep": 0,
        "pay_delay_aug": 0,
        "bill_amt_sep": 1000.0,
        "bill_amt_aug": 1000.0,
        "pay_amt_sep": 100.0,
        "pay_amt_aug": 100.0
    }

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_prediction_response_valid():
    payload = {
        "prediction": 1,
        "probability": 0.82
    }

    resp = PredictionResponse(**payload)
    assert resp.prediction in (0, 1)
    assert 0.0 <= resp.probability <= 1.0
