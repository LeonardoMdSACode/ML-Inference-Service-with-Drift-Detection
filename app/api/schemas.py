# app\api\schemas.py
# Pydantic input/output schemas

from pydantic import BaseModel
from typing import Dict


class PredictionRequest(BaseModel):
    credit_limit: float
    age: int
    pay_delay_sep: int
    pay_delay_aug: int
    bill_amt_sep: float
    bill_amt_aug: float
    pay_amt_sep: float
    pay_amt_aug: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
