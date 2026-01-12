# model.predict wrapper

import json
import joblib
import numpy as np
from app.core.config import MODEL_PATH, FEATURES_PATH


class Predictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, "r") as f:
            self.features = json.load(f)

    def predict(self, payload: dict):
        X = np.array([[payload[f] for f in self.features]])
        proba = self.model.predict_proba(X)[0, 1]
        pred = int(proba >= 0.5)
        return pred, float(proba)
