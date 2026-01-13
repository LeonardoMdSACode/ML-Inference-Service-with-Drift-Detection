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

    def predict(self, df):
        X = df[self.features]
        probas = self.model.predict_proba(X)[:, 1]
        preds = (probas >= 0.5).astype(int)
        return preds.tolist(), probas.tolist()
