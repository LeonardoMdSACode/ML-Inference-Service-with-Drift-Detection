# offline training
import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/processed/credit_default_clean.csv"
MODEL_DIR = "models/v1"

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")


# -----------------------------
# Columns
# -----------------------------
FEATURE_COLUMNS = [
    "credit_limit",
    "age",
    "pay_delay_sep",
    "pay_delay_aug",
    "bill_amt_sep",
    "bill_amt_aug",
    "pay_amt_sep",
    "pay_amt_aug",
]

TARGET_COLUMN = "target"


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_proba)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation ROC-AUC: {roc:.4f}")

    # Persist artifacts
    joblib.dump(model, MODEL_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print("Model and features saved successfully.")


if __name__ == "__main__":
    main()
