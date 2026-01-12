# Preparing data

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------
# Paths
# -----------------------------
RAW_DATA_PATH = "data/raw/credit_default.csv"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models/v1"

CLEAN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "credit_default_clean.csv")
CURRENT_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "current_data.csv")
REFERENCE_DATA_PATH = os.path.join(MODELS_DIR, "reference_data.csv")


# -----------------------------
# Column mapping
# -----------------------------
COLUMN_RENAME_MAP = {
    "LIMIT_BAL": "credit_limit",
    "AGE": "age",

    "PAY_0": "pay_delay_sep",
    "PAY_2": "pay_delay_aug",
    "PAY_3": "pay_delay_jul",
    "PAY_4": "pay_delay_jun",
    "PAY_5": "pay_delay_may",
    "PAY_6": "pay_delay_apr",

    "BILL_AMT1": "bill_amt_sep",
    "BILL_AMT2": "bill_amt_aug",
    "BILL_AMT3": "bill_amt_jul",
    "BILL_AMT4": "bill_amt_jun",
    "BILL_AMT5": "bill_amt_may",
    "BILL_AMT6": "bill_amt_apr",

    "PAY_AMT1": "pay_amt_sep",
    "PAY_AMT2": "pay_amt_aug",
    "PAY_AMT3": "pay_amt_jul",
    "PAY_AMT4": "pay_amt_jun",
    "PAY_AMT5": "pay_amt_may",
    "PAY_AMT6": "pay_amt_apr",

    "default.payment.next.month": "target"
}


# -----------------------------
# Feature selection (frozen)
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
# Main logic
# -----------------------------
def main():
    # Create directories if missing
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop ID column (not a feature)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Rename columns
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Keep only selected features + target
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[required_columns]

    # Basic sanity checks
    if df.isnull().any().any():
        raise ValueError("Null values detected after preprocessing.")

    # Save fully cleaned dataset
    df.to_csv(CLEAN_DATA_PATH, index=False)

    # Reference / current split (time-simulated, deterministic)
    reference_df, current_df = train_test_split(
        df,
        test_size=0.3,
        shuffle=False
    )

    # Persist splits
    reference_df.to_csv(REFERENCE_DATA_PATH, index=False)
    current_df.to_csv(CURRENT_DATA_PATH, index=False)

    print("Data preparation completed successfully.")
    print(f"Clean data saved to: {CLEAN_DATA_PATH}")
    print(f"Reference data saved to: {REFERENCE_DATA_PATH}")
    print(f"Current data saved to: {CURRENT_DATA_PATH}")


if __name__ == "__main__":
    main()
