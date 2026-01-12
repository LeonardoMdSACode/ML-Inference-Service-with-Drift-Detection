import sys
import os
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.monitoring.drift import run_drift_check
from app.monitoring.governance import run_governance_checks

def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Load current and reference data
    current_path = os.path.join(root_dir, "data", "processed", "current_data.csv")
    reference_path = os.path.join(root_dir, "models", "v1", "reference_data.csv")

    if not os.path.exists(current_path):
        raise FileNotFoundError(f"{current_path} does not exist.")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"{reference_path} does not exist.")

    current_df = pd.read_csv(current_path)
    reference_df = pd.read_csv(reference_path)

    # Run drift check
    report = run_drift_check(current_df, reference_df, model_version="v1")

    # Run drift check
    print("Metrics from Evidently report:", report)



if __name__ == "__main__":
    main()
