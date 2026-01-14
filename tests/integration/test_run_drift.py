import os
from pathlib import Path
import pandas as pd

from app.monitoring.drift import run_drift_check

def test_run_drift_check_outputs_metrics():
    repo_root = Path(__file__).resolve().parents[2]

    current_path = repo_root / "data" / "processed" / "current_data.csv"
    reference_path = repo_root / "models" / "v1" / "reference_data.csv"

    assert current_path.exists()
    assert reference_path.exists()

    current_df = pd.read_csv(current_path)
    reference_df = pd.read_csv(reference_path)

    report = run_drift_check(
        current_df,
        reference_df,
        model_version="v1"
    )

    assert report is not None
