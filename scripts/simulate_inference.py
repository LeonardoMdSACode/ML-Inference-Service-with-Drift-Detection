# scripts/simulate_inference.py
import pandas as pd
import requests
import random
import time
import sys

API_URL = "http://localhost:8000/predict"
SOURCE_DATA = "data/processed/current_data.csv"

# Traffic behavior (tune freely)
MIN_SLEEP = 2      # seconds
MAX_SLEEP = 8      # seconds
MIN_BATCH = 1
MAX_BATCH = 5

print("Starting inference traffic daemon...")
print(f"Target API: {API_URL}")
print(f"Source data: {SOURCE_DATA}")
print("Press Ctrl+C to stop.\n")

# Load once (realistic: upstream feature store snapshot)
try:
    df = pd.read_csv(SOURCE_DATA)
except Exception as e:
    print("Failed to load source data:", e)
    sys.exit(1)

required_cols = set(df.columns)

while True:
    try:
        # ---- Random batch size ----
        batch_size = random.randint(MIN_BATCH, MAX_BATCH)
        sample = df.sample(batch_size)

        # ---- Serialize to CSV ----
        csv_bytes = sample.to_csv(index=False).encode("utf-8")

        # ---- Send request ----
        response = requests.post(
            API_URL,
            files={"file": ("sample.csv", csv_bytes, "text/csv")},
            timeout=10,
        )

        if response.status_code == 200:
            payload = response.json()
            print(
                f"[OK] rows={payload['n_rows']} "
                f"predictions_logged=True"
            )
        else:
            print(
                f"[WARN] HTTP {response.status_code} "
                f"{response.text}"
            )

    except KeyboardInterrupt:
        print("\nTraffic daemon stopped by user.")
        break

    except Exception as e:
        print("[ERROR] Inference request failed:", e)

    # ---- Sleep (non-uniform traffic) ----
    sleep_time = random.uniform(MIN_SLEEP, MAX_SLEEP)
    time.sleep(sleep_time)
