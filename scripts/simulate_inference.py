# scripts/simulate_inference.py
import pandas as pd
import requests
import random
import time

df = pd.read_csv("data/processed/current_data.csv")

# Sample 1-5 rows randomly
sample = df.sample(random.randint(1,5))
csv_bytes = sample.to_csv(index=False).encode("utf-8")

# POST to FastAPI predict endpoint
response = requests.post(
    "http://localhost:8000/predict",
    files={"file": ("sample.csv", csv_bytes, "text/csv")}
)

print("Status:", response.status_code)
try:
    print("Response:", response.json())
except Exception:
    print("Server returned non-JSON response:")
    print(response.text)

