# app/api/traffic_daemon.py
import asyncio
import pandas as pd
import random
import requests
import os

API_URL = "http://localhost:8000/predict"
SOURCE_DATA = "data/processed/current_data.csv"

MIN_SLEEP = 2
MAX_SLEEP = 8
MIN_BATCH = 1
MAX_BATCH = 5
STARTUP_DELAY = 10  # allow server startup

async def traffic_loop():
    await asyncio.sleep(STARTUP_DELAY)

    if not os.path.exists(SOURCE_DATA):
        print("Traffic daemon: source data not found, disabled.")
        return

    df = pd.read_csv(SOURCE_DATA)
    print("Traffic daemon started.")

    while True:
        try:
            batch_size = random.randint(MIN_BATCH, MAX_BATCH)
            sample = df.sample(batch_size)
            csv_bytes = sample.to_csv(index=False).encode("utf-8")

            # ---- Increased timeout to avoid ReadTimeout ----
            response = requests.post(
                API_URL,
                files={"file": ("sample.csv", csv_bytes, "text/csv")},
                timeout=60,
            )

            if response.status_code != 200:
                print("Traffic daemon warning:", response.status_code)

        except Exception as e:
            print("Traffic daemon error:", e)

        await asyncio.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
