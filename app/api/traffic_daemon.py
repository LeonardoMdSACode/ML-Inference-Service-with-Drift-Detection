# app/api/traffic_daemon.py
import asyncio
import pandas as pd
import random
import os
import httpx

API_URL = "http://localhost:8000/predict"
SOURCE_DATA = "data/processed/current_data.csv"

MIN_SLEEP = 2
MAX_SLEEP = 8
MIN_BATCH = 1
MAX_BATCH = 5
STARTUP_DELAY = 7

async def traffic_loop():
    await asyncio.sleep(STARTUP_DELAY)

    if not os.path.exists(SOURCE_DATA):
        print("Traffic daemon: source data not found, disabled.")
        return

    df = pd.read_csv(SOURCE_DATA)
    print("Traffic daemon started.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                batch_size = random.randint(MIN_BATCH, MAX_BATCH)
                sample = df.sample(batch_size)
                csv_bytes = sample.to_csv(index=False).encode("utf-8")

                resp = await client.post(
                    API_URL,
                    files={"file": ("sample.csv", csv_bytes, "text/csv")}
                )

                if resp.status_code != 200:
                    print("Traffic daemon warning:", resp.status_code)

            except Exception as e:
                print("Traffic daemon error:", e)

            await asyncio.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
