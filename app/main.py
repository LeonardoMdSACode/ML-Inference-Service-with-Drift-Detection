# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import asyncio
from contextlib import asynccontextmanager

from app.api.routes import router
from app.api.dashboard_data import router as dashboard_data_router
from app.core.logging import init_db
from app.api.background_drift import drift_loop
from app.api.traffic_daemon import traffic_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    init_db()

    # Start drift detection loop
    drift_task = asyncio.create_task(drift_loop(interval_seconds=10))

    # Start traffic daemon (delayed internally, HF-safe)
    traffic_task = asyncio.create_task(traffic_loop())

    yield

    # ---- Shutdown ----
    for task in (drift_task, traffic_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="ML Inference Service",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

app.include_router(router)
app.include_router(dashboard_data_router)
