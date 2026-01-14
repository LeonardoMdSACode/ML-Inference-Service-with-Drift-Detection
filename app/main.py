# app/main.py (no other changes)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import asyncio
from contextlib import asynccontextmanager

from app.api.routes import router
from app.api.dashboard_data import router as dashboard_data_router
from app.core.logging import init_db
from app.api.background_drift import drift_loop

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    task = asyncio.create_task(drift_loop(interval_seconds=10))
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="ML Inference Service", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

app.include_router(router)
app.include_router(dashboard_data_router)
