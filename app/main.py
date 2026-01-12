# FastAPI entrypoint

from fastapi import FastAPI
from app.api.routes import router
from app.core.logging import init_db
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="ML Inference Service")

init_db()
app.include_router(router)

app.mount(
    "/reports",
    StaticFiles(directory="reports"),
    name="reports"
)
