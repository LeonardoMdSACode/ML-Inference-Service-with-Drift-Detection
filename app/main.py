# FastAPI entrypoint

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.core.logging import init_db

app = FastAPI(title="ML Inference Service")

# Init DB
init_db()

# Mount static FIRST
app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)

# Mount reports
app.mount(
    "/reports",
    StaticFiles(directory="reports"),
    name="reports"
)

# Include API routes
app.include_router(router)

