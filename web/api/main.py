from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.api.config import settings
from web.api.database import init_db
from web.api.routers import auth, devices, files, models, runs


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.results_dir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="nitid API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(models.router, tags=["models"])
app.include_router(runs.router, tags=["runs"])
app.include_router(files.router, tags=["files"])
app.include_router(devices.router, tags=["devices"])
