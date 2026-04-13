"""FastAPI-Entrypoint fuer Orpheus Acoustic Sense."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import router
from api.streaming import router as streaming_router
from config import settings

app = FastAPI(
    title="Orpheus Acoustic Sense",
    description="Beide Ohren in einem System: Humanness + Paralinguistik aus Audio.",
    version=settings.version,
)
app.include_router(router)
app.include_router(streaming_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
