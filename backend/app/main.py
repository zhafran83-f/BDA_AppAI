"""Local REST API. Run from the repository root with one worker:

backend/e-commerce/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from .api.errors import register_error_handlers
from .api.router import router
from .api.runtime import InferenceRuntime
from .core.model_loader import ModelLoader


def create_app(*, loader_factory=None) -> FastAPI:
    factory = ModelLoader if loader_factory is None else loader_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = InferenceRuntime(factory())
        app.state.runtime = runtime
        try:
            await run_in_threadpool(runtime.warmup)
            yield
        finally:
            runtime.close()
            app.state.runtime = None

    app = FastAPI(
        title="BDA AI Analytics API",
        version="0.1.0",
        description=(
            "Local inference API. Review and Purchase load at startup. "
            "Fashion loads on its first valid request; /ready stays 503 until "
            "all three model bundles are loaded. Cluster IDs are not sentiment labels."
        ),
        lifespan=lifespan,
    )
    app.state.runtime = None
    register_error_handlers(app)
    app.include_router(router)
    return app


# Constructing the app does not load model artifacts.
app = create_app()
