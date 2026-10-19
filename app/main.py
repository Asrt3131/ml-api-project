"""
ML Prediction API — Production-Grade FastAPI Service
====================================================

Serves Iris classification predictions from a pickled scikit-learn model.

Author: A_srt343
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.model import (
    FEATURE_NAMES,
    IRIS_CLASS_NAMES,
    model_service,
)
from app.schemas import (
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthOutput,
    ModelInfoOutput,
    PredictionInput,
    PredictionOutput,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ml-api")


# ---------------------------------------------------------------------------
# Middleware — تعریف قبل از create_app
# ---------------------------------------------------------------------------
class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID and log latency for every request."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "req_id=%s method=%s path=%s latency=%.1fms EXCEPTION",
                req_id, request.method, request.url.path, latency_ms,
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"
        logger.info(
            "req_id=%s method=%s path=%s status=%d latency=%.1fms",
            req_id, request.method, request.url.path,
            response.status_code, latency_ms,
        )
        return response


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting up [env=%s]", settings.environment)
    try:
        model_service.load()
    except FileNotFoundError as exc:
        logger.error("Startup warning: %s — /health will report model_loaded=False", exc)
    yield
    logger.info("Shutting down...")
    model_service.unload()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error_envelope(request: Request, code: int, message: str) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "path": request.url.path,
            "request_id": request.headers.get("X-Request-ID"),
        }
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Iris classification service with FastAPI + Docker.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(HTTPException)
    async def http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_envelope(request, code=exc.status_code, message=str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def unhandled_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content=_error_envelope(request, code=500, message="Internal server error"),
        )

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, Any]:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "online",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthOutput, tags=["meta"])
    async def health() -> HealthOutput:
        return HealthOutput(status="healthy", model_loaded=model_service.is_loaded)

    @app.get("/ready", tags=["meta"])
    async def ready() -> dict[str, str]:
        if not model_service.is_loaded:
            raise HTTPException(503, "Model not ready")
        return {"status": "ready"}

    @app.get("/model/info", response_model=ModelInfoOutput, tags=["meta"])
    async def model_info() -> ModelInfoOutput:
        bundle = model_service.bundle
        return ModelInfoOutput(
            model_type=type(bundle.model).__name__,
            path=str(bundle.path),
            loaded_at_utc=bundle.loaded_at_utc,
            sklearn_version=bundle.sklearn_version,
            n_features=settings.expected_feature_count,
            feature_names=FEATURE_NAMES,
            classes=list(IRIS_CLASS_NAMES.values()),
        )

    @app.post("/predict", response_model=PredictionOutput, tags=["inference"])
    async def predict(payload: PredictionInput) -> PredictionOutput:
        pred, conf = model_service.predict(payload.features)
        return PredictionOutput(
            prediction=pred,
            class_name=IRIS_CLASS_NAMES.get(pred, f"class_{pred}"),
            confidence=round(conf, 4),
        )

    @app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["inference"])
    async def predict_batch(payload: BatchPredictionInput) -> BatchPredictionOutput:
        results = model_service.predict_batch([s.features for s in payload.samples])
        return BatchPredictionOutput(
            count=len(results),
            predictions=[
                PredictionOutput(
                    prediction=p,
                    class_name=IRIS_CLASS_NAMES.get(p, f"class_{p}"),
                    confidence=round(c, 4),
                )
                for p, c in results
            ],
        )

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )