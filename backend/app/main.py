import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.db.session import engine
from app.logging_config import configure_logging
from app.routers import analyze, evaluation

configure_logging(settings.log_level)
logger = logging.getLogger("managerlens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "startup",
        extra={"environment": settings.environment, "gemini_model": settings.gemini_model},
    )
    yield
    logger.info("shutdown")
    engine.dispose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs method/path/status/duration for every request.

    Deliberately never logs the request body — that's where a manager's situation text
    (and, in a future auth phase, tokens) would live. Errors are logged with their type
    and a truncated message, not full user content.
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.exception(
            "request_failed",
            extra={"method": request.method, "path": request.url.path, "duration_ms": duration_ms},
        )
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


app.include_router(analyze.router)
app.include_router(evaluation.router)


@app.get("/health")
def health_check():
    """Liveness check: is the process up? Deliberately does not touch the database or
    any LLM provider — those are checked by /health/ready — so it stays fast and cheap
    for orchestrators that poll it frequently."""
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}


@app.get("/health/ready")
def readiness_check():
    """Readiness check: can this instance actually serve traffic? Verifies the database
    is reachable. Does not call Groq or Gemini (that would cost quota on every poll)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.warning("readiness_check_failed", extra={"error_type": type(exc).__name__})
        return JSONResponse(status_code=503, content={"status": "not_ready", "database": "unreachable"})

    return {"status": "ready", "database": "reachable"}
