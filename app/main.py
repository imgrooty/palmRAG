from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response, status
from sqlalchemy import text

from app.api.v1.router import api_v1_router
from app.core.logging import logger
from app.db.database import engine, init_db
from app.integrations.qdrant import qdrant_service
from app.integrations.redis import redis_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Palm Mind RAG application...")
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database init warning (may require active DB container): {e}")

    try:
        await qdrant_service.init_collection()
    except Exception as e:
        logger.warning(
            f"Qdrant init warning (may require active Qdrant container): {e}"
        )

    yield

    logger.info("Shutting down Palm Mind RAG application...")
    await redis_service.close()
    await engine.dispose()


app = FastAPI(
    title="Palm Mind RAG Backend API",
    description="Document Q&A RAG + Interview Booking Conversational Agent Service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(api_v1_router)


@app.get("/health", tags=["health"])
async def health_check(response: Response) -> dict[str, Any]:
    postgres_status = "disconnected"
    redis_status = "disconnected"
    qdrant_status = "disconnected"
    is_healthy = True

    # 1. Test Postgres
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            postgres_status = "connected"
    except Exception as e:
        logger.warning(
            f"Postgres health check failed [{type(e).__name__}]: {e}",
            exc_info=True,
        )
        is_healthy = False

    # 2. Test Redis
    try:
        if await redis_service.is_healthy():
            redis_status = "connected"
        else:
            is_healthy = False
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        is_healthy = False

    # 3. Test Qdrant
    try:
        if await qdrant_service.is_healthy():
            qdrant_status = "connected"
        else:
            is_healthy = False
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        is_healthy = False

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if is_healthy else "error",
        "postgres": postgres_status,
        "redis": redis_status,
        "qdrant": qdrant_status,
    }
