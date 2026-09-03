from contextlib import asynccontextmanager
import sys
from typing import Any

from fastapi import FastAPI, Response, status

if sys.platform != "win32":
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.logging import logger
from app.db.mongo import get_motor_client, init_collections
from app.integrations.qdrant import qdrant_service
from app.integrations.redis import redis_service
from app.services.embeddings import warm_up as embedding_warm_up


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Palm Mind RAG application...")
    try:
        await init_collections()
    except Exception as e:
        logger.warning(f"Database init warning (may require active DB container): {e}")

    # Determine embedding backend (Gemini vs fastembed) before Qdrant init
    # so the collection is created with the correct vector dimension.
    try:
        await embedding_warm_up()
    except Exception as e:
        logger.warning(f"Embedding warm-up warning: {e}")

    from app.services.embeddings import embedding_service  # noqa: PLC0415

    try:
        await qdrant_service.init_collection(vector_size=embedding_service.vector_size)
    except Exception as e:
        logger.warning(
            f"Qdrant init warning (may require active Qdrant container): {e}"
        )

    yield

    logger.info("Shutting down Palm Mind RAG application...")
    await redis_service.close()
    try:
        await qdrant_service.close()
    except Exception as e:
        logger.warning(f"Qdrant close warning: {e}")
    try:
        get_motor_client().close()
    except Exception as e:
        logger.warning(f"MongoDB client close warning: {e}")


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
    mongo_status = "disconnected"
    redis_status = "disconnected"
    qdrant_status = "disconnected"
    is_healthy = True

    # 1. Test MongoDB
    try:
        client = get_motor_client()
        await client[settings.mongodb_db_name].command("ping")
        mongo_status = "connected"
    except Exception as e:
        logger.warning(
            f"MongoDB health check failed [{type(e).__name__}]: {e}",
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
        "mongodb": mongo_status,
        "redis": redis_status,
        "qdrant": qdrant_status,
    }
