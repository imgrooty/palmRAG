"""MongoDB client, session dependency, and collection initialisation.

Replaces the SQLAlchemy engine + async_session_maker stack.  Motor is an
async-native driver — no ORM layer, no connection-pool configuration
needed beyond the URI.

Collections
-----------
* documents  — one document per uploaded file
* chunks     — one document per text chunk (for audit/debug; Qdrant holds the vectors)
* bookings   — confirmed + in-progress interview bookings
"""

from collections.abc import AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.core.config import settings
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_motor_client: AsyncIOMotorClient | None = None


def get_motor_client() -> AsyncIOMotorClient:
    """Return the process-wide Motor client, creating it on first call."""
    global _motor_client  # noqa: PLW0603
    if _motor_client is None:
        _motor_client = AsyncIOMotorClient(
            settings.mongodb_url,
            uuidRepresentation="standard",
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return _motor_client


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """Yield the Motor database for the duration of one request.

    Motor connections are inherently connection-pooled at the client level;
    no explicit session management or commit/rollback is needed.
    """
    yield get_motor_client()[settings.mongodb_db_name]


# ---------------------------------------------------------------------------
# Startup initialisation
# ---------------------------------------------------------------------------


async def init_collections() -> None:
    """Create collections and indexes on startup (idempotent).

    Motor creates the collection implicitly on first write; this function
    only ensures the indexes exist so queries are fast from day one.
    """
    db: AsyncIOMotorDatabase = get_motor_client()[settings.mongodb_db_name]

    # --- documents ---
    await db["documents"].create_indexes(
        [
            IndexModel([("status", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
        ]
    )

    # --- chunks ---
    await db["chunks"].create_indexes(
        [
            IndexModel([("document_id", ASCENDING)]),
            IndexModel([("document_id", ASCENDING), ("chunk_index", ASCENDING)]),
        ]
    )

    # --- bookings ---
    await db["bookings"].create_indexes(
        [
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
        ]
    )

    logger.info("MongoDB collections and indexes initialised.")
