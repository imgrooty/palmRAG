from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    # In-memory SQLite for testing DB models and repositories
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def mock_groq_service():
    mock = MagicMock()
    mock.rewrite_query = AsyncMock(side_effect=lambda q, h: q)
    mock.generate_rag_answer = AsyncMock(
        return_value="This is a test answer based on retrieved context."
    )
    mock.extract_booking_info = AsyncMock()
    mock.transcribe_audio = AsyncMock(
        return_value="This is a test transcript from video audio."
    )
    return mock


@pytest.fixture
def mock_qdrant_service():
    mock = MagicMock()
    mock.upsert_chunks = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.is_healthy = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_redis_service():
    mock = MagicMock()
    mock.get_history = AsyncMock(return_value=[])
    mock.append_turn = AsyncMock()
    mock.get_partial_booking = AsyncMock(return_value={})
    mock.save_partial_booking = AsyncMock()
    mock.clear_partial_booking = AsyncMock()
    mock.is_healthy = AsyncMock(return_value=True)
    return mock
