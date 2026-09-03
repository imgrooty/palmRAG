from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[MagicMock, None]:
    """Mock Motor database for unit tests."""
    mock_db = MagicMock()

    # Default mock responses for document operations
    mock_db["documents"].insert_one = AsyncMock()
    mock_db["documents"].find_one_and_update = AsyncMock(
        return_value={"_id": "test_id", "status": "completed"}
    )
    mock_db["documents"].find_one = AsyncMock(return_value=None)

    # Default mock responses for chunk operations
    mock_db["chunks"].insert_many = AsyncMock()

    # Default mock responses for booking operations
    mock_db["bookings"].insert_one = AsyncMock()

    yield mock_db


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
