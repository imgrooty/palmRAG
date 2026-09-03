from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient
import pytest

from app.db.mongo import get_db
from app.main import app


@pytest.mark.asyncio
async def test_api_ingest_multiple_files(async_session):
    app.dependency_overrides[get_db] = lambda: async_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = [
            ("files", ("test1.txt", b"Content for document 1", "text/plain")),
            ("files", ("test2.txt", b"Content for document 2", "text/plain")),
        ]
        data = {"chunking_strategy": "recursive"}

        with (
            patch(
                "app.services.ingestion.qdrant_service.upsert_chunks",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.ingestion.embedding_service.embed_texts",
                new_callable=AsyncMock,
                return_value=[[0.1] * 384],
            ),
        ):
            response = await client.post(
                "/api/v1/documents/ingest",
                files=files,
                data=data,
            )

            assert response.status_code == 200
            json_data = response.json()
            assert json_data["total"] == 2
            assert json_data["successful"] == 2
            assert json_data["failed"] == 0
            assert len(json_data["documents"]) == 2
            assert json_data["documents"][0]["filename"] == "test1.txt"
            assert json_data["documents"][1]["filename"] == "test2.txt"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_ingest_single_file(async_session):
    app.dependency_overrides[get_db] = lambda: async_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = [("files", ("single.txt", b"Single document content", "text/plain"))]
        data = {"chunking_strategy": "fixed"}

        with (
            patch(
                "app.services.ingestion.qdrant_service.upsert_chunks",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.ingestion.embedding_service.embed_texts",
                new_callable=AsyncMock,
                return_value=[[0.1] * 384],
            ),
        ):
            response = await client.post(
                "/api/v1/documents/ingest",
                files=files,
                data=data,
            )

            assert response.status_code == 200
            json_data = response.json()
            assert json_data["total"] == 1
            assert json_data["successful"] == 1
            assert json_data["documents"][0]["filename"] == "single.txt"

    app.dependency_overrides.clear()
