from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.document import ChunkingStrategy
from app.services.ingestion import ingestion_service


def test_validate_file_extensions():
    assert ingestion_service.validate_file("resume.pdf", 1000) == "pdf"
    assert ingestion_service.validate_file("notes.txt", 500) == "txt"
    assert ingestion_service.validate_file("intro.mp4", 5000) == "video"
    assert ingestion_service.validate_file("clip.MOV", 5000) == "video"

    with pytest.raises(ValueError, match="Unsupported file type"):
        ingestion_service.validate_file("malicious.exe", 100)


def test_validate_file_size_limit():
    with pytest.raises(ValueError, match="exceeds limit"):
        ingestion_service.validate_file("huge.pdf", 100 * 1024 * 1024)


def test_extract_text_from_txt():
    content = "Sample plain text document content for testing ingestion.".encode(
        "utf-8"
    )
    extracted = ingestion_service.extract_text_from_txt(content)
    assert "Sample plain text" in extracted


@pytest.mark.asyncio
async def test_ingest_sync_document(async_session):
    content = "John Doe is a Senior Backend Engineer with 5 years of Python experience.".encode(
        "utf-8"
    )

    with (
        patch(
            "app.services.ingestion.qdrant_service.upsert_chunks",
            new_callable=AsyncMock,
        ) as mock_qdrant,
        patch(
            "app.services.ingestion.embedding_service.embed_texts",
            new_callable=AsyncMock,
            return_value=[[0.1] * 384],
        ),
    ):
        doc_id, count = await ingestion_service.ingest_sync_document(
            db_session=async_session,
            filename="john_resume.txt",
            file_type="txt",
            content_bytes=content,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
        )

        assert doc_id is not None
        assert count > 0
        mock_qdrant.assert_called_once()
