import os
from pathlib import Path
import uuid

import pymupdf as fitz  # PyMuPDF
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.db.database import async_session_maker
from app.integrations.qdrant import qdrant_service
from app.repositories.documents import document_repository
from app.schemas.document import ChunkingStrategy
from app.services.chunking import ChunkResult, chunking_service
from app.services.embeddings import embedding_service
from app.services.transcription import transcription_service

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".mp4": "video",
    ".mov": "video",
    ".mkv": "video",
}


class IngestionService:
    def validate_file(self, filename: str, file_size: int) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Allowed types: PDF, TXT, MP4, MOV, MKV."
            )

        if file_size > settings.max_upload_bytes:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds limit of {settings.max_upload_mb} MB."
            )

        return SUPPORTED_EXTENSIONS[ext]

    def extract_text_from_pdf(self, content_bytes: bytes) -> list[tuple[int, str]]:
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        pages: list[tuple[int, str]] = []
        for i, page in enumerate(doc):
            page_text = page.get_text("text") or ""
            pages.append((i + 1, page_text))
        doc.close()
        return pages

    def extract_text_from_txt(self, content_bytes: bytes) -> str:
        try:
            return content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return content_bytes.decode("latin-1", errors="replace")

    async def ingest_sync_document(
        self,
        db_session: AsyncSession,
        filename: str,
        file_type: str,
        content_bytes: bytes,
        chunking_strategy: ChunkingStrategy,
    ) -> tuple[uuid.UUID, int]:
        # 1. Create document record in DB
        doc = await document_repository.create_document(
            db_session,
            filename=filename,
            file_type=file_type,
            chunking_strategy=chunking_strategy.value,
            status="processing",
        )

        try:
            # 2. Extract text
            if file_type == "pdf":
                text_or_pages = self.extract_text_from_pdf(content_bytes)
            elif file_type == "txt":
                text_or_pages = self.extract_text_from_txt(content_bytes)
            else:
                raise ValueError(f"Invalid file type for sync ingestion: {file_type}")

            # 3. Chunk text
            chunk_results: list[ChunkResult] = chunking_service.chunk_document(
                text_or_pages=text_or_pages,
                strategy=chunking_strategy,
            )

            if not chunk_results:
                await document_repository.update_document_status(
                    db_session, doc.id, status="completed", chunk_count=0
                )
                return doc.id, 0

            # 4. Generate chunk IDs and payload
            chunk_ids = [uuid.uuid4() for _ in chunk_results]
            chunks_data = [
                {
                    "id": cid,
                    "chunk_index": cr.chunk_index,
                    "content": cr.content,
                    "page_number": cr.page_number,
                }
                for cid, cr in zip(chunk_ids, chunk_results, strict=True)
            ]

            # 5. Embed chunks
            texts_to_embed = [cr.content for cr in chunk_results]
            vectors = embedding_service.embed_texts(texts_to_embed)

            # 6. Upsert vectors to Qdrant
            payloads = [
                {
                    "document_id": str(doc.id),
                    "chunk_index": cr.chunk_index,
                    "filename": filename,
                    "page_number": cr.page_number,
                    "content": cr.content,
                }
                for cr in chunk_results
            ]
            await qdrant_service.upsert_chunks(
                chunk_ids=chunk_ids,
                vectors=vectors,
                payloads=payloads,
            )

            # 7. Write chunks to PostgreSQL
            await document_repository.create_chunks(
                db_session, document_id=doc.id, chunks_data=chunks_data
            )

            # 8. Mark document as completed
            await document_repository.update_document_status(
                db_session, doc.id, status="completed", chunk_count=len(chunk_results)
            )

            return doc.id, len(chunk_results)

        except Exception as e:
            logger.error(f"Failed sync document ingestion for {filename}: {e}")
            await document_repository.update_document_status(
                db_session, doc.id, status="failed", chunk_count=0
            )
            raise

    async def process_video_background(
        self,
        document_id: uuid.UUID,
        filename: str,
        temp_video_path: str,
        chunking_strategy: ChunkingStrategy,
    ) -> None:
        logger.info(
            f"Starting background video ingestion for document {document_id} ({filename})..."
        )
        try:
            transcript = await transcription_service.transcribe_video(temp_video_path)

            chunk_results: list[ChunkResult] = chunking_service.chunk_document(
                text_or_pages=transcript,
                strategy=chunking_strategy,
            )

            chunk_ids = [uuid.uuid4() for _ in chunk_results]
            chunks_data = [
                {
                    "id": cid,
                    "chunk_index": cr.chunk_index,
                    "content": cr.content,
                    "page_number": None,
                }
                for cid, cr in zip(chunk_ids, chunk_results, strict=True)
            ]

            texts_to_embed = [cr.content for cr in chunk_results]
            vectors = embedding_service.embed_texts(texts_to_embed)

            payloads = [
                {
                    "document_id": str(document_id),
                    "chunk_index": cr.chunk_index,
                    "filename": filename,
                    "page_number": None,
                    "content": cr.content,
                }
                for cr in chunk_results
            ]

            await qdrant_service.upsert_chunks(
                chunk_ids=chunk_ids,
                vectors=vectors,
                payloads=payloads,
            )

            async with async_session_maker() as session:
                await document_repository.create_chunks(
                    session, document_id=document_id, chunks_data=chunks_data
                )
                await document_repository.update_document_status(
                    session,
                    document_id=document_id,
                    status="completed",
                    chunk_count=len(chunk_results),
                )
                await session.commit()

            logger.info(
                f"Successfully finished background video processing for {document_id}"
            )

        except Exception as e:
            logger.error(f"Error in background video processing for {document_id}: {e}")
            async with async_session_maker() as session:
                await document_repository.update_document_status(
                    session, document_id=document_id, status="failed", chunk_count=0
                )
                await session.commit()
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)


ingestion_service = IngestionService()
