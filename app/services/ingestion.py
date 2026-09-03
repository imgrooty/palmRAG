"""Document ingestion service.

PDF extraction uses pypdfium2 (Google PDFium via CFFI) instead of PyMuPDF.
The Motor database is passed in from the route layer; the background video
task obtains its own Motor database reference directly from get_motor_client().
"""

import os
from pathlib import Path
import tempfile
import uuid

from fastapi import BackgroundTasks, UploadFile
import pypdfium2 as pdfium
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings
from app.core.logging import logger
from app.db.mongo import get_motor_client
from app.integrations.qdrant import qdrant_service
from app.repositories.documents import document_repository
from app.schemas.document import (
    ChunkingStrategy,
    DocumentBatchIngestResponse,
    DocumentIngestResponse,
)
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
        """Validate file extension and size.

        Args:
            filename: Original upload filename.
            file_size: File size in bytes (pass 0 for extension-only check).

        Returns:
            Normalised file type string: 'pdf', 'txt', or 'video'.

        Raises:
            ValueError: If the extension is unsupported or file exceeds the size limit.
        """
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Allowed types: PDF, TXT, MP4, MOV, MKV."
            )

        if file_size > 0 and file_size > settings.max_upload_bytes:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds limit of {settings.max_upload_mb} MB."
            )

        return SUPPORTED_EXTENSIONS[ext]

    def _get_ocr_engine(self):
        """Lazy initializer for RapidOCR ONNX model instance."""
        if not hasattr(self, "_ocr_engine"):
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore

                self._ocr_engine = RapidOCR()
            except Exception as e:
                logger.warning(f"Failed to initialize RapidOCR engine: {e}")
                self._ocr_engine = None
        return self._ocr_engine

    def _ocr_page(self, page) -> str:
        """Perform OCR on a single pypdfium2 PDF page object using RapidOCR."""
        ocr_engine = self._get_ocr_engine()
        if ocr_engine is None:
            return ""

        try:
            pil_img = page.render(scale=2.0).to_pil()
            result, _ = ocr_engine(pil_img)
            if result:
                lines = [
                    line[1] for line in result if line and len(line) > 1 and line[1]
                ]
                return "\n".join(lines)
        except Exception as e:
            logger.error(f"RapidOCR extraction failed on page: {e}")
        return ""

    def extract_text_from_pdf(self, content_bytes: bytes) -> list[tuple[int, str]]:
        """Extract per-page text from a PDF byte string using pypdfium2,

        falling back to RapidOCR for scanned image pages with no embedded text.

        Args:
            content_bytes: Raw PDF bytes.

        Returns:
            List of (1-based page number, page text) tuples.
        """
        doc = pdfium.PdfDocument(content_bytes)
        pages: list[tuple[int, str]] = []
        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            text = (textpage.get_text_range() or "").strip()

            # Fallback to RapidOCR if extracted digital text is minimal / empty
            if len(text) < 10:
                ocr_text = self._ocr_page(page)
                if ocr_text.strip():
                    text = ocr_text.strip()

            pages.append((i + 1, text))
        return pages

    def extract_text_from_txt(self, content_bytes: bytes) -> str:
        """Decode raw TXT bytes, falling back to latin-1 on UTF-8 errors.

        Args:
            content_bytes: Raw file bytes.

        Returns:
            Decoded string.
        """
        try:
            return content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return content_bytes.decode("latin-1", errors="replace")

    async def ingest_sync_document(
        self,
        db_session: AsyncIOMotorDatabase,
        filename: str,
        file_type: str,
        content_bytes: bytes,
        chunking_strategy: ChunkingStrategy,
    ) -> tuple[uuid.UUID, int]:
        """Ingest a PDF or TXT document synchronously (called from the request handler).

        Args:
            db_session: Motor database instance.
            filename: Original upload filename.
            file_type: 'pdf' or 'txt'.
            content_bytes: Raw file bytes.
            chunking_strategy: Which chunking algorithm to apply.

        Returns:
            (document_id, chunks_created) tuple.

        Raises:
            ValueError: On unsupported file type.
            Exception: On any downstream failure (status set to 'failed' before re-raise).
        """
        # 1. Create document record in MongoDB
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
                    db_session, doc["_id"], status="completed", chunk_count=0
                )
                return doc["_id"], 0

            # 4. Build IDs + both payload lists in a single pass
            chunk_ids: list[uuid.UUID] = []
            chunks_data: list[dict] = []
            qdrant_payloads: list[dict] = []
            for cr in chunk_results:
                cid = uuid.uuid4()
                chunk_ids.append(cid)
                chunks_data.append(
                    {
                        "id": cid,
                        "chunk_index": cr.chunk_index,
                        "content": cr.content,
                        "page_number": cr.page_number,
                    }
                )
                qdrant_payloads.append(
                    {
                        "document_id": str(doc["_id"]),
                        "chunk_index": cr.chunk_index,
                        "filename": filename,
                        "page_number": cr.page_number,
                        "content": cr.content,
                    }
                )

            # 5. Embed (async — dispatched to ONNX thread pool)
            texts_to_embed = [cr.content for cr in chunk_results]
            vectors = await embedding_service.embed_texts(texts_to_embed)

            # 6. Upsert vectors to Qdrant
            await qdrant_service.upsert_chunks(
                chunk_ids=chunk_ids,
                vectors=vectors,
                payloads=qdrant_payloads,
            )

            # 7. Write chunks to MongoDB
            await document_repository.create_chunks(
                db_session, document_id=doc["_id"], chunks_data=chunks_data
            )

            # 8. Mark document completed
            await document_repository.update_document_status(
                db_session,
                doc["_id"],
                status="completed",
                chunk_count=len(chunk_results),
            )

            return doc["_id"], len(chunk_results)

        except Exception as e:
            logger.error("Failed sync document ingestion for %s: %s", filename, e)
            await document_repository.update_document_status(
                db_session, doc["_id"], status="failed", chunk_count=0
            )
            raise

    async def process_video_background(
        self,
        document_id: uuid.UUID,
        filename: str,
        temp_video_path: str,
        chunking_strategy: ChunkingStrategy,
    ) -> None:
        """Ingest a video document as a background task.

        Obtains its own Motor database reference (not shared with the HTTP
        request that spawned it, which has already completed).

        Args:
            document_id: Pre-created document _id in MongoDB.
            filename: Original upload filename.
            temp_video_path: Absolute path to the temporary video file on disk.
            chunking_strategy: Which chunking algorithm to apply.
        """
        logger.info(
            "Starting background video ingestion for document %s (%s)…",
            document_id,
            filename,
        )
        db: AsyncIOMotorDatabase = get_motor_client()[settings.mongodb_db_name]
        try:
            transcript = await transcription_service.transcribe_video(temp_video_path)

            chunk_results: list[ChunkResult] = chunking_service.chunk_document(
                text_or_pages=transcript,
                strategy=chunking_strategy,
            )

            chunk_ids_bg: list[uuid.UUID] = []
            chunks_data_bg: list[dict] = []
            qdrant_payloads_bg: list[dict] = []
            for cr in chunk_results:
                cid = uuid.uuid4()
                chunk_ids_bg.append(cid)
                chunks_data_bg.append(
                    {
                        "id": cid,
                        "chunk_index": cr.chunk_index,
                        "content": cr.content,
                        "page_number": None,
                    }
                )
                qdrant_payloads_bg.append(
                    {
                        "document_id": str(document_id),
                        "chunk_index": cr.chunk_index,
                        "filename": filename,
                        "page_number": None,
                        "content": cr.content,
                    }
                )

            texts_to_embed_bg = [cr.content for cr in chunk_results]
            vectors_bg = await embedding_service.embed_texts(texts_to_embed_bg)

            await qdrant_service.upsert_chunks(
                chunk_ids=chunk_ids_bg,
                vectors=vectors_bg,
                payloads=qdrant_payloads_bg,
            )

            await document_repository.create_chunks(
                db, document_id=document_id, chunks_data=chunks_data_bg
            )
            await document_repository.update_document_status(
                db,
                document_id=document_id,
                status="completed",
                chunk_count=len(chunk_results),
            )

            logger.info(
                "Successfully finished background video processing for %s", document_id
            )

        except Exception as e:
            logger.error(
                "Error in background video processing for %s: %s", document_id, e
            )
            await document_repository.update_document_status(
                db, document_id=document_id, status="failed", chunk_count=0
            )
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    async def ingest_file_item(
        self,
        background_tasks: BackgroundTasks,
        file: UploadFile,
        chunking_strategy: ChunkingStrategy,
        db_session: AsyncIOMotorDatabase,
    ) -> DocumentIngestResponse:
        """Process an individual uploaded file item for ingestion.

        Args:
            background_tasks: FastAPI background tasks.
            file: The uploaded file.
            chunking_strategy: Chunking strategy enum.
            db_session: Motor database reference.

        Returns:
            DocumentIngestResponse detailing status and chunks or error.
        """
        filename = file.filename or "unknown"
        if not file.filename:
            return DocumentIngestResponse(
                filename=filename,
                file_type="unknown",
                chunking_strategy=chunking_strategy,
                chunks_created=0,
                status="failed",
                error_message="Uploaded file must have a valid filename.",
            )

        if file.size is not None and file.size > settings.max_upload_bytes:
            return DocumentIngestResponse(
                filename=filename,
                file_type="unknown",
                chunking_strategy=chunking_strategy,
                chunks_created=0,
                status="failed",
                error_message=f"File size ({file.size} bytes) exceeds limit of {settings.max_upload_mb} MB.",
            )

        try:
            file_type = self.validate_file(filename, 0)
        except ValueError as ve:
            return DocumentIngestResponse(
                filename=filename,
                file_type="unknown",
                chunking_strategy=chunking_strategy,
                chunks_created=0,
                status="failed",
                error_message=str(ve),
            )

        try:
            content_bytes = await file.read()
            file_size = len(content_bytes)
            file_type = self.validate_file(filename, file_size)
        except ValueError as ve:
            return DocumentIngestResponse(
                filename=filename,
                file_type=file_type,
                chunking_strategy=chunking_strategy,
                chunks_created=0,
                status="failed",
                error_message=str(ve),
            )

        if file_type == "video":
            try:
                doc = await document_repository.create_document(
                    db_session,
                    filename=filename,
                    file_type=file_type,
                    chunking_strategy=chunking_strategy.value,
                    status="processing",
                )
                ext = Path(filename).suffix.lower()
                temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
                with os.fdopen(temp_fd, "wb") as f:
                    f.write(content_bytes)

                background_tasks.add_task(
                    self.process_video_background,
                    document_id=doc["_id"],
                    filename=filename,
                    temp_video_path=temp_path,
                    chunking_strategy=chunking_strategy,
                )
                return DocumentIngestResponse(
                    document_id=doc["_id"],
                    filename=filename,
                    file_type=file_type,
                    chunking_strategy=chunking_strategy,
                    chunks_created=0,
                    status="processing",
                )
            except Exception as e:
                logger.error(f"Failed video ingestion setup for {filename}: {e}")
                return DocumentIngestResponse(
                    filename=filename,
                    file_type=file_type,
                    chunking_strategy=chunking_strategy,
                    chunks_created=0,
                    status="failed",
                    error_message=str(e),
                )

        try:
            doc_id, chunks_created = await self.ingest_sync_document(
                db_session=db_session,
                filename=filename,
                file_type=file_type,
                content_bytes=content_bytes,
                chunking_strategy=chunking_strategy,
            )
            return DocumentIngestResponse(
                document_id=doc_id,
                filename=filename,
                file_type=file_type,
                chunking_strategy=chunking_strategy,
                chunks_created=chunks_created,
                status="completed",
            )
        except Exception as e:
            logger.error(f"Failed sync ingestion for {filename}: {e}")
            return DocumentIngestResponse(
                filename=filename,
                file_type=file_type,
                chunking_strategy=chunking_strategy,
                chunks_created=0,
                status="failed",
                error_message=str(e),
            )

    async def ingest_batch_documents(
        self,
        background_tasks: BackgroundTasks,
        files: list[UploadFile],
        chunking_strategy: ChunkingStrategy,
        db_session: AsyncIOMotorDatabase,
    ) -> DocumentBatchIngestResponse:
        """Ingest a batch of files (PDF, TXT, Video) and return batch summary.

        Args:
            background_tasks: FastAPI background tasks context.
            files: List of uploaded files.
            chunking_strategy: Selected chunking algorithm strategy.
            db_session: Motor database reference.

        Returns:
            DocumentBatchIngestResponse summarizing results per file and overall totals.
        """
        results: list[DocumentIngestResponse] = []
        for file in files:
            res = await self.ingest_file_item(
                background_tasks=background_tasks,
                file=file,
                chunking_strategy=chunking_strategy,
                db_session=db_session,
            )
            results.append(res)

        successful = sum(1 for r in results if r.status in ("completed", "processing"))
        failed = sum(1 for r in results if r.status == "failed")
        return DocumentBatchIngestResponse(
            documents=results,
            total=len(results),
            successful=successful,
            failed=failed,
        )


ingestion_service = IngestionService()
