import os
from pathlib import Path
import tempfile
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.db.database import get_db
from app.repositories.documents import document_repository
from app.schemas.document import (
    ChunkingStrategy,
    DocumentIngestResponse,
    DocumentStatusResponse,
)
from app.services.ingestion import ingestion_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/ingest", response_model=DocumentIngestResponse, status_code=status.HTTP_200_OK
)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentIngestResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    # Read content bytes to validate size
    content_bytes = await file.read()
    file_size = len(content_bytes)

    try:
        file_type = ingestion_service.validate_file(file.filename, file_size)
    except ValueError as ve:
        err_msg = str(ve)
        if "limit" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=err_msg,
            ) from ve
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        ) from ve

    if file_type == "video":
        # Video ingestion runs asynchronously via FastAPI BackgroundTasks
        # 1. Create DB record with status "processing"
        doc = await document_repository.create_document(
            db,
            filename=file.filename,
            file_type=file_type,
            chunking_strategy=chunking_strategy.value,
            status="processing",
        )

        # 2. Save video content to temp file for background worker
        ext = Path(file.filename).suffix.lower()
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content_bytes)

        # 3. Add background task
        background_tasks.add_task(
            ingestion_service.process_video_background,
            document_id=doc.id,
            filename=file.filename,
            temp_video_path=temp_path,
            chunking_strategy=chunking_strategy,
        )

        return DocumentIngestResponse(
            document_id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            chunking_strategy=chunking_strategy,
            chunks_created=0,
            status="processing",
        )

    # Synchronous ingestion for PDF and TXT files
    try:
        doc_id, chunks_created = await ingestion_service.ingest_sync_document(
            db_session=db,
            filename=file.filename,
            file_type=file_type,
            content_bytes=content_bytes,
            chunking_strategy=chunking_strategy,
        )

        return DocumentIngestResponse(
            document_id=doc_id,
            filename=file.filename,
            file_type=file_type,
            chunking_strategy=chunking_strategy,
            chunks_created=chunks_created,
            status="completed",
        )
    except Exception as e:
        logger.error(f"Ingestion endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {e}",
        ) from e


@router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentStatusResponse:
    doc = await document_repository.get_document_by_id(db, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found.",
        )

    return DocumentStatusResponse.model_validate(doc)
