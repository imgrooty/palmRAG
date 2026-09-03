
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.mongo import get_db
from app.schemas.document import (
    ChunkingStrategy,
    DocumentBatchIngestResponse,
)
from app.services.ingestion import ingestion_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/ingest",
    response_model=DocumentBatchIngestResponse,
    status_code=status.HTTP_200_OK,
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files", "chunking_strategy"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "One or more files to ingest (PDF, TXT, or MP4)",
                            },
                            "chunking_strategy": {
                                "type": "string",
                                "enum": ["fixed", "recursive", "semantic"],
                            },
                        },
                    }
                }
            }
        }
    },
)
async def ingest_document(
    background_tasks: BackgroundTasks,
    files: list[UploadFile],
    chunking_strategy: ChunkingStrategy = Form(...),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> DocumentBatchIngestResponse:
    # Filter out any entries that lack a real filename (e.g. empty multipart parts)
    upload_files = [f for f in files if f.filename]

    if not upload_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid file(s) provided for ingestion.",
        )

    batch_response = await ingestion_service.ingest_batch_documents(
        background_tasks=background_tasks,
        files=upload_files,
        chunking_strategy=chunking_strategy,
        db_session=db,
    )

    # For single-file requests that failed validation, raise expected HTTP error status
    if len(upload_files) == 1 and batch_response.failed == 1:
        err = batch_response.documents[0].error_message or "Ingestion failed"
        if "exceeds limit" in err.lower():
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=err,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err,
        )

    return batch_response


