from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class DocumentIngestResponse(BaseModel):
    document_id: UUID
    filename: str
    file_type: str
    chunking_strategy: ChunkingStrategy
    chunks_created: int
    status: str

    model_config = ConfigDict(from_attributes=True)


class DocumentStatusResponse(BaseModel):
    document_id: UUID
    filename: str
    file_type: str
    chunking_strategy: ChunkingStrategy
    chunks_created: int = Field(alias="chunk_count")
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
