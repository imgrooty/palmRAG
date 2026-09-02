import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import ChunkModel
from app.models.document import DocumentModel


class DocumentRepository:
    async def create_document(
        self,
        session: AsyncSession,
        filename: str,
        file_type: str,
        chunking_strategy: str,
        status: str = "processing",
    ) -> DocumentModel:
        doc = DocumentModel(
            id=uuid.uuid4(),
            filename=filename,
            file_type=file_type,
            chunking_strategy=chunking_strategy,
            status=status,
            chunk_count=0,
        )
        session.add(doc)
        await session.flush()
        return doc

    async def update_document_status(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
        status: str,
        chunk_count: int | None = None,
    ) -> DocumentModel | None:
        result = await session.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if doc is not None:
            doc.status = status
            if chunk_count is not None:
                doc.chunk_count = chunk_count
            await session.flush()
        return doc

    async def get_document_by_id(
        self, session: AsyncSession, document_id: uuid.UUID
    ) -> DocumentModel | None:
        result = await session.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        return result.scalar_one_or_none()

    async def create_chunks(
        self,
        session: AsyncSession,
        document_id: uuid.UUID,
        chunks_data: list[dict[str, Any]],
    ) -> list[ChunkModel]:
        chunk_models = []
        for data in chunks_data:
            chunk = ChunkModel(
                id=data.get("id") or uuid.uuid4(),
                document_id=document_id,
                chunk_index=data["chunk_index"],
                content=data["content"],
                page_number=data.get("page_number"),
            )
            chunk_models.append(chunk)

        session.add_all(chunk_models)
        await session.flush()
        return chunk_models

    async def get_chunks_by_document_id(
        self, session: AsyncSession, document_id: uuid.UUID
    ) -> list[ChunkModel]:
        result = await session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.chunk_index)
        )
        return list(result.scalars().all())


document_repository = DocumentRepository()
