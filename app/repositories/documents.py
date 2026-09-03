"""Document and chunk repository — Motor (MongoDB) implementation.

Public method signatures are identical to the old SQLAlchemy version so
services and routes require no changes.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


class DocumentRepository:
    async def create_document(
        self,
        db: AsyncIOMotorDatabase,
        filename: str,
        file_type: str,
        chunking_strategy: str,
        status: str = "processing",
    ) -> dict[str, Any]:
        """Insert a new document record and return the inserted document dict.

        Args:
            db: Motor database instance.
            filename: Original upload filename.
            file_type: 'pdf', 'txt', or 'video'.
            chunking_strategy: 'fixed' or 'recursive'.
            status: Initial status, defaults to 'processing'.

        Returns:
            The inserted document as a dict (includes '_id').
        """
        doc = {
            "_id": uuid.uuid4(),
            "filename": filename,
            "file_type": file_type,
            "chunking_strategy": chunking_strategy,
            "status": status,
            "chunk_count": 0,
            "created_at": datetime.now(tz=timezone.utc),
        }
        await db["documents"].insert_one(doc)
        return doc

    async def update_document_status(
        self,
        db: AsyncIOMotorDatabase,
        document_id: uuid.UUID,
        status: str,
        chunk_count: int | None = None,
    ) -> dict[str, Any] | None:
        """Update the status (and optionally chunk_count) of a document.

        Args:
            db: Motor database instance.
            document_id: Document _id.
            status: New status string.
            chunk_count: If provided, also updates chunk_count.

        Returns:
            Updated document dict, or None if not found.
        """
        update: dict[str, Any] = {"status": status}
        if chunk_count is not None:
            update["chunk_count"] = chunk_count

        result = await db["documents"].find_one_and_update(
            {"_id": document_id},
            {"$set": update},
            return_document=True,
        )
        return result

    async def get_document_by_id(
        self, db: AsyncIOMotorDatabase, document_id: uuid.UUID
    ) -> dict[str, Any] | None:
        """Fetch a single document by its _id.

        Args:
            db: Motor database instance.
            document_id: Document _id.

        Returns:
            Document dict or None.
        """
        return await db["documents"].find_one({"_id": document_id})

    async def create_chunks(
        self,
        db: AsyncIOMotorDatabase,
        document_id: uuid.UUID,
        chunks_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Bulk-insert chunk records for a document.

        Args:
            db: Motor database instance.
            document_id: Parent document _id.
            chunks_data: List of dicts with keys: id, chunk_index, content, page_number.

        Returns:
            List of inserted chunk dicts.
        """
        if not chunks_data:
            return []

        now = datetime.now(tz=timezone.utc)
        docs = [
            {
                "_id": d.get("id") or uuid.uuid4(),
                "document_id": document_id,
                "chunk_index": d["chunk_index"],
                "content": d["content"],
                "page_number": d.get("page_number"),
                "created_at": now,
            }
            for d in chunks_data
        ]
        await db["chunks"].insert_many(docs, ordered=False)
        return docs

    async def get_chunks_by_document_id(
        self, db: AsyncIOMotorDatabase, document_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        """Retrieve all chunks for a document, ordered by chunk_index.

        Args:
            db: Motor database instance.
            document_id: Parent document _id.

        Returns:
            List of chunk dicts.
        """
        cursor = db["chunks"].find(
            {"document_id": document_id},
            sort=[("chunk_index", 1)],
        )
        return await cursor.to_list(length=None)


document_repository = DocumentRepository()
