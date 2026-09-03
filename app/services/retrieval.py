from dataclasses import dataclass

from app.integrations.qdrant import qdrant_service
from app.services.embeddings import embedding_service


@dataclass
class RetrievedChunk:
    content: str
    filename: str
    page_number: int | None = None
    score: float = 0.0


class RetrievalService:
    async def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        # 1. Embed query (async — dispatched to thread pool)
        query_vector = await embedding_service.embed_text(query)

        # 2. Search Qdrant
        scored_points = await qdrant_service.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # 3. Format results
        retrieved: list[RetrievedChunk] = []
        for point in scored_points:
            if isinstance(point, dict):
                payload = point.get("payload") or {}
                score = point.get("score", 0.0)
            else:
                payload = point.payload or {}
                score = point.score

            retrieved.append(
                RetrievedChunk(
                    content=payload.get("content", ""),
                    filename=payload.get("filename", "unknown"),
                    page_number=payload.get("page_number"),
                    score=score,
                )
            )

        return retrieved


retrieval_service = RetrievalService()
