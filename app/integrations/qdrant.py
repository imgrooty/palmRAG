from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.core.logging import logger


class QdrantService:
    def __init__(self) -> None:
        self.url = settings.qdrant_url
        self.api_key = settings.qdrant_api_key
        self.collection_name = settings.qdrant_collection
        self._client: AsyncQdrantClient | None = None

    async def get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            if self.url:
                self._client = AsyncQdrantClient(
                    url=self.url,
                    api_key=self.api_key or None,
                    check_compatibility=False,
                )
            else:
                self._client = AsyncQdrantClient(
                    host="localhost",
                    port=6333,
                    check_compatibility=False,
                )
        return self._client

    async def init_collection(self, vector_size: int = 384) -> None:
        client = await self.get_client()
        try:
            collections = await client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(
                    f"Creating Qdrant collection '{self.collection_name}'..."
                )
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(
                    f"Qdrant collection '{self.collection_name}' created."
                )
            else:
                logger.info(
                    f"Qdrant collection '{self.collection_name}' already exists."
                )
        except Exception as e:
            logger.warning(
                f"Qdrant init warning (may require active Qdrant container): {e}"
            )

    async def upsert_chunks(
        self,
        points: list[dict[str, Any]] | None = None,
        chunk_ids: list[Any] | None = None,
        vectors: list[list[float]] | None = None,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        client = await self.get_client()
        try:
            if points is None:
                points = []
                if chunk_ids and vectors and payloads:
                    for cid, vec, pl in zip(
                        chunk_ids, vectors, payloads, strict=True
                    ):
                        points.append(
                            {
                                "id": str(cid),
                                "vector": vec,
                                "payload": pl,
                            }
                        )

            qdrant_points = [
                models.PointStruct(
                    id=str(pt["id"]),
                    vector=pt["vector"],
                    payload=pt["payload"],
                )
                for pt in points
            ]
            await client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
            )
            logger.info(
                f"Successfully upserted {len(qdrant_points)} points to Qdrant."
            )
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant: {e}")
            raise

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        client = await self.get_client()
        try:
            if hasattr(client, "query_points"):
                res = await client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
                hits = res.points
            else:
                hits = await client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
            return [
                {
                    "id": str(hit.id),
                    "score": hit.score,
                    "payload": hit.payload or {},
                }
                for hit in hits
            ]
        except Exception as e:
            logger.error(f"Failed to search Qdrant collection: {e}")
            raise

    async def health_check(self) -> bool:
        try:
            client = await self.get_client()
            await client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    is_healthy = health_check


qdrant_service = QdrantService()
