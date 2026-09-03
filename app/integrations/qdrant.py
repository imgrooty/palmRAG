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

    async def init_collection(self, vector_size: int) -> None:
        client = await self.get_client()
        try:
            collections = await client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection '{self.collection_name}'...")
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Qdrant collection '{self.collection_name}' created.")
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
        """Upsert a batch of vector points into the collection.

        Accepts either a pre-built ``points`` list or the three parallel
        sequences ``chunk_ids``, ``vectors``, ``payloads``.  Builds
        PointStruct objects in a single pass to minimise peak memory.
        """
        client = await self.get_client()
        try:
            if points is not None:
                qdrant_points = [
                    models.PointStruct(
                        id=str(pt["id"]),
                        vector=pt["vector"],
                        payload=pt["payload"],
                    )
                    for pt in points
                ]
            elif chunk_ids and vectors and payloads:
                # Single-pass construction — no intermediate ``points`` list.
                qdrant_points = [
                    models.PointStruct(
                        id=str(cid),
                        vector=vec,
                        payload=pl,
                    )
                    for cid, vec, pl in zip(chunk_ids, vectors, payloads, strict=True)
                ]
            else:
                qdrant_points = []

            if not qdrant_points:
                return

            await client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
            )
            logger.info(
                "Successfully upserted %d points to Qdrant.", len(qdrant_points)
            )
        except Exception as e:
            logger.error("Failed to upsert points to Qdrant: %s", e)
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
            logger.warning("Qdrant health check failed: %s", e)
            return False

    is_healthy = health_check

    async def close(self) -> None:
        """Close the underlying async Qdrant client and release connections."""
        if self._client is not None:
            await self._client.close()
            self._client = None


qdrant_service = QdrantService()
