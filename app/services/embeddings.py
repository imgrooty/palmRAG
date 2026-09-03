"""Embedding service backed by fastembed (ONNX Runtime).

Design notes
------------
* fastembed uses ONNX Runtime instead of PyTorch — no 2 GB torch install,
  ~80 MB RAM footprint vs ~600 MB for sentence-transformers.
* The same model (all-MiniLM-L6-v2) produces identical 384-dim vectors,
  so existing Qdrant data requires no re-indexing.
* ONNX Runtime releases the GIL during inference, so we still dispatch to a
  ThreadPoolExecutor to keep the asyncio event loop fully free.
* warm_up() is called from the FastAPI lifespan to avoid cold-start latency
  on the first live request.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.core.config import settings
from app.core.logging import logger

# Single-worker executor: ONNX already parallelises internally via OpenMP.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")

_model_instance: Any = None


def _get_model() -> Any:
    """Return the singleton TextEmbedding model, loading it on first call.

    Always called inside the executor thread, never in async context.
    """
    global _model_instance  # noqa: PLW0603
    if _model_instance is None:
        from fastembed import TextEmbedding  # noqa: PLC0415

        logger.info("Loading fastembed model '%s'…", settings.embedding_model)
        _model_instance = TextEmbedding(
            model_name=f"sentence-transformers/{settings.embedding_model}",
            max_length=512,
            threads=1,
        )
        logger.info("fastembed model loaded successfully.")
    return _model_instance


def _sync_embed(texts: list[str]) -> list[list[float]]:
    """Run inference synchronously in the thread-pool thread.

    fastembed.embed() returns a generator of numpy arrays; we materialise
    and convert to list[list[float]] in one pass here so the async caller
    receives a plain Python structure with no numpy dependency at the call site.
    """
    model = _get_model()
    return [vec.tolist() for vec in model.embed(texts)]


class EmbeddingService:
    """Async-safe wrapper around a fastembed TextEmbedding model."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            list of float vectors, one per input string.

        Raises:
            RuntimeError: If the underlying model call fails.
        """
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(_executor, _sync_embed, texts)
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: String to embed.

        Returns:
            A single float vector.
        """
        if not text:
            return []
        results = await self.embed_texts([text])
        return results[0] if results else []

    # Explicit alias kept for call-sites that use the longer name.
    embed_texts_batch = embed_texts


async def warm_up() -> None:
    """Pre-load the ONNX model during application startup.

    Called from the FastAPI lifespan so the first real request is never
    penalised by the model-download / ONNX-compilation cold-start.
    """
    logger.info("Warming up fastembed model…")
    await embedding_service.embed_texts(["warm-up"])
    logger.info("fastembed model warm-up complete.")


embedding_service = EmbeddingService()
