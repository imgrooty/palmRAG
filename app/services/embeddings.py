"""Embedding service with Gemini API primary and fastembed (ONNX) fallback.

Design notes
------------
* At startup, ``warm_up()`` probes the Gemini API.  If the key is set and the
  API responds, the service uses Gemini for all embeddings (768-dim).
  Otherwise it falls back to local fastembed / ONNX (384-dim).
* The chosen backend is fixed for the lifetime of the process — mixing
  dimensions in a single Qdrant collection is invalid.
* ``vector_size`` exposes the active dimension so Qdrant ``init_collection``
  can use the correct value.
* Public interface is unchanged: ``embed_texts()``, ``embed_text()``,
  ``warm_up()``.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.core.config import settings
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Backend: Gemini API
# ---------------------------------------------------------------------------

GEMINI_VECTOR_SIZE = 768


class _GeminiBackend:
    """Calls Google Gemini ``embed_content`` via the ``google-genai`` SDK."""

    def __init__(self) -> None:
        from google import genai  # noqa: PLC0415

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch using Gemini API (async-safe via run_in_executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_embed, texts)

    def _sync_embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types, errors  # noqa: PLC0415
        import time

        all_embeddings = []
        # Gemini free tier sometimes limits 100 requests per minute
        # We use a batch size of 90 to stay slightly under the limit
        batch_size = 90
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Retry loop for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self._client.models.embed_content(
                        model=self._model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            output_dimensionality=GEMINI_VECTOR_SIZE,
                        ),
                    )
                    all_embeddings.extend([emb.values for emb in response.embeddings])
                    
                    # Add a small delay between batches to smooth out rate limits
                    if i + batch_size < len(texts):
                        time.sleep(2.0)
                        
                    break # Success, break out of retry loop
                except errors.APIError as e:
                    if e.code == 429 and attempt < max_retries - 1:
                        logger.warning(f"Gemini API rate limit hit. Waiting 25s before retry (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(25.0)
                    else:
                        raise e
            
        return all_embeddings

    @property
    def vector_size(self) -> int:
        return GEMINI_VECTOR_SIZE

    @property
    def name(self) -> str:
        return f"Gemini ({self._model}, {self.vector_size}-dim)"


# ---------------------------------------------------------------------------
# Backend: fastembed (ONNX Runtime, local)
# ---------------------------------------------------------------------------

FASTEMBED_VECTOR_SIZE = 384

# Single-worker executor: ONNX already parallelises internally via OpenMP.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
_fastembed_model: Any = None


def _get_fastembed_model() -> Any:
    """Return the singleton TextEmbedding model, loading it on first call."""
    global _fastembed_model  # noqa: PLW0603
    if _fastembed_model is None:
        from fastembed import TextEmbedding  # noqa: PLC0415

        logger.info("Loading fastembed model '%s'…", settings.embedding_model)
        _fastembed_model = TextEmbedding(
            model_name=f"sentence-transformers/{settings.embedding_model}",
            max_length=512,
            threads=1,
        )
        logger.info("fastembed model loaded successfully.")
    return _fastembed_model


def _sync_fastembed(texts: list[str]) -> list[list[float]]:
    """Run fastembed inference synchronously in the thread-pool thread."""
    model = _get_fastembed_model()
    return [vec.tolist() for vec in model.embed(texts)]


class _FastembedBackend:
    """Local ONNX Runtime embeddings via fastembed."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _sync_fastembed, texts)

    @property
    def vector_size(self) -> int:
        return FASTEMBED_VECTOR_SIZE

    @property
    def name(self) -> str:
        return f"fastembed ({settings.embedding_model}, {self.vector_size}-dim)"


# ---------------------------------------------------------------------------
# Unified service
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Async-safe embedding service with Gemini primary / fastembed fallback.

    The active backend is determined once during ``warm_up()`` and stays fixed.
    """

    def __init__(self) -> None:
        self._backend: _GeminiBackend | _FastembedBackend | None = None

    @property
    def vector_size(self) -> int:
        """Active embedding dimension (768 for Gemini, 384 for fastembed)."""
        if self._backend is None:
            # Before warm_up, return Gemini size if key is configured,
            # fastembed size otherwise — so Qdrant init gets a reasonable default.
            if settings.gemini_api_key:
                return GEMINI_VECTOR_SIZE
            return FASTEMBED_VECTOR_SIZE
        return self._backend.vector_size

    @property
    def backend_name(self) -> str:
        if self._backend is None:
            return "not initialised"
        return self._backend.name

    async def _init_backend(self) -> None:
        """Probe Gemini, fall back to fastembed on failure."""
        if settings.gemini_api_key:
            try:
                backend = _GeminiBackend()
                # Probe with a tiny request to verify the key works.
                await backend.embed_texts(["warm-up"])
                self._backend = backend
                logger.info("Embedding backend: %s (primary)", self._backend.name)
                return
            except Exception as exc:
                logger.warning(
                    "Gemini embedding probe failed (%s), falling back to fastembed.",
                    exc,
                )
        else:
            logger.info(
                "GEMINI_API_KEY not set — using fastembed as embedding backend."
            )

        self._backend = _FastembedBackend()
        # Pre-load ONNX model to avoid cold-start on first request.
        await self._backend.embed_texts(["warm-up"])
        logger.info("Embedding backend: %s (fallback)", self._backend.name)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            list of float vectors, one per input string.

        Raises:
            RuntimeError: If the underlying backend call fails.
        """
        if not texts:
            return []
        if self._backend is None:
            await self._init_backend()
        try:
            return await self._backend.embed_texts(texts)  # type: ignore[union-attr]
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


async def warm_up() -> None:
    """Probe the embedding backend during application startup.

    Called from the FastAPI lifespan so the first real request is never
    penalised by cold-start latency.  Also determines Gemini vs fastembed.
    """
    logger.info("Initialising embedding backend…")
    await embedding_service._init_backend()
    logger.info(
        "Embedding backend ready: %s (vector_size=%d)",
        embedding_service.backend_name,
        embedding_service.vector_size,
    )


embedding_service = EmbeddingService()
