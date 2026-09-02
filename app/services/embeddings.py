import os
import sys
from typing import Any

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Safe patch for transformers BACKENDS_MAPPING in Python 3.14
try:
    import transformers.utils.import_utils as tu

    class SafeBackendsDict(dict):
        def __contains__(self, key):
            return True

        def __getitem__(self, key):
            if dict.__contains__(self, key):
                return dict.__getitem__(self, key)
            return (lambda: False, f"Backend {key} disabled")

    tu.BACKENDS_MAPPING = SafeBackendsDict(tu.BACKENDS_MAPPING)
except Exception:
    pass

from app.core.config import settings
from app.core.logging import logger

_model_instance = None


def get_model() -> Any:
    """Lazy load SentenceTransformer model."""
    global _model_instance
    if _model_instance is None:
        logger.info(
            f"Loading SentenceTransformer model '{settings.embedding_model}'..."
        )

        try:
            import transformers.utils.import_utils as tu

            class SafeBackendsDict(dict):
                def __contains__(self, key):
                    return True

                def __getitem__(self, key):
                    if dict.__contains__(self, key):
                        return dict.__getitem__(self, key)
                    return (lambda: False, f"Backend {key} disabled")

            tu.BACKENDS_MAPPING = SafeBackendsDict(tu.BACKENDS_MAPPING)
        except Exception:
            pass

        from sentence_transformers import SentenceTransformer

        _model_instance = SentenceTransformer(settings.embedding_model)
        logger.info("SentenceTransformer model loaded successfully.")
    return _model_instance


class EmbeddingService:
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def generate_embedding(self, text: str) -> list[float]:
        if not text:
            return []

        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    embed_texts = generate_embeddings
    embed_text = generate_embedding


embedding_service = EmbeddingService()
