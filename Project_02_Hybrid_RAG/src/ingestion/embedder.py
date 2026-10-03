"""Generate embeddings with fastembed."""
from fastembed import TextEmbedding

from .config import EMBEDDING_MODEL


class Embedder:
    """Wraps fastembed for bge-small."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Cannot embed empty list")
        return [v.tolist() for v in self.model.embed(texts)]