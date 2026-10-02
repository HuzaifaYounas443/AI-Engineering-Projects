"""Store chunks + vectors in Qdrant."""
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import COLLECTION_NAME, QDRANT_URL, VECTOR_DIM


class QdrantIndexer:
    """Handles Qdrant collection + upsert."""

    def __init__(self, url: str = QDRANT_URL):
        self.client = QdrantClient(url=url, timeout=120)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME in existing:
            return
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks / vectors length mismatch")

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=vec,
                payload={
                    "text": ch["text"],
                    "source": ch["source"],
                    "type": ch["type"],
                    "chunk_index": ch["chunk_index"],
                },
            )
            for ch, vec in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)