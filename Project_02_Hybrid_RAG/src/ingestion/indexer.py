"""Store chunks + vectors in Qdrant.

Handles:
  - Collection creation
  - Batched upsert
  - Collection reset (for clean re-ingestion)
"""
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import COLLECTION_NAME, QDRANT_URL, VECTOR_DIM


class QdrantIndexer:
    """Qdrant client wrapper for the RAG pipeline.

    Attributes:
        client: The underlying QdrantClient instance.
    """

    def __init__(self, url: str = QDRANT_URL):
        """Connect to Qdrant and ensure the collection exists.

        Args:
            url: Qdrant server URL. Defaults to config.QDRANT_URL.
        """
        self.client = QdrantClient(
            url=url,
            timeout=120,              # generous timeout for slow Docker on Windows
            check_compatibility=False, # skip version warning noise
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME in existing:
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(f"  ✓ Created collection '{COLLECTION_NAME}'")

    def reset(self) -> None:
        """Drop the collection and recreate it empty.

        Use this before re-ingesting to prevent duplicate points. Safe to
        call even if the collection doesn't exist yet.
        """
        try:
            self.client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"  ✓ Dropped collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # collection didn't exist — nothing to drop
        self._ensure_collection()

    def upsert(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
        batch_size: int = 64,
    ) -> int:
        """Insert or update points in Qdrant.

        Args:
            chunks: List of chunk dicts. Each must have keys:
                text, source, type, chunk_index.
            vectors: List of embedding vectors. Must match len(chunks).
            batch_size: Number of points per upsert call.

        Returns:
            Total number of points written.

        Raises:
            ValueError: If chunks and vectors have mismatched lengths.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) "
                "must have the same length"
            )

        written = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]

            points = [
                PointStruct(
                    id=str(uuid4()),
                    vector=vec,
                    payload={
                        "text": ch["text"],
                        "source": ch["source"],
                        "type": ch.get("type", ""),
                        "chunk_index": ch.get("chunk_index", -1),
                    },
                )
                for ch, vec in zip(batch_chunks, batch_vectors)
            ]

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            written += len(points)

        return written