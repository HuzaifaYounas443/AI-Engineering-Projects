"""Dense retriever — semantic search via Qdrant vector similarity."""
from src.ingestion import Embedder, QdrantIndexer


class DenseRetriever:
    """Vector-based retriever using the same embeddings as ingestion.

    The query is embedded with the same model that embedded the chunks,
    then Qdrant returns the top_k most similar vectors by cosine similarity.
    """

    def __init__(self, collection_name: str = "website_knowledge"):
        self.embedder = Embedder()
        self.indexer = QdrantIndexer()
        self.collection_name = collection_name

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return the top_k chunks most semantically similar to query.

        Args:
            query: The user's search string.
            top_k: Number of chunks to return.

        Returns:
            List of dicts with keys: text, source, type, chunk_index, score.
            Sorted by cosine similarity descending.
        """
        # Embed the query using the same model used during ingestion
        vector = self.embedder.embed([query])[0]

        # Query Qdrant (newer API uses query_points, not search)
        response = self.indexer.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": point.payload["text"],
                "source": point.payload["source"],
                "type": point.payload.get("type", ""),
                "chunk_index": point.payload.get("chunk_index", -1),
                "score": float(point.score),
            }
            for point in response.points
        ]