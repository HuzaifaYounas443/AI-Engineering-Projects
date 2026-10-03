"""Step 9a: Semantic search over the Qdrant index."""
from src.ingestion import Embedder, QdrantIndexer


def search(question: str, top_k: int = 5) -> list[dict]:
    """Embed the question and find the most similar chunks."""
    embedder = Embedder()
    vector = embedder.embed([question])[0]

    indexer = QdrantIndexer()
    
    # query_points replaces the removed search() method
    results = indexer.client.query_points(
        collection_name="website_knowledge",
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    
    # query_points returns a QueryResponse; access .points
    return [
        {
            "score": r.score,
            "text": r.payload["text"],
            "source": r.payload["source"],
            "chunk_index": r.payload["chunk_index"],
        }
        for r in results.points
    ]


if __name__ == "__main__":
    question = "How do I reset my password?"
    print(f"Question: {question}\n")

    results = search(question, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f}")
        print(f"    Source: {r['source']}")
        print(f"    Text: {r['text'][:200]}...")
        print()