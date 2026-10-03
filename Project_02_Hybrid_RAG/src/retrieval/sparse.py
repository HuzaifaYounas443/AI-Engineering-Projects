"""BM25 sparse retriever — keyword-based ranking.

BM25 scores chunks by how often query terms appear in them, weighted by
term rarity across the corpus. Complements dense retrieval, which matches
meaning rather than exact words.
"""
from pathlib import Path

from rank_bm25 import BM25Okapi

from chunk_store import load_chunks


def _tokenize(text: str) -> list[str]:
    """Simple lowercase whitespace tokenizer.

    Good enough for English content. For multilingual or code-heavy corpora,
    swap in a proper tokenizer (spaCy, nltk, etc.).
    """
    return text.lower().split()


class BM25Retriever:
    """Keyword-based retriever using Okapi BM25."""

    def __init__(self, chunks: list[dict] | None = None):
        """Build the BM25 index.

        Args:
            chunks: Optional pre-loaded chunks. If None, loads from
                chunks_store.json (the file saved during ingestion).
        """
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise ValueError("Cannot build BM25 index from empty chunks")

        self.tokenized_corpus = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return the top_k chunks most relevant to query.

        Args:
            query: The user's search string.
            top_k: Number of chunks to return.

        Returns:
            List of dicts with keys: text, source, type, chunk_index, score.
            Sorted by score descending.
        """
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair chunks with scores, sort by score, take top_k
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        return [
            {
                "text": c["text"],
                "source": c["source"],
                "type": c["type"],
                "chunk_index": c["chunk_index"],
                "score": float(s),
            }
            for c, s in ranked
        ]