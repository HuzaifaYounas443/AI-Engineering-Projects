"""Hybrid retriever — combines dense and sparse retrieval with RRF.

Why hybrid:
  - Dense handles meaning/paraphrase ("change credentials" → "reset password")
  - Sparse handles exact terms ("Starter Plan", error codes, product names)
  - Neither alone is optimal; fusing them improves both recall and precision.
"""
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .sparse import BM25Retriever


class HybridRetriever:
    """Dense + Sparse retriever fused with Reciprocal Rank Fusion.

    Retrieves from both sources independently, then merges the two ranked
    lists into a single fused ranking using RRF.
    """

    def __init__(
        self,
        dense: DenseRetriever | None = None,
        sparse: BM25Retriever | None = None,
        rrf_k: int = 60,
    ):
        """Initialize the hybrid retriever.

        Args:
            dense: DenseRetriever instance. If None, one is created.
            sparse: BM25Retriever instance. If None, one is created.
            rrf_k: RRF constant (standard value: 60).
        """
        self.dense = dense if dense is not None else DenseRetriever()
        self.sparse = sparse if sparse is not None else BM25Retriever()
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        per_retriever_k: int = 20,
        return_sources: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        """Search using both retrievers and fuse results.

        Args:
            query: The user's search string.
            top_k: Number of chunks to return after fusion.
            per_retriever_k: How many chunks each retriever returns
                internally before fusion. Larger = better recall, more compute.
            return_sources: If True, also returns the individual dense/sparse
                result lists for debugging and inspection.

        Returns:
            If return_sources=False: list of top_k fused chunks.
            If return_sources=True: (fused_chunks, {"dense": [...], "sparse": [...]}).
        """
        dense_results = self.dense.search(query, top_k=per_retriever_k)
        sparse_results = self.sparse.search(query, top_k=per_retriever_k)

        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=self.rrf_k,
        )

        top = fused[:top_k]

        if return_sources:
            return top, {"dense": dense_results, "sparse": sparse_results}
        return top