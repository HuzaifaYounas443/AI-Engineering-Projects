"""Reciprocal Rank Fusion — combine multiple ranked lists into one.

Based on: Cormack, Clarke, Buettcher (2009).
"Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods."

The formula for each document is:
    RRF(d) = sum over rankers of 1 / (k + rank(d))

Where k is a constant (default 60) that dampens the effect of very high ranks.
"""
from collections import defaultdict


def _chunk_key(chunk: dict) -> tuple:
    """Identity for a chunk — used to deduplicate across retrievers.

    Two chunks are the same document if their source and chunk_index match.
    """
    return (chunk.get("source", ""), chunk.get("chunk_index", -1))


def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        *ranked_lists: Any number of ranked result lists. Each list must be
            ordered best-first. Each item must be a dict with at least
            'source' and 'chunk_index' keys.
        k: RRF constant. 60 is the standard value from the original paper.

    Returns:
        A single merged list, sorted by RRF score descending. Each item is
        the original chunk dict plus a 'rrf_score' field. Duplicates across
        lists are merged (their RRF scores are summed).
    """
    if not ranked_lists:
        raise ValueError("At least one ranked list is required")

    rrf_scores: dict[tuple, float] = defaultdict(float)
    chunk_lookup: dict[tuple, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            key = _chunk_key(chunk)
            # rank starts at 0; formula uses 1-based rank
            rrf_scores[key] += 1.0 / (k + rank + 1)
            # Keep the first-seen version of the chunk (they should be identical)
            if key not in chunk_lookup:
                chunk_lookup[key] = chunk

    # Sort by fused score descending
    merged = sorted(
        rrf_scores.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return [
        {**chunk_lookup[key], "rrf_score": score}
        for key, score in merged
    ]