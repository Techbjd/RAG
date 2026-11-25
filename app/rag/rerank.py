"""Cross-encoder reranking used by Step 5.

Input comes from:
    retrieval.hybrid_search() -> wide Candidate list

Output:
    the best ``top_k`` Candidates, with ``score`` replaced by the
    cross-encoder score. Step 5 then applies min_score and deduplication.
"""

from __future__ import annotations

from typing import Sequence
from app.core.config import get_settings
from app.rag.retrieval import Candidate
from functools import lru_cache
from sentence_transformers import CrossEncoder
import numpy as np
from rank_bm25 import BM25Okapi

@lru_cache(maxsize=2)
def _load(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name, device="cpu")




def load_reranker(model_name: str):
    """Load a cross-encoder reranker on CPU.

    This is loaded once in the FastAPI lifespan, not per request. The reranker
    is independent of the embedding model; it never touches stored vectors.
    """
    return _load(model_name)



def rerank(
    query: str,
    candidates: Sequence[Candidate],
    *,
    top_k: int = 5,
    model=None,
) -> list[Candidate]:
    """Reorder candidates with a cross-encoder and keep the best ``top_k``.

    Each ``(query, candidate.text)`` pair is scored jointly by the
    cross-encoder, which is more precise than bi-encoder cosine similarity.
    The returned candidates keep their original metadata; their ``score`` is
    replaced with the reranker score. Pass a fake ``model`` in tests to avoid
    downloading weights.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    candidates = list(candidates)
    if not candidates:
        return []

    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    if model is None:
        model = load_reranker(get_settings().reranker_model)

    pairs = [(query, candidate.text) for candidate in candidates]
    raw_scores = np.asarray(model.predict(pairs), dtype=np.float64).reshape(-1)

    if raw_scores.size != len(candidates):
        raise ValueError(
            f"reranker returned {raw_scores.size} scores "
            f"for {len(candidates)} candidates"
        )

    rescored = [
        rescored(candidate, score=float(score))
        for candidate, score in zip(candidates, raw_scores)
    ]
    rescored.sort(key=lambda candidate: candidate.score, reverse=True)
    return rescored[:top_k]
