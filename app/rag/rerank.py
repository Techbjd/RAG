from __future__ import annotations

from typing import Sequence

from app.rag.retrieval import Candidate


def load_reranker(model_name: str):
    """Load a cross-encoder reranker on CPU.

    This is loaded once in the FastAPI lifespan, not per request. The reranker
    is independent of the embedding model; it never touches stored vectors.
    """



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
    ...
