from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.rag.step2_chunk import Chunk



@dataclass
class Candidate:
    """A retrieved chunk before or after hybrid scoring.

    Attributes:
        chunk_id: Stable chunk identifier (``Chunk.id``).
        text: Chunk text content.
        chapter: Chapter title the chunk came from.
        page_start: First page covered by the chunk, if known.
        page_end: Last page covered by the chunk, if known.
        vector_score: Cosine similarity against the query vector.
        keyword_score: BM25 score, normalized to ``[0, 1]``.
        score: Combined hybrid score.
    """

    chunk_id: str
    text: str
    chapter: str
    page_start: int | None
    page_end: int | None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    score: float = 0.0


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalize scores to ``[0, 1]``.

    BM25 scores are unbounded, so they must be normalized before being mixed
    with cosine similarities. If every score is equal, return zeros.
    """
    ...



def hybrid_search(
    query: str,
    query_vector: np.ndarray,
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    *,
    candidate_k: int = 30,
    alpha: float = 0.6,
) -> list[Candidate]:
    """Retrieve candidates with vector search plus BM25 keyword search.

    ``chunks`` and ``vectors`` must be row-aligned (one vector per chunk).
    Vector scores and normalized BM25 scores are combined as
    ``alpha * cosine + (1 - alpha) * bm25_normalized``. Returns at most
    ``candidate_k`` candidates, highest combined score first.
    """
    ...


def deduplicate_candidates(
    candidates: Sequence[Candidate],
    *,
    max_per_chapter: int = 2,
) -> list[Candidate]:
    """Drop near-duplicate candidates while preserving score order.

    Adjacent chunks overlap, so the same passage can appear several times in
    the top results. Keeping a cap per chapter (or by page range) prevents the
    answer model from receiving five copies of the same text.
    """
    ...
