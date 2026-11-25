"""Hybrid retrieval used by Step 5.

Input comes from:
    Step 4 load_index() -> chunks + row-aligned vectors
    Step 5 _embed_question() -> query vector

Output:
    list[Candidate] ranked by alpha * cosine + (1 - alpha) * normalized BM25,
    then deduplicated for the answer step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rank_bm25 import BM25Okapi

from app.rag.step2_chunk import Chunk
import re


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
    values =[float(score) for score in scores]
    if not values:
        return[]

    low,high=min(values),max(values)
    if high-low <=1e-12:
        return [0.0]*len(values)

    return [(value-low)/(high-low) for value in values]


#make sure that the  text having upper case ans lowecase same and no concatianton bettwen sentence
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    n_chunks=len(chunks)
    if n_chunks == 0:
        return []
    vectors =np.asarray(vectors,dtype=np.float32)
    if vectors.ndim !=2 or vectors.shape[0] != n_chunks:
        raise ValueError(
            f"vectors must have shape ({n_chunks}, dim), got {vectors.shape}"
        )

    query_vector=np.asarray(query_vector,dtype=np.float32).reshape(-1)

    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"query vector dim {query_vector.shape[0]} "
            f"!= index dim {vectors.shape[1]}"
        )
# cosine similarity dot product of unit vectors
    unit_vectors = _unit_rows(vectors)
    query_norm = float(np.linalg.norm(query_vector))
    unit_query = query_vector / max(query_norm, 1e-12)
    vector_scores = unit_vectors @ unit_query



    corpus =[_tokenize(chunk.text) or ["__empty__"] for chunk in chunks]
    query_tokens=_tokenize(query)

    if query_tokens:
        bm25 = BM25Okapi(corpus)
        keyword_raw = np.asarray(
            bm25.get_scores(query_tokens), dtype=np.float32
        )
    else:
        keyword_raw = np.zeros(n_chunks, dtype=np.float32)

    keyword_scores = np.asarray(
        normalize_scores(keyword_raw.tolist()), dtype=np.float32
    )

    # 3. combine
    combined = alpha * vector_scores + (1.0 - alpha) * keyword_scores

    # 4. take the best candidate_k
    top_indices = np.argsort(-combined)[:candidate_k]

    candidates: list[Candidate] = []
    for index in top_indices:
        chunk = chunks[int(index)]
        candidates.append(
            Candidate(
                chunk_id=chunk.id,
                text=chunk.text,
                chapter=chunk.chapter,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                vector_score=float(vector_scores[index]),
                keyword_score=float(keyword_scores[index]),
                score=float(combined[index]),
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates

def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)

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
    if max_per_chapter <= 0:
        return  []
    seen_ids:set[str]=set()
    per_chapter:dict[str,int]={}
    kept:list[Candidate]=[]

    for candidate in candidates:
        if candidate.chunk_id in seen_ids:
            continue
        used =per_chapter.get(candidate.chapter,0)
        if used >= max_per_chapter:
            continue

        seen_ids.add(candidate.chunk_id)
        per_chapter[candidate.chapter]=used +1
        kept.append(candidate)

    return kept
