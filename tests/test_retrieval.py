"""Offline tests for the optional hybrid retrieval module.

No embedding or reranker model is loaded: vectors are injected directly.
"""

import numpy as np
import pytest

from app.rag import retrieval
from app.rag.step2_chunk import Chunk


def _chunk(ordinal, text, chapter="Chapter One"):
    return Chunk(
        id=f"chunk-{ordinal}",
        book_id="book-1",
        ordinal=ordinal,
        chapter=chapter,
        page_start=1,
        page_end=1,
        text=text,
        n_chars=len(text),
        n_tokens=1,
    )


def test_retrieval_module_exposes_public_api():
    assert callable(retrieval.normalize_scores)
    assert callable(retrieval.hybrid_search)
    assert callable(retrieval.deduplicate_candidates)
    assert retrieval.Candidate is not None


def test_normalize_scores_maps_range_to_zero_and_one():
    assert retrieval.normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert retrieval.normalize_scores([3.0, 3.0]) == [0.0, 0.0]
    assert retrieval.normalize_scores([]) == []


def test_hybrid_search_returns_best_candidate_first():
    chunks = [
        _chunk(0, "the tide clock was built in 1887"),
        _chunk(1, "a storm arrived at midnight", "Chapter Two"),
    ]
    vectors = np.eye(2, dtype=np.float32)

    candidates = retrieval.hybrid_search(
        "who built the tide clock",
        vectors[0],
        chunks,
        vectors,
        candidate_k=2,
    )

    assert [candidate.chunk_id for candidate in candidates] == [
        "chunk-0",
        "chunk-1",
    ]
    assert candidates[0].score > candidates[1].score


def test_hybrid_search_rejects_wrong_query_dimension():
    chunks = [_chunk(0, "some text")]
    vectors = np.ones((1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="query vector dim"):
        retrieval.hybrid_search(
            "question",
            np.ones(3, dtype=np.float32),
            chunks,
            vectors,
        )


def test_hybrid_search_returns_empty_without_chunks():
    assert retrieval.hybrid_search(
        "question",
        np.ones(4, dtype=np.float32),
        [],
        np.empty((0, 4), dtype=np.float32),
    ) == []


def test_deduplicate_keeps_multiple_candidates():
    candidates = [
        retrieval.Candidate(
            chunk_id=f"chunk-{index}",
            text=f"text {index}",
            chapter=chapter,
            page_start=1,
            page_end=1,
        )
        for index, chapter in enumerate(
            ["One", "One", "One", "Two", "Two", "Two"]
        )
    ]

    kept = retrieval.deduplicate_candidates(candidates, max_per_chapter=2)

    assert [candidate.chunk_id for candidate in kept] == [
        "chunk-0",
        "chunk-1",
        "chunk-3",
        "chunk-4",
    ]


def test_deduplicate_default_cap_allows_four_per_chapter():
    candidates = [
        retrieval.Candidate(
            chunk_id=f"chunk-{index}",
            text=f"text {index}",
            chapter=chapter,
            page_start=1,
            page_end=1,
        )
        for index, chapter in enumerate(
            ["One", "One", "One", "Two", "Two", "Two"]
        )
    ]

    kept = retrieval.deduplicate_candidates(candidates)

    assert [candidate.chunk_id for candidate in kept] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
        "chunk-5",
    ]


def test_deduplicate_drops_near_duplicate_vectors():
    candidates = [
        retrieval.Candidate(
            chunk_id=f"chunk-{index}",
            text=f"text {index}",
            chapter="One",
            page_start=1,
            page_end=1,
        )
        for index in range(3)
    ]
    vectors = {
        "chunk-0": np.array([1.0, 0.0], dtype=np.float32),
        "chunk-1": np.array([0.999, 0.001], dtype=np.float32),
        "chunk-2": np.array([0.0, 1.0], dtype=np.float32),
    }

    kept = retrieval.deduplicate_candidates(
        candidates,
        vectors=vectors,
        max_per_chapter=0,
    )

    assert [candidate.chunk_id for candidate in kept] == ["chunk-0", "chunk-2"]
