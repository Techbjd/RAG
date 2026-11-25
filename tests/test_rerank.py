"""Offline tests for the optional cross-encoder reranker.

A fake model supplies scores so no weights are downloaded in CI.
"""

from app.rag import rerank
from app.rag.retrieval import Candidate


def _candidate(index):
    return Candidate(
        chunk_id=f"chunk-{index}",
        text=f"passage {index}",
        chapter="Chapter One",
        page_start=1,
        page_end=1,
    )


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = []

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.scores


def test_rerank_module_exposes_public_api():
    assert callable(rerank.load_reranker)
    assert callable(rerank.rerank)


def test_rerank_orders_by_model_score_and_replaces_score():
    candidates = [_candidate(0), _candidate(1), _candidate(2)]
    model = FakeReranker([0.1, 2.0, -1.0])

    result = rerank.rerank("query", candidates, top_k=2, model=model)

    assert [candidate.chunk_id for candidate in result] == [
        "chunk-1",
        "chunk-0",
    ]
    assert [candidate.score for candidate in result] == [2.0, 0.1]
    assert model.pairs == [
        ("query", "passage 0"),
        ("query", "passage 1"),
        ("query", "passage 2"),
    ]
    assert [candidate.score for candidate in candidates] == [0.0, 0.0, 0.0]


def test_rerank_returns_empty_for_no_candidates():
    assert rerank.rerank("query", [], model=FakeReranker([])) == []
