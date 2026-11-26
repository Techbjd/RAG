"""Step 5 - Answer: question + stored index -> grounded answer + sources.

Where the input comes from:
    question : the user's text, passed by the API layer (POST /api/query)
    session  : a SQLAlchemy Session owned by the API layer
    settings : app.core.config.Settings

What it reads:
    SQLite rows written by Step 4, loaded through step4_store.load_index().

Pipeline:
    1. load every stored document index
    2. embed the question with the same model used in Step 3
    3. hybrid retrieval (vector + BM25) -> Candidate list
    4. cross-encoder rerank
    5. score threshold + deduplicate -> final passages
    6. answer from those passages (extractive or Gemini), or refuse

What it returns:
    Answer(answer: str, sources: list[Source]) for the API to serialize.

Design rule: answer only from retrieved passages. If nothing clears the
threshold, return REFUSAL instead of using outside knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from google import genai

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.models import Document
from app.rag.rerank import rerank
from app.rag.retrieval import Candidate, deduplicate_candidates, hybrid_search
from app.rag.step3_embed import load_embedder
from app.rag.step4_store import load_index
from google.genai import errors as genai_errors
from google.genai import errors as genai_errors
import httpx
import logging

# Create or get a logger instance
logger = logging.getLogger(__name__)
@dataclass
class Source:
    """A citable passage used in an answer.

    Attributes:
        chunk_id: Stable chunk ID from Step 2.
        chapter: Chapter title the passage came from.
        page_start: First page covered by the chunk, if known.
        page_end: Last page covered by the chunk, if known.
        score: Final score after reranking/thresholding.
        text: Display text for the passage.
    """

    chunk_id: str
    chapter: str
    page_start: int | None
    page_end: int | None
    score: float
    text: str


@dataclass
class Answer:
    """The final answer plus its supporting sources."""

    answer: str
    sources: list[Source]


def _embed_question(question: str, settings: Settings) -> np.ndarray:
    """Encode the question with the same frozen embedder used for chunks.

    Input comes from answer_question(). The encoder is the Step 3 embedder
    (step3_embed.load_embedder), so the query vector lives in the same space
    as the stored vectors.
    """

    question = question.strip()

    if not question:
        raise ValueError("question must not be empty")
    if settings.embedding_backend != "sentence-transformers":
        raise ValueError(
            f"unsupported embedding_backend={settings.embedding_backend!r}"
        )

    # load the same encoder used in Step 3 and return a normalized 1D vector
    model = load_embedder(settings.embedding_model)
    vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    return np.asarray(vector, dtype=np.float32)


# Candidate is a retrieval object; Source is what the API shows the user.
def _source_from_candidate(candidate: Candidate) -> Source:
    """Convert a reranked ``Candidate`` into a citable ``Source``."""
    return Source(
        chunk_id=candidate.chunk_id,
        chapter=candidate.chapter,
        page_start=candidate.page_start,
        page_end=candidate.page_end,
        score=float(candidate.score),
        text=candidate.text,
    )


REFUSAL = "That does not appear in this book."


def _extractive_answer(sources: list[Source]) -> str:
    """Return the best passage when ``LLM_BACKEND=extractive``."""
    if not sources:
        return REFUSAL
    return sources[0].text


def _llm_answer(question: str, sources: list[Source], settings: Settings) -> str:
    """Ask the configured LLM to answer only from numbered passages.

    Input comes from answer_question(): the already-filtered Source list
    (Step 5 retrieval output) and the original question. Each source becomes
    a numbered passage "[n] chapter (pages)\\ntext"; the model is instructed
    to cite those numbers and to refuse when the passages do not answer.
    """

    if settings.llm_backend == "extractive":
        return _extractive_answer(sources)

    if not sources:
        return REFUSAL

    passages = []
    for number, source in enumerate(sources, start=1):
        if source.page_start is None:
            pages = "pages unknown"
        elif source.page_end is None or source.page_end == source.page_start:
            pages = f"page {source.page_start}"
        else:
            pages = f"pages {source.page_start}-{source.page_end}"

        passages.append(
            f"[{number}] {source.chapter} ({pages})\n{source.text}"
        )

    system = (
        "You are a careful book assistant. Answer only from the passages below. "
        "Cite every claim with the passage number, like [1]. "
        'If the passages do not contain the answer, say "The book does not say." '
        "Never use outside knowledge. "
        "Answer in at most 5 sentences or 3 bullets."
    )
    prompt = "\n\n".join(passages) + f"\n\nQuestion: {question}"

    if settings.llm_backend == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        try:
            client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options=genai.types.HttpOptions(
                    timeout=30_000,  # milliseconds
                    retry_options=genai.types.HttpRetryOptions(
                        attempts=2,
                        initial_delay=0.5,
                        max_delay=2.0,
                        exp_base=2.0,
                        jitter=0.2,
                        http_status_codes=[429, 500, 502, 503, 504],
                    ),
                ),
            )
            response = client.models.generate_content(
                model=settings.llm_model,
                contents=f"{system}\n\n{prompt}",
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=1024,
                    temperature=0.2,
                    # thinking_config=genai.types.ThinkingConfig(
                    #     thinking_level=genai.types.ThinkingLevel.MINIMAL,
                    # ),
                ),
            )
            text = (response.text or "").strip()
        except (genai_errors.APIError , httpx.TimeoutException):
                logger.warning("Gemini failed, using extractive fallback: %s", genai_errors.APIError )
                return _extractive_answer(sources)

    else:
        raise ValueError(f"unsupported llm_backend={settings.llm_backend!r}")

    if not text:
        return REFUSAL
    return text



def answer_question(
    question: str,
    top_k: int = 5,
    *,
    settings: Settings | None = None,
    session: Session | None = None,
) -> Answer:
    """Retrieve, rerank, threshold, and answer with sources.

    Inputs:
        question: user question from the API layer.
        top_k: how many final passages to keep/send to the answer model.
        settings: app settings (defaults to get_settings()).
        session: SQLAlchemy Session from the API layer (required).

    Data flow:
        Step 4 rows -> load_index() -> chunks + vectors
        Step 3 embedder -> query vector
        retrieval.hybrid_search() -> wide Candidate list
        rerank.rerank() -> narrow Candidate list
        min_score + dedupe -> final Source list
        extractive/Gemini -> Answer

    Workflow:
        1. Load the stored index.
        2. Embed the question.
        3. Run hybrid vector + keyword search.
        4. Rerank candidates.
        5. Apply score threshold and deduplication.
        6. Return a refusal if nothing clears the threshold.
        7. Otherwise build an extractive or LLM-grounded answer.
    """
    if settings is None:
        settings = get_settings()

    if session is None:
        raise ValueError(
            "session is required; pass a Session from the API layer"
        )

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    # load every indexed book
    document_ids = session.scalars(select(Document.id)).all()
    indexes = [
        load_index(session, document_id)
        for document_id in document_ids
    ]
    indexes = [index for index in indexes if index is not None]
    if not indexes:
        return Answer(answer=REFUSAL, sources=[])

    chunks = [
        chunk
        for index in indexes
        for chunk in index.chunks
    ]

    vectors = np.vstack([index.vectors for index in indexes])

    # 3. embed the question
    query_vector = _embed_question(question, settings)

    # 4. retrieve wide
    candidates = hybrid_search(
        question,
        query_vector,
        chunks,
        vectors,
        candidate_k=settings.candidate_k,
        alpha=settings.hybrid_alpha,
    )

    # 5. rerank narrow, with a little spare room for deduplication
    reranked = rerank(
        question,
        candidates,
        top_k=settings.candidate_k,
    )

    # 6. threshold + deduplicate
    filtered = [
        candidate
        for candidate in reranked
        if candidate.score >= settings.min_score
    ]
    vector_by_id = {chunk.id: vectors[i] for i, chunk in enumerate(chunks)}
    final = deduplicate_candidates(
        filtered,
        vectors=vector_by_id,
        max_per_chapter=settings.max_per_chapter,
        similarity_threshold=settings.dedupe_similarity,
    )[:top_k]

    if not final:
        return Answer(answer=REFUSAL, sources=[])

    # 7. answer
    sources = [
        _source_from_candidate(candidate)
        for candidate in final
    ]

    if settings.llm_backend == "extractive":
        text = _extractive_answer(sources)
    else:
        text = _llm_answer(question, sources, settings)

    return Answer(answer=text, sources=sources)
