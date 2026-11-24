from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.rag.retrieval import Candidate
from app.rag.step3_embed import load_embedder
from app.rag.step4_store import load_index
from google import genai

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
    """Encode the question with the same frozen embedder used for chunks."""

    question=question.strip()

    if not question:
        raise ValueError("question mus no tbe empty")
    if settings.embedding_backend != "sentence-trandformers":
        raise ValueError(
            f"unsupported embedding_backend={settings.embedding_backend!r}"
        )
    model=load_embedder(settings.embedding_model) #load same encoder used in step 3 and embedded the question and save the normalized format in the 1D vector

    vector=model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    return np.asarray(vector,dtype=np.float32)

#candiate is a retrival object and source is what api shows to user
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
    if not Source:
        return REFUSAL
    return sources[0].text



def _llm_answer(question: str, sources: list[Source], settings: Settings) -> str:
    """Ask the configured LLM to answer only from numbered passages."""

    if settings.llm_backend =="extractive":
        return _extractive_answer(sources)

    if not sources:
        return REFUSAL

    passages=[]

    for number,source in enumerate(source,start=1):
        if source.page_start is None :
            pages="pages unknown"

        elif source.page.end is None or source.page_end ==source.page_Start:
            pages=f"pages{source.page_start}"

        else:
            pages = f"pages {source.page_start}-{source.page_end}"

        passages.append(
            f"[{number}] {source.chapter} ({pages})\n{source.text})"
        )

        system = (
        "You are a careful book assistant. Answer only from the passages below. "
        "Cite every claim with the passage number, like [1]. "
        'If the passages do not contain the answer, say "The book does not say." '
        "Never use outside knowledge."
    )
        prompt = "\n\n".join(passages) + f"\n\nQuestion: {question}"
        if settings.llm_backend == "gemini":
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set")



        client = genai.Client(api_key=settings.gemini_api_key)

        response = client.models.generate_content(
            model=settings.llm_model,
            contents=f"{system}\n\n{prompt}",
        )

        text = (response.text or "").strip()
    else:
        raise ValueError(
            f"unsupported llm_backend ={settings.llm_backend}"
        )
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
        settings=get_settings()

    if session is None:
        raise ValueError(
            f" session is required ; pass a Session  from the Api layer "
        )

    if top_k <= 0:
        raise ValueError("top_K must be grater that zero")

    #load every indexed book
    document_ids =session.scalars(select(document.id)).all()
    indexes =[
        load_index(session,document_id)
        for document_id in document_ids
    ]
    if not indexes:
        return Answer(answer=REFUSAL,source=[])

    chunks=[
        chunk
        for index in indexes
        for chunk in index.chunks
    ]

    vectors=np.vstack([index.vectors for index in indexes])
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
        top_k=max(top_k * 2, top_k),
    )
    # 6. threshold + deduplicate
    filtered = [
        candidate
        for candidate in reranked
        if candidate.score >= settings.min_score
    ]
    final = deduplicate_candidates(filtered)[:top_k]

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
