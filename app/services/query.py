"""Query orchestration: short DB phase, then CPU/LLM answering.

Keeping the DB phase here lets routes stay thin without changing how answers
are produced: the index is loaded with a session that is closed before
inference, then ``answer_question`` works on the detached snapshot.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.rag.step5_answer import Answer, answer_question, load_indexes


def ask_question(
    question: str,
    top_k: int,
    session_factory: Callable[[], Session],
    settings: Settings,
) -> Answer:
    """Load the index in a short session and answer from the snapshot."""
    with session_factory() as session:
        indexes = load_indexes(session)

    return answer_question(question, top_k, settings=settings, indexes=indexes)
