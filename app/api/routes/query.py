"""HTTP endpoint for asking a question against the stored index."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.rag import AnswerOut, QueryRequest, SourceOut
from app.services.query import ask_question

router = APIRouter()


@router.post("/query", response_model=AnswerOut)
def query(request: Request, body: QueryRequest):
    """Retrieve, rerank, and answer; never hold a DB connection while answering."""
    settings = request.app.state.settings
    top_k = body.top_k or settings.top_k

    try:
        answer = ask_question(
            body.question,
            top_k,
            request.app.state.session_factory,
            settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnswerOut(
        answer=answer.answer,
        sources=[SourceOut(**source.__dict__) for source in answer.sources],
    )
