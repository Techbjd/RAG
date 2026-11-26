"""Book RAG HTTP endpoints: upload/index a book and ask questions.

All tunables (upload directory, allowed suffixes, size limit, top_k, ...) come
from ``app.core.config.Settings``, which reads ``.env``. Routes only read them
from ``request.app.state.settings``.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.db.models import Document
from app.rag.pipeline import index_book
from app.rag.step4_store import reset_index
from app.rag.step5_answer import answer_question
from app.schemas.rag import (
    AnswerOut,
    DocumentOut,
    IngestOut,
    QueryRequest,
    ResetOut,
    SourceOut,
)

router = APIRouter()

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(filename: str | None) -> str:
    """Strip directories and unsafe characters from an uploaded filename."""
    name = Path(filename or "upload").name
    name = _UNSAFE_CHARS.sub("_", name).strip("._")
    return name or "upload"


@router.post("/books", response_model=IngestOut, status_code=201)
def upload_book(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Save an uploaded book, then run Steps 1-4 against it."""
    settings = request.app.state.settings

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_upload_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type {suffix!r}",
        )

    dest = settings.raw_dir / _safe_name(file.filename)
    dest.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    size = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if max_bytes > 0 and size > max_bytes:
                break
            out.write(chunk)

    if max_bytes > 0 and size > max_bytes:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {settings.max_upload_mb} MB limit",
        )

    try:
        result = index_book(dest, settings=settings, session=session)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestOut(
        document_id=result.stored.document_id,
        title=result.book.title,
        chapters=len(result.book.chapters),
        chunks=len(result.chunks),
        dim=result.embedded.dim,
        replaced=result.replaced,
    )


@router.post("/query", response_model=AnswerOut)
def query(
    request: Request,
    body: QueryRequest,
    session: Session = Depends(get_session),
):
    """Run Step 5: retrieve, rerank, and answer from the stored index."""
    settings = request.app.state.settings
    top_k = body.top_k or settings.top_k

    try:
        answer = answer_question(
            body.question,
            top_k,
            settings=settings,
            session=session,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnswerOut(
        answer=answer.answer,
        sources=[SourceOut(**source.__dict__) for source in answer.sources],
    )


@router.get("/books", response_model=list[DocumentOut])
def list_books(session: Session = Depends(get_session)):
    """List every indexed book."""
    rows = session.scalars(
        select(Document).order_by(Document.created_at)
    ).all()
    return [
        DocumentOut(id=row.id, title=row.title, source_path=row.source_path)
        for row in rows
    ]


@router.delete("/books", response_model=ResetOut)
def reset_books(session: Session = Depends(get_session)):
    """Delete every indexed book and clear the embedding signature.

    Use this after changing EMBEDDING_MODEL or EMBEDDING_DIM; otherwise
    Step 4 refuses new uploads because the stored vectors are incompatible.
    """
    deleted = session.scalar(select(func.count()).select_from(Document)) or 0
    reset_index(session)
    return ResetOut(deleted=deleted)
