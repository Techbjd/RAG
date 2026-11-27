"""HTTP endpoints for uploading, listing, and resetting books."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.schemas.rag import DocumentOut, IngestOut, ResetOut
from app.services.books import (
    FileTooLarge,
    UnsupportedFileType,
    delete_documents,
    index_upload,
    list_documents,
)

router = APIRouter()


@router.post("/books", response_model=IngestOut, status_code=201)
def upload_book(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Save an uploaded book, then run Steps 1-4 against it."""
    settings = request.app.state.settings

    try:
        result = index_upload(file.file, file.filename, settings, session)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
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


@router.get("/books", response_model=list[DocumentOut])
def get_books(session: Session = Depends(get_session)):
    """List every indexed book."""
    return [
        DocumentOut(id=row.id, title=row.title, source_path=row.source_path)
        for row in list_documents(session)
    ]


@router.delete("/books", response_model=ResetOut)
def reset_books(session: Session = Depends(get_session)):
    """Delete every indexed book and clear the embedding signature.

    Use this after changing EMBEDDING_MODEL or EMBEDDING_DIM; otherwise
    Step 4 refuses new uploads because the stored vectors are incompatible.
    """
    return ResetOut(deleted=delete_documents(session))
