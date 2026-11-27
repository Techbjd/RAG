"""Application logic for uploaded books (no HTTP or status-code concerns).

Routes pass in a binary stream, filename, settings, and a SQLAlchemy session;
this module owns file storage, validation, and indexing. Errors raised here
are translated into HTTP responses by the route layer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Document
from app.rag.pipeline import index_book
from app.rag.step4_store import reset_index

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class UnsupportedFileType(ValueError):
    """The uploaded suffix is not in the configured allowlist."""


class FileTooLarge(ValueError):
    """The uploaded file exceeds the configured size limit."""


def safe_name(filename: str | None) -> str:
    """Strip directories and unsafe characters from an uploaded filename."""
    name = Path(filename or "upload").name
    name = _UNSAFE_CHARS.sub("_", name).strip("._")
    return name or "upload"


def save_upload(stream: BinaryIO, filename: str | None, settings: Settings) -> Path:
    """Validate and stream one upload to ``DATA_DIR/raw``."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in settings.allowed_upload_suffixes:
        raise UnsupportedFileType(f"unsupported file type {suffix!r}")

    dest = settings.raw_dir / safe_name(filename)
    dest.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    size = 0
    with dest.open("wb") as out:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if max_bytes > 0 and size > max_bytes:
                break
            out.write(chunk)

    if max_bytes > 0 and size > max_bytes:
        dest.unlink(missing_ok=True)
        raise FileTooLarge(f"file exceeds {settings.max_upload_mb} MB limit")

    return dest


def index_upload(
    stream: BinaryIO,
    filename: str | None,
    settings: Settings,
    session: Session,
):
    """Save an upload and run Steps 1-4; return the pipeline result."""
    path = save_upload(stream, filename, settings)
    return index_book(path, settings=settings, session=session)


def list_documents(session: Session) -> list[Document]:
    """Return indexed documents in creation order."""
    return list(session.scalars(select(Document).order_by(Document.created_at)).all())


def delete_documents(session: Session) -> int:
    """Delete every indexed book and return the number removed."""
    deleted = session.scalar(select(func.count()).select_from(Document)) or 0
    reset_index(session)
    return deleted
